import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd

from scipy.signal import find_peaks

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


RANDOM_STATE = 42
TARGET_EVENTS = 1594
N_SPLITS = 5
NEG_RATIO = 10
MIN_DISTANCE = 3
NMS_WINDOW = 4
SMOOTH_WINDOW = 2
WINDOW_HARD = 8
WINDOW_REFINE = 5

USE_FIXED_TIME_THRESHOLD = True
BEST_TIME_THRESHOLD = 0.10


print("LOAD DATA")

train_df = pd.read_parquet("train.parquet")
test_df = pd.read_parquet("test.parquet")
video_meta = pd.read_csv("videos.csv")

print("TRAIN:", train_df.shape)
print("TEST:", test_df.shape)
print("META:", video_meta.shape)

print("\nTrain is_punch distribution:")
print(train_df["is_punch"].value_counts(normalize=True))

train_df = train_df.fillna(0)
test_df = test_df.fillna(0)


meta_cols = [
    "video_key",
    "video_id",
    "agn_index",
    "round_number",
    "dataset_type",
    "fight_index",
    "frame_count",
    "width",
    "height",
]

video_meta = video_meta[meta_cols]

train_df = train_df.merge(video_meta, on="video_key", how="left")
test_df = test_df.merge(video_meta, on="video_key", how="left")

print("\nAfter merge:")
print("TRAIN:", train_df.shape)
print("TEST:", test_df.shape)


def add_features(df):

    df = df.copy()

    group_cols = ["video_key", "fighter"]
    if "track_id" in df.columns:
        group_cols.append("track_id")

    df = df.sort_values(group_cols + ["frame"])

    if {"center_x", "center_y", "bbox_h"}.issubset(df.columns):
        cx = df["center_x"].values
        cy = df["center_y"].values
        bh = df["bbox_h"].values + 1e-3

        for i in range(17):
            x_col = f"kp_{i}_x"
            y_col = f"kp_{i}_y"

            if x_col in df.columns:
                df[f"{x_col}_rel"] = (df[x_col].values - cx) / bh

            if y_col in df.columns:
                df[f"{y_col}_rel"] = (df[y_col].values - cy) / bh

    if "frame_count" in df.columns:
        fn = df["frame"] / (df["frame_count"] + 1e-3)
        df["frame_norm"] = fn
        df["frame_sin"] = np.sin(2 * np.pi * fn)
        df["frame_cos"] = np.cos(2 * np.pi * fn)

    numeric_cols = [
        c for c in df.columns
        if df[c].dtype.kind in "biufc" and c not in ["is_punch"]
    ]

    grouped = df.groupby(group_cols, sort=False)

    for col in numeric_cols:
        s = df[col]
        prev = grouped[col].shift(1)
        nxt = grouped[col].shift(-1)

        df[f"{col}_prev"] = prev
        df[f"{col}_next"] = nxt
        df[f"{col}_d_prev"] = s - prev
        df[f"{col}_d_next"] = nxt - s

    important_cols = [
        c for c in numeric_cols
        if ("speed" in c or "distance" in c or "accel" in c)
    ]

    for col in important_cols:
        g = grouped[col]

        roll3 = g.rolling(3, center=True, min_periods=1)
        roll5 = g.rolling(5, center=True, min_periods=1)
        roll7 = g.rolling(7, center=True, min_periods=1)
        roll9 = g.rolling(9, center=True, min_periods=1)

        df[f"{col}_roll_mean_3"] = roll3.mean().reset_index(level=group_cols, drop=True)
        df[f"{col}_roll_std_3"]  = roll3.std().reset_index(level=group_cols, drop=True)

        df[f"{col}_roll_mean_5"] = roll5.mean().reset_index(level=group_cols, drop=True)
        df[f"{col}_roll_std_5"]  = roll5.std().reset_index(level=group_cols, drop=True)

        df[f"{col}_roll_mean_7"] = roll7.mean().reset_index(level=group_cols, drop=True)
        df[f"{col}_roll_std_7"]  = roll7.std().reset_index(level=group_cols, drop=True)

        df[f"{col}_roll_mean_9"] = roll9.mean().reset_index(level=group_cols, drop=True)
        df[f"{col}_roll_std_9"]  = roll9.std().reset_index(level=group_cols, drop=True)

    key_ids = [0, 5, 6, 7, 8, 9, 10]
    for i in key_ids:
        x_col = f"kp_{i}_x"
        y_col = f"kp_{i}_y"
        if x_col in df.columns and y_col in df.columns:
            dx = grouped[x_col].diff()
            dy = grouped[y_col].diff()
            df[f"kp_{i}_speed"] = np.sqrt(dx**2 + dy**2).fillna(0)

    df = df.fillna(0)
    return df


print("\nAdding advanced features...")
train_df = add_features(train_df)
test_df = add_features(test_df)
print("TRAIN with features:", train_df.shape)
print("TEST with features:", test_df.shape)


print("HARD NEGATIVE MINING")

punches = train_df[train_df["is_punch"] == 1].copy()
punch_idx = punches.index

hard_mask = np.zeros(len(train_df), dtype=bool)
for idx in punch_idx:
    left = max(0, idx - WINDOW_HARD)
    right = min(len(train_df) - 1, idx + WINDOW_HARD)
    hard_mask[left:right+1] = True

hard_neg = train_df[hard_mask & (train_df["is_punch"] == 0)].copy()
easy_neg = train_df[~hard_mask & (train_df["is_punch"] == 0)].copy()

easy_neg_sample = easy_neg.sample(
    min(len(easy_neg), len(punches) * (NEG_RATIO - 1)),
    random_state=RANDOM_STATE
)

detector_df = pd.concat([punches, hard_neg, easy_neg_sample])
detector_df = detector_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print("Punches:", len(punches))
print("Hard negatives:", len(hard_neg))
print("Easy negatives sample:", len(easy_neg_sample))
print("Detector DF:", detector_df.shape)

print("\nDetector is_punch distribution:")
print(detector_df["is_punch"].value_counts(normalize=True))


drop_cols = [
    "is_punch",
    "fighter",
    "hand",
    "target",
    "effectiveness",
    "clear",
    "punch_type",
    "video_id",
    "agn_index",
]

feature_cols = [
    c for c in train_df.columns
    if (c not in drop_cols and train_df[c].dtype.kind in "biufc")
]

print("\nFEATURE COUNT:", len(feature_cols))

X = detector_df[feature_cols]
y = detector_df["is_punch"].astype(int)

X_test = test_df[feature_cols]

print("X shape:", X.shape)
print("X_test shape:", X_test.shape)


def build_models(pos_weight):

    cat = CatBoostClassifier(
        iterations=1200,
        depth=8,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=False,
        task_type="GPU",
        devices="0",
        auto_class_weights="Balanced"
    )

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=pos_weight
    )

    lgbm = LGBMClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        device_type="gpu",
        verbosity=1,
        scale_pos_weight=pos_weight
    )

    return cat, xgb, lgbm


print("CV TRAINING")

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

cat_models = []
xgb_models = []
lgbm_models = []

oof_proba = np.zeros(len(X))

pos_weight = (y == 0).sum() / max(1, (y == 1).sum())
print(f"\nEstimated positive weight for XGB/LGBM: {pos_weight:.2f}")

for fold, (tr, va) in enumerate(skf.split(X, y)):

    print(f"\nFOLD {fold+1}/{N_SPLITS}")
    print("Train size:", len(tr), "Val size:", len(va))

    X_tr = X.iloc[tr]
    X_va = X.iloc[va]

    y_tr = y.iloc[tr]
    y_va = y.iloc[va]

    cat, xgb, lgbm = build_models(pos_weight)

    print("  CatBoost training on GPU...")
    cat.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        verbose=100
    )

    print("  XGBoost training on GPU...")
    xgb.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=100
    )

    print("  LightGBM training on GPU...")
    lgbm.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)]
    )

    p1 = cat.predict_proba(X_va)[:, 1]
    p2 = xgb.predict_proba(X_va)[:, 1]
    p3 = lgbm.predict_proba(X_va)[:, 1]

    final = p1 * 0.4 + p2 * 0.3 + p3 * 0.3

    oof_proba[va] = final

    auc = roc_auc_score(y_va, final)
    print("  FOLD AUC:", auc)

    cat_models.append(cat)
    xgb_models.append(xgb)
    lgbm_models.append(lgbm)


full_auc = roc_auc_score(y, oof_proba)
print("\nFULL OOF AUC:", full_auc)

print("\nOOF proba stats (all):")
print(pd.Series(oof_proba).describe())

print("\nOOF proba stats by class:")
print("  positives:")
print(pd.Series(oof_proba[y.values == 1]).describe())
print("  negatives:")
print(pd.Series(oof_proba[y.values == 0]).describe())


def extract_events_from_df(df, threshold):
    events = []

    for (video_key, fighter), video_df in df.groupby(["video_key", "fighter"]):
        video_df = video_df.sort_values("frame").reset_index()

        probs = video_df["smooth_proba"].values
        raw_probs = video_df["punch_proba"].values
        frames = video_df["frame"].values

        peaks, _ = find_peaks(
            probs,
            height=threshold,
            distance=MIN_DISTANCE
        )

        for p in peaks:
            w = WINDOW_REFINE + int(raw_probs[p] > 0.5)
            left = max(0, p - w)
            right = min(len(raw_probs) - 1, p + w)
            local_idx = left + np.argmax(raw_probs[left:right+1])

            events.append({
                "video_key": video_key,
                "fighter": fighter,
                "frame": int(frames[local_idx]),
            })

    return pd.DataFrame(events)


print("TIME-OPTIMIZED THRESHOLD SEARCH")

gt_events = detector_df[detector_df["is_punch"] == 1][["video_key", "fighter", "frame"]]

detector_df = detector_df.sort_values(["video_key", "fighter", "frame"]).reset_index(drop=True)
detector_df["punch_proba"] = oof_proba

detector_df["smooth_proba"] = (
    detector_df
    .groupby(["video_key", "fighter"])["punch_proba"]
    .transform(lambda x: x.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean())
)

def evaluate_threshold(thr):
    pred_events = extract_events_from_df(detector_df, thr)

    matched = 0
    total_error = 0
    errors = []

    for _, gt in gt_events.iterrows():
        gk, gf, gf_frame = gt["video_key"], gt["fighter"], gt["frame"]

        candidates = pred_events[
            (pred_events["video_key"] == gk) &
            (pred_events["fighter"] == gf)
        ]

        if len(candidates) == 0:
            continue

        candidates["err"] = (candidates["frame"] - gf_frame).abs()
        best = candidates.loc[candidates["err"].idxmin()]

        if best["err"] <= 6:
            matched += 1
            total_error += best["err"]
            errors.append(best["err"])

    recall = matched / len(gt_events)
    mean_err = np.mean(errors) if errors else 999
    p95_err = np.percentile(errors, 95) if errors else 999
    diff_events = abs(len(pred_events) - len(gt_events))

    score = (
        recall * 3.0
        - mean_err * 0.15
        - p95_err * 0.05
        - diff_events * 0.0005
    )

    return score, recall, mean_err, p95_err, diff_events, len(pred_events)


if USE_FIXED_TIME_THRESHOLD:
    best_thr = BEST_TIME_THRESHOLD
    print(f"\n[INFO] Using FIXED time threshold: {best_thr}")
else:
    best_thr = None
    best_score = -1e9
    best_stats = None

    print("\nSearching best threshold for time metric...")

    for thr in np.arange(0.05, 0.40, 0.005):
        score, recall, mean_err, p95_err, diff_events, pred_count = evaluate_threshold(thr)

        print(
            f"TH={thr:.3f} | "
            f"score={score:.4f} | "
            f"recall={recall:.4f} | "
            f"mean_err={mean_err:.2f} | "
            f"p95={p95_err:.2f} | "
            f"pred={pred_count} | "
            f"diff={diff_events}"
        )

        if score > best_score:
            best_score = score
            best_thr = thr
            best_stats = (recall, mean_err, p95_err, diff_events, pred_count)

    print("\nBEST TIME-OPTIMIZED THRESHOLD:", best_thr)
    print("Recall:", best_stats[0])
    print("Mean error:", best_stats[1])
    print("95th percentile:", best_stats[2])
    print("Event diff:", best_stats[3])
    print("Pred events:", best_stats[4])


print("LOCAL EVENT METRIC")

gt_events = detector_df[detector_df["is_punch"] == 1][["video_key", "fighter", "frame"]]

detector_df = detector_df.sort_values(["video_key", "fighter", "frame"]).reset_index(drop=True)
detector_df["punch_proba"] = oof_proba

detector_df["smooth_proba"] = (
    detector_df
    .groupby(["video_key", "fighter"])["punch_proba"]
    .transform(lambda x: x.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean())
)

pred_events = extract_events_from_df(detector_df, best_thr)

matched = 0
total_error = 0
errors = []

for _, gt in gt_events.iterrows():
    gk, gf, gf_frame = gt["video_key"], gt["fighter"], gt["frame"]

    candidates = pred_events[
        (pred_events["video_key"] == gk) &
        (pred_events["fighter"] == gf)
    ]

    if len(candidates) == 0:
        continue

    candidates["err"] = (candidates["frame"] - gf_frame).abs()
    best = candidates.loc[candidates["err"].idxmin()]

    if best["err"] <= 6:
        matched += 1
        total_error += best["err"]
        errors.append(best["err"])

recall = matched / len(gt_events)
mean_err = total_error / matched if matched > 0 else 999

print(f"GT events: {len(gt_events)}")
print(f"Pred events: {len(pred_events)}")
print(f"Matched: {matched}")
print(f"Recall: {recall:.4f}")
print(f"Mean frame error: {mean_err:.3f}")
print(f"Median frame error: {np.median(errors):.3f}")
print(f"95th percentile error: {np.percentile(errors, 95):.3f}")


print("TEST PREDICTIONS")

test_proba = np.zeros(len(X_test))

for i, (cat, xgb, lgbm) in enumerate(zip(cat_models, xgb_models, lgbm_models), 1):
    print(f"  Predicting fold {i}...")
    p1 = cat.predict_proba(X_test)[:, 1]
    p2 = xgb.predict_proba(X_test)[:, 1]
    p3 = lgbm.predict_proba(X_test)[:, 1]

    fold_proba = p1 * 0.4 + p2 * 0.3 + p3 * 0.3
    test_proba += fold_proba

test_proba /= N_SPLITS

test_df["punch_proba"] = test_proba
print("Test proba stats:", test_df["punch_proba"].describe())


print("SMOOTHING")

test_df = test_df.sort_values(["video_key", "fighter", "frame"])

test_df["smooth_proba"] = (
    test_df
    .groupby(["video_key", "fighter"])["punch_proba"]
    .transform(
        lambda x: x.rolling(
            SMOOTH_WINDOW,
            center=True,
            min_periods=1
        ).mean()
    )
)

print("Smooth proba stats:", test_df["smooth_proba"].describe())


print("EVENT EXTRACTION")

def extract_events(threshold):

    events = []

    for (video_key, fighter), video_df in test_df.groupby(["video_key", "fighter"]):

        video_df = video_df.sort_values("frame").reset_index()

        probs = video_df["smooth_proba"].values
        raw_probs = video_df["punch_proba"].values
        frames = video_df["frame"].values
        row_idx = video_df["index"].values

        raw_smooth = pd.Series(raw_probs).rolling(3, center=True, min_periods=1).mean().values

        peaks, _ = find_peaks(
            probs,
            height=threshold,
            distance=MIN_DISTANCE
        )

        for p in peaks:

            w = WINDOW_REFINE + int(raw_probs[p] > 0.5)

            left = max(0, p - w)
            right = min(len(raw_probs) - 1, p + w)

            local_idx = left + np.argmax(raw_smooth[left:right+1])

            events.append({
                "video_key": video_key,
                "fighter": fighter,
                "frame": int(frames[local_idx]),
                "confidence": float(raw_probs[local_idx]),
                "row_idx": int(row_idx[local_idx]),
            })

    events_df = pd.DataFrame(events)

    print(f"[DEBUG] Raw events before NMS: {len(events_df)}")

    final_events = []

    for (video_key, fighter), video_df in events_df.groupby(["video_key", "fighter"]):

        video_df = video_df.sort_values("confidence", ascending=False)
        selected = []

        for _, row in video_df.iterrows():
            frame = row["frame"]
            if all(abs(frame - kept["frame"]) > NMS_WINDOW for kept in selected):
                selected.append(row)

        final_events.extend(selected)

    final_df = pd.DataFrame(final_events)
    print(f"[DEBUG] Events after NMS: {len(final_df)}")

    return len(final_df), final_df


print("\nSearching threshold for 1594 events...")

low = 0.05
high = 0.60

best_df = None
best_diff = 1e9

for step in range(20):

    mid = (low + high) / 2

    count, tmp_df = extract_events(mid)

    diff = abs(count - TARGET_EVENTS)

    print(
        f"STEP={step+1:02d} TH={mid:.4f} "
        f"EVENTS={count} DIFF={diff}"
    )

    if diff < best_diff:
        best_diff = diff
        best_df = tmp_df.copy()

    if count > TARGET_EVENTS:
        low = mid
    else:
        high = mid

events_df = best_df.copy()

print("\nFINAL EVENTS:", len(events_df))


print("ATTRIBUTE MODELS")

attr_train = train_df[train_df["is_punch"] == 1].copy()
attr_train["punch_proba"] = 1.0
attr_train["smooth_proba"] = 1.0

attr_features = [c for c in feature_cols if c in attr_train.columns]
attr_features += ["punch_proba", "smooth_proba"]

agents = {}

labels = [
    "hand",
    "target",
    "punch_type",
    "effectiveness"
]

for label in labels:

    print(f"\nTraining attribute model: {label}")

    y_raw = attr_train[label].fillna("unknown")

    le = LabelEncoder()
    y_attr = le.fit_transform(y_raw.astype(str))

    model = CatBoostClassifier(
        iterations=700,
        depth=6,
        learning_rate=0.03,
        loss_function="MultiClass",
        eval_metric="TotalF1",
        random_seed=RANDOM_STATE,
        verbose=False,
        task_type="GPU",
        devices="0"
    )

    model.fit(
        attr_train[attr_features],
        y_attr,
        verbose=False
    )

    agents[label] = {
        "model": model,
        "encoder": le
    }


print("ATTRIBUTE PREDICTION")

event_rows = test_df.loc[events_df["row_idx"]].copy()

for label in labels:

    print(f"  Predicting {label}...")
    model = agents[label]["model"]
    encoder = agents[label]["encoder"]

    pred = model.predict(event_rows[attr_features]).flatten()
    pred = pred.astype(int)

    decoded = encoder.inverse_transform(pred)

    events_df[label] = decoded

events_df["clear"] = True


print("GENERATE SUBMISSION")

submission = events_df.merge(
    video_meta[["video_key", "video_id", "agn_index"]],
    on="video_key",
    how="left"
)

submission = submission.sort_values([
    "video_key",
    "frame"
]).reset_index(drop=True)

submission["id"] = np.arange(len(submission)) + 1

submission = submission[
    [
        "id",
        "video_id",
        "agn_index",
        "video_key",
        "frame",
        "fighter",
        "punch_type",
        "hand",
        "target",
        "effectiveness",
        "clear",
    ]
]

print("Submission shape:", submission.shape)
print(submission.head(10))

submission.to_csv(
    "submission_one.csv",
    index=False
)

print("\nSUBMISSION SAVED -> submission.csv")

with open("boxing_pipelinepkl", "wb") as f:

    pickle.dump({

        "cat_models": cat_models,
        "xgb_models": xgb_models,
        "lgbm_models": lgbm_models,

        "attribute_models": agents,

        "features": feature_cols,
        "threshold": best_thr,

    }, f)

print("\nMODELS SAVED")
print("\nDONE.")
