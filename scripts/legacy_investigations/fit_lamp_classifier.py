#!/usr/bin/env python3
"""fit_lamp_classifier.py -- throwaway: fit classical ML (logistic regression
+ random forest, no NNs) on analyze_lamp_features.py's per-blob feature table
to find out what actually discriminates a real lamp-row element from
background noise / real controller LEDs on this real recording. Frame-level
train/test split (not blob-level) to avoid leaking near-identical blobs from
the same physical object across train and test. NOT committed.

Usage: python3 fit_lamp_classifier.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

SEQ_DIR = Path("/home/nikitakarpuks/Documents/lamp_sequences/seq1")

FEATURES = ["area", "max_pix", "radius", "circularity", "elongation",
            "rpix", "nn_dist", "neighbor_count", "row_residual"]


def main():
    rows = json.load(open(SEQ_DIR / "blob_features.json"))
    image_ids = sorted(set(r["image_id"] for r in rows))
    n_test_frames = max(1, int(len(image_ids) * 0.25))
    test_ids = set(image_ids[-n_test_frames:])  # last 25% of the sequence, chronologically held out

    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=np.float64)
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    is_test = np.array([r["image_id"] in test_ids for r in rows])

    X_train, X_test = X[~is_test], X[is_test]
    y_train, y_test = y[~is_test], y[is_test]
    print(f"train: {len(X_train)} blobs ({y_train.sum()} lamp) | test: {len(X_test)} blobs ({y_test.sum()} lamp)")
    print(f"train frames: {len(image_ids) - n_test_frames}, test frames (held out, chronological tail): {n_test_frames}")

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    print("\n=== Logistic Regression ===")
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(X_train_s, y_train)
    pred = lr.predict(X_test_s)
    proba = lr.predict_proba(X_test_s)[:, 1]
    print(classification_report(y_test, pred, target_names=["not_lamp", "lamp"]))
    print("AUC:", roc_auc_score(y_test, proba))
    print("confusion matrix [ [TN FP] [FN TP] ]:\n", confusion_matrix(y_test, pred))
    print("\ncoefficients (standardized -- larger |coef| = more discriminating):")
    for f, c in sorted(zip(FEATURES, lr.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {f:>16s}: {c:+.3f}")

    print("\n=== Random Forest ===")
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight="balanced",
                                 random_state=0, min_samples_leaf=5)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    proba = rf.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, pred, target_names=["not_lamp", "lamp"]))
    print("AUC:", roc_auc_score(y_test, proba))
    print("confusion matrix [ [TN FP] [FN TP] ]:\n", confusion_matrix(y_test, pred))
    print("\nfeature importances:")
    for f, imp in sorted(zip(FEATURES, rf.feature_importances_), key=lambda t: -t[1]):
        print(f"  {f:>16s}: {imp:.3f}")

    # Compare against the CURRENT production heuristic's own brightness gate
    # alone (lamp_blob_filter.max_brightness=210, i.e. "dim enough to be a
    # lamp") as a baseline -- how good is brightness ALONE at separating
    # lamp from not-lamp on this real data?
    print("\n=== Baseline: brightness <= 210 alone (today's own max_brightness gate) ===")
    bright_col = FEATURES.index("max_pix")
    pred_baseline = (X_test[:, bright_col] <= 210).astype(int)
    print(classification_report(y_test, pred_baseline, target_names=["not_lamp", "lamp"]))
    print("confusion matrix [ [TN FP] [FN TP] ]:\n", confusion_matrix(y_test, pred_baseline))

    # Print real per-class feature distributions for interpretability (not
    # just model internals) -- lets a human sanity-check the ML findings
    # directly against real numbers.
    print("\n=== Per-class real feature distributions (train set) ===")
    for f in FEATURES:
        col = FEATURES.index(f)
        lamp_vals = X_train[y_train == 1, col]
        other_vals = X_train[y_train == 0, col]
        print(f"  {f:>16s}: lamp  mean={lamp_vals.mean():8.2f} median={np.median(lamp_vals):8.2f} "
              f"p10={np.percentile(lamp_vals,10):8.2f} p90={np.percentile(lamp_vals,90):8.2f}")
        print(f"  {'':>16s}  other mean={other_vals.mean():8.2f} median={np.median(other_vals):8.2f} "
              f"p10={np.percentile(other_vals,10):8.2f} p90={np.percentile(other_vals,90):8.2f}")


if __name__ == "__main__":
    main()
