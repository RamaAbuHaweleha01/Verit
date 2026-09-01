#!/usr/bin/env python3
"""
Verit NIDS - Hybrid Model Training & Evaluation Pipeline
------------------------------------------------------------------
Trains BOTH halves of the hybrid detector on your labeled dataset
(e.g. Database/all_data_combined.csv) with a proper 3-way split:

    TRAIN  -> fits the FeatureProcessor (scaler/imputer/encoder), the
              XGBoost classifier, and the autoencoder (benign rows only)
    VAL    -> used for early stopping and anomaly-threshold calibration
              (never touched during fitting)
    TEST   -> touched ONLY at the very end, for the reported metrics

Produces, under --report-dir:
    - evaluation_report.md         human-readable summary + error analysis
    - xgboost_confusion_matrix.png
    - xgboost_roc_curves.png
    - xgboost_per_class_metrics.csv
    - autoencoder_roc_curve.png
    - autoencoder_error_distribution.png
    - metrics.json                 every number in this report, machine-readable

Saves, under --out-dir:
    - processor.joblib             fitted FeatureProcessor (fit on TRAIN only)
    - xgboost_model.joblib
    - autoencoder/                 (autoencoder.keras + metadata)

Usage:
    python3 train_hybrid_models.py --csv Database/all_data_combined.csv \
        --label-column Label --out-dir models/artifacts --report-dir models/reports
"""

import argparse
import json
import sys
from pathlib import Path

# Dependency check MUST run before anything below it -- numpy/pandas/sklearn/
# matplotlib/xgboost/tensorflow are all imported further down, so if any of
# them are missing, this needs to install them first or those imports crash
# before ever reaching ensure_dependencies().
from models.dependency_manager import ensure_dependencies
ensure_dependencies()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize

from processing import FeatureProcessor
from processing.feature_processor import DEFAULT_IDENTITY_COLUMNS
from models.xgboost_classifier import XGBoostAttackClassifier
from models.autoencoder import AutoencoderAnomalyDetector
from extract_features import load_precomputed_csv


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------

def safe_stratified_split(df, y, test_size, random_state=42):
    """Falls back to a non-stratified split (with a warning) if any class
    is too small to stratify -- common in CICIDS2017, where classes like
    Heartbleed have only a handful of samples total."""
    try:
        return train_test_split(df, y, test_size=test_size, stratify=y, random_state=random_state)
    except ValueError as e:
        print(f"[!] Stratified split failed ({e}); falling back to a non-stratified split "
              f"for this split. Very rare classes may end up under/over-represented.")
        return train_test_split(df, y, test_size=test_size, random_state=random_state)


def three_way_split(df, y, test_size, val_size, random_state=42):
    df_trainval, df_test, y_trainval, y_test = safe_stratified_split(
        df, y, test_size=test_size, random_state=random_state
    )
    relative_val = val_size / (1.0 - test_size)
    df_train, df_val, y_train, y_val = safe_stratified_split(
        df_trainval, y_trainval, test_size=relative_val, random_state=random_state
    )
    return df_train, df_val, df_test, y_train, y_val, y_test


# --------------------------------------------------------------------------
# XGBoost evaluation artifacts
# --------------------------------------------------------------------------

def plot_confusion_matrix(cm, label_classes, out_path, normalize=True):
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
    else:
        cm_norm = cm

    n = len(label_classes)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), max(6, n * 0.55)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1 if normalize else cm.max())
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(label_classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(label_classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("XGBoost Confusion Matrix" + (" (row-normalized)" if normalize else ""))
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            text = f"{val:.2f}" if normalize else f"{int(cm[i, j])}"
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=6, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curves(y_true, y_proba, label_classes, out_path):
    n_classes = len(label_classes)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 7))
    aucs = {}
    for i, cls_name in enumerate(label_classes):
        if y_bin[:, i].sum() == 0:
            continue  # class absent from this split entirely, can't compute ROC
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        aucs[cls_name] = roc_auc
        ax.plot(fpr, tpr, lw=1, alpha=0.7, label=f"{cls_name} (AUC={roc_auc:.3f})")

    # micro-average
    valid_cols = [i for i in range(n_classes) if y_bin[:, i].sum() > 0]
    if valid_cols:
        fpr_micro, tpr_micro, _ = roc_curve(y_bin[:, valid_cols].ravel(), y_proba[:, valid_cols].ravel())
        auc_micro = auc(fpr_micro, tpr_micro)
        ax.plot(fpr_micro, tpr_micro, "k--", lw=2, label=f"micro-average (AUC={auc_micro:.3f})")
        aucs["__micro_average__"] = auc_micro

    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("XGBoost ROC Curves (One-vs-Rest, per class)")
    ax.legend(fontsize=6, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return aucs


def xgboost_error_analysis(cm, label_classes, y_true, y_pred, class_counts):
    """Investigates *why* the classifier is wrong where it's wrong: which
    class pairs get confused, and whether that correlates with how few
    training examples a class had."""
    n = len(label_classes)
    confusions = []
    for i in range(n):
        row_total = cm[i].sum()
        if row_total == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            count = cm[i, j]
            if count > 0:
                confusions.append({
                    "true_class": label_classes[i],
                    "predicted_as": label_classes[j],
                    "count": int(count),
                    "pct_of_true_class": float(count / row_total * 100),
                })
    confusions.sort(key=lambda d: d["count"], reverse=True)
    top_confusions = confusions[:15]

    per_class_recall = {}
    for i, cls in enumerate(label_classes):
        row_total = cm[i].sum()
        per_class_recall[cls] = float(cm[i, i] / row_total) if row_total > 0 else None

    # correlate weak recall with small training-set representation
    low_recall_small_sample = []
    for cls, recall in per_class_recall.items():
        if recall is not None and recall < 0.80 and class_counts.get(cls, 0) < 200:
            low_recall_small_sample.append((cls, recall, class_counts.get(cls, 0)))
    low_recall_small_sample.sort(key=lambda t: t[1])

    return {
        "top_confusions": top_confusions,
        "per_class_recall": per_class_recall,
        "low_recall_small_sample_classes": low_recall_small_sample,
    }


# --------------------------------------------------------------------------
# Autoencoder evaluation artifacts
# --------------------------------------------------------------------------

def plot_autoencoder_roc(y_true_binary, errors, out_path):
    fpr, tpr, thresholds = roc_curve(y_true_binary, errors)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Attack Detection Rate)")
    ax.set_title("Autoencoder ROC Curve (reconstruction error as anomaly score)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return roc_auc


def plot_error_distribution(benign_errors, attack_errors, threshold, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, np.percentile(np.concatenate([benign_errors, attack_errors]), 99), 60)
    ax.hist(benign_errors, bins=bins, alpha=0.6, label="BENIGN (test)", density=True)
    ax.hist(attack_errors, bins=bins, alpha=0.6, label="Attack (test, all types)", density=True)
    ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold ({threshold:.4f})")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Density")
    ax.set_title("Autoencoder Reconstruction Error Distribution -- Test Set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def autoencoder_error_analysis(test_identity, y_test_labels, errors, threshold, benign_label):
    """Per attack-type detection rate -- explains WHICH attacks the
    autoencoder catches easily vs. which ones look statistically close to
    benign traffic (and are therefore hard to catch via reconstruction
    error alone -- exactly the case for a real hybrid system, since those
    are the ones you'd want the supervised classifier to have learned
    explicitly instead)."""
    df = pd.DataFrame({"label": y_test_labels, "error": errors})
    df["flagged"] = df["error"] > threshold

    rows = []
    for label, group in df.groupby("label"):
        if label == benign_label:
            # for benign, "flagged" means false positive
            rows.append({
                "class": label,
                "n_samples": len(group),
                "detection_or_fp_rate": float(group["flagged"].mean()),
                "mean_error": float(group["error"].mean()),
                "role": "false_positive_rate",
            })
        else:
            rows.append({
                "class": label,
                "n_samples": len(group),
                "detection_or_fp_rate": float(group["flagged"].mean()),
                "mean_error": float(group["error"].mean()),
                "role": "detection_rate",
            })
    rows.sort(key=lambda r: (r["role"] != "detection_rate", r["detection_or_fp_rate"]))
    return rows


# --------------------------------------------------------------------------
# Report writing
# --------------------------------------------------------------------------

def write_markdown_report(report_dir, context):
    lines = []
    lines.append("# Verit NIDS - Hybrid Model Evaluation Report\n")
    lines.append(f"Dataset: `{context['csv_path']}`  \n")
    lines.append(f"Total rows loaded: {context['n_total']}  \n")
    lines.append(f"Split sizes -- train: {context['n_train']}, val: {context['n_val']}, "
                  f"test: {context['n_test']}\n")

    lines.append("\n## Class distribution (full dataset)\n")
    lines.append("| Class | Count | % of total |")
    lines.append("|---|---|---|")
    for cls, count in context["class_counts"].items():
        pct = count / context["n_total"] * 100
        lines.append(f"| {cls} | {count} | {pct:.2f}% |")

    lines.append("\n## XGBoost (Known-Attack Classifier) -- Test Set Results\n")
    lines.append("```\n" + context["xgb_report_str"] + "\n```\n")
    lines.append(f"![Confusion Matrix](xgboost_confusion_matrix.png)\n")
    lines.append(f"![ROC Curves](xgboost_roc_curves.png)\n")

    lines.append("\n### Per-class AUC\n")
    lines.append("| Class | AUC |")
    lines.append("|---|---|")
    for cls, a in context["xgb_aucs"].items():
        lines.append(f"| {cls} | {a:.4f} |")

    lines.append("\n### Error Analysis -- XGBoost\n")
    lines.append("**Top confused class pairs (true \u2192 predicted):**\n")
    lines.append("| True Class | Predicted As | Count | % of True Class |")
    lines.append("|---|---|---|---|")
    for c in context["xgb_error_analysis"]["top_confusions"]:
        lines.append(f"| {c['true_class']} | {c['predicted_as']} | {c['count']} | "
                      f"{c['pct_of_true_class']:.1f}% |")

    low_recall = context["xgb_error_analysis"]["low_recall_small_sample_classes"]
    if low_recall:
        lines.append("\n**Classes with recall < 0.80 AND fewer than 200 training samples "
                      "(the most likely explanation for their errors is simply insufficient "
                      "training data, not a fundamental feature limitation):**\n")
        lines.append("| Class | Recall | Training-set-relative sample count |")
        lines.append("|---|---|---|")
        for cls, recall, count in low_recall:
            lines.append(f"| {cls} | {recall:.3f} | {count} |")
    else:
        lines.append("\nNo classes showed both low recall and a small sample count -- "
                      "errors are more likely due to genuine feature overlap between attack "
                      "types (see confusion pairs above) than lack of data.\n")

    lines.append("\n## Autoencoder (Zero-Day / Anomaly Detector) -- Test Set Results\n")
    lines.append(f"Anomaly threshold (from validation, p{context['ae_threshold_percentile']}): "
                  f"{context['ae_threshold']:.6f}\n")
    lines.append(f"ROC-AUC (benign vs. all attacks, reconstruction error as score): "
                  f"{context['ae_roc_auc']:.4f}\n")
    lines.append(f"![Autoencoder ROC Curve](autoencoder_roc_curve.png)\n")
    lines.append(f"![Error Distribution](autoencoder_error_distribution.png)\n")

    lines.append("\n### Per-class detection / false-positive rate\n")
    lines.append("| Class | Role | Rate | Mean Reconstruction Error | n samples (test) |")
    lines.append("|---|---|---|---|---|")
    for r in context["ae_error_analysis"]:
        lines.append(f"| {r['class']} | {r['role']} | {r['detection_or_fp_rate']:.3f} | "
                      f"{r['mean_error']:.6f} | {r['n_samples']} |")

    lines.append("\n### Error Analysis -- Autoencoder\n")
    low_detect = [r for r in context["ae_error_analysis"]
                  if r["role"] == "detection_rate" and r["detection_or_fp_rate"] < 0.5]
    if low_detect:
        lines.append("Attack types the autoencoder catches poorly (detection rate < 50%) -- "
                      "these attack types' flow statistics apparently resemble normal traffic "
                      "closely enough that reconstruction error alone doesn't separate them. "
                      "This is expected and is exactly why the hybrid design pairs this model "
                      "with the supervised XGBoost classifier above, which can be trained "
                      "explicitly on these classes if they're known attack types:\n")
        for r in low_detect:
            lines.append(f"- **{r['class']}** ({r['n_samples']} test samples): "
                          f"{r['detection_or_fp_rate']*100:.1f}% detected, "
                          f"mean error {r['mean_error']:.6f} vs. threshold {context['ae_threshold']:.6f}")
    else:
        lines.append("All attack classes were detected at better than 50% by reconstruction "
                      "error alone -- a good sign for zero-day generalization, since these "
                      "attacks were never given to the autoencoder during training.\n")

    fp_row = next((r for r in context["ae_error_analysis"] if r["class"] == context["benign_label"]), None)
    if fp_row:
        lines.append(f"\nBenign false-positive rate on the test set: "
                      f"{fp_row['detection_or_fp_rate']*100:.2f}% "
                      f"(calibrated at the p{context['ae_threshold_percentile']} threshold from "
                      f"validation data -- raise this percentile to trade detection sensitivity "
                      f"for fewer false alarms, or lower it for the opposite trade-off).\n")

    report_path = Path(report_dir) / "evaluation_report.md"
    report_path.write_text("\n".join(lines))
    print(f"[*] Wrote evaluation report -> {report_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train + evaluate the Verit hybrid NIDS models")
    p.add_argument("--csv", type=str, required=True, help="Path to labeled flow CSV (e.g. CICIDS2017 combined)")
    p.add_argument("--label-column", type=str, default="Label")
    p.add_argument("--benign-label", type=str, default="BENIGN")
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--csv-chunksize", type=int, default=200_000)
    p.add_argument("--out-dir", type=str, default="models/artifacts")
    p.add_argument("--report-dir", type=str, default="models/reports")
    p.add_argument("--nan-strategy", choices=["median", "mean", "zero", "drop"], default="median")
    p.add_argument("--ae-threshold-percentile", type=float, default=99.0)
    p.add_argument("--ae-epochs", type=int, default=100)
    p.add_argument("--ae-patience", type=int, default=8)
    p.add_argument("--xgb-early-stopping-rounds", type=int, default=25)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. Load -----------------------------------------------------------
    df = load_precomputed_csv([args.csv], label_column=args.label_column, chunksize=args.csv_chunksize)
    df = df.dropna(subset=[args.label_column])
    n_total = len(df)
    class_counts = df[args.label_column].value_counts().to_dict()
    print(f"[*] Loaded {n_total} labeled rows across {len(class_counts)} classes.")

    # -- 2. Global label encoding (fixed class->int mapping across all splits) --
    global_label_encoder = LabelEncoder()
    y_all = global_label_encoder.fit_transform(df[args.label_column].astype(str))
    label_classes = list(global_label_encoder.classes_)
    if args.benign_label not in label_classes:
        print(f"[!] Warning: benign label '{args.benign_label}' not found in classes {label_classes}")
    benign_idx = label_classes.index(args.benign_label) if args.benign_label in label_classes else None

    # -- 3. Train / Val / Test split (BEFORE fitting anything) -------------
    df_train, df_val, df_test, y_train_i, y_val_i, y_test_i = three_way_split(
        df, y_all, test_size=args.test_size, val_size=args.val_size
    )
    print(f"[*] Split sizes -- train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)}")
    del df

    # -- 4. Fit FeatureProcessor on TRAIN ONLY ------------------------------
    identity_cols = list(DEFAULT_IDENTITY_COLUMNS) + [args.label_column]
    processor = FeatureProcessor(
        identity_columns=identity_cols, label_column=None, nan_strategy=args.nan_strategy
    )
    train_result = processor.fit_transform(df_train)
    X_train, identity_train = train_result["X"], train_result["identity"]
    val_result = processor.transform(df_val)
    X_val, identity_val = val_result["X"], val_result["identity"]
    test_result = processor.transform(df_test)
    X_test, identity_test = test_result["X"], test_result["identity"]

    # re-derive integer labels aligned to whatever rows survived cleaning/dedup
    y_train = global_label_encoder.transform(identity_train[args.label_column].astype(str))
    y_val = global_label_encoder.transform(identity_val[args.label_column].astype(str))
    y_test = global_label_encoder.transform(identity_test[args.label_column].astype(str))
    y_test_labels = identity_test[args.label_column].values

    processor.save(out_dir / "processor.joblib")
    del df_train, df_val, df_test

    # -- 5. Train XGBoost ----------------------------------------------------
    print("\n" + "=" * 70 + "\n[*] Training XGBoost known-attack classifier\n" + "=" * 70)
    xgb_clf = XGBoostAttackClassifier(label_classes=label_classes)
    xgb_clf.fit(X_train, y_train, X_val=X_val, y_val=y_val,
                early_stopping_rounds=args.xgb_early_stopping_rounds)
    xgb_clf.save(out_dir / "xgboost_model.joblib")

    print("\n[*] Evaluating XGBoost on held-out TEST set")
    xgb_eval = xgb_clf.evaluate(X_test, y_test, dataset_name="TEST")

    cm_path = report_dir / "xgboost_confusion_matrix.png"
    plot_confusion_matrix(xgb_eval["confusion_matrix"], label_classes, cm_path)

    roc_path = report_dir / "xgboost_roc_curves.png"
    xgb_aucs = plot_roc_curves(xgb_eval["y_true"], xgb_eval["y_proba"], label_classes, roc_path)

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        xgb_eval["y_true"], xgb_eval["y_pred"], labels=list(range(len(label_classes))), zero_division=0
    )
    per_class_df = pd.DataFrame({
        "class": label_classes, "precision": precisions, "recall": recalls,
        "f1_score": f1s, "support": supports,
    })
    per_class_df.to_csv(report_dir / "xgboost_per_class_metrics.csv", index=False)

    xgb_err_analysis = xgboost_error_analysis(
        xgb_eval["confusion_matrix"], label_classes, xgb_eval["y_true"], xgb_eval["y_pred"], class_counts
    )

    # -- 6. Train Autoencoder (BENIGN only) -----------------------------------
    print("\n" + "=" * 70 + "\n[*] Training autoencoder zero-day detector (BENIGN traffic only)\n" + "=" * 70)
    if benign_idx is None:
        print("[!] Cannot train autoencoder without a valid benign label. Skipping.")
        ae = None
    else:
        benign_mask_train = y_train == benign_idx
        benign_mask_val = y_val == benign_idx
        X_train_benign = X_train[benign_mask_train]
        X_val_benign = X_val[benign_mask_val]
        print(f"[*] Training autoencoder on {len(X_train_benign)} benign flows "
              f"(validating threshold on {len(X_val_benign)} held-out benign flows)")

        ae = AutoencoderAnomalyDetector(threshold_percentile=args.ae_threshold_percentile)
        ae.fit(X_train_benign, X_val_benign=X_val_benign, epochs=args.ae_epochs, patience=args.ae_patience)
        ae.save(out_dir / "autoencoder")

        print("\n[*] Evaluating autoencoder on held-out TEST set")
        test_errors = ae.reconstruction_error(X_test)
        y_test_binary = (y_test != benign_idx).astype(int)  # 1 = any attack, 0 = benign

        ae_roc_path = report_dir / "autoencoder_roc_curve.png"
        ae_roc_auc = plot_autoencoder_roc(y_test_binary, test_errors, ae_roc_path)

        benign_test_errors = test_errors[y_test == benign_idx]
        attack_test_errors = test_errors[y_test != benign_idx]
        dist_path = report_dir / "autoencoder_error_distribution.png"
        plot_error_distribution(benign_test_errors, attack_test_errors, ae.threshold_, dist_path)

        ae_err_analysis = autoencoder_error_analysis(
            identity_test, y_test_labels, test_errors, ae.threshold_, args.benign_label
        )

    # -- 7. Report -----------------------------------------------------------
    context = {
        "csv_path": args.csv,
        "n_total": n_total,
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "class_counts": class_counts,
        "xgb_report_str": xgb_eval["classification_report_str"],
        "xgb_aucs": xgb_aucs,
        "xgb_error_analysis": xgb_err_analysis,
        "benign_label": args.benign_label,
        "ae_threshold_percentile": args.ae_threshold_percentile,
    }
    if ae is not None:
        context.update({
            "ae_threshold": ae.threshold_,
            "ae_roc_auc": ae_roc_auc,
            "ae_error_analysis": ae_err_analysis,
        })
    write_markdown_report(report_dir, context)

    metrics_json = {
        "n_total": n_total, "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "class_counts": class_counts,
        "xgboost": {
            "classification_report": xgb_eval["classification_report_dict"],
            "per_class_auc": xgb_aucs,
            "error_analysis": xgb_err_analysis,
        },
        "autoencoder": ({
            "threshold": ae.threshold_,
            "roc_auc": ae_roc_auc,
            "error_analysis": ae_err_analysis,
        } if ae is not None else None),
    }
    with open(report_dir / "metrics.json", "w") as f:
        json.dump(metrics_json, f, indent=2, default=str)
    print(f"\n[*] Done. Models -> {out_dir}, Report -> {report_dir}")


if __name__ == "__main__":
    main()
