"""Generate the Markdown model report and its static chart previews."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_backend import ARTIFACT_DIR, REPORT_FIGURE_DIR, REPORT_FILE

os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACT_DIR / "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREEN = "#1b9b72"
RED = "#cf5c55"
BLUE = "#347bd1"
INK = "#1b2d40"
MUTED = "#75869a"


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _save_chart(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()


def _make_figures(metadata: dict[str, Any]) -> None:
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics = metadata["metrics"]
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    curves = metadata["curves"]

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Normal", "Anomaly"])
    ax.set_yticks([0, 1], labels=["Normal", "Anomaly"])
    ax.set_xlabel("Predicted label", color=MUTED)
    ax.set_ylabel("Actual label", color=MUTED)
    ax.set_title("Holdout confusion matrix", loc="left", color=INK, fontweight="bold")
    midpoint = matrix.max() / 2 if matrix.size else 0
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > midpoint else INK,
                fontsize=13,
                fontweight="bold",
            )
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04, label="Connections")
    _save_chart(REPORT_FIGURE_DIR / "confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.plot(curves["roc"]["x"], curves["roc"]["y"], color=BLUE, linewidth=2.4, label=f"Model · AUC {_percent(metrics['roc_auc'])}")
    ax.plot([0, 1], [0, 1], color="#aab5c0", linewidth=1.2, linestyle="--", label="Chance")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate")
    ax.set_title("ROC curve", loc="left", color=INK, fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    _save_chart(REPORT_FIGURE_DIR / "roc_curve.png")

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.plot(curves["precision_recall"]["x"], curves["precision_recall"]["y"], color=GREEN, linewidth=2.4, label=f"Model · AP {_percent(metrics['average_precision'])}")
    ax.axhline(metadata["validation_anomaly_rows"] / metadata["validation_rows"], color="#aab5c0", linewidth=1.2, linestyle="--", label="Anomaly prevalence")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision")
    ax.set_title("Precision–recall curve", loc="left", color=INK, fontweight="bold")
    ax.legend(loc="lower left", frameon=False)
    _save_chart(REPORT_FIGURE_DIR / "precision_recall.png")

    top_features = metadata["feature_importances"][:12]
    names = [item["feature"] for item in reversed(top_features)]
    values = [item["importance"] for item in reversed(top_features)]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.barh(names, values, color=BLUE, height=0.68)
    ax.set_xlabel("Aggregated impurity importance", color=MUTED)
    ax.set_title("Top 12 input features", loc="left", color=INK, fontweight="bold")
    ax.grid(axis="y", visible=False)
    _save_chart(REPORT_FIGURE_DIR / "feature_importance.png")

    groups = ["Training fit", "Validation"]
    normal = [metadata["fit_normal_rows"], metadata["validation_normal_rows"]]
    anomaly = [metadata["fit_anomaly_rows"], metadata["validation_anomaly_rows"]]
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.bar(x, normal, color=GREEN, label="Normal")
    ax.bar(x, anomaly, bottom=normal, color=RED, label="Anomaly")
    ax.set_xticks(x, labels=groups)
    ax.set_ylabel("Connections")
    ax.set_title("Class balance by split", loc="left", color=INK, fontweight="bold")
    ax.legend(frameon=False, ncols=2)
    for index, total in enumerate([sum(normal[:1] + anomaly[:1]), normal[1] + anomaly[1]]):
        ax.text(index, total, f"{total:,}", ha="center", va="bottom", color=MUTED, fontsize=9)
    _save_chart(REPORT_FIGURE_DIR / "class_balance.png")


def _build_markdown(metadata: dict[str, Any]) -> str:
    metrics = metadata["metrics"]
    matrix = metrics["confusion_matrix"]
    importances = metadata["feature_importances"]
    params = metadata["model_parameters"]
    all_features = ", ".join(f"`{name}`" for name in metadata["feature_columns"])
    feature_table = "\n".join(
        f"| {index} | `{item['feature']}` | {item['importance']:.6f} |"
        for index, item in enumerate(importances[:15], start=1)
    )
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""# Network Intrusion Detection — Model Report

Generated from the model metadata on **{generated}**. The interactive charts are available in the local app at [Model Insights](http://127.0.0.1:5000/insights) while the app is running.

## At a glance

| Item | Value |
|---|---:|
| Model | {metadata['model_name']} |
| Target | `{metadata['target']}` (`anomaly` is the positive class) |
| Labeled training rows | {metadata['training_rows']:,} |
| Training fit rows | {metadata['fit_rows']:,} |
| Holdout validation rows | {metadata['validation_rows']:,} |
| Input features used | {len(metadata['feature_columns'])} |
| Holdout accuracy | **{_percent(metrics['accuracy'])}** |
| ROC-AUC | **{_percent(metrics['roc_auc'])}** |
| Average precision | **{_percent(metrics['average_precision'])}** |

This is a stratified random holdout from the supplied labeled training CSV. `Test_data.csv` has no labels and is used only as an unlabeled prediction sample.

## Dataset and split

The source file is `data/Train_data.csv`. It contains {metadata['training_rows']:,} labeled connections: {metadata['normal_rows']:,} normal and {metadata['anomaly_rows']:,} anomaly. The split uses `train_test_split(test_size={metadata['validation_split']}, stratify=class, random_state={metadata['random_state']})`, producing {metadata['fit_rows']:,} fit rows and {metadata['validation_rows']:,} validation rows. Class stratification preserves the approximate class proportions in each split.

The model uses these {len(metadata['feature_columns'])} input columns (the constant `num_outbound_cmds` column is excluded):

{all_features}

### Split class balance

![Training and validation class balance](static/reports/class_balance.png)

## Preprocessing and classifier

- Categorical columns: `{', '.join(metadata['categorical_columns'])}`. Missing values are filled with the most frequent training value, then one-hot encoded. Categories not seen during training are ignored by the encoder rather than causing prediction to fail.
- Numeric columns: {len(metadata['numeric_columns'])} numeric features. Missing values are filled with the median learned from the fit partition.
- Classifier: Random Forest with {params['n_estimators']} trees, `max_features={params['max_features']}`, `max_depth={params['max_depth']}`, `min_samples_leaf={params['min_samples_leaf']}`, `class_weight={params['class_weight']}`, and `random_state={params['random_state']}`.
- Evaluation: metrics and curves are measured on the held-out 20% before the deployable pipeline is refit using all labeled rows.
- Implementation: scikit-learn {metadata['sklearn_version']} pipeline; curve samples are stored with the local model metadata.

## Holdout evaluation

| Metric | Result | Meaning |
|---|---:|---|
| Accuracy | {_percent(metrics['accuracy'])} | Share of all connections classified correctly |
| Precision (anomaly) | {_percent(metrics['precision_anomaly'])} | Share of predicted anomalies that are actual anomalies |
| Recall / sensitivity (anomaly) | {_percent(metrics['recall_anomaly'])} | Share of actual anomalies detected |
| Specificity (normal) | {_percent(metrics['specificity'])} | Share of actual normal connections correctly recognized |
| F1 (anomaly) | {_percent(metrics['f1_anomaly'])} | Harmonic mean of anomaly precision and recall |
| False-positive rate | {_percent(metrics['false_positive_rate'])} | Share of normal connections incorrectly flagged as anomalies |
| ROC-AUC | {_percent(metrics['roc_auc'])} | Threshold-independent ranking across true-positive and false-positive rates |
| Average precision | {_percent(metrics['average_precision'])} | Area-style summary of the precision–recall curve, weighted across recall increments |

### Confusion matrix

Rows are actual classes; columns are predicted classes.

| Actual \\ Predicted | Normal | Anomaly |
|---|---:|---:|
| Normal | TN = {matrix[0][0]:,} | FP = {matrix[0][1]:,} |
| Anomaly | FN = {matrix[1][0]:,} | TP = {matrix[1][1]:,} |

![Confusion matrix](static/reports/confusion_matrix.png)

### ROC and precision–recall curves

![ROC curve](static/reports/roc_curve.png)

![Precision–recall curve](static/reports/precision_recall.png)

## Metric definitions and formulas

For anomaly as the positive class: **TP** is an anomaly correctly flagged; **TN** is normal traffic correctly cleared; **FP** is normal traffic incorrectly flagged; **FN** is an anomaly missed by the model.

| Term | Formula | Interpretation |
|---|---|---|
| Accuracy | `(TP + TN) / (TP + TN + FP + FN)` | Overall fraction of correct predictions. |
| Precision | `TP / (TP + FP)` | Of all connections flagged as anomalies, how many are anomalies. |
| Recall / sensitivity | `TP / (TP + FN)` | Of all actual anomalies, how many the model detects. |
| Specificity | `TN / (TN + FP)` | Of all actual normal connections, how many the model clears. |
| F1 score | `2 × Precision × Recall / (Precision + Recall)` | Harmonic mean that balances precision and recall. |
| False-positive rate | `FP / (FP + TN)` | Fraction of normal connections incorrectly flagged; equals `1 − Specificity`. |
| ROC curve | Plot `TPR = TP/(TP+FN)` against `FPR = FP/(FP+TN)` over score thresholds. | Shows the detection and false-alarm tradeoff as the threshold changes. |
| ROC-AUC | `∫ TPR d(FPR)` | Probability-style ranking summary; 1 is perfect ranking and 0.5 is chance-level ranking. |
| Precision–recall curve | Plot `Precision = TP/(TP+FP)` against `Recall = TP/(TP+FN)` over score thresholds. | Focuses on anomaly detection quality, especially when class frequencies differ. |
| Average precision | `Σ (Recallₙ − Recallₙ₋₁) × Precisionₙ` | Summarizes precision over recall increments; higher values are better. |

Precision, recall, specificity, and F1 use the model's default class decision. ROC-AUC and average precision use the predicted anomaly probabilities across thresholds.

## Feature importance

The chart aggregates the Random Forest's impurity-based importance across one-hot columns back to each original input feature. These values describe how often and how effectively the fitted trees use a feature to reduce node impurity; they are not causal explanations and can favor features with many possible values.

![Top feature importances](static/reports/feature_importance.png)

| Rank | Input feature | Aggregated importance |
|---:|---|---:|
{feature_table}

The local app's [Model Insights page](http://127.0.0.1:5000/insights) includes interactive versions of these charts with hover values and zoom controls.

## Limitations

- The holdout is randomly drawn from the supplied training CSV. It is not a time-based or independently collected network test, so the score may be optimistic for different networks or future traffic.
- The supplied test CSV has no ground-truth class labels; no accuracy claim is made for its predictions.
- Feature importance is a model diagnostic, not a causal explanation for an individual prediction.
- Accuracy can hide class-specific errors. Review precision, recall, specificity, false-positive rate, and the confusion matrix together when assessing the operating tradeoff.
"""


def generate_report(metadata: dict[str, Any]) -> None:
    """Create all static figures and write the root-level Markdown report."""
    _make_figures(metadata)
    REPORT_FILE.write_text(_build_markdown(metadata), encoding="utf-8")


def ensure_report(metadata: dict[str, Any]) -> None:
    expected = [
        REPORT_FIGURE_DIR / "confusion_matrix.png",
        REPORT_FIGURE_DIR / "roc_curve.png",
        REPORT_FIGURE_DIR / "precision_recall.png",
        REPORT_FIGURE_DIR / "feature_importance.png",
        REPORT_FIGURE_DIR / "class_balance.png",
    ]
    if not REPORT_FILE.is_file() or not all(path.is_file() for path in expected):
        generate_report(metadata)
