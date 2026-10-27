"""Training and inference pipeline for the Network Intrusion Detection app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
TRAIN_FILE = DATA_DIR / "Train_data.csv"
MODEL_FILE = ARTIFACT_DIR / "intrusion_model.joblib"
METADATA_FILE = ARTIFACT_DIR / "model_metadata.json"
REPORT_FILE = ROOT / "txt.md"
REPORT_FIGURE_DIR = ROOT / "static" / "reports"
TARGET = "class"
CONSTANT_FEATURES = {"num_outbound_cmds"}
METADATA_SCHEMA_VERSION = 2


def _make_pipeline(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[Pipeline, list[str], list[str]]:
    categorical = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    numeric = [column for column in feature_columns if column not in categorical]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            ("numeric", SimpleImputer(strategy="median"), numeric),
        ],
        remainder="drop",
    )
    classifier = RandomForestClassifier(
        n_estimators=220,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("classifier", classifier)]), categorical, numeric


def _sample_curve(x: np.ndarray, y: np.ndarray, limit: int = 300) -> dict[str, list[float]]:
    """Keep curve payloads compact while preserving their overall shape."""
    if len(x) <= limit:
        indices = np.arange(len(x))
    else:
        indices = np.unique(np.linspace(0, len(x) - 1, limit, dtype=int))
    return {
        "x": [float(value) for value in x[indices]],
        "y": [float(value) for value in y[indices]],
    }


def _group_feature_importances(
    pipeline: Pipeline, categorical: list[str], numeric: list[str]
) -> list[dict[str, Any]]:
    """Aggregate one-hot split importances back to their original columns."""
    preprocess = pipeline.named_steps["preprocess"]
    classifier = pipeline.named_steps["classifier"]
    encoded = preprocess.named_transformers_["categorical"].named_steps["one_hot"]
    values = classifier.feature_importances_
    offset = 0
    grouped: list[dict[str, Any]] = []

    for column, categories in zip(categorical, encoded.categories_):
        end = offset + len(categories)
        grouped.append({"feature": column, "importance": float(values[offset:end].sum())})
        offset = end
    for column in numeric:
        grouped.append({"feature": column, "importance": float(values[offset])})
        offset += 1

    return sorted(grouped, key=lambda item: item["importance"], reverse=True)


def train_and_save() -> tuple[Any, dict[str, Any]]:
    """Validate on a held-out split, refit on all rows, and save the model."""
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Training data not found: {TRAIN_FILE}")

    data = pd.read_csv(TRAIN_FILE)
    if TARGET not in data.columns:
        raise ValueError(f"Training data must contain a '{TARGET}' column.")

    labels = data[TARGET].astype(str).str.strip().str.lower()
    allowed_labels = {"normal", "anomaly"}
    if not set(labels.unique()).issubset(allowed_labels):
        raise ValueError("Training labels must be 'normal' or 'anomaly'.")

    feature_columns = [
        column
        for column in data.columns
        if column != TARGET and column not in CONSTANT_FEATURES
    ]
    features = data[feature_columns].copy()
    for column in features.select_dtypes(include=["object", "str"]).columns:
        features[column] = features[column].fillna("unknown").astype(str)

    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    validation_pipeline, categorical, numeric = _make_pipeline(features, feature_columns)
    validation_pipeline.fit(x_train, y_train)
    predicted = validation_pipeline.predict(x_valid)
    classes = list(validation_pipeline.classes_)
    anomaly_index = classes.index("anomaly")
    anomaly_probability = validation_pipeline.predict_proba(x_valid)[:, anomaly_index]
    y_valid_binary = (y_valid.to_numpy() == "anomaly").astype(int)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_valid_binary, anomaly_probability)
    pr_precision, pr_recall, _ = precision_recall_curve(y_valid_binary, anomaly_probability)
    matrix = confusion_matrix(y_valid, predicted, labels=["normal", "anomaly"])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_valid, predicted)),
        "precision_anomaly": float(precision_score(y_valid, predicted, pos_label="anomaly", zero_division=0)),
        "recall_anomaly": float(recall_score(y_valid, predicted, pos_label="anomaly", zero_division=0)),
        "f1_anomaly": float(f1_score(y_valid, predicted, pos_label="anomaly", zero_division=0)),
        "specificity": float(true_negative / (true_negative + false_positive)) if true_negative + false_positive else 0.0,
        "false_positive_rate": float(false_positive / (true_negative + false_positive)) if true_negative + false_positive else 0.0,
        "roc_auc": float(roc_auc_score(y_valid_binary, anomaly_probability)),
        "average_precision": float(average_precision_score(y_valid_binary, anomaly_probability)),
        "validation_rows": int(len(y_valid)),
        "confusion_matrix": matrix.tolist(),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }
    curves = {
        "roc": _sample_curve(false_positive_rate, true_positive_rate),
        "precision_recall": _sample_curve(pr_recall, pr_precision),
    }
    feature_importances = _group_feature_importances(validation_pipeline, categorical, numeric)

    # Keep the deployed model trained on every labeled row after measuring the
    # holdout score. The score above remains tied to data the model did not see.
    final_pipeline, _, _ = _make_pipeline(features, feature_columns)
    final_pipeline.fit(features, labels)
    metadata: dict[str, Any] = {
        "model_name": "Random Forest Classifier",
        "target": TARGET,
        "feature_columns": feature_columns,
        "categorical_columns": categorical,
        "numeric_columns": numeric,
        "metrics": metrics,
        "training_rows": int(len(data)),
        "fit_rows": int(len(x_train)),
        "validation_rows": int(len(x_valid)),
        "normal_rows": int((labels == "normal").sum()),
        "anomaly_rows": int((labels == "anomaly").sum()),
        "fit_normal_rows": int((y_train == "normal").sum()),
        "fit_anomaly_rows": int((y_train == "anomaly").sum()),
        "validation_normal_rows": int((y_valid == "normal").sum()),
        "validation_anomaly_rows": int((y_valid == "anomaly").sum()),
        "validation_split": 0.2,
        "random_state": 42,
        "model_parameters": {
            "n_estimators": 220,
            "max_features": "sqrt",
            "max_depth": None,
            "min_samples_leaf": 1,
            "class_weight": "balanced_subsample",
            "random_state": 42,
        },
        "curves": curves,
        "feature_importances": feature_importances,
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "sklearn_version": sklearn.__version__,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_model = MODEL_FILE.with_suffix(".tmp")
    temporary_metadata = METADATA_FILE.with_suffix(".tmp")
    joblib.dump(final_pipeline, temporary_model)
    temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary_model.replace(MODEL_FILE)
    temporary_metadata.replace(METADATA_FILE)
    from report_generator import generate_report

    generate_report(metadata)
    return final_pipeline, metadata


def load_or_train() -> tuple[Any, dict[str, Any]]:
    """Load a locally trained model, training it on first launch if needed."""
    if MODEL_FILE.exists() and METADATA_FILE.exists():
        try:
            saved_metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
            if saved_metadata.get("metadata_schema_version") != METADATA_SCHEMA_VERSION:
                return train_and_save()
            loaded_model = joblib.load(MODEL_FILE)
            from report_generator import ensure_report

            ensure_report(saved_metadata)
            return loaded_model, saved_metadata
        except (OSError, ValueError, EOFError, KeyError):
            # A damaged or stale local artifact can be recreated from the bundled data.
            pass
    return train_and_save()
