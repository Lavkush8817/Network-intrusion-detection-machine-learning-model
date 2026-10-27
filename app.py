"""Local web application for network intrusion detection."""

from __future__ import annotations

import io
import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request, send_file, url_for
from plotly.offline import get_plotlyjs
from werkzeug.exceptions import RequestEntityTooLarge

from model_backend import DATA_DIR, REPORT_FILE, load_or_train


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "artifacts" / "predictions"
TEST_FILE = DATA_DIR / "Test_data.csv"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.logger.setLevel(logging.INFO)

model = None
metadata = None


def ensure_model():
    global model, metadata
    if model is None or metadata is None:
        model, metadata = load_or_train()
    return model, metadata


def _prepare_features(frame: pd.DataFrame, model_metadata: dict) -> pd.DataFrame:
    required = model_metadata["feature_columns"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        preview = ", ".join(missing[:8])
        remainder = " …" if len(missing) > 8 else ""
        raise ValueError(f"CSV is missing {len(missing)} required feature column(s): {preview}{remainder}")

    features = frame[required].copy()
    for column in model_metadata["categorical_columns"]:
        features[column] = features[column].fillna("unknown").astype(str)
    for column in model_metadata["numeric_columns"]:
        converted = pd.to_numeric(features[column], errors="coerce")
        invalid = features[column].notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"Column '{column}' contains nonnumeric values.")
        features[column] = converted
    return features


@app.get("/")
def index():
    ensure_model()
    return render_template("index.html")


@app.get("/insights")
def insights():
    ensure_model()
    return render_template("insights.html")


@app.get("/api/model-insights")
def model_insights():
    _, model_metadata = ensure_model()
    return jsonify(model_metadata)


@app.get("/assets/plotly.min.js")
def plotly_bundle():
    return Response(get_plotlyjs(), mimetype="application/javascript", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/txt.md")
def download_model_report():
    if not REPORT_FILE.is_file():
        _, model_metadata = ensure_model()
        from report_generator import generate_report

        generate_report(model_metadata)
    return send_file(REPORT_FILE, mimetype="text/markdown", as_attachment=True, download_name="txt.md")


@app.get("/api/health")
def health():
    _, model_metadata = ensure_model()
    return jsonify(
        {
            "status": "ready",
            "model_name": model_metadata["model_name"],
            "metrics": model_metadata["metrics"],
            "training_rows": model_metadata["training_rows"],
            "feature_count": len(model_metadata["feature_columns"]),
        }
    )


@app.get("/api/sample.csv")
def sample_csv():
    if not TEST_FILE.exists():
        return jsonify({"error": "The bundled sample dataset is missing."}), 404
    sample = pd.read_csv(TEST_FILE).head(10)
    output = io.BytesIO(sample.to_csv(index=False).encode("utf-8"))
    output.seek(0)
    return send_file(output, mimetype="text/csv", as_attachment=True, download_name="sample_network_traffic.csv")


@app.post("/api/predict")
def predict():
    active_model, model_metadata = ensure_model()
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "Choose a CSV file to analyze."}), 400
    if Path(upload.filename).suffix.lower() != ".csv":
        return jsonify({"error": "Upload a .csv file."}), 400

    try:
        source = pd.read_csv(upload.stream)
    except Exception as exc:
        return jsonify({"error": f"Could not read this CSV: {exc}"}), 400
    if source.empty:
        return jsonify({"error": "The CSV has no data rows."}), 400

    try:
        features = _prepare_features(source, model_metadata)
        predictions = active_model.predict(features)
        probabilities = active_model.predict_proba(features)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check the CSV columns and values, then try again."}), 500

    confidence = np.max(probabilities, axis=1) * 100
    results = source.drop(
        columns=["detection_result", "confidence_percent", "prediction_matches"], errors="ignore"
    ).copy()
    results.insert(0, "detection_result", predictions)
    results.insert(1, "confidence_percent", np.round(confidence, 2))

    upload_accuracy = None
    if "class" in source.columns:
        actual = source["class"].astype(str).str.strip().str.lower()
        if not set(actual.unique()).issubset({"normal", "anomaly"}):
            return jsonify({"error": "If the CSV includes 'class', its values must be 'normal' or 'anomaly'."}), 400
        results["prediction_matches"] = (actual.to_numpy() == predictions).astype(bool)
        upload_accuracy = float((actual.to_numpy() == predictions).mean())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    output_path = OUTPUT_DIR / f"{token}.csv"
    results.to_csv(output_path, index=False)

    anomaly_count = int(np.count_nonzero(predictions == "anomaly"))
    total = int(len(predictions))
    preview_frame = results.head(12).astype(object)
    preview_frame = preview_frame.where(pd.notna(preview_frame), None)
    return jsonify(
        {
            "summary": {
                "total": total,
                "normal": total - anomaly_count,
                "anomaly": anomaly_count,
                "anomaly_percent": round(100 * anomaly_count / total, 2),
                "upload_accuracy": upload_accuracy,
            },
            "preview": preview_frame.to_dict(orient="records"),
            "columns": results.columns.tolist(),
            "download_url": url_for("download_result", token=token),
        }
    )


@app.get("/api/download/<token>")
def download_result(token: str):
    if len(token) != 32 or any(character not in "0123456789abcdef" for character in token):
        return jsonify({"error": "Result not found."}), 404
    path = OUTPUT_DIR / f"{token}.csv"
    if not path.is_file():
        return jsonify({"error": "Result not found. Run the analysis again to create a new download."}), 404
    return send_file(path, mimetype="text/csv", as_attachment=True, download_name="intrusion_detection_results.csv")


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return jsonify({"error": "File is larger than the 25 MB upload limit."}), 413


if __name__ == "__main__":
    trained_model, trained_metadata = ensure_model()
    score = trained_metadata["metrics"]["accuracy"] * 100
    print(f"{trained_metadata['model_name']} ready · holdout accuracy {score:.2f}%")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(host="127.0.0.1", port=5000, debug=False)
