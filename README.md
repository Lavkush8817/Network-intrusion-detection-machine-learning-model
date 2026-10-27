# Sentinel · Network Intrusion Detection

A local end-to-end machine learning app for classifying network connections as **normal** or **anomaly**. It includes a Python model backend, a browser-based dashboard, CSV batch predictions, interactive model insights, and a downloadable technical report.

## Model and accuracy

The app trains a Random Forest pipeline from the included `data/Train_data.csv`. Categorical fields are one-hot encoded, unseen categories are handled safely, and numeric gaps are imputed. The reported accuracy is measured on a stratified 20% holdout split with a fixed random seed; after measuring it, the deployed pipeline is refit on all labeled training rows.

With the included data and configuration, the holdout accuracy is around **99.8%** (5,039 validation rows). Exact metric values are written to `txt.md` on each training run; small changes can occur across scikit-learn versions. This is a dataset validation score, not a guarantee of performance on a different network or future traffic.

Open **Model Insights** in the dashboard to explore the confusion matrix, ROC and precision–recall curves, class balance, feature importance, and the full training configuration. Download `txt.md` from that page for the metric formulas, holdout results, confusion matrix, limitations, and static PNG chart previews.

## Run on macOS or Linux

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create an environment and install dependencies:

   ```bash
   python3 -m venv .venv
   .venv/bin/python -m pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   bash run.sh
   ```

   Or run `.venv/bin/python app.py` directly.

5. Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser.

## Run on Windows

1. Install Python 3.10 or newer and open Command Prompt in this folder.
2. Set up the environment:

   ```bat
   py -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   ```

3. Start the app with `run.bat` and open [http://127.0.0.1:5000](http://127.0.0.1:5000).

On first launch the model is trained and saved in `artifacts/`. Later launches reuse that local model. To retrain it, run `.venv/bin/python train_model.py` on macOS/Linux or `.venv\Scripts\python train_model.py` on Windows; this refreshes `txt.md` and the static chart previews as well.

## Analyze traffic

- Use **Download sample CSV** in the dashboard to get a correctly formatted example.
- Upload a CSV containing the network feature columns from `Train_data.csv`. The constant `num_outbound_cmds` field is not required.
- The dashboard reports normal/anomaly counts and shows the first 12 predictions. Select **Download full report** to save all rows with `detection_result` and `confidence_percent` columns.
- If an uploaded CSV also includes a `class` column with `normal`/`anomaly` values, the report includes row-level `prediction_matches` and the dashboard shows the accuracy on that labeled file.

The included `data/Test_data.csv` has no ground-truth labels, so predictions on it cannot be scored as test accuracy.

## Project files

- `app.py` — local Flask web server and CSV prediction API
- `model_backend.py` — training, validation, preprocessing, and model persistence
- `train_model.py` — retrain the model and print validation metrics
- `txt.md` — generated technical report with metric formulas and chart previews
- `templates/` and `static/` — browser dashboard
- `data/` — supplied train and test datasets
