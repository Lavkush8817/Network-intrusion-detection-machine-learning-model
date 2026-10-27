const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const fileSelected = document.getElementById("file-selected");
const fileName = document.getElementById("file-name");
const fileSize = document.getElementById("file-size");
const analyzeButton = document.getElementById("analyze-button");
const errorMessage = document.getElementById("error-message");

function formatNumber(value) {
  return new Intl.NumberFormat().format(value);
}

function formatPercent(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function setFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".csv")) {
    showError("Please choose a CSV file.");
    return;
  }
  fileName.textContent = file.name;
  fileSize.textContent = `${(file.size / 1024).toFixed(file.size < 1024 * 1024 ? 0 : 1)} ${file.size < 1024 * 1024 ? "KB" : "MB"}`;
  dropZone.hidden = true;
  fileSelected.hidden = false;
  analyzeButton.disabled = false;
  errorMessage.hidden = true;
}

function clearFile() {
  fileInput.value = "";
  fileSelected.hidden = true;
  dropZone.hidden = false;
  analyzeButton.disabled = true;
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function setLoading(loading) {
  analyzeButton.classList.toggle("loading", loading);
  analyzeButton.disabled = loading || !fileInput.files.length;
  document.getElementById("button-label").textContent = loading ? "Analyzing traffic…" : "Analyze traffic";
}

function escapeText(value) {
  return value === null || value === undefined ? "—" : String(value);
}

function renderPreview(data) {
  const head = document.getElementById("preview-head");
  const body = document.getElementById("preview-body");
  head.replaceChildren();
  body.replaceChildren();
  const visibleColumns = data.columns.slice(0, 7);
  const headingRow = document.createElement("tr");
  visibleColumns.forEach((column) => {
    const th = document.createElement("th");
    th.textContent = column.replaceAll("_", " ").toUpperCase();
    headingRow.append(th);
  });
  head.append(headingRow);

  data.preview.forEach((row) => {
    const tr = document.createElement("tr");
    visibleColumns.forEach((column) => {
      const td = document.createElement("td");
      if (column === "detection_result") {
        const badge = document.createElement("span");
        const isAnomaly = row[column] === "anomaly";
        badge.className = `prediction-badge ${isAnomaly ? "anomaly" : "normal"}`;
        badge.textContent = isAnomaly ? "● Anomaly" : "● Normal";
        td.append(badge);
      } else {
        td.textContent = escapeText(row[column]);
      }
      tr.append(td);
    });
    body.append(tr);
  });
}

function renderResults(data) {
  const summary = data.summary;
  document.getElementById("result-total").textContent = formatNumber(summary.total);
  document.getElementById("result-normal").textContent = formatNumber(summary.normal);
  document.getElementById("result-anomaly").textContent = formatNumber(summary.anomaly);
  document.getElementById("normal-meter").style.width = `${(summary.normal / summary.total) * 100}%`;
  document.getElementById("anomaly-meter").style.width = `${(summary.anomaly / summary.total) * 100}%`;
  const accuracyCard = document.getElementById("upload-accuracy-card");
  if (summary.upload_accuracy !== null) {
    accuracyCard.hidden = false;
    document.getElementById("upload-accuracy").textContent = formatPercent(summary.upload_accuracy);
  } else {
    accuracyCard.hidden = true;
  }
  const download = document.getElementById("download-results");
  download.href = data.download_url;
  document.getElementById("results-section").hidden = false;
  renderPreview(data);
  document.getElementById("results-section").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Model is unavailable.");
    const score = data.metrics.accuracy;
    document.getElementById("system-status-text").textContent = "Model ready";
    document.getElementById("accuracy-value").innerHTML = `${(score * 100).toFixed(2)}<small>%</small>`;
    document.getElementById("accuracy-track").style.width = `${score * 100}%`;
    document.getElementById("training-rows").textContent = formatNumber(data.training_rows);
    document.getElementById("model-name").textContent = data.model_name;
    document.getElementById("feature-count").textContent = data.feature_count;
    document.getElementById("score-label").textContent = formatPercent(score);
    document.getElementById("score-track").style.width = `${score * 100}%`;
    document.getElementById("validation-rows").textContent = formatNumber(data.metrics.validation_rows);
    document.getElementById("anomaly-precision").textContent = formatPercent(data.metrics.precision_anomaly);
    document.getElementById("anomaly-recall").textContent = formatPercent(data.metrics.recall_anomaly);
  } catch (_error) {
    document.getElementById("system-status-text").textContent = "Model unavailable";
  }
}

fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
document.getElementById("remove-file").addEventListener("click", clearFile);
dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragover");
  const file = event.dataTransfer.files[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  setFile(file);
});

analyzeButton.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  errorMessage.hidden = true;
  setLoading(true);
  const form = new FormData();
  form.append("file", file);
  try {
    const response = await fetch("/api/predict", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Analysis failed.");
    renderResults(data);
  } catch (error) {
    showError(error.message || "Could not analyze this file. Please try again.");
  } finally {
    setLoading(false);
  }
});

loadHealth();
