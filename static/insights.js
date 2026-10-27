const chartFont = '"DM Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
const plotConfig = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"] };
const plotBase = {
  paper_bgcolor: "#ffffff",
  plot_bgcolor: "#ffffff",
  font: { family: chartFont, size: 11, color: "#66798d" },
  margin: { l: 54, r: 20, t: 16, b: 48 },
  hoverlabel: { bgcolor: "#10253b", bordercolor: "#10253b", font: { color: "#fff", family: chartFont } },
};

function pct(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function setMetric(id, value) {
  document.getElementById(id).innerHTML = `${(value * 100).toFixed(2)}<small>%</small>`;
}

function axisStyle(title) {
  return {
    title: { text: title, standoff: 10 },
    gridcolor: "#edf1f5",
    zerolinecolor: "#dfe6ed",
    linecolor: "#dfe6ed",
    ticks: "outside",
    tickcolor: "#dfe6ed",
    automargin: true,
  };
}

function drawConfusion(metrics) {
  const matrix = metrics.confusion_matrix;
  const labels = ["Normal", "Anomaly"];
  const text = matrix.map((row) => row.map((value) => value.toLocaleString()));
  Plotly.newPlot("confusion-chart", [{
    type: "heatmap",
    z: matrix,
    x: labels,
    y: labels,
    text,
    texttemplate: "%{text}",
    textfont: { size: 18, color: "#17324b" },
    colorscale: [[0, "#edf6ff"], [0.55, "#8bc4ee"], [1, "#347bd1"]],
    showscale: false,
    hovertemplate: "Actual %{y}<br>Predicted %{x}<br>Connections: %{z:,}<extra></extra>",
  }], {
    ...plotBase,
    margin: { l: 84, r: 16, t: 12, b: 55 },
    xaxis: { ...axisStyle("Predicted label"), side: "bottom" },
    yaxis: { ...axisStyle("Actual label"), autorange: "reversed" },
  }, plotConfig);
}

function drawCurves(data) {
  const metrics = data.metrics;
  Plotly.newPlot("roc-chart", [
    { x: data.curves.roc.x, y: data.curves.roc.y, type: "scatter", mode: "lines", name: `Model · AUC ${pct(metrics.roc_auc)}`, line: { color: "#347bd1", width: 3 }, hovertemplate: "FPR %{x:.3f}<br>TPR %{y:.3f}<extra></extra>" },
    { x: [0, 1], y: [0, 1], type: "scatter", mode: "lines", name: "Chance", line: { color: "#aebbc8", width: 1.5, dash: "dash" }, hoverinfo: "skip" },
  ], { ...plotBase, xaxis: { ...axisStyle("False-positive rate"), range: [0, 1] }, yaxis: { ...axisStyle("True-positive rate"), range: [0, 1.02] }, legend: { orientation: "h", y: -0.28, x: 0.02, font: { size: 9 } } }, plotConfig);

  Plotly.newPlot("pr-chart", [{
    x: data.curves.precision_recall.x,
    y: data.curves.precision_recall.y,
    type: "scatter",
    mode: "lines",
    name: `Model · AP ${pct(metrics.average_precision)}`,
    line: { color: "#1b9b72", width: 3 },
    hovertemplate: "Recall %{x:.3f}<br>Precision %{y:.3f}<extra></extra>",
  }], { ...plotBase, xaxis: { ...axisStyle("Recall"), range: [0, 1] }, yaxis: { ...axisStyle("Precision"), range: [0, 1.02] }, showlegend: false }, plotConfig);

  document.getElementById("roc-score").textContent = `AUC ${pct(metrics.roc_auc)}`;
  document.getElementById("ap-score").textContent = `AP ${pct(metrics.average_precision)}`;
}

function drawClassBalance(data) {
  Plotly.newPlot("class-chart", [
    { x: ["Training fit", "Validation"], y: [data.fit_normal_rows, data.validation_normal_rows], type: "bar", name: "Normal", marker: { color: "#1b9b72" }, hovertemplate: "%{x}<br>Normal: %{y:,}<extra></extra>" },
    { x: ["Training fit", "Validation"], y: [data.fit_anomaly_rows, data.validation_anomaly_rows], type: "bar", name: "Anomaly", marker: { color: "#cf5c55" }, hovertemplate: "%{x}<br>Anomaly: %{y:,}<extra></extra>" },
  ], { ...plotBase, barmode: "stack", xaxis: { ...axisStyle("Data split") }, yaxis: { ...axisStyle("Connections"), rangemode: "tozero" }, legend: { orientation: "h", y: 1.13, x: 0.02, font: { size: 9 } } }, plotConfig);
}

function drawImportance(data) {
  const top = data.feature_importances.slice(0, 15).reverse();
  Plotly.newPlot("importance-chart", [{
    x: top.map((item) => item.importance),
    y: top.map((item) => item.feature),
    type: "bar",
    orientation: "h",
    marker: { color: top.map((_item, index) => `rgba(52,123,209,${0.42 + (index / Math.max(top.length - 1, 1)) * 0.52})`) },
    hovertemplate: "%{y}<br>Importance: %{x:.5f}<extra></extra>",
  }], { ...plotBase, margin: { l: 150, r: 20, t: 8, b: 42 }, xaxis: { ...axisStyle("Aggregated impurity importance"), rangemode: "tozero" }, yaxis: { ...axisStyle(""), automargin: true } }, plotConfig);
}

function fillDetails(data) {
  const params = data.model_parameters;
  document.getElementById("detail-model").textContent = data.model_name;
  document.getElementById("detail-parameters").textContent = `${params.n_estimators} trees · max features ${params.max_features} · leaf size ${params.min_samples_leaf} · class weight ${params.class_weight} · seed ${params.random_state}`;
  document.getElementById("detail-split").textContent = `${data.training_rows.toLocaleString()} rows · ${(data.validation_split * 100).toFixed(0)}% stratified holdout`;
  document.getElementById("detail-class-counts").textContent = `Normal ${data.normal_rows.toLocaleString()} · Anomaly ${data.anomaly_rows.toLocaleString()} · ${data.feature_columns.length} input features`;
}

async function loadInsights() {
  try {
    const response = await fetch("/api/model-insights");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not load model insights.");
    const metrics = data.metrics;
    setMetric("metric-accuracy", metrics.accuracy);
    setMetric("metric-precision", metrics.precision_anomaly);
    setMetric("metric-recall", metrics.recall_anomaly);
    setMetric("metric-roc-auc", metrics.roc_auc);
    fillDetails(data);
    drawConfusion(metrics);
    drawCurves(data);
    drawClassBalance(data);
    drawImportance(data);
    document.getElementById("insight-status").textContent = "Model ready";
  } catch (error) {
    const box = document.getElementById("insights-error");
    box.textContent = error.message || "Could not load model insights.";
    box.hidden = false;
    document.getElementById("insight-status").textContent = "Model unavailable";
  }
}

loadInsights();
