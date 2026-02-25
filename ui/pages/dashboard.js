/* ═══════════════════════════════════════════════════════════════════
   Radio-Cortex — Dashboard Page Module
   Extracted from dashboard.html
   ═══════════════════════════════════════════════════════════════════ */

(function () {
  let allData = [];
  let filteredData = [];
  let headers = [];
  let radarChartInstance = null;
  let barChartInstance = null;
  const selectedIndices = new Set();
  const MAX_SELECTION = 100;
  let currentSort = { col: "Timestamp", asc: false };

  const SCORE_COLS = [
    "QoS_Score",
    "Reliability_Score",
    "Resource_Score",
    "Buffer_Score",
    "PHY_Score",
    "Architecture_Score",
  ];
  const CHART_COLORS = [
    "#00e5c8",
    "#ff5555",
    "#50fa7b",
    "#ffb86c",
    "#9580ff",
    "#ff6eb4",
  ];
  const HIDE_COLS = [
    "Controller",
    "_originalIndex",
    "Config_model_type",
    "Config_model_path",
  ];
  const PRIORITY_COLS = ["Model", "Scenario", "Throughput_Mbps"];
  const CARD_COLORS = [
    "#00e5c8",
    "#9580ff",
    "#50fa7b",
    "#ffb86c",
    "#ff6eb4",
    "#ff5555",
    "#3b82f6",
  ];

  // CSV parsing
  function parseCSV(text) {
    const lines = text.trim().split("\n");
    let rawHeaders = lines[0].split(",").map((h) => h.trim());
    const rows = [];
    for (let i = 1; i < lines.length; i++) {
      const vals = lines[i].split(",");
      const row = { _originalIndex: i - 1 };
      rawHeaders.forEach((h, j) => {
        const v = vals[j]?.trim();
        row[h] = isNaN(v) || v === "" ? v : parseFloat(v);
      });
      if (row["Config_model_type"]) row["Model"] = row["Config_model_type"];
      else row["Model"] = row["Controller"] || "Unknown";
      rows.push(row);
    }
    headers = rawHeaders;
    if (!headers.includes("Model")) headers.push("Model");
    return rows;
  }

  // File upload
  const uploadZone = document.getElementById("dashUploadZone");
  const fileInput = document.getElementById("dashFileInput");

  uploadZone.addEventListener("click", () => fileInput.click());
  uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.classList.add("dragover");
  });
  uploadZone.addEventListener("dragleave", () =>
    uploadZone.classList.remove("dragover"),
  );
  uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) loadFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) loadFile(e.target.files[0]);
  });

  function loadFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      processCsvText(e.target.result);
    };
    reader.readAsText(file);
  }

  function processCsvText(text) {
    allData = parseCSV(text);
    if (!allData.length) return alert("No data found in CSV");
    filteredData = [...allData];
    uploadZone.classList.add("hidden");
    document.getElementById("dashContent").classList.remove("hidden");
    selectedIndices.clear();
    populateFilters();
    render();
  }

  // Expose for button bindings
  document.getElementById("dashResetBtn").addEventListener("click", () => {
    selectedIndices.clear();
    render();
  });
  document
    .getElementById("dashDownloadBtn")
    .addEventListener("click", downloadFiltered);

  // Filter controls
  document.getElementById("dashApplyFilter").addEventListener("click", () => {
    const modelFilter = document.getElementById("dashFilterModel").value;
    const scenarioFilter = document.getElementById("dashFilterScenario").value;
    filteredData = allData.filter((r) => {
      if (modelFilter && (r.Model || r.Controller) !== modelFilter)
        return false;
      if (scenarioFilter && r.Scenario !== scenarioFilter) return false;
      return true;
    });
    render();
  });

  document.getElementById("dashClearFilter").addEventListener("click", () => {
    document.getElementById("dashFilterModel").value = "";
    document.getElementById("dashFilterScenario").value = "";
    filteredData = [...allData];
    render();
  });

  function populateFilters() {
    const modelSel = document.getElementById("dashFilterModel");
    const scenarioSel = document.getElementById("dashFilterScenario");
    modelSel.innerHTML = '<option value="">All Models</option>';
    scenarioSel.innerHTML = '<option value="">All Scenarios</option>';

    const models = [
      ...new Set(allData.map((r) => r.Model || r.Controller).filter(Boolean)),
    ].sort();
    const scenarios = [
      ...new Set(allData.map((r) => r.Scenario).filter(Boolean)),
    ].sort();

    models.forEach((m) => modelSel.add(new Option(m, m)));
    scenarios.forEach((s) => scenarioSel.add(new Option(s, s)));
  }

  function render() {
    renderTable();
    renderCards();
    renderRadar();
    renderBar();
    // Update row count
    const countEl = document.getElementById("dashRowCount");
    if (countEl)
      countEl.textContent = `(${filteredData.length} of ${allData.length} rows)`;
  }

  function renderCards() {
    const grid = document.getElementById("dashScoreGrid");
    if (selectedIndices.size === 0) {
      const cards = [
        {
          label: "Avg Score",
          value: "—",
          unit: "/ 100",
          color: CARD_COLORS[0],
        },
        {
          label: "Avg Throughput",
          value: "0.00",
          unit: "Mbps",
          color: CARD_COLORS[1],
        },
        {
          label: "Avg Satisfaction",
          value: "0.0",
          unit: "%",
          color: CARD_COLORS[2],
        },
        {
          label: "Avg Packet Loss",
          value: "0.00",
          unit: "%",
          color: CARD_COLORS[3],
        },
        { label: "Avg Delay", value: "0.0", unit: "ms", color: CARD_COLORS[4] },
        { label: "Avg SINR", value: "0.0", unit: "dB", color: CARD_COLORS[5] },
        { label: "Selected", value: "0", unit: "rows", color: CARD_COLORS[6] },
      ];
      grid.innerHTML = cards
        .map(
          (c) => `
                <div class="score-card" style="--accent:${c.color}">
                    <div class="label">${c.label}</div>
                    <div class="value" style="color:${c.color}">${c.value}</div>
                    <div class="unit">${c.unit}</div>
                </div>
            `,
        )
        .join("");
      return;
    }

    const selectedRows = [...selectedIndices].map((idx) => allData[idx]);
    const count = selectedRows.length;
    const getAvg = (key, scale = 1) => {
      let sum = 0,
        n = 0;
      selectedRows.forEach((r) => {
        const val = parseFloat(r[key]);
        if (!isNaN(val)) {
          sum += val;
          n++;
        }
      });
      return n ? (sum / n) * scale : 0;
    };

    let sumOfRowAvgs = 0,
      validRowAvgs = 0;
    selectedRows.forEach((r) => {
      let rowSum = 0,
        rowCnt = 0;
      SCORE_COLS.forEach((c) => {
        const val = parseFloat(r[c]);
        if (!isNaN(val)) {
          rowSum += val;
          rowCnt++;
        }
      });
      if (rowCnt > 0) {
        sumOfRowAvgs += rowSum / rowCnt;
        validRowAvgs++;
      }
    });
    const avgCompositeScore = validRowAvgs ? sumOfRowAvgs / validRowAvgs : 0;

    const cards = [
      {
        label: `Avg Score (Selection)`,
        value: avgCompositeScore.toFixed(1),
        unit: "/ 100",
        color: CARD_COLORS[0],
      },
      {
        label: "Avg Throughput",
        value: getAvg("Throughput_Mbps").toFixed(2),
        unit: "Mbps",
        color: CARD_COLORS[1],
      },
      {
        label: "Avg Satisfaction",
        value: getAvg("Satisfaction_Percent").toFixed(1),
        unit: "%",
        color: CARD_COLORS[2],
      },
      {
        label: "Avg Packet Loss",
        value: (getAvg("PacketLoss_Ratio") * 100).toFixed(2),
        unit: "%",
        color: CARD_COLORS[3],
      },
      {
        label: "Avg Delay",
        value: getAvg("Avg_Delay_ms").toFixed(1),
        unit: "ms",
        color: CARD_COLORS[4],
      },
      {
        label: "Avg SINR",
        value: getAvg("Avg_SINR_dB").toFixed(1),
        unit: "dB",
        color: CARD_COLORS[5],
      },
      { label: "Selected", value: count, unit: "rows", color: CARD_COLORS[6] },
    ];
    grid.innerHTML = cards
      .map(
        (c) => `
            <div class="score-card" style="--accent:${c.color}">
                <div class="label">${c.label}</div>
                <div class="value" style="color:${c.color}">${c.value}</div>
                <div class="unit">${c.unit}</div>
            </div>
        `,
      )
      .join("");
  }

  function renderRadar() {
    const canvas = document.getElementById("dashRadarChart");
    const legendDiv = document.getElementById("dashRadarLegend");
    const availableScores = SCORE_COLS.filter((c) => c in (allData[0] || {}));
    let labels = availableScores.map((c) =>
      c.replace("_Score", "").replace("_", " "),
    );
    let datasets = [];

    if (selectedIndices.size > 0) {
      datasets = [...selectedIndices].map((idx, i) => {
        const row = allData[idx];
        const scores = availableScores.map((c) => row[c] || 0);
        const color = CHART_COLORS[i % CHART_COLORS.length];
        let timeShort = "";
        if (row.Timestamp) {
          if (row.Timestamp.includes("T"))
            timeShort = row.Timestamp.split("T")[1].split(".")[0];
          else if (row.Timestamp.includes(" "))
            timeShort = row.Timestamp.split(" ")[1].split(".")[0];
          else timeShort = row.Timestamp;
        }
        const modelName = row.Model || row.Controller;
        return {
          label: `${modelName} @ ${timeShort}`,
          data: scores,
          borderColor: color,
          backgroundColor: color + "22",
          pointBackgroundColor: color,
          borderWidth: 2,
          pointRadius: 4,
        };
      });
    }

    if (radarChartInstance) radarChartInstance.destroy();
    radarChartInstance = new Chart(canvas, {
      type: "radar",
      data: { labels, datasets },
      options: {
        responsive: true,
        scales: {
          r: {
            beginAtZero: true,
            max: 100,
            grid: { color: "#252a3a" },
            angleLines: { color: "#252a3a" },
            pointLabels: { color: "#8891ab", font: { size: 11 } },
            ticks: {
              color: "#5a6280",
              backdropColor: "transparent",
              stepSize: 25,
            },
          },
        },
        plugins: { legend: { display: false } },
      },
    });

    legendDiv.innerHTML = datasets.length
      ? datasets
          .map(
            (ds) => `
            <span style="display:flex;align-items:center;gap:4px;font-size:11px;color:var(--tx2)">
                <span style="width:10px;height:10px;border-radius:50%;background:${ds.borderColor}"></span>${ds.label}
            </span>
        `,
          )
          .join("")
      : '<span style="color:var(--tx3);font-size:12px">Select rows from table to compare</span>';
  }

  function renderBar() {
    const canvas = document.getElementById("dashBarChart");
    let labels = [],
      datasets = [];
    if (selectedIndices.size > 0) {
      const selectedRows = [...selectedIndices].map((idx) => allData[idx]);
      labels = selectedRows.map((r) => {
        let t = "";
        if (r.Timestamp) {
          if (r.Timestamp.includes("T"))
            t = r.Timestamp.split("T")[1].split(".")[0];
          else if (r.Timestamp.includes(" "))
            t = r.Timestamp.split(" ")[1].split(".")[0];
          else t = r.Timestamp;
        }
        return `${r.Model || r.Controller} (${t})`;
      });
      datasets = [
        {
          label: "Throughput (Mbps)",
          data: selectedRows.map((r) => r.Throughput_Mbps || 0),
          backgroundColor: [...selectedIndices].map(
            (_, i) => CHART_COLORS[i % CHART_COLORS.length],
          ),
          borderColor: [...selectedIndices].map(
            (_, i) => CHART_COLORS[i % CHART_COLORS.length],
          ),
          borderWidth: 1,
          borderRadius: 4,
          barPercentage: 0.6,
        },
      ];
    }
    if (barChartInstance) barChartInstance.destroy();
    barChartInstance = new Chart(canvas, {
      type: "bar",
      data: { labels, datasets },
      options: {
        responsive: true,
        scales: {
          x: {
            grid: { color: "#252a3a" },
            ticks: { color: "#8891ab", font: { size: 10 } },
          },
          y: {
            grid: { color: "#252a3a" },
            ticks: { color: "#8891ab" },
            title: { display: true, text: "Mbps", color: "#5a6280" },
            beginAtZero: true,
          },
        },
        plugins: { legend: { display: false } },
      },
    });
  }

  function renderTable() {
    let tableCols = headers.filter((c) => !HIDE_COLS.includes(c));
    const orderedCols = [];
    if (headers.includes("Model")) orderedCols.push("Model");
    else if (headers.includes("Controller")) orderedCols.push("Controller");
    if (headers.includes("Scenario")) orderedCols.push("Scenario");
    if (headers.includes("Throughput_Mbps"))
      orderedCols.push("Throughput_Mbps");
    tableCols.forEach((c) => {
      if (!orderedCols.includes(c) && !c.startsWith("Config_"))
        orderedCols.push(c);
    });

    const selectionColors = {};
    [...selectedIndices].forEach((idx, i) => {
      selectionColors[idx] = CHART_COLORS[i % CHART_COLORS.length];
    });

    document.getElementById("dashTableHead").innerHTML =
      '<tr><th style="width:40px">Compare</th>' +
      orderedCols
        .map((c) => {
          const isSorted = currentSort.col === c;
          const arrow = isSorted ? (currentSort.asc ? " ▲" : " ▼") : "";
          return `<th style="cursor:pointer;user-select:none" onclick="window._dashToggleSort('${c}')">${c.replace(/_/g, " ")}${arrow}</th>`;
        })
        .join("") +
      "</tr>";

    document.getElementById("dashTableBody").innerHTML = filteredData
      .map((r) => {
        const isSelected = selectedIndices.has(r._originalIndex);
        const color = isSelected
          ? selectionColors[r._originalIndex]
          : "transparent";
        const style = isSelected
          ? `style="background:${color}22;border-left:4px solid ${color}"`
          : "";
        const tooltipContent = headers
          .filter((h) => h.startsWith("Config_"))
          .map((h) => {
            const val = r[h];
            if (val === undefined || val === "") return null;
            return `${h}: ${val}`;
          })
          .filter(Boolean)
          .join("\n");
        return (
          `<tr ${style} title="${tooltipContent.replace(/"/g, "&quot;")}">
                <td style="text-align:center"><input type="checkbox" ${isSelected ? "checked" : ""} onchange="window._dashToggleSelection(${r._originalIndex})" ${!isSelected && selectedIndices.size >= MAX_SELECTION ? "disabled" : ""}></td>` +
          orderedCols
            .map((c) => {
              let v = r[c];
              if (typeof v === "number") v = v.toFixed(4).replace(/\.?0+$/, "");
              return `<td>${v ?? "—"}</td>`;
            })
            .join("") +
          "</tr>"
        );
      })
      .join("");
  }

  // Expose sort/selection globally for inline handlers
  window._dashToggleSort = function (col) {
    if (currentSort.col === col) currentSort.asc = !currentSort.asc;
    else {
      currentSort.col = col;
      currentSort.asc = true;
      if (
        col === "Timestamp" ||
        col.includes("Score") ||
        col === "Throughput_Mbps"
      )
        currentSort.asc = false;
    }
    filteredData.sort((a, b) => {
      let va = a[currentSort.col],
        vb = b[currentSort.col];
      if (va === undefined) va = "";
      if (vb === undefined) vb = "";
      if (typeof va === "number" && typeof vb === "number")
        return currentSort.asc ? va - vb : vb - va;
      va = String(va).toLowerCase();
      vb = String(vb).toLowerCase();
      if (va < vb) return currentSort.asc ? -1 : 1;
      if (va > vb) return currentSort.asc ? 1 : -1;
      return 0;
    });
    render();
  };

  window._dashToggleSelection = function (idx) {
    if (selectedIndices.has(idx)) selectedIndices.delete(idx);
    else {
      if (selectedIndices.size >= MAX_SELECTION)
        return alert(`Max ${MAX_SELECTION} rows`);
      selectedIndices.add(idx);
    }
    render();
  };

  function downloadFiltered() {
    const csvContent = [headers.join(",")]
      .concat(filteredData.map((r) => headers.map((h) => r[h] ?? "").join(",")))
      .join("\n");
    const blob = new Blob([csvContent], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "filtered_results.csv";
    a.click();
  }
  // Auto-fetch data from backend
  async function fetchAvailableFiles() {
    try {
      const res = await fetch("/api/results", { cache: "no-store" });
      const data = await res.json();

      const select = document.getElementById("dashFileSelect");
      if (!select) return;

      if (data.csvs && data.csvs.length > 0) {
        select.innerHTML = data.csvs
          .map((c) => `<option value="${c}">${c}</option>`)
          .join("");
        let targetCsv = data.csvs.includes("experiment_results.csv")
          ? "experiment_results.csv"
          : data.csvs[data.csvs.length - 1];
        select.value = targetCsv;

        loadSelectedFile(targetCsv);

        select.addEventListener("change", (e) => {
          if (e.target.value) loadSelectedFile(e.target.value);
        });
      } else {
        select.innerHTML = `<option value="">No CSVs found</option>`;
      }
    } catch (e) {
      console.log("[Dashboard] Auto-load failed or API unavailable", e);
      const select = document.getElementById("dashFileSelect");
      if (select) select.innerHTML = `<option value="">API Offline</option>`;
    }
  }

  async function loadSelectedFile(filename) {
    try {
      const res = await fetch("/results/" + filename, { cache: "no-store" });
      if (!res.ok) throw new Error("Failed to load");
      const text = await res.text();
      processCsvText(text);
      console.log("[Dashboard] Loaded:", filename);
    } catch (e) {
      console.error("[Dashboard] Error loading file:", e);
      alert("Failed to load " + filename);
    }
  }

  fetchAvailableFiles();
})();
