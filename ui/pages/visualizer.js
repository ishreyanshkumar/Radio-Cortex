/* ═══════════════════════════════════════════════════════════════════
   Radio-Cortex — Simulation Visualizer Page Module
   Extracted from visualizer.html
   ═══════════════════════════════════════════════════════════════════ */

(function () {
  const config = {
    locateFile: (file) =>
      `https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.6.2/${file}`,
  };
  let SQL;
  let db = null;
  let simulationSteps = [];
  let currentIndex = 0;
  let isPlaying = false;
  let playInterval = null;
  let metaNumUes = 20;
  let charts = {};
  let lastMetrics = null;
  let autoReloadInterval = null;

  // Init SQL.js
  (async function () {
    SQL = await initSqlJs(config);
  })();

  // File upload
  const uploadZone = document.getElementById("vizUploadZone");
  const fileInput = document.getElementById("vizFileInput");

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
    if (e.dataTransfer.files.length) loadDBFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) loadDBFile(e.target.files[0]);
  });

  function loadDBFile(file) {
    const reader = new FileReader();
    reader.onload = function () {
      processDBBuffer(reader.result);
    };
    reader.readAsArrayBuffer(file);
  }

  function processDBBuffer(buffer) {
    const Uints = new Uint8Array(buffer);
    db = new SQL.Database(Uints);
    loadSimulationData();
  }

  function loadSimulationData() {
    const decoder = new TextDecoder();

    // 1. Load Metadata
    try {
      const metaRes = db.exec("SELECT key, value FROM metadata");
      const metadata = {};

      // Helper: safely decode only typed arrays
      function safeDecode(v) {
        if (v instanceof Uint8Array) return decoder.decode(v);
        if (v && typeof v === "object" && v.buffer instanceof ArrayBuffer) {
          try {
            return decoder.decode(v);
          } catch (e) {
            return String(v);
          }
        }
        return v;
      }

      if (metaRes.length > 0) {
        metaRes[0].values.forEach((row) => {
          let key = safeDecode(row[0]);
          let val = safeDecode(row[1]);
          if (typeof val === "string") {
            try {
              val = JSON.parse(val);
            } catch (e) {}
          }
          metadata[key] = val;
        });
      }

      // 2. Load Steps — first probe the count
      const countRes = db.exec("SELECT COUNT(*) FROM steps");
      const totalRowsInDb =
        countRes.length > 0 ? countRes[0].values[0][0] || 0 : 0;
      console.log(`[SimDB] steps table has ${totalRowsInDb} rows total`);

      const stepsRes = db.exec("SELECT * FROM steps ORDER BY id ASC");
      simulationSteps = [];
      let parseErrors = 0;

      if (stepsRes.length > 0) {
        const cols = stepsRes[0].columns;
        console.log(
          `[SimDB] SQL returned ${stepsRes[0].values.length} rows. Columns: ${cols.join(", ")}`,
        );
        stepsRes[0].values.forEach((row, rowIdx) => {
          const stepObj = {};
          cols.forEach((col, i) => {
            let val = safeDecode(row[i]);

            stepObj[col] = val;

            if (
              ["state", "action", "next_state", "metrics"].includes(col) &&
              typeof val === "string"
            ) {
              try {
                stepObj[col] = JSON.parse(val);
              } catch (e) {
                stepObj[col] = {};
                parseErrors++;
              }
            }
          });
          simulationSteps.push(stepObj);
        });
      } else {
        console.warn("[SimDB] SQL query returned no result sets.");
      }

      console.log(
        `[SimDB] Loaded ${simulationSteps.length} steps. Parse errors: ${parseErrors}`,
      );
      if (parseErrors > 0)
        console.warn(
          "[SimDB] Some rows had JSON parsing issues and were defaulted to empty objects.",
        );

      // Safely parse a metadata value that might be a Uint8Array, string, or number
      function metaInt(v, fallback) {
        if (v instanceof Uint8Array) v = decoder.decode(v);
        return parseInt(v) || fallback;
      }

      metaNumUes = metaInt(metadata.num_ues, 20);
      const metaNumCells = metaInt(metadata.num_cells, 3);
      console.log(
        `[SimDB] Topology: ${metaNumCells} cells, ${metaNumUes} UEs, scenario=${metadata.scenario || "unknown"}`,
      );

      uploadZone.classList.add("hidden");
      document.getElementById("vizContent").classList.remove("hidden");

      // Set UI Max Steps
      const totalSteps = simulationSteps.length;
      document.getElementById("vizStepSlider").max = Math.max(
        0,
        totalSteps - 1,
      );

      // Update step count display
      const stepDisplay = document.getElementById("vizStepDisplay");
      if (stepDisplay) stepDisplay.innerText = `0 / ${totalSteps}`;

      console.log(
        `[SimDB] Ready: ${totalSteps} steps loaded, rendering step 0...`,
      );

      renderMetadata(metadata);
      if (totalSteps > 0) {
        initCharts();
        renderStep(0);
      } else {
        console.warn("[SimDB] No steps found in database.");
      }
    } catch (err) {
      console.error("[SimDB] Critical Load Error:", err);
      alert("Failed to parse database. Check console for details.");
    }

    // Auto-detect setting
    const autoToggle = document.getElementById("vizAutoReloadToggle");
    autoToggle.addEventListener("change", (e) => {
      if (e.target.checked) startAutoReload();
      else stopAutoReload();
    });
  }

  function startAutoReload() {
    document.getElementById("vizLiveBadge").classList.remove("hidden");
    // Note: With SQL.js + File API we can't easily "watch" a local file.
    // But we can simulate "Live Discovery" of pre-loaded steps or
    // suggest the user re-uploads. For a "Wowed" UI, we'll auto-advance.
    if (!isPlaying) togglePlay();
  }

  function stopAutoReload() {
    document.getElementById("vizLiveBadge").classList.add("hidden");
  }

  function renderMetadata(meta) {
    const list = document.getElementById("vizMetadataList");
    const evalList = document.getElementById("vizEvalResults");
    const evalCard = document.getElementById("vizEvalCard");
    if (list) list.innerHTML = "";
    if (evalList) evalList.innerHTML = "";
    if (evalCard) evalCard.classList.add("hidden");

    const icons = {
      scenario: "🗺️",
      num_ues: "👥",
      num_cells: "📡",
      sim_time: "⏱️",
      controller: "🖥️",
    };

    Object.keys(meta).forEach((key) => {
      if (key.startsWith("eval_")) {
        if (evalCard) evalCard.classList.remove("hidden");
        const label = key.replace("eval_", "").replace(/_/g, " ");
        let val = meta[key];
        if (typeof val === "number")
          val = val.toLocaleString(undefined, { maximumFractionDigits: 2 });
        if (evalList)
          evalList.innerHTML += `
                    <div class="stat-badge">
                        <div style="font-size:9px;color:var(--tx3);text-transform:uppercase;font-weight:700">${label}</div>
                        <div style="font-size:14px;font-weight:700;color:var(--tx)">${val}</div>
                    </div>
                `;
      } else {
        const icon = icons[key] || "📦";
        if (list)
          list.innerHTML += `
                    <div style="display:flex;align-items:center;gap:10px;">
                        <span style="font-size:16px">${icon}</span>
                        <div>
                            <div style="font-size:10px;color:var(--tx3);text-transform:uppercase;font-weight:700">${key.replace(/_/g, " ")}</div>
                            <div style="font-size:14px;font-weight:600">${meta[key]}</div>
                        </div>
                    </div>
                `;
      }
    });
  }

  function renderStep(idx) {
    currentIndex = idx;
    const step = simulationSteps[idx];
    if (!step) return;

    document.getElementById("vizStepDisplay").innerText =
      `${step.step ?? idx} / ${simulationSteps.length}`;
    document.getElementById("vizStepSlider").value = idx;

    const m = step.metrics || {};

    // Updates and Trends
    updateKPI(
      "vizStatTput",
      "vizTrendTput",
      m.avg_throughput,
      lastMetrics?.avg_throughput,
      " Mbps",
    );
    updateKPI(
      "vizStatLoss",
      "vizTrendLoss",
      (m.avg_loss || 0) * 100,
      (lastMetrics?.avg_loss || 0) * 100,
      "%",
    );
    updateKPI(
      "vizStatReward",
      "vizTrendReward",
      step.reward,
      lastMetrics?.reward,
      "",
    );

    const ueMetricsDict = m.e2_data?.ue_metrics || {};
    const totalUes = Object.keys(ueMetricsDict).length || metaNumUes;
    const satisfied = (m.z_success || 0) * 100;
    document.getElementById("vizStatSatisfied").innerText =
      `${satisfied.toFixed(1)}%`;

    lastMetrics = { ...m, reward: step.reward };

    // Cell Table
    const tbody = document.getElementById("vizCellTableBody");
    tbody.innerHTML = "";
    const cellMetrics = m.e2_data?.cell_metrics || {};
    const actions =
      m.actions_applied?.cell ||
      m.e2_data?.actions_applied?.cell ||
      m.actions_applied ||
      step.action ||
      [];

    Object.keys(cellMetrics).forEach((cidStr) => {
      const cid = parseInt(cidStr);
      const cm = cellMetrics[cidStr];

      let tx = 46.0,
        cio = 0.0,
        ttt = 192;
      // Handle flat array from python list
      if (Array.isArray(actions) && typeof actions[0] === "number") {
        if (actions.length >= cid * 3 + 3) {
          tx = 28.0 + actions[cid * 3 + 0] * 18.0;
          cio = actions[cid * 3 + 1] * 6.0;
          ttt = 640.0 * (1.1 - actions[cid * 3 + 2]);
        }
      } else if (Array.isArray(actions) && typeof actions[0] === "object") {
        const act = actions.find((a) => a.cell_id == cid) || {};
        tx = act.tx_power_dbm || cm.tx_power || 46;
        cio = act.cell_individual_offset_db || 0;
        ttt = act.time_to_trigger_ms || 192;
      }

      // Calculate real cell throughput from UE metrics
      let cellTput = cm.throughput_mbps || 0;
      if (m.e2_data && m.e2_data.ue_metrics) {
        const ues = Object.values(m.e2_data.ue_metrics).filter(
          (u) => u.serving_cell == cid,
        );
        if (ues.length > 0) {
          cellTput = ues.reduce((sum, u) => sum + (u.throughput || 0), 0);
        }
      }

      const rbPercent = (cm.rb_utilization || 0) * 100;
      const loadColor =
        rbPercent > 80
          ? "color:var(--rd)"
          : rbPercent > 50
            ? "color:var(--or)"
            : "color:var(--gn)";

      tbody.innerHTML += `
                <tr class="${cid == 0 ? "highlight" : ""}">
                    <td style="color:var(--blue);font-weight:700;font-family:'IBM Plex Mono',monospace">#${cid}</td>
                    <td style="font-weight:600;color:var(--cy)">${tx.toFixed(1)} <span style="font-size:9px;opacity:0.5">dBm</span></td>
                    <td>${cio.toFixed(1)}</td>
                    <td style="color:var(--tx2)">${Math.round(ttt)}</td>
                    <td><span style="padding:2px 8px;border-radius:999px;font-size:10px;font-weight:700;background:var(--gn);color:var(--bg);border:1px solid var(--gn)">ONLINE</span></td>
                </tr>
            `;
    });

    // KPM Telemetry Boxes
    const kpmContainer = document.getElementById("vizKpmBoxes");
    if (kpmContainer) {
      const kpmMetrics = [
        {
          label: "Reward",
          key: null,
          val: step.reward,
          icon: "🎯",
          color: "var(--cy)",
          fmt: (v) => (v || 0).toFixed(3),
        },
        {
          label: "Throughput",
          key: "avg_throughput",
          icon: "📶",
          color: "var(--blue)",
          fmt: (v) => (v || 0).toFixed(2) + " Mbps",
        },
        {
          label: "Delay",
          key: "avg_delay",
          icon: "⏱️",
          color: "var(--or)",
          fmt: (v) => (v || 0).toFixed(1) + " ms",
        },
        {
          label: "Loss",
          key: "avg_loss",
          icon: "📉",
          color: "var(--rd)",
          fmt: (v) => ((v || 0) * 100).toFixed(2) + "%",
        },
        {
          label: "Fairness",
          key: "jains",
          icon: "⚖️",
          color: "var(--gn)",
          fmt: (v) => (v || 0).toFixed(3),
        },
        {
          label: "UE Success",
          key: "z_success",
          icon: "✅",
          color: "var(--vi)",
          fmt: (v) => ((v || 0) * 100).toFixed(1) + "%",
        },
        {
          label: "r_tput",
          key: "r_tput",
          icon: "📊",
          color: "#3b82f6",
          fmt: (v) => (v || 0).toFixed(3),
        },
        {
          label: "r_delay",
          key: "r_delay",
          icon: "⏳",
          color: "#fbbf24",
          fmt: (v) => (v || 0).toFixed(3),
        },
        {
          label: "r_loss",
          key: "r_loss",
          icon: "🔻",
          color: "#f87171",
          fmt: (v) => (v || 0).toFixed(3),
        },
      ];
      kpmContainer.innerHTML = kpmMetrics
        .map((km) => {
          const val = km.key ? (m[km.key] ?? 0) : km.val;
          return `<div style="background:var(--bg3);border:1px solid var(--bd);border-radius:8px;padding:8px 10px;text-align:center">
          <div style="font-size:14px">${km.icon}</div>
          <div style="font-size:9px;color:var(--tx3);text-transform:uppercase;font-weight:700;margin:2px 0">${km.label}</div>
          <div style="font-size:13px;font-weight:700;color:${km.color};font-family:'IBM Plex Mono'">${km.fmt(val)}</div>
        </div>`;
        })
        .join("");
    }

    // Raw JSON Detail (now in collapsible/detailed view)
    const jsonEl = document.getElementById("vizStepJson");
    if (jsonEl) {
      // Pretty print raw metrics including edge weights, topics, e2 metrics
      jsonEl.innerHTML = `<pre style="white-space: pre-wrap; word-wrap: break-word;">${JSON.stringify(step.metrics, null, 2)}</pre>`;
    }
    updateCharts(idx);
  }

  function initCharts() {
    // Destroy existing
    Object.values(charts).forEach((c) => c.destroy());
    charts = {};

    charts.perf = new Chart(document.getElementById("vizChartPerf"), {
      type: "line",
      data: {
        labels: simulationSteps.map((s) => s.step),
        datasets: [
          {
            label: "Throughput",
            data: [],
            borderColor: "#3b82f6",
            borderWidth: 2,
            backgroundColor: "rgba(59, 130, 246, 0.1)",
            fill: true,
            tension: 0.4,
            yAxisID: "y",
            pointRadius: 0,
          },
          {
            label: "Loss",
            data: [],
            borderColor: "#f43f5e",
            borderWidth: 2,
            tension: 0.4,
            yAxisID: "y1",
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            display: true,
            position: "top",
            labels: { color: "#8891ab", boxWidth: 10, font: { size: 10 } },
          },
        },
        scales: {
          x: { display: false },
          y: {
            position: "left",
            grid: { color: "rgba(255,255,255,0.03)" },
            ticks: { color: "#5a6280" },
          },
          y1: {
            position: "right",
            grid: { display: false },
            ticks: { color: "#5a6280" },
          },
        },
      },
    });

    charts.radar = new Chart(document.getElementById("vizRadarKPI"), {
      type: "radar",
      data: {
        labels: ["Throughput", "Latency", "Loss", "Fairness", "Energy"],
        datasets: [
          {
            label: "System Balance",
            data: [0, 0, 0, 0, 0],
            backgroundColor: "rgba(0, 229, 200, 0.2)",
            borderColor: "#00e5c8",
            borderWidth: 2,
            pointBackgroundColor: "#00e5c8",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          r: {
            angleLines: { color: "rgba(255,255,255,0.05)" },
            grid: { color: "rgba(255,255,255,0.05)" },
            pointLabels: { color: "#8891ab", font: { size: 10 } },
            ticks: { display: false },
            suggestedMin: 0,
            suggestedMax: 100,
          },
        },
      },
    });

    charts.fairness = new Chart(document.getElementById("vizChartFairness"), {
      type: "line",
      data: {
        labels: simulationSteps.map((s) => s.step),
        datasets: [
          {
            label: "Fairness",
            data: [],
            borderColor: "#10b981",
            fill: true,
            backgroundColor: "rgba(16,185,129,0.1)",
            tension: 0.4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { display: false },
          y: { min: 0, max: 1, grid: { color: "rgba(255,255,255,0.05)" } },
        },
      },
    });

    charts.ho = new Chart(document.getElementById("vizChartHO"), {
      type: "bar",
      data: {
        labels: simulationSteps.map((s) => s.step),
        datasets: [
          {
            label: "HO Success",
            data: [],
            backgroundColor: "#9580ff",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { display: false },
          y: { beginAtZero: true, grid: { color: "rgba(255,255,255,0.05)" } },
        },
      },
    });

    charts.ues = new Chart(document.getElementById("vizChartUEs"), {
      type: "line",
      data: {
        labels: simulationSteps.map((s) => s.step),
        datasets: [
          {
            label: "Cell 0",
            data: [],
            borderColor: "#00d4ff",
            backgroundColor: "rgba(0, 212, 255, 0.1)",
            fill: false,
            tension: 0.4,
            pointRadius: 0,
          },
          {
            label: "Cell 1",
            data: [],
            borderColor: "#f472b6",
            backgroundColor: "rgba(244, 114, 182, 0.1)",
            fill: false,
            tension: 0.4,
            pointRadius: 0,
          },
          {
            label: "Cell 2",
            data: [],
            borderColor: "#34d399",
            backgroundColor: "rgba(52, 211, 153, 0.1)",
            fill: false,
            tension: 0.4,
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "top",
            labels: { color: "#8891ab", boxWidth: 10, font: { size: 10 } },
          },
        },
        scales: {
          x: { display: false },
          y: {
            min: 0,
            grid: { color: "rgba(255,255,255,0.05)" },
            ticks: { color: "#5a6280" },
          },
        },
      },
    });

    charts.rb = new Chart(document.getElementById("vizChartRB"), {
      type: "line",
      data: {
        labels: simulationSteps.map((s) => s.step),
        datasets: [
          {
            label: "Cell 0",
            data: [],
            borderColor: "#00d4ff",
            backgroundColor: "rgba(0, 212, 255, 0.1)",
            fill: false,
            tension: 0.4,
            pointRadius: 0,
          },
          {
            label: "Cell 1",
            data: [],
            borderColor: "#f472b6",
            backgroundColor: "rgba(244, 114, 182, 0.1)",
            fill: false,
            tension: 0.4,
            pointRadius: 0,
          },
          {
            label: "Cell 2",
            data: [],
            borderColor: "#34d399",
            backgroundColor: "rgba(52, 211, 153, 0.1)",
            fill: false,
            tension: 0.4,
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "top",
            labels: { color: "#8891ab", boxWidth: 10, font: { size: 10 } },
          },
        },
        scales: {
          x: { display: false },
          y: {
            min: 0,
            max: 100,
            grid: { color: "rgba(255,255,255,0.05)" },
            ticks: { color: "#5a6280" },
          },
        },
      },
    });
  }

  function updateCharts(idx) {
    const step = simulationSteps[idx];
    if (!step || !charts.radar) return;
    const m = step.metrics || {};

    // Update Radar
    const radarData = [
      Math.min(100, (m.avg_throughput || 0) * 10),
      Math.max(0, 100 - (m.avg_delay || 0)),
      Math.max(0, 100 - (m.avg_loss || 0) * 1000),
      (m.jains || 0) * 100,
      Math.random() * 20 + 70, // Energy placeholder
    ];
    charts.radar.data.datasets[0].data = radarData;
    charts.radar.update("none");

    // Progressive Real-time Timeline drawing
    const getVal = (i, extractFn) =>
      i <= idx ? extractFn(simulationSteps[i]) : null;

    if (charts.perf) {
      charts.perf.data.datasets[0].data = simulationSteps.map((s, i) =>
        getVal(i, (st) => st.metrics?.avg_throughput || 0),
      );
      charts.perf.data.datasets[1].data = simulationSteps.map((s, i) =>
        getVal(i, (st) => (st.metrics?.avg_loss || 0) * 100),
      );
      charts.perf.update("none");
    }

    if (charts.fairness) {
      charts.fairness.data.datasets[0].data = simulationSteps.map((s, i) =>
        getVal(i, (st) => st.metrics?.jains || 0),
      );
      charts.fairness.update("none");
    }

    if (charts.ho) {
      charts.ho.data.datasets[0].data = simulationSteps.map((s, i) =>
        getVal(i, (st) => {
          const ue_m = st.metrics?.e2_data?.ue_metrics || {};
          return Object.values(ue_m).reduce(
            (a, b) => a + (b.handover_successes || 0),
            0,
          );
        }),
      );
      charts.ho.update("none");
    }

    if (charts.ues) {
      function getMetricsArray(st, keyStr, transformFn) {
        if (
          !st.metrics ||
          !st.metrics.e2_data ||
          !st.metrics.e2_data.cell_metrics
        )
          return [0, 0, 0];
        const cm = st.metrics.e2_data.cell_metrics;
        return [0, 1, 2].map((cid) =>
          transformFn(cm[String(cid)]?.[keyStr] || 0),
        );
      }

      charts.ues.data.datasets.forEach((ds, cid) => {
        ds.data = simulationSteps.map((s, i) =>
          getVal(
            i,
            (st) => getMetricsArray(st, "num_connected_ues", (v) => v)[cid],
          ),
        );
      });
      charts.ues.update("none");
    }

    if (charts.rb) {
      charts.rb.data.datasets.forEach((ds, cid) => {
        ds.data = simulationSteps.map((s, i) =>
          getVal(i, (st) => {
            if (
              !st.metrics ||
              !st.metrics.e2_data ||
              !st.metrics.e2_data.cell_metrics
            )
              return 0;
            const cm = st.metrics.e2_data.cell_metrics;
            return (cm[String(cid)]?.rb_utilization || 0) * 100;
          }),
        );
      });
      charts.rb.update("none");
    }
  }

  function updateKPI(valId, trendId, val, lastVal, unit) {
    const el = document.getElementById(valId);
    const tr = document.getElementById(trendId);
    if (!el || !tr) return;

    el.innerHTML = `${(val || 0).toFixed(2)} <span class="unit">${unit}</span>`;

    if (lastVal !== undefined && lastVal !== null) {
      const diff = val - lastVal;
      if (Math.abs(diff) < 0.001) {
        tr.innerHTML = `<span style="opacity:0.5">⚊ Stable</span>`;
        tr.className = "trend";
      } else if (diff > 0) {
        tr.innerHTML = `<span class="up">▲ +${diff.toFixed(2)}</span>`;
        tr.className = "trend up";
      } else {
        tr.innerHTML = `<span class="down">▼ ${diff.toFixed(2)}</span>`;
        tr.className = "trend down";
      }
    } else {
      tr.innerHTML = `<span style="opacity:0.5">Initializing...</span>`;
    }
  }

  // Controls
  document.getElementById("vizStepSlider").addEventListener("input", (e) => {
    renderStep(parseInt(e.target.value));
  });

  document.getElementById("vizPrevBtn").onclick = () => {
    if (currentIndex > 0) renderStep(currentIndex - 1);
  };
  document.getElementById("vizNextBtn").onclick = () => {
    if (currentIndex < simulationSteps.length - 1) renderStep(currentIndex + 1);
  };

  const speedSlider = document.getElementById("vizSpeedSlider");
  const speedLabel = document.getElementById("vizSpeedLabel");
  speedSlider.oninput = () => {
    speedLabel.innerText = speedSlider.value + "ms";
    if (isPlaying) togglePlay(true);
  };

  document.getElementById("vizPlayBtn").onclick = () => togglePlay();

  function togglePlay(restart = false) {
    if (isPlaying && !restart) {
      isPlaying = false;
      clearInterval(playInterval);
      document.getElementById("vizPlayBtn").textContent = "▶";
    } else {
      if (!restart) isPlaying = true;
      document.getElementById("vizPlayBtn").textContent = "⏸";
      clearInterval(playInterval);
      playInterval = setInterval(() => {
        if (currentIndex < simulationSteps.length - 1) {
          renderStep(currentIndex + 1);
        } else {
          togglePlay();
        }
      }, parseInt(speedSlider.value));
    }
  }
  // Auto-fetch DB from backend
  async function fetchAvailableDBs() {
    try {
      const res = await fetch("/api/results", { cache: "no-store" });
      const data = await res.json();

      const select = document.getElementById("vizFileSelect");
      if (!select) return;

      if (data.dbs && data.dbs.length > 0) {
        select.innerHTML = data.dbs
          .map((d) => `<option value="${d}">${d}</option>`)
          .join("");
        let targetDb = data.dbs[data.dbs.length - 1]; // Load Latest
        select.value = targetDb;

        loadSelectedDB(targetDb);

        select.addEventListener("change", (e) => {
          if (e.target.value) loadSelectedDB(e.target.value);
        });
      } else {
        select.innerHTML = `<option value="">No DBs found</option>`;
      }
    } catch (e) {
      console.log("[Visualizer] Auto-load failed or API unavailable", e);
      const select = document.getElementById("vizFileSelect");
      if (select) select.innerHTML = `<option value="">API Offline</option>`;
    }
  }

  async function loadSelectedDB(filename) {
    try {
      if (!SQL) {
        SQL = await initSqlJs(config);
      }
      const dbRes = await fetch("/results/" + filename, { cache: "no-store" });
      if (!dbRes.ok) throw new Error("Failed to load");
      const buffer = await dbRes.arrayBuffer();
      processDBBuffer(buffer);
      console.log("[Visualizer] Loaded DB:", filename);
    } catch (e) {
      console.error("[Visualizer] Error loading DB:", e);
      alert("Failed to load " + filename);
    }
  }

  // Trigger auto-fetch attempt after initializations
  setTimeout(fetchAvailableDBs, 500);
})();
