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
      const Uints = new Uint8Array(reader.result);
      db = new SQL.Database(Uints);
      loadSimulationData();
    };
    reader.readAsArrayBuffer(file);
  }

  function loadSimulationData() {
    const decoder = new TextDecoder();

    // 1. Load Metadata
    const metaRes = db.exec("SELECT key, value FROM metadata");
    const metadata = {};
    if (metaRes.length > 0) {
      metaRes[0].values.forEach((row) => {
        let key = row[0];
        let val = row[1];
        if (key instanceof Uint8Array) key = decoder.decode(key);
        if (val instanceof Uint8Array) val = decoder.decode(val);
        try {
          val = JSON.parse(val);
        } catch (e) {}
        metadata[key] = val;
      });
    }

    // 2. Load Steps
    const stepsRes = db.exec("SELECT * FROM steps ORDER BY id ASC");
    simulationSteps = [];
    if (stepsRes.length > 0) {
      const cols = stepsRes[0].columns;
      stepsRes[0].values.forEach((row) => {
        const stepObj = {};
        cols.forEach((col, i) => {
          let val = row[i];
          if (val instanceof Uint8Array) val = decoder.decode(val);
          stepObj[col] = val;
          if (["state", "action", "next_state", "metrics"].includes(col)) {
            try {
              stepObj[col] = JSON.parse(val);
            } catch (e) {
              stepObj[col] = {};
            }
          }
        });
        simulationSteps.push(stepObj);
      });
    }

    metaNumUes = parseInt(metadata.num_ues) || 20;

    uploadZone.classList.add("hidden");
    document.getElementById("vizContent").classList.remove("hidden");

    // Set UI Max Steps
    document.getElementById("vizStepSlider").max = simulationSteps.length - 1;

    renderMetadata(metadata);
    initCharts();
    renderStep(0);
  }

  function renderMetadata(meta) {
    const list = document.getElementById("vizMetadataList");
    const evalList = document.getElementById("vizEvalResults");
    const evalCard = document.getElementById("vizEvalCard");
    list.innerHTML = "";
    evalList.innerHTML = "";
    evalCard.classList.add("hidden");

    const icons = {
      scenario: "🗺️",
      num_ues: "👥",
      num_cells: "📡",
      sim_time: "⏱️",
      controller: "🖥️",
    };

    Object.keys(meta).forEach((key) => {
      if (key.startsWith("eval_")) {
        evalCard.classList.remove("hidden");
        const label = key.replace("eval_", "").replace(/_/g, " ");
        let val = meta[key];
        if (typeof val === "number")
          val = val.toLocaleString(undefined, { maximumFractionDigits: 2 });
        evalList.innerHTML += `
                    <div class="stat-badge">
                        <div style="font-size:9px;color:var(--tx3);text-transform:uppercase;font-weight:700">${label}</div>
                        <div style="font-size:14px;font-weight:700;color:var(--tx)">${val}</div>
                    </div>
                `;
      } else {
        const icon = icons[key] || "📦";
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

    document.getElementById("vizStepDisplay").innerText = step.step;
    document.getElementById("vizStepSlider").value = idx;

    const m = step.metrics || {};
    document.getElementById("vizStatTput").innerHTML =
      `${(m.avg_throughput || 0).toFixed(2)} <span style="font-size:12px;color:var(--tx2)">Mbps</span>`;
    document.getElementById("vizStatLoss").innerText =
      `${((m.avg_loss || 0) * 100).toFixed(1)}%`;
    document.getElementById("vizStatReward").innerText = (
      step.reward || 0
    ).toFixed(2);

    const ueMetricsDict = m.e2_data?.ue_metrics || {};
    const totalUes = Object.keys(ueMetricsDict).length || metaNumUes;
    const satisfied = (m.z_success || 0) * totalUes;
    document.getElementById("vizStatSatisfied").innerText =
      `${Math.round(satisfied)} / ${totalUes}`;

    // Cell Table
    const tbody = document.getElementById("vizCellTableBody");
    tbody.innerHTML = "";
    const cellMetrics = m.e2_data?.cell_metrics || {};
    const actions =
      m.actions_applied?.cell || m.e2_data?.actions_applied?.cell || [];

    Object.keys(cellMetrics).forEach((cid) => {
      const cm = cellMetrics[cid];
      const act = actions.find((a) => a.cell_id == cid) || {};
      const load = (cm.rb_utilization || 0) * 100;
      const loadColor =
        load > 80
          ? "color:var(--rd)"
          : load > 50
            ? "color:var(--or)"
            : "color:var(--gn)";

      tbody.innerHTML += `
                <tr>
                    <td style="color:var(--blue);font-weight:700;font-family:'IBM Plex Mono',monospace">#${cid}</td>
                    <td style="font-weight:600">${(act.tx_power_dbm || 46).toFixed(1)}</td>
                    <td>${(act.cell_individual_offset_db || 0).toFixed(1)}</td>
                    <td style="color:var(--tx2)">${Math.round(act.time_to_trigger_ms || 192)}</td>
                    <td>${cm.num_connected_ues || 0}</td>
                    <td style="font-weight:700;${loadColor}">${load.toFixed(1)}%</td>
                    <td><span style="padding:2px 8px;border-radius:999px;font-size:10px;font-weight:700;background:var(--gn);color:var(--bg);border:1px solid var(--gn)">ACTIVE</span></td>
                </tr>
            `;
    });

    // JSON Detail
    document.getElementById("vizStepJson").innerText = JSON.stringify(
      step.metrics,
      null,
      2,
    );
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
            label: "Tput (Mbps)",
            data: simulationSteps.map((s) => s.metrics?.avg_throughput || 0),
            borderColor: "#3b82f6",
            tension: 0.4,
            yAxisID: "y",
          },
          {
            label: "Loss (%)",
            data: simulationSteps.map((s) => (s.metrics?.avg_loss || 0) * 100),
            borderColor: "#f43f5e",
            tension: 0.4,
            yAxisID: "y1",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { display: false },
          y: { position: "left", grid: { color: "rgba(255,255,255,0.05)" } },
          y1: { position: "right", grid: { display: false } },
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
            data: simulationSteps.map((s) => s.metrics?.jains || 0),
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
            data: simulationSteps.map((s) => {
              const ue_m = s.metrics?.e2_data?.ue_metrics || {};
              return Object.values(ue_m).reduce(
                (a, b) => a + (b.handover_successes || 0),
                0,
              );
            }),
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
  }

  function updateCharts(idx) {
    // Could add vertical line indicator — skipped for performance
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
})();
