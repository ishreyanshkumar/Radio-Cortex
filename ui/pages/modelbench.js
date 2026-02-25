/* ═══════════════════════════════════════════════════════════════════
   Radio-Cortex — ModelBench Page Module
   Extracted from modelbench.html
   ═══════════════════════════════════════════════════════════════════ */

(function () {
  const MC = "Config_model_path",
    SC = "Scenario";
  const HI = [
    "Throughput_Mbps",
    "Jains_Fairness",
    "Satisfaction_Percent",
    "Cell_Edge_Tput",
    "Avg_SINR_dB",
    "Avg_RSRP_dBm",
    "EnergyEfficiency_Mbps_W",
    "HO_Success_Rate",
    "Control_Stability",
    "QoS_Score",
    "Reliability_Score",
    "Resource_Score",
    "Buffer_Score",
    "PHY_Score",
    "Architecture_Score",
  ];
  const LO = [
    "Avg_Delay_ms",
    "PacketLoss_Ratio",
    "p95_Delay_ms",
    "Max_PacketLoss",
    "QoS_Violations",
    "Total_Downtime_s",
    "Congestion_Intensity",
    "Avg_HO_per_UE",
    "Avg_Inference_ms",
    "Model_Params",
  ];
  const AM = [...HI, ...LO];
  const SH = {};
  AM.forEach((m) => {
    SH[m] = m
      .replace("_Mbps", "")
      .replace("_ms", "")
      .replace("_Ratio", "%")
      .replace("_Percent", "%")
      .replace("_dB", "dB")
      .replace("_dBm", "dBm")
      .replace("Avg_", "")
      .replace("Config_", "");
  });
  const P = [
    "#00e5c8",
    "#9580ff",
    "#50fa7b",
    "#ffb86c",
    "#ff5555",
    "#ff6eb4",
    "#f1fa8c",
    "#8be9fd",
    "#bd93f9",
    "#ff79c6",
    "#6272a4",
    "#44475a",
    "#00b8a0",
    "#e6db74",
    "#66d9ef",
    "#a6e22e",
    "#fd971f",
    "#ae81ff",
    "#f92672",
    "#75715e",
  ];
  let raw = [],
    A = {},
    Ch = {};

  // File upload
  const uploadZone = document.getElementById("mbUploadZone");
  const fileInput = document.getElementById("mbFileInput");

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
    if (e.dataTransfer.files.length) go(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", (e) => {
    if (e.target.files[0]) go(e.target.files[0]);
  });

  document.getElementById("mbNewFileBtn").addEventListener("click", reset);

  function go(file) {
    if (!file) return;
    const em = document.getElementById("mbErrorMsg");
    em.classList.add("hidden");
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (r) => {
        raw = r.data.filter((row) => row[MC]);
        if (!raw.length) {
          em.textContent = "No rows with Config_model_path found.";
          em.classList.remove("hidden");
          return;
        }
        document.getElementById("loadingOverlay").classList.add("on");
        setTimeout(() => {
          try {
            analyze();
            render();
          } catch (e) {
            alert("Error: " + e.message);
            console.error(e);
          }
          document.getElementById("loadingOverlay").classList.remove("on");
        }, 80);
      },
      error: (e) => {
        em.textContent = "Parse error: " + e.message;
        em.classList.remove("hidden");
      },
    });
  }

  function sn(m) {
    const s = String(m);
    if (s === "base" || s === "Baseline") return "Baseline";
    if (s.includes("/"))
      return s
        .split("/")
        .pop()
        .replace(/\.(pt|pth|ckpt|bin|onnx|h5)$/i, "");
    return s;
  }

  function analyze() {
    const models = [...new Set(raw.map((r) => r[MC]))].sort();
    const scenarios = [
      ...new Set(raw.map((r) => r[SC]).filter(Boolean)),
    ].sort();
    const snames = models.map(sn);
    const nc = {};
    snames.forEach((n, i) => {
      nc[n] = (nc[n] || 0) + 1;
      if (nc[n] > 1) snames[i] = n + "_" + nc[n];
    });
    const present = AM.filter((m) =>
      raw.some((r) => typeof r[m] === "number" && !isNaN(r[m])),
    );
    const hi = present.filter((m) => HI.includes(m)),
      lo = present.filter((m) => LO.includes(m));

    const means = {};
    present.forEach((m) => {
      means[m] = [];
    });
    models.forEach((model, mi) => {
      const rows = raw.filter((r) => r[MC] === model);
      present.forEach((m) => {
        const v = rows
          .map((r) => r[m])
          .filter((v) => typeof v === "number" && !isNaN(v));
        means[m][mi] = v.length ? v.reduce((a, b) => a + b, 0) / v.length : 0;
      });
    });

    const rk = {};
    present.forEach((m) => {
      const isH = hi.includes(m);
      const s = means[m]
        .map((v, i) => ({ v, i }))
        .sort((a, b) => (isH ? b.v - a.v : a.v - b.v));
      rk[m] = Array(models.length);
      s.forEach((it, r) => {
        rk[m][it.i] = r + 1;
      });
    });
    rk["AVG_RANK"] = models.map((_, mi) => {
      const r = present.map((m) => rk[m][mi]);
      return Math.round((r.reduce((a, b) => a + b, 0) / r.length) * 100) / 100;
    });

    const nm = {};
    present.forEach((m) => {
      const v = means[m],
        mn = Math.min(...v),
        mx = Math.max(...v),
        isH = hi.includes(m);
      nm[m] = v.map((x) =>
        mx === mn
          ? 50
          : isH
            ? ((x - mn) / (mx - mn)) * 100
            : ((mx - x) / (mx - mn)) * 100,
      );
    });

    const wins = models.map(() => 0);
    scenarios.forEach((sc) => {
      const sr = raw.filter((r) => r[SC] === sc);
      present.forEach((m) => {
        let bi = -1,
          bv = null;
        const isH = hi.includes(m);
        models.forEach((model, mi) => {
          const rows = sr.filter((r) => r[MC] === model);
          if (!rows.length) return;
          const vals = rows
            .map((r) => r[m])
            .filter((v) => typeof v === "number" && !isNaN(v));
          if (!vals.length) return;
          const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
          if (bv === null || (isH ? avg > bv : avg < bv)) {
            bv = avg;
            bi = mi;
          }
        });
        if (bi >= 0) wins[bi]++;
      });
    });

    const sd = {},
      sw = {};
    scenarios.forEach((sc) => {
      sd[sc] = {};
      sw[sc] = {};
      const sr = raw.filter((r) => r[SC] === sc);
      models.forEach((model, mi) => {
        const rows = sr.filter((r) => r[MC] === model);
        sd[sc][mi] = {};
        present.forEach((m) => {
          const v = rows
            .map((r) => r[m])
            .filter((v) => typeof v === "number" && !isNaN(v));
          sd[sc][mi][m] = v.length
            ? v.reduce((a, b) => a + b, 0) / v.length
            : null;
        });
      });
      present.forEach((m) => {
        const isH = hi.includes(m);
        let bi = -1,
          bv = null;
        models.forEach((_, mi) => {
          const v = sd[sc][mi]?.[m];
          if (v == null) return;
          if (bv === null || (isH ? v > bv : v < bv)) {
            bv = v;
            bi = mi;
          }
        });
        sw[sc][m] = bi;
      });
    });

    const fam = {};
    models.forEach((m, i) => {
      const s = String(m);
      let f;
      if (s === "base" || s === "Baseline") f = "Baseline";
      else if (s.includes("/")) {
        const p = s.split("/");
        f = p.length > 2 ? p.slice(0, -1).join("/") : p[0];
      } else f = s.replace(/[_-]?\d+$/, "");
      if (!fam[f]) fam[f] = [];
      fam[f].push(i);
    });

    A = {
      models,
      snames,
      scenarios,
      present,
      hi,
      lo,
      means,
      rk,
      nm,
      wins,
      sd,
      sw,
      fam,
    };
  }

  function mc(i) {
    return P[i % P.length];
  }
  function dc(id) {
    if (Ch[id]) {
      Ch[id].destroy();
      delete Ch[id];
    }
  }
  function fv(v) {
    return v < 0.01
      ? v.toFixed(6)
      : v < 1
        ? v.toFixed(4)
        : v < 100
          ? v.toFixed(2)
          : v.toFixed(1);
  }

  function render() {
    uploadZone.classList.add("hidden");
    document.getElementById("mbContent").classList.remove("hidden");
    document.getElementById("mbStatModels").textContent = A.models.length;
    document.getElementById("mbStatScenarios").textContent = A.scenarios.length;
    document.getElementById("mbStatRows").textContent = raw.length;
    Chart.defaults.font.family = "IBM Plex Mono";
    Chart.defaults.color = "#8891ab";
    rWin();
    rRank();
    rAR();
    rWC();
    rRadar();
    rExp();
    rSc();
    rScat();
    rCat();
    rSW();
  }

  function rWin() {
    const el = document.getElementById("mbWinners");
    el.innerHTML = "";
    const show = A.present.slice(0, Math.min(8, A.present.length));
    el.className =
      show.length > 6 ? "grid-4" : show.length > 3 ? "grid-3" : "grid-2";
    show.forEach((m) => {
      const isH = A.hi.includes(m);
      const bi = A.rk[m].indexOf(1);
      const d = document.createElement("div");
      d.className = "stat-badge";
      d.innerHTML = `<div class="stat-value">${fv(A.means[m][bi])}</div><div class="stat-label">${SH[m] || m} ${isH ? "↑" : "↓"}</div><div style="font-size:9px;color:var(--vi);margin-top:3px;word-break:break-all;font-family:'IBM Plex Mono',monospace">${A.snames[bi]}</div>`;
      el.appendChild(d);
    });
  }

  function rRank() {
    const t = document.getElementById("mbRankTable");
    const ms = [...A.present, "AVG_RANK"];
    const n = A.models.length;
    let h = "<thead><tr><th>Model</th>";
    ms.forEach((m) => {
      const d = m === "AVG_RANK" ? "" : A.hi.includes(m) ? " ↑" : " ↓";
      h += `<th title="${m}">${(SH[m] || m).slice(0, 14)}${d}</th>`;
    });
    h += "</tr></thead><tbody>";
    const si = A.snames
      .map((_, i) => i)
      .sort((a, b) => A.rk["AVG_RANK"][a] - A.rk["AVG_RANK"][b]);
    si.forEach((i) => {
      h += `<tr><td title="${A.models[i]}" style="font-weight:600">${A.snames[i]}</td>`;
      ms.forEach((m) => {
        const r = A.rk[m][i];
        let c = "";
        if (r === 1) c = "rank-1";
        else if (r === 2) c = "rank-2";
        else if (r === 3) c = "rank-3";
        else if (r === n) c = "rank-worst";
        else if (r >= n - 1) c = "rank-bad";
        h += `<td class="${c}">${m === "AVG_RANK" ? r.toFixed(2) : r}</td>`;
      });
      h += "</tr>";
    });
    t.innerHTML = h + "</tbody>";
  }

  function hb(id, lbl, dat, col, ttl) {
    dc(id);
    Ch[id] = new Chart(document.getElementById(id), {
      type: "bar",
      data: {
        labels: lbl,
        datasets: [{ data: dat, backgroundColor: col, borderRadius: 4 }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        plugins: {
          legend: { display: false },
          title: ttl
            ? { display: true, text: ttl, color: "#8891ab", font: { size: 11 } }
            : undefined,
        },
        scales: {
          x: { grid: { color: "#252a3a" } },
          y: { grid: { display: false }, ticks: { font: { size: 10 } } },
        },
      },
    });
  }

  function rAR() {
    const si = A.snames
      .map((_, i) => i)
      .sort((a, b) => A.rk["AVG_RANK"][a] - A.rk["AVG_RANK"][b]);
    hb(
      "mbChartAR",
      si.map((i) => A.snames[i]),
      si.map((i) => A.rk["AVG_RANK"][i]),
      si.map((i) => mc(i)),
    );
  }
  function rWC() {
    const si = A.snames.map((_, i) => i).sort((a, b) => A.wins[b] - A.wins[a]);
    hb(
      "mbChartW",
      si.map((i) => A.snames[i]),
      si.map((i) => A.wins[i]),
      si.map((i) => mc(i)),
    );
  }

  function mR(id, idx) {
    const rm = A.present.slice(0, Math.min(12, A.present.length));
    dc(id);
    Ch[id] = new Chart(document.getElementById(id), {
      type: "radar",
      data: {
        labels: rm.map((m) => (SH[m] || m).slice(0, 16)),
        datasets: idx.map((i) => ({
          label: A.snames[i],
          data: rm.map((m) => A.nm[m][i]),
          borderColor: mc(i),
          backgroundColor: mc(i) + "18",
          borderWidth: 2,
          pointRadius: 3,
          pointBackgroundColor: mc(i),
        })),
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            position: "bottom",
            labels: { boxWidth: 10, font: { size: 9 } },
          },
        },
        scales: {
          r: {
            angleLines: { color: "#252a3a" },
            grid: { color: "#252a3a" },
            pointLabels: { font: { size: 8 } },
            ticks: { display: false },
            min: 0,
            max: 100,
          },
        },
      },
    });
  }
  function rRadar() {
    const si = A.snames
      .map((_, i) => i)
      .sort((a, b) => A.rk["AVG_RANK"][a] - A.rk["AVG_RANK"][b]);
    mR("mbRadarTop", si.slice(0, 5));
    mR("mbRadarBot", si.slice(-5));
  }

  function rExp() {
    const sel = document.getElementById("mbExMetric");
    sel.innerHTML = "";
    A.present.forEach((m) => sel.add(new Option(m, m)));
    const draw = () => {
      const m = sel.value,
        isH = A.hi.includes(m),
        sb = document.getElementById("mbExSort").value;
      let idx = A.snames.map((_, i) => i);
      if (sb === "v")
        idx.sort((a, b) =>
          isH ? A.means[m][b] - A.means[m][a] : A.means[m][a] - A.means[m][b],
        );
      else idx.sort((a, b) => A.snames[a].localeCompare(A.snames[b]));
      dc("mbChartEx");
      Ch["mbChartEx"] = new Chart(document.getElementById("mbChartEx"), {
        type: "bar",
        data: {
          labels: idx.map((i) => A.snames[i]),
          datasets: [
            {
              data: idx.map((i) => A.means[m][i]),
              backgroundColor: idx.map((i) => {
                const r = A.rk[m][i];
                if (r === 1) return "#50fa7b";
                if (r <= 3) return "#86efac";
                if (r >= A.models.length) return "#ff5555";
                return "#8be9fd";
              }),
              borderRadius: 4,
            },
          ],
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            title: {
              display: true,
              text: `${m} ${isH ? "(higher=better)" : "(lower=better)"}`,
              color: "#8891ab",
              font: { size: 11 },
            },
          },
          scales: {
            x: { grid: { color: "#252a3a" } },
            y: { grid: { display: false }, ticks: { font: { size: 10 } } },
          },
        },
      });
    };
    sel.onchange = draw;
    document.getElementById("mbExSort").onchange = draw;
    draw();
  }

  function rSc() {
    const sel = document.getElementById("mbScMetric"),
      ns = document.getElementById("mbScN");
    sel.innerHTML = "";
    A.present.forEach((m) => sel.add(new Option(m, m)));
    const draw = () => {
      const m = sel.value,
        tn = parseInt(ns.value) || A.models.length;
      const si = A.snames
        .map((_, i) => i)
        .sort((a, b) => A.rk["AVG_RANK"][a] - A.rk["AVG_RANK"][b]);
      const show = si.slice(0, tn);
      dc("mbChartSc");
      Ch["mbChartSc"] = new Chart(document.getElementById("mbChartSc"), {
        type: "bar",
        data: {
          labels: A.scenarios,
          datasets: show.map((mi) => ({
            label: A.snames[mi],
            data: A.scenarios.map((sc) => A.sd[sc][mi]?.[m] ?? 0),
            backgroundColor: mc(mi) + "cc",
            borderRadius: 3,
          })),
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: "bottom",
              labels: { boxWidth: 10, font: { size: 9 } },
            },
            title: {
              display: true,
              text: m,
              color: "#8891ab",
              font: { size: 11 },
            },
          },
          scales: {
            x: {
              grid: { display: false },
              ticks: { maxRotation: 45, font: { size: 8 } },
            },
            y: { grid: { color: "#252a3a" } },
          },
        },
      });
    };
    sel.onchange = draw;
    ns.onchange = draw;
    draw();
  }

  function rScat() {
    const xs = document.getElementById("mbScatX"),
      ys = document.getElementById("mbScatY"),
      ss = document.getElementById("mbScatSc");
    xs.innerHTML = "";
    ys.innerHTML = "";
    ss.innerHTML = "";
    A.present.forEach((m) => {
      xs.add(new Option(m, m));
      ys.add(new Option(m, m));
    });
    if (A.present.includes("Throughput_Mbps")) xs.value = "Throughput_Mbps";
    if (A.present.includes("Avg_Delay_ms")) ys.value = "Avg_Delay_ms";
    else if (A.present.length > 1) ys.value = A.present[1];
    A.scenarios.forEach((s) => ss.add(new Option(s, s)));
    const draw = () => {
      const mx = xs.value,
        my = ys.value,
        sc = ss.value,
        sd = A.sd[sc];
      if (!sd) return;
      dc("mbChartScat");
      Ch["mbChartScat"] = new Chart(document.getElementById("mbChartScat"), {
        type: "bubble",
        data: {
          datasets: A.models
            .map((_, i) => {
              const xv = sd[i]?.[mx],
                yv = sd[i]?.[my];
              if (xv == null || yv == null) return null;
              const sat = sd[i]?.["Satisfaction_Percent"];
              const r = sat != null ? Math.max(5, sat / 6) : 7;
              return {
                label: A.snames[i],
                data: [{ x: xv, y: yv, r }],
                backgroundColor: mc(i) + "aa",
                borderColor: mc(i),
                borderWidth: 1.5,
              };
            })
            .filter(Boolean),
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: "bottom",
              labels: { boxWidth: 8, font: { size: 9 } },
            },
            title: {
              display: true,
              text: `${sc}: ${SH[mx] || mx} vs ${SH[my] || my}`,
              color: "#8891ab",
              font: { size: 11 },
            },
          },
          scales: {
            x: {
              title: { display: true, text: mx, color: "#8891ab" },
              grid: { color: "#252a3a" },
            },
            y: {
              title: { display: true, text: my, color: "#8891ab" },
              grid: { color: "#252a3a" },
            },
          },
        },
      });
    };
    xs.onchange = draw;
    ys.onchange = draw;
    ss.onchange = draw;
    draw();
  }

  function rCat() {
    const cats = Object.keys(A.fam);
    if (cats.length <= 1 || cats.length >= A.models.length) return;
    const rm = A.present.slice(0, Math.min(10, A.present.length));
    const cn = {};
    rm.forEach((m) => {
      const cm = cats.map((c) => {
        const idx = A.fam[c];
        return idx.reduce((s, i) => s + A.means[m][i], 0) / idx.length;
      });
      const mn = Math.min(...cm),
        mx = Math.max(...cm),
        isH = A.hi.includes(m);
      cn[m] = cm.map((v) =>
        mx === mn
          ? 50
          : isH
            ? ((v - mn) / (mx - mn)) * 100
            : ((mx - v) / (mx - mn)) * 100,
      );
    });
    dc("mbChartCat");
    Ch["mbChartCat"] = new Chart(document.getElementById("mbChartCat"), {
      type: "radar",
      data: {
        labels: rm.map((m) => (SH[m] || m).slice(0, 16)),
        datasets: cats.map((cat, ci) => ({
          label: cat,
          data: rm.map((m) => cn[m][ci]),
          borderColor: P[ci % P.length],
          backgroundColor: P[ci % P.length] + "20",
          borderWidth: 2.5,
          pointRadius: 4,
          pointBackgroundColor: P[ci % P.length],
        })),
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: "bottom",
            labels: { boxWidth: 12, font: { size: 10 }, padding: 14 },
          },
        },
        scales: {
          r: {
            angleLines: { color: "#252a3a" },
            grid: { color: "#252a3a" },
            pointLabels: { font: { size: 9 } },
            ticks: { display: false },
            min: 0,
            max: 100,
          },
        },
      },
    });
  }

  function rSW() {
    const t = document.getElementById("mbWinnerTable");
    const km = A.present.slice(0, Math.min(10, A.present.length));
    let h = "<thead><tr><th>Scenario</th>";
    km.forEach(
      (m) => (h += `<th title="${m}">${(SH[m] || m).slice(0, 12)}</th>`),
    );
    h += "</tr></thead><tbody>";
    A.scenarios.forEach((sc) => {
      h += `<tr><td style="font-weight:600">${sc}</td>`;
      km.forEach((m) => {
        const wi = A.sw[sc]?.[m];
        if (wi >= 0) {
          const v = A.sd[sc][wi]?.[m];
          h += `<td><span class="rank-1" title="${A.snames[wi]}: ${v != null ? fv(v) : "—"}">${A.snames[wi].slice(0, 14)}</span></td>`;
        } else h += "<td>—</td>";
      });
      h += "</tr>";
    });
    t.innerHTML = h + "</tbody>";
  }

  function reset() {
    uploadZone.classList.remove("hidden");
    document.getElementById("mbContent").classList.add("hidden");
    Object.keys(Ch).forEach((k) => {
      Ch[k].destroy();
      delete Ch[k];
    });
    fileInput.value = "";
    raw = [];
    A = {};
  }
  // Auto-fetch data from backend
  async function fetchAvailableFiles() {
    try {
      const res = await fetch("/api/results", { cache: "no-store" });
      const data = await res.json();

      const select = document.getElementById("mbFileSelect");
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
      console.log("[ModelBench] Auto-load failed or API unavailable", e);
      const select = document.getElementById("mbFileSelect");
      if (select) select.innerHTML = `<option value="">API Offline</option>`;
    }
  }

  async function loadSelectedFile(filename) {
    try {
      const res = await fetch("/results/" + filename, { cache: "no-store" });
      if (!res.ok) throw new Error("Failed to load");
      const text = await res.text();
      go(text);
      console.log("[ModelBench] Loaded:", filename);
    } catch (e) {
      console.error("[ModelBench] Error loading file:", e);
      alert("Failed to load " + filename);
    }
  }

  fetchAvailableFiles();
})();
