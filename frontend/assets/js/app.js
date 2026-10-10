// ==========================================================================
// Adaptive Evolution Research Console v4.2 — Client Application Logic
// ==========================================================================

const $ = id => document.getElementById(id);

let S = null;
let selectedGen = 0;
let selectedDetail = null;
let pollTimer = null;
let elapsedTimer = null;
let autoFollow = true;
let pdfLoadedForRun = null;
let pdfPage = 1;
let pdfZoom = 100;
let genLoadToken = 0;
const historyByGen = new Map();

// Helpers
async function api(url, opt = {}) {
  const r = await fetch(url, opt);
  const d = await r.json();
  if (!r.ok) throw new Error(d.detail || d.message || "Request failed");
  return d;
}

const fmt = (x, d = 2) => {
  if (x == null || isNaN(x)) return "—";
  return Number(x).toLocaleString(undefined, { maximumFractionDigits: d });
};

const esc = x => String(x ?? "").replace(/[&<>"']/g, m => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
}[m]));

const sec = s => {
  if (s == null || isNaN(s)) return "00:00";
  s = Math.max(0, Math.floor(s));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const q = s % 60;
  if (h > 0) {
    return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(q).padStart(2, "0")}`;
  }
  return `${String(m).padStart(2, "0")}:${String(q).padStart(2, "0")}`;
};

function getElapsed() {
  if (!S?.run?.started) return 0;
  return (S.run.finished ?? (Date.now() / 1000)) - S.run.started;
}

// --------------------------------------------------------------------------
// Persistent Coordinate Store & Registration
// --------------------------------------------------------------------------
const CITY_COORDINATES_MAP = new Map();

function registerCoordinates(coordsList) {
  if (!Array.isArray(coordsList)) return;
  coordsList.forEach((p, idx) => {
    if (!p) return;
    const city = p.city != null ? Number(p.city) : idx;
    const x = Number(p.x);
    const y = Number(p.y);
    if (!isNaN(x) && !isNaN(y)) {
      CITY_COORDINATES_MAP.set(city, { city, x, y });
    }
  });
}

// --------------------------------------------------------------------------
// Core UI Render
// --------------------------------------------------------------------------
function render(s) {
  if (!s) return;
  // Preserve coordinates, route, and pipeline data across lightweight polling cycles
  if (S) {
    if (!s.route && S.route) s.route = S.route;
    if (!s.coordinates && S.coordinates) s.coordinates = S.coordinates;
    if (!s.initial && S.initial) s.initial = S.initial;
  }
  S = s;
  registerCoordinates(s.coordinates || s.dataset?.coordinates || s.route);
  const d = s.dataset || {};
  const st = s.status || {};
  const cfg = s.config || {};
  const run = s.run || {};
  const records = s.records || [];

  // Left Column: Dataset Meta
  $("dsName").textContent = d.name || "—";
  $("dsCities").textContent = d.cities ? `${d.cities} cities` : "—";
  $("dsWeight").textContent = d.weight_type || "—";
  $("dsOpt").textContent = d.best_known != null ? fmt(d.best_known) : "Unknown";
  $("dsBadge").textContent = d.name ? "Loaded" : "None";
  $("dsBadge").className = `badge ${d.name ? "badge-accent" : ""}`;

  // Config Info
  $("popInfo").textContent = `${cfg.population_size || 50}`;
  $("maxInfo").textContent = `${cfg.max_generations || 300}`;
  $("stagLimitInfo").textContent = `${cfg.no_improvement_limit || 30} gens`;
  $("kInfo").textContent = `${cfg.k_neighbors || 15}`;
  $("mutInfo").textContent = `${cfg.mutation_rate || 0.12}`;
  $("lkhInfo").textContent = `every ${cfg.lkh_cadence || 15}`;

  // Floating Header: Live identity + status
  $("hdDataset").textContent = d.name ? `${d.name}${d.cities ? ` · ${d.cities} cities` : ""}` : "No dataset";
  $("hdRunid").textContent = s.run_id ? `Run ${esc(s.run_id)}` : "Run —";
  const chip = $("hdStatus");
  const hdDot = $("hdStatusDot");
  chip.className = "hd-status-chip";
  if (run.active) {
    chip.classList.add("is-running");
    chip.lastElementChild.textContent = "Running";
    hdDot.style.animation = "none";
  } else if (st.is_stopped) {
    chip.classList.add(st.converged ? "is-done" : "is-stopped");
    chip.lastElementChild.textContent = st.converged ? "Converged" : "Stopped";
    hdDot.style.animation = "none";
  } else {
    chip.classList.add("is-idle");
    chip.lastElementChild.textContent = "Idle";
    hdDot.style.animation = "none";
  }

  // Center Column: Metrics Ribbon
  $("mCost").textContent = fmt(st.best_cost);
  if (st.gap != null) {
    $("mGap").textContent = `${fmt(st.gap, 4)}%`;
    $("mGap").style.color = st.gap < 0.0001 ? "var(--green)" : "var(--text)";
  } else {
    $("mGap").textContent = "—";
    $("mGap").style.color = "var(--text)";
  }
  $("mGen").textContent = st.generation ?? 0;
  $("mBest").textContent = st.best_generation ?? 0;
  $("mDiv").textContent = s.latest?.diversity != null ? fmt(s.latest.diversity, 4) : "—";
  $("mElapsed").textContent = sec(getElapsed());
  $("mEta").textContent = run.active ? `${st.eta_text || "ETA: calculating..."}` : "Run idle";

  // Priority metric emphasis: highlight cost/gap when at/near known optimum
  const opt = d.best_known;
  $("mCostSub").textContent = opt != null ? `Known optimum ${fmt(opt, 0)}` : "Best tour found";
  $("cardCost").classList.toggle("is-optimal", st.gap != null && st.gap <= 0.0001);
  $("cardGap").classList.toggle("is-optimal", st.gap != null && st.gap <= 0.0001);

  // Stagnation Watchdog Meter (Critical Requirement)
  const patience = Number(st.stagnation_patience || cfg.no_improvement_limit || 30);
  const noImp = Number(st.no_improvement ?? 0);
  const stagPct = Math.min(100, Math.max(0, (noImp / patience) * 100));

  $("stagBadge").textContent = `${noImp} / ${patience} GENS`;
  $("stagFill").style.width = `${stagPct}%`;

  const dot = $("watchdogDot");
  const badge = $("stagBadge");
  const fill = $("stagFill");

  dot.className = "watchdog-pulse";
  badge.className = "stagnation-badge";
  fill.className = "stagnation-bar-fill";

  if (noImp >= 20) {
    dot.classList.add("critical");
    badge.classList.add("critical");
    fill.classList.add("critical");
    $("stagDesc").textContent = `Critical: ${patience - noImp} generation(s) remaining before automatic termination!`;
  } else if (noImp >= 10) {
    dot.classList.add("warning");
    badge.classList.add("warning");
    fill.classList.add("warning");
    $("stagDesc").textContent = `Watchdog warning: No global improvement for ${noImp} consecutive generations.`;
  } else {
    $("stagDesc").textContent = noImp === 0
      ? `Global best improved at generation ${st.best_generation ?? 0}. Counter is reset.`
      : `Algorithm evolving normally (${noImp} generations without improvement).`;
  }

  // Live Status Bar
  const maxGen = Math.max(1, cfg.max_generations || 300);
  const progPct = Math.min(100, (st.generation / maxGen) * 100);
  $("progressFill").style.width = `${progPct}%`;
  $("progressText").textContent = `${st.generation} / ${maxGen} generations (${fmt(progPct, 1)}%)`;

  const liveDot = $("liveDot");
  if (run.active) {
    liveDot.style.background = "var(--blue)";
    liveDot.style.boxShadow = "0 0 10px var(--blue)";
    $("liveTitle").textContent = `Generation ${st.generation + 1} evolving...`;
    $("liveMeta").textContent = `Generation ${st.generation} committed. Background worker active.`;
  } else if (st.is_stopped || !run.active && st.generation > 0) {
    liveDot.style.background = st.converged ? "var(--green)" : "var(--amber)";
    liveDot.style.boxShadow = `0 0 10px ${st.converged ? "var(--green)" : "var(--amber)"}`;
    $("liveTitle").textContent = st.stop_reason || (st.converged ? "Optimal Solution Found" : "Experiment Stopped");
    $("liveMeta").textContent = `${s.generations_available?.length || records.length} completed generation record(s) persisted.`;
  } else {
    liveDot.style.background = "var(--text-dim)";
    liveDot.style.boxShadow = "none";
    $("liveTitle").textContent = "Ready to start";
    $("liveMeta").textContent = "Upload a dataset or click Run Full Experiment to begin.";
  }

  // Terminal Stop Banner (Phase 1 & Phase 2 requirement)
  renderTerminalBanner(s);

  // Button States
  $("runBtn").disabled = !!run.active;
  $("stopBtn").disabled = !run.active;

  // Dynamic Views
  updateGenerationPicker();
  renderFeed();
  renderCharts();
  renderTables();
  renderInitial();
  renderOperators();
  renderConsoleLog(s);

  // Synchronize PDF Download Links
  updatePdfLinks(s.run_id);

  // Auto-load generation details or render current best tour
  if (selectedDetail && Number(selectedDetail.generation) === selectedGen) {
    renderGenerationDetail();
  } else {
    loadGeneration(selectedGen);
    // Immediately draw current best tour while generation details load so canvas is never blank
    if (s.route || s.coordinates) {
      renderRouteCanvas(s.route || []);
    }
  }

  // Polling loop
  if (run.active) {
    startPolling();
  } else {
    stopPolling();
    // If run finished with completed generations and we haven't loaded the PDF yet, auto-refresh PDF preview
    if (st.is_stopped && st.generation > 0 && s.run_id && pdfLoadedForRun !== s.run_id) {
      loadPdfReport(s.run_id);
    }
  }
}

// --------------------------------------------------------------------------
// Terminal Stop Banner
// --------------------------------------------------------------------------
function renderTerminalBanner(s) {
  const banner = $("terminalBanner");
  const st = s.status || {};
  const run = s.run || {};

  // Show banner only when run has completed/stopped with at least 1 generation
  if (!run.active && st.generation > 0 && st.is_stopped) {
    banner.style.display = "flex";

    const isStag = st.stop_reason?.toLowerCase().includes("stagnation") || (st.no_improvement >= 30);
    const isOpt = st.converged || (st.gap != null && st.gap <= 0.0001);

    if (isStag) {
      banner.className = "terminal-banner stagnation-stopped";
      $("bannerIcon").textContent = "⚠";
      $("bannerTitle").textContent = "STOPPED: STAGNATION THRESHOLD REACHED (30/30 GENS)";
      $("bannerText").textContent = "The genetic algorithm reached 30 consecutive generations without global-best improvement and safely halted execution.";
    } else if (isOpt) {
      banner.className = "terminal-banner";
      $("bannerIcon").textContent = "★";
      $("bannerTitle").textContent = "SUCCESS: KNOWN OPTIMUM REACHED";
      $("bannerText").textContent = `Global optimum found at generation ${st.best_generation}. Zero optimality gap achieved!`;
    } else {
      banner.className = "terminal-banner";
      $("bannerIcon").textContent = "✓";
      $("bannerTitle").textContent = `EXPERIMENT TERMINATED: ${(st.stop_reason || "Completed").toUpperCase()}`;
      $("bannerText").textContent = `Algorithm halted after ${st.generation} generations. All telemetry records saved.`;
    }

    $("bannerStats").innerHTML = `
      <div class="banner-stat-chip"><span>Final Best:</span><b>${fmt(st.best_cost)}</b></div>
      <div class="banner-stat-chip"><span>Optimality Gap:</span><b>${st.gap != null ? fmt(st.gap, 4) + "%" : "—"}</b></div>
      <div class="banner-stat-chip"><span>Cutoff Gen:</span><b>${st.generation}</b></div>
      <div class="banner-stat-chip"><span>Best Gen:</span><b>${st.best_generation}</b></div>
      <div class="banner-stat-chip"><span>Total Duration:</span><b>${sec(getElapsed())}</b></div>
    `;
  } else {
    banner.style.display = "none";
  }
}

// --------------------------------------------------------------------------
// Generation Navigation & Feed
// --------------------------------------------------------------------------
function updateGenerationPicker() {
  const gens = S?.generations_available || [0];
  if (!gens.includes(selectedGen)) {
    selectedGen = gens[gens.length - 1] ?? 0;
  }
  const select = $("genSelect");
  select.innerHTML = gens.map(g => `<option value="${g}">Generation ${g}</option>`).join("");
  select.value = String(selectedGen);
}

function renderFeed() {
  const gens = S?.generations_available || [];
  const records = S?.records || [];
  const recMap = new Map(records.map(r => [r.generation, r]));
  const bestGen = S?.status?.best_generation ?? 0;

  const html = gens.map(g => {
    const r = recMap.get(g) || {};
    const isAct = g === selectedGen;
    const isBest = g === bestGen;
    return `
      <div class="carousel-card ${isAct ? "active" : ""} ${isBest ? "is-best" : ""}" data-gen="${g}">
        <div class="card-gen-num">GEN ${g}${isBest ? " ★" : ""}</div>
        <div class="card-cost-num">${fmt(r.best_cost)}</div>
        <div class="card-delta-num">${fmt(r.execution_time, 3)} s · -${fmt(r.improvement, 1)}</div>
      </div>
    `;
  }).join("");

  $("generationFeed").innerHTML = html;
  document.querySelectorAll(".carousel-card").forEach(card => {
    card.onclick = () => selectGeneration(Number(card.dataset.gen), false);
  });
}

function selectGeneration(g, follow = true) {
  selectedGen = g;
  autoFollow = follow;
  selectedDetail = null;
  updateGenerationPicker();
  renderFeed();
  loadGeneration(g);
}

async function loadGeneration(g) {
  if (!S?.generations_available?.includes(g)) return;
  const token = ++genLoadToken;
  try {
    const detail = await api(`/api/generation/${g}`);
    // Discard stale responses if the user moved on or a poll re-selected a generation.
    if (token !== genLoadToken) return;
    if (detail.route_coords) registerCoordinates(detail.route_coords);
    selectedDetail = detail;
    renderGenerationDetail();
  } catch (e) {
    if (token !== genLoadToken) return;
    $("console").textContent += `\n[Error loading generation ${g}]: ${e.message}`;
  }
}

function renderGenerationDetail() {
  if (!selectedDetail) return;
  const d = selectedDetail;
  const r = d.record || {};
  const pop = d.population || [];

  $("routeCaption").textContent = `Visualizing best route of Generation ${d.generation}`;
  $("routeInfo").innerHTML = `
    <span><b>Gen:</b> ${d.generation}</span>
    <span><b>Pop:</b> ${d.population_count}</span>
    <span><b>Cost:</b> ${fmt(r.best_cost)}</span>
    <span><b>Gap:</b> ${r.gap != null ? fmt(r.gap, 4) + "%" : "—"}</span>
    <span><b>Improvement:</b> ${fmt(r.improvement)}</span>
    <span><b>Time:</b> ${fmt(r.execution_time, 4)} s</span>
  `;

  $("genSummary").innerHTML = [
    ["Population Size", d.population_count],
    ["Best Tour Cost", fmt(r.best_cost)],
    ["Optimality Gap", r.gap != null ? fmt(r.gap, 4) + "%" : "—"],
    ["Applied Operator", r.operator || "Initialization"],
    ["Population Diversity", r.diversity != null ? fmt(r.diversity, 4) : "—"],
    ["PMG Memory Trust", fmt(r.memory_trust, 3)],
    ["Selective LKH-3", r.lkh_applied ? "Polished Elite" : "Not scheduled"],
    ["Double-Bridge 4-Opt", r.double_bridge_applied ? "Perturbation Active" : "Normal"],
    ["Best Individual ID", r.best_id ?? "—"],
    ["New Global Best", r.new_global_best ? "Yes (★)" : "No"],
    ["Execution Time", `${fmt(r.execution_time, 4)} s`],
    ["No-Improvement Streak", r.no_improvement_count ?? "0"]
  ].map(([label, val]) => `
    <div class="diag-item">
      <span>${esc(label)}</span>
      <b>${esc(val)}</b>
    </div>
  `).join("");

  $("selectedRoute").textContent = (d.route || []).join(" → ");
  renderRouteCanvas(d.route || [], d.route_coords);

  $("popCaption").textContent = `Generation ${d.generation} · ${d.population_count} ranked tours`;
  $("populationTable").innerHTML = pop.map((x, i) => `
    <tr>
      <td>${i + 1}</td>
      <td>${esc(x.id)}</td>
      <td>${fmt(x.cost)}</td>
      <td>${fmt(x.fitness, 8)}</td>
      <td>${esc(x.method)}</td>
      <td style="max-width:240px;overflow:hidden;text-overflow:ellipsis;">${esc((x.tour || []).join(" → "))}</td>
    </tr>
  `).join("");
}

// --------------------------------------------------------------------------
// Tour Visualization (High-DPI Crystal SVG)
// --------------------------------------------------------------------------
function renderRouteCanvas(routeOrCoords, explicitCoords) {
  const container = $("routeViz");
  if (!container) return;

  let pts = [];

  // 1. Explicit coordinate objects passed directly
  if (Array.isArray(explicitCoords) && explicitCoords.length >= 2) {
    pts = explicitCoords.map((p, idx) => ({
      city: p.city != null ? Number(p.city) : idx,
      x: Number(p.x),
      y: Number(p.y)
    })).filter(p => !isNaN(p.x) && !isNaN(p.y));
  }

  // 2. Direct coordinate objects passed in first argument
  if (pts.length < 2 && Array.isArray(routeOrCoords) && routeOrCoords.length >= 2 && typeof routeOrCoords[0] === "object" && routeOrCoords[0] !== null && "x" in routeOrCoords[0]) {
    pts = routeOrCoords.map((p, idx) => ({
      city: p.city != null ? Number(p.city) : idx,
      x: Number(p.x),
      y: Number(p.y)
    })).filter(p => !isNaN(p.x) && !isNaN(p.y));
  }

  // 3. Permutation array resolved through persistent CITY_COORDINATES_MAP
  if (pts.length < 2 && Array.isArray(routeOrCoords) && routeOrCoords.length >= 2) {
    pts = routeOrCoords.map(c => CITY_COORDINATES_MAP.get(Number(c))).filter(p => p && !isNaN(p.x) && !isNaN(p.y));
  }

  // 4. Fallback to S.route if current route could not resolve all coordinates
  if (pts.length < 2 && Array.isArray(S?.route) && S.route.length >= 2) {
    pts = S.route.map((p, idx) => ({
      city: p.city != null ? Number(p.city) : idx,
      x: Number(p.x),
      y: Number(p.y)
    })).filter(p => !isNaN(p.x) && !isNaN(p.y));
  }

  // 5. If pure distance matrix without spatial coordinates, generate circular layout
  if (pts.length < 2 && Array.isArray(routeOrCoords) && routeOrCoords.length >= 2) {
    const N = routeOrCoords.length;
    const radius = 135;
    const cx = 360, cy = 170;
    pts = routeOrCoords.map((c, i) => {
      const cityId = Number(c);
      const angle = (2 * Math.PI * i) / N - Math.PI / 2;
      return {
        city: cityId,
        x: cx + radius * Math.cos(angle),
        y: cy + radius * Math.sin(angle)
      };
    });
  }

  if (pts.length < 2) {
    container.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-muted);font-size:12px;">Tour contains ${routeOrCoords?.length || 0} cities (loading geometry...).</div>`;
    return;
  }

  const W = 720, H = 340, P = 38;
  const xs = pts.map(p => p.x);
  const ys = pts.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);

  const spanX = Math.max(1e-4, maxX - minX);
  const spanY = Math.max(1e-4, maxY - minY);
  const scale = Math.min((W - 2 * P) / spanX, (H - 2 * P) / spanY);

  const X = x => P + (x - minX) * scale + (W - 2 * P - spanX * scale) / 2;
  const Y = y => H - P - (y - minY) * scale - (H - 2 * P - spanY * scale) / 2;

  const closed = [...pts, pts[0]];
  const polyPoints = closed.map(p => `${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join(" ");

  const showLabels = pts.length <= 60;
  const dotRadius = pts.length > 150 ? 2.0 : pts.length > 70 ? 2.8 : 3.8;

  let svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:100%;display:block;">
    <defs>
      <linearGradient id="crystalRouteGrad" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#58e6f5"/>
        <stop offset="40%" stop-color="#818cf8"/>
        <stop offset="75%" stop-color="#c084fc"/>
        <stop offset="100%" stop-color="#f472b6"/>
      </linearGradient>
      <radialGradient id="startGlow" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#fef08a"/>
        <stop offset="50%" stop-color="#f59e0b"/>
        <stop offset="100%" stop-color="#b45309"/>
      </radialGradient>
      <pattern id="gridPattern" width="28" height="28" patternUnits="userSpaceOnUse">
        <line x1="0" y1="0" x2="28" y2="0" stroke="rgba(255,255,255,0.035)" stroke-width="1"/>
        <line x1="0" y1="0" x2="0" y2="28" stroke="rgba(255,255,255,0.035)" stroke-width="1"/>
      </pattern>
    </defs>
    <!-- Background grid -->
    <rect width="${W}" height="${H}" fill="url(#gridPattern)"/>

    <!-- Ambient luminous aura for tour -->
    <polyline points="${polyPoints}" fill="none" stroke="#58e6f5" stroke-width="6" opacity="0.30" stroke-linejoin="round" stroke-linecap="round"/>

    <!-- Core crystal prismatic tour line -->
    <polyline points="${polyPoints}" fill="none" stroke="url(#crystalRouteGrad)" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>
  `;

  // Draw city nodes
  pts.forEach((p, idx) => {
    const isStart = idx === 0;
    const cx = X(p.x).toFixed(1);
    const cy = Y(p.y).toFixed(1);

    if (isStart) {
      svg += `
        <!-- Start Node Pulsing Radar Halo -->
        <circle cx="${cx}" cy="${cy}" r="12" fill="none" stroke="rgba(251, 191, 36, 0.5)" stroke-width="1.8">
          <animate attributeName="r" values="8;14;8" dur="2.4s" repeatCount="indefinite"/>
          <animate attributeName="opacity" values="0.85;0.15;0.85" dur="2.4s" repeatCount="indefinite"/>
        </circle>
        <circle cx="${cx}" cy="${cy}" r="7" fill="url(#startGlow)" stroke="#ffffff" stroke-width="2"/>
        <text x="${cx}" y="${Number(cy) - 11}" font-size="10" font-weight="800" fill="#fbbf24" text-anchor="middle" font-family="'JetBrains Mono', monospace" filter="drop-shadow(0 2px 4px rgba(0,0,0,0.8))">START (${p.city})</text>
      `;
    } else {
      svg += `
        <circle cx="${cx}" cy="${cy}" r="${dotRadius}" fill="#e2e8f0" stroke="#0f172a" stroke-width="1.2">
          <title>City ${p.city} (${p.x.toFixed(1)}, ${p.y.toFixed(1)})</title>
        </circle>
      `;
      if (showLabels) {
        svg += `
          <text x="${cx}" y="${Number(cy) - (dotRadius + 3)}" font-size="8" fill="#94a3b8" text-anchor="middle" font-family="'JetBrains Mono', monospace" opacity="0.85">${p.city}</text>
        `;
      }
    }
  });

  svg += `</svg>`;
  container.innerHTML = svg;
}

// --------------------------------------------------------------------------
// Publication Research Charts (High-DPI SVG)
// --------------------------------------------------------------------------
function drawSvgChart(elemId, series, opts = {}) {
  const el = $(elemId);
  if (!el) return;
  const values = series.flatMap(s => s.values).filter(Number.isFinite);
  if (!values.length) {
    el.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;color:var(--text-muted);font-size:11px;">Waiting for generation telemetry...</div>`;
    return;
  }

  const W = 760, H = 240, L = 58, R = 18, T = 20, B = 40;
  const iw = W - L - R, ih = H - T - B;

  let lo = opts.zero ? 0 : Math.min(...values);
  let hi = Math.max(...values);
  if (opts.refLine != null && Number.isFinite(opts.refLine)) {
    lo = Math.min(lo, opts.refLine);
    hi = Math.max(hi, opts.refLine);
  }
  if (Math.abs(hi - lo) < 1e-9) {
    hi += 1;
    lo = Math.max(0, lo - 1);
  }

  const n = Math.max(...series.map(s => s.values.length));
  const X = i => L + (n <= 1 ? iw / 2 : (i * iw) / Math.max(1, n - 1));
  const Y = v => T + ((hi - v) * ih) / (hi - lo);

  // Unique gradient ids so multiple charts on one page never collide.
  const uid = `cg-${elemId}`;

  let out = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="${uid}-fill" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#58e6f5" stop-opacity="0.30"/>
        <stop offset="100%" stop-color="#58e6f5" stop-opacity="0"/>
      </linearGradient>
    </defs>`;

  // Horizontal Grid Lines
  for (let j = 0; j <= 4; j++) {
    const val = lo + ((hi - lo) * j) / 4;
    const y = Y(val);
    out += `<line x1="${L}" y1="${y}" x2="${W - R}" y2="${y}" class="gridline"/>`;
    out += `<text x="${L - 8}" y="${y + 3}" text-anchor="end" class="axis-text">${fmt(val, opts.yDigits ?? 2)}</text>`;
  }

  // X-Axis Ticks
  const ticks = Math.min(8, n);
  for (let j = 0; j < ticks; j++) {
    const i = Math.round((j * (n - 1)) / Math.max(1, ticks - 1));
    const x = X(i);
    out += `<line x1="${x}" y1="${T}" x2="${x}" y2="${H - B}" class="gridline" opacity="0.3"/>`;
    out += `<text x="${x}" y="${H - B + 15}" text-anchor="middle" class="axis-text">${i}</text>`;
  }

  // Axes lines & Titles
  out += `<line x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}" stroke="var(--border-metal)" stroke-width="1.2"/>`;
  out += `<line x1="${L}" y1="${T}" x2="${L}" y2="${H - B}" stroke="var(--border-metal)" stroke-width="1.2"/>`;
  out += `<text x="${L + iw / 2}" y="${H - 8}" text-anchor="middle" class="axis-title-svg">${esc(opts.xLabel || "Generation")}</text>`;

  // Reference Line (e.g. Known Optimum or Stagnation Limit)
  if (opts.refLine != null && Number.isFinite(opts.refLine)) {
    const yRef = Y(opts.refLine);
    out += `<line x1="${L}" y1="${yRef}" x2="${W - R}" y2="${yRef}" class="chart-line-red"/>`;
    out += `<text x="${W - R - 6}" y="${yRef - 5}" text-anchor="end" font-size="9.5" fill="#f87171" font-weight="bold" font-family="JetBrains Mono">${esc(opts.refLabel || "Threshold")}</text>`;
  }

  // Draw Series Polylines
  series.forEach((s, idx) => {
    const pts = s.values.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
    const cls = s.colorClass || (idx === 0 ? "chart-line-blue" : (idx === 1 ? "chart-line-orange" : "chart-line-purple"));

    // Soft area fill under the primary convergence series only (keeps science legible).
    if (idx === 0 && opts.area) {
      out += `<polygon points="${L},${H - B} ${pts} ${W - R},${H - B}" fill="url(#${uid}-fill)" stroke="none"/>`;
    }

    out += `<polyline points="${pts}" class="${cls}"/>`;

    if (opts.showPoints && s.values.length <= 60) {
      s.values.forEach((v, i) => {
        out += `<circle cx="${X(i).toFixed(1)}" cy="${Y(v).toFixed(1)}" r="3" fill="${idx === 0 ? "#58e6f5" : "#fbbf24"}" stroke="#141a24" stroke-width="1.2"/>`;
      });
    }
  });

  // Improvement-event markers on the convergence chart (real per-generation improvements only).
  if (opts.markersOn && opts.markersOn.length) {
    opts.markersOn.forEach(m => {
      if (m.i < 0 || m.i >= n) return;
      const mx = X(m.i).toFixed(1);
      const my = Y(m.v).toFixed(1);
      out += `<circle cx="${mx}" cy="${my}" r="4.6" fill="none" stroke="#f3b8de" stroke-width="1.6" opacity="0.9"/>`;
      out += `<circle cx="${mx}" cy="${my}" r="2.1" fill="#f3b8de"/>`;
    });
  }

  // Series Legend
  if (series.length > 1) {
    let lx = W - R - 180;
    series.forEach((s, idx) => {
      const col = idx === 0 ? "#3b82f6" : (idx === 1 ? "#f59e0b" : "#8b5cf6");
      out += `<rect x="${lx}" y="${T - 12}" width="10" height="3" fill="${col}"/>`;
      out += `<text x="${lx + 14}" y="${T - 8}" font-size="9" fill="var(--text-muted)">${esc(s.name)}</text>`;
      lx += 85;
    });
  }

  out += `</svg>`;
  el.innerHTML = out + `<div class="chart-tip"></div>`;

  // Bind hover tooltips against the real data series for this chart.
  el.style.position = el.style.position || "relative";
  attachChartInteraction(elemId, {
    W, H, L, R, n,
    series: series.map((s, i) => ({ name: s.name, values: s.values, color: s.tipColor || ["#58e6f5", "#b9a5ff", "#f3b8de"][i % 3] })),
    genAt: opts.genAt || (i => i),
    digits: opts.yDigits
  });
}

// Tracks chart geometry so hover can map pointer position back to real data points.
const chartIndex = new Map();

function hideChartTip() {
  document.querySelectorAll(".chart-tip").forEach(t => t.classList.remove("show"));
}

function attachChartInteraction(elemId, meta) {
  const el = $(elemId);
  if (!el) return;
  const tip = el.querySelector(".chart-tip");
  if (!tip || !meta.n) return;
  const svg = el.querySelector("svg");
  if (!svg) return;
  const vb = { W: meta.W, H: meta.H };

  el.onmousemove = ev => {
    const rect = el.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * vb.W;
    const i = Math.round(((px - meta.L) / (vb.W - meta.L - meta.R)) * (meta.n - 1 < 1 ? 0 : meta.n - 1));
    const idx = Math.max(0, Math.min(meta.n - 1, i));
    const h = historyByGen.get(meta.genAt(idx)) || {};
    const title = meta.series.map(s => {
      const v = s.values[idx];
      return `<div class="ct-row"><span class="ct-dot" style="background:${s.color}"></span>${esc(s.name)}<b>${fmt(v, meta.digits ?? 3)}</b></div>`;
    }).join("");
    tip.innerHTML = `<div class="ct-head">Generation ${h.generation ?? meta.genAt(idx)}</div>${title}` +
      (h.best_cost != null ? `<div class="ct-meta">Global best <b>${fmt(h.best_cost)}</b>${h.gap != null ? ` · gap <b>${fmt(h.gap, 4)}%</b>` : ""}</div>` : "");
    const lx = Math.min(Math.max(8, (meta.L + (idx * (vb.W - meta.L - meta.R)) / Math.max(1, meta.n - 1)) - 60), vb.W - 190);
    tip.style.left = `${(lx / vb.W) * 100}%`;
    tip.classList.add("show");
  };
  el.onmouseleave = hideChartTip;
}

function renderCharts() {
  if (!S) return;
  const records = S.records || [];
  const history = S.history || [];
  const opt = S.dataset?.best_known;
  historyByGen.clear();
  records.forEach(r => historyByGen.set(Number(r.generation), r));

  // Maps a chart index position back to the true generation number for the active dataset.
  const genAt = i => (records[i]?.generation ?? i);

  // Real improvement events only (net cost reduction > 0 recorded in a generation).
  const improvementMarkers = records
    .map((r, i) => ({ i, v: history[i], imp: Number(r.improvement ?? 0) }))
    .filter(m => m.imp > 0 && Number.isFinite(m.v));

  // 1. Convergence Chart (primary research centerpiece)
  drawSvgChart("convergence", [
    { name: "Global Best Cost", values: history, colorClass: "chart-line-blue" }
  ], {
    xLabel: "Generation",
    refLine: opt,
    refLabel: `Optimum (${opt != null ? fmt(opt, 0) : ""})`,
    yDigits: 0,
    zero: false,
    area: true,
    markersOn: improvementMarkers,
    genAt,
    showPoints: history.length <= 50
  });

  // 2. Stagnation Monitor (consecutive non-improving generations vs 30 limit)
  const stagValues = records.map(r => Number(r.no_improvement_count ?? 0));
  drawSvgChart("stagnationChart", [
    { name: "Non-Improving Streak", values: stagValues, colorClass: "chart-line-orange" }
  ], {
    xLabel: "Generation",
    refLine: 30,
    refLabel: "Termination Limit (30)",
    yDigits: 0,
    zero: true,
    genAt,
    showPoints: stagValues.length <= 50
  });

  // 3. Improvement Delta
  const impValues = records.map(r => Number(r.improvement ?? 0));
  drawSvgChart("improvement", [
    { name: "Cost Reduction", values: impValues, colorClass: "chart-line-blue" }
  ], {
    xLabel: "Generation",
    yDigits: 2,
    zero: true,
    genAt,
    showPoints: impValues.length <= 50
  });

  // 4. Runtime per Generation
  const timeValues = records.map(r => Number(r.execution_time ?? 0));
  drawSvgChart("timeChart", [
    { name: "Generation Runtime (s)", values: timeValues, colorClass: "chart-line-purple" }
  ], {
    xLabel: "Generation",
    yDigits: 3,
    zero: true,
    genAt,
    showPoints: timeValues.length <= 50
  });

  // 5. Diversity & Memory Trust
  const divValues = records.map(r => Number(r.diversity ?? 0));
  const trustValues = records.map(r => Number(r.memory_trust ?? 0));
  drawSvgChart("diversityChart", [
    { name: "Diversity Entropy", values: divValues, colorClass: "chart-line-blue" },
    { name: "PMG Trust", values: trustValues, colorClass: "chart-line-orange", tipColor: "#b9a5ff" }
  ], {
    xLabel: "Generation",
    yDigits: 3,
    zero: true,
    genAt
  });
}

// --------------------------------------------------------------------------
// Tables & Summary Grids
// --------------------------------------------------------------------------
function renderTables() {
  const records = S?.records || [];
  const tbody = $("generationTable");
  tbody.innerHTML = records.map(r => `
    <tr data-gen="${r.generation}">
      <td>${r.generation}</td>
      <td>${S?.population_counts?.[String(r.generation)] ?? S?.status?.population_size ?? 50}</td>
      <td>${fmt(r.best_cost)}</td>
      <td>${r.gap != null ? fmt(r.gap, 4) + "%" : "—"}</td>
      <td>${fmt(r.improvement)}</td>
      <td>${esc(r.operator || "Initialization")}</td>
      <td>${fmt(r.execution_time, 4)}</td>
    </tr>
  `).join("");

  document.querySelectorAll("#generationTable tr").forEach(tr => {
    tr.onclick = () => selectGeneration(Number(tr.dataset.gen), false);
  });
}

function renderInitial() {
  const init = S?.initial || {};
  $("initialOutputs").innerHTML = [
    ["Greedy Seed", init.greedy_cost, init.greedy_time],
    ["2-Opt Local Search", init.two_opt_local_cost, init.two_opt_local_time],
    ["2-Opt Refinement", init.two_opt_refined_cost, init.two_opt_refined_time],
    ["Population Construction", init.two_opt_refined_cost, init.population_time]
  ].map(([title, cost, t]) => `
    <div class="init-cell">
      <span>${title}</span>
      <b>${fmt(cost)}</b>
      <small>${fmt(t, 4)} s</small>
    </div>
  `).join("");
}

function renderOperators() {
  const plays = S?.operator_activity?.plays || {};
  const credit = S?.operator_activity?.credit || {};
  const keys = [...new Set([...Object.keys(plays), ...Object.keys(credit)])];

  $("operatorActivity").innerHTML = keys.map(k => `
    <div class="op-badge">
      <span>${esc(k.replace(/_/g, " "))}</span>
      <b>${plays[k] || 0} plays · ${fmt(credit[k] || 0, 1)} pts</b>
    </div>
  `).join("") || `<div style="grid-column:1/-1;color:var(--text-muted);font-size:11px;">No operator iterations recorded yet.</div>`;
}

function renderConsoleLog(s) {
  const d = s.dataset || {};
  const st = s.status || {};
  const run = s.run || {};
  const lines = [
    `=== APGGS ADAPTIVE EVOLUTION RESEARCH CONSOLE ===`,
    `Instance Name       : ${d.name || "None"}`,
    `Number of Cities    : ${d.cities || 0}`,
    `Known Optimum       : ${d.best_known != null ? fmt(d.best_known) : "Unknown"}`,
    `Current Generation  : ${st.generation}`,
    `Global Best Cost    : ${fmt(st.best_cost)}`,
    `Optimality Gap      : ${st.gap != null ? fmt(st.gap, 4) + "%" : "Unknown"}`,
    `Best Generation     : ${st.best_generation}`,
    `Stagnation Streak   : ${st.no_improvement ?? 0} / ${st.stagnation_patience ?? 30}`,
    `Elapsed Run Time    : ${sec(getElapsed())}`,
    `Lifecycle State     : ${(st.state || (run.active ? "running" : "idle")).toUpperCase()}`,
    `Termination Status  : ${st.stop_reason || "In progress"}`,
    run.error ? `FATAL ERROR: ${run.error}` : null
  ].filter(Boolean);

  if (s.diagnostics && s.diagnostics.length) {
    lines.push(`\n--- Engine Diagnostics ---`);
    s.diagnostics.forEach(diag => lines.push(`[DIAG] ${diag}`));
  }

  $("console").textContent = lines.join("\n");
}

// --------------------------------------------------------------------------
// Side-by-Side PDF Viewer Controller
// --------------------------------------------------------------------------
function updatePdfLinks(runId) {
  if (!runId) return;
  const pdfUrl = `/api/report/pdf?run_id=${encodeURIComponent(runId)}`;
  $("downloadPdfBtn").href = pdfUrl;
  $("downloadPdfBtn2").href = pdfUrl;
  $("downloadZipBtn").href = `/api/artifacts/zip?run_id=${encodeURIComponent(runId)}`;
  $("hdRunid").title = runId;
  $("pdfRunLabel").textContent = runId;
}

// Builds the browser PDF-viewer URL fragment for page navigation and zoom.
function pdfViewParams() {
  return `#page=${pdfPage}&zoom=${pdfZoom}`;
}

function syncPdfControls() {
  $("pdfPageLabel").textContent = `Page ${pdfPage}`;
  $("pdfZoomLabel").textContent = `${pdfZoom}%`;
  const frame = $("pdfFrame");
  if (frame.style.display === "none" || !frame.src || frame.src.startsWith("about:")) return;
  // Re-apply only the URL fragment so the embedded viewer navigates
  // without re-fetching the document (avoids reload flicker/socket errors).
  frame.src = frame.src.split("#")[0] + pdfViewParams();
}

function loadPdfReport(runId) {
  if (!runId) return;
  const frame = $("pdfFrame");
  const empty = $("pdfEmptyState");
  const loading = $("pdfLoadingState");

  pdfPage = 1;
  pdfZoom = 100;
  empty.style.display = "none";
  if (loading) loading.style.display = "flex";

  frame.onload = () => { if (loading) loading.style.display = "none"; };
  frame.src = `/api/report/pdf?run_id=${encodeURIComponent(runId)}&t=${Date.now()}${pdfViewParams()}`;
  frame.style.display = "block";
  pdfLoadedForRun = runId;
  $("pdfPanel").classList.add("has-report");
  updatePdfLinks(runId);
  syncPdfControls();
}

// --------------------------------------------------------------------------
// Polling & Lifecycle Control
// --------------------------------------------------------------------------
function startPolling() {
  if (pollTimer) return;
  pollTimer = setInterval(async () => {
    try {
      const liveData = await api("/api/live");
      const prevGen = S?.status?.generation ?? 0;
      const wasFollowing = autoFollow && selectedGen === prevGen;

      render(liveData);

      if (liveData.status.generation > prevGen && wasFollowing) {
        selectedGen = liveData.status.generation;
        selectedDetail = null;
        updateGenerationPicker();
        renderFeed();
        loadGeneration(selectedGen);
      }
    } catch (e) {
      console.warn("Polling cycle error:", e);
    }
  }, 550);
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function startElapsedTimer() {
  if (elapsedTimer) return;
  elapsedTimer = setInterval(() => {
    $("mElapsed").textContent = sec(getElapsed());
  }, 500);
}

// --------------------------------------------------------------------------
// Event Listeners
// --------------------------------------------------------------------------
$("genSelect").onchange = e => selectGeneration(Number(e.target.value), false);
$("prevGen").onclick = () => {
  const gens = S?.generations_available || [];
  const i = gens.indexOf(selectedGen);
  if (i > 0) selectGeneration(gens[i - 1], false);
};
$("nextGen").onclick = () => {
  const gens = S?.generations_available || [];
  const i = gens.indexOf(selectedGen);
  if (i >= 0 && i < gens.length - 1) selectGeneration(gens[i + 1], false);
};
$("jumpLatest").onclick = () => {
  const gens = S?.generations_available || [];
  if (gens.length) selectGeneration(gens[gens.length - 1], true);
};

// Execution Toolbar
$("runBtn").onclick = async () => {
  try {
    const res = await api("/api/run-all/start", { method: "POST" });
    render(res);
    startPolling();
  } catch (e) {
    alert(`Could not start run: ${e.message}`);
  }
};

$("stopBtn").onclick = async () => {
  try {
    const res = await api("/api/run-all/stop", { method: "POST" });
    render(res);
  } catch (e) {
    alert(`Stop request error: ${e.message}`);
  }
};

$("resetBtn").onclick = async () => {
  try {
    stopPolling();
    const res = await api("/api/reset", { method: "POST" });
    selectedGen = 0;
    selectedDetail = null;
    autoFollow = true;
    pdfLoadedForRun = null;
    $("pdfFrame").style.display = "none";
    $("pdfEmptyState").style.display = "flex";
    render(res);
  } catch (e) {
    alert(`Reset error: ${e.message}`);
  }
};

// Theme Toggle
$("themeBtn").onclick = () => {
  const isLight = document.body.classList.toggle("light");
  $("themeText").textContent = isLight ? "Dark mode" : "Light mode";
  $("themeIcon").textContent = isLight ? "🌙" : "☀️";
  localStorage.setItem("ae-theme", isLight ? "light" : "dark");
};
function applyStoredTheme() {
  if (localStorage.getItem("ae-theme") === "light") {
    document.body.classList.add("light");
    $("themeText").textContent = "Dark mode";
    $("themeIcon").textContent = "🌙";
  }
}
applyStoredTheme();

// PDF Controls
$("generatePdfBtn").onclick = async () => {
  try {
    $("generatePdfBtn").disabled = true;
    $("generatePdfBtn").textContent = "Generating...";
    const res = await api("/api/report/generate", { method: "POST" });
    loadPdfReport(res.run_id || S?.run_id);
  } catch (e) {
    alert(`PDF generation failed: ${e.message}`);
  } finally {
    $("generatePdfBtn").disabled = false;
    $("generatePdfBtn").textContent = "Generate PDF Report";
  }
};

$("refreshPdfBtn").onclick = () => {
  if (S?.run_id) {
    loadPdfReport(S.run_id);
  }
};

// PDF page navigation & zoom (applied to the embedded viewer via URL fragment)
$("pdfPrevPage").onclick = () => {
  pdfPage = Math.max(1, pdfPage - 1);
  syncPdfControls();
};
$("pdfNextPage").onclick = () => {
  pdfPage += 1;
  syncPdfControls();
};
$("pdfZoomIn").onclick = () => {
  pdfZoom = Math.min(300, pdfZoom + 25);
  syncPdfControls();
};
$("pdfZoomOut").onclick = () => {
  pdfZoom = Math.max(50, pdfZoom - 25);
  syncPdfControls();
};
$("pdfZoomReset").onclick = () => {
  pdfPage = 1;
  pdfZoom = 100;
  syncPdfControls();
};

// Report section quick-navigation chips
function gotoPdfPage(page) {
  pdfPage = Math.max(1, page);
  syncPdfControls();
}
document.querySelectorAll(".pdf-section-chips .s-chip").forEach(chip => {
  chip.onclick = e => {
    e.preventDefault();
    gotoPdfPage(Number(chip.dataset.page) || 1);
  };
});

$("bannerPdfBtn").onclick = () => {
  if (S?.run_id) {
    loadPdfReport(S.run_id);
  }
};

$("bannerZipBtn").onclick = () => {
  if (S?.run_id) {
    window.location.href = `/api/artifacts/zip?run_id=${encodeURIComponent(S.run_id)}`;
  }
};

// Panel Collapse / Toggle
$("togglePdfBtn").onclick = () => {
  const layout = $("mainLayout");
  layout.classList.toggle("hide-pdf");
};

$("collapsePdfBtn").onclick = () => {
  $("mainLayout").classList.add("hide-pdf");
};

// Reopen the PDF panel from the floating edge tab
$("showPdfTab").onclick = () => {
  $("mainLayout").classList.remove("hide-pdf");
};

$("clearConsoleBtn").onclick = () => {
  $("console").textContent = "Console cleared.";
};

// File Upload
const dropZone = $("dropZone");
const fileInput = $("fileInput");
dropZone.onclick = () => fileInput.click();
fileInput.onchange = () => fileInput.files[0] && uploadFile(fileInput.files[0]);
dropZone.ondragover = e => e.preventDefault();
dropZone.ondrop = e => {
  e.preventDefault();
  if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]);
};

async function uploadFile(file) {
  try {
    const fd = new FormData();
    fd.append("file", file);
    const s = await api("/api/upload", { method: "POST", body: fd });
    selectedGen = 0;
    selectedDetail = null;
    autoFollow = true;
    pdfLoadedForRun = null;
    $("pdfFrame").style.display = "none";
    $("pdfEmptyState").style.display = "flex";
    render(s);
    startElapsedTimer();
    await loadGeneration(0);
  } catch (e) {
    alert(`Upload error: ${e.message}`);
  }
}

const DEFAULT_PREVIEW_STATE = {
  run_id: "demo10_crystal_init",
  dataset: {
    name: "DEMO10",
    cities: 10,
    weight_type: "EUC_2D",
    best_known: 2474,
    has_coordinates: true,
    filename: "demo10.tsp"
  },
  config: {
    population_size: 50,
    max_generations: 300,
    no_improvement_limit: 30,
    stagnation_patience: 30,
    k_neighbors: 15,
    mutation_rate: 0.12,
    lkh_cadence: 15
  },
  status: {
    generation: 0,
    best_cost: 2474.0,
    gap: 0.0,
    best_generation: 0,
    no_improvement: 0,
    stagnation_patience: 30,
    stagnation_display: "0 / 30",
    converged: false,
    is_stopped: false,
    state: "idle",
    stop_reason: "Ready to start",
    population_size: 50,
    eta_text: "Ready"
  },
  run: {
    active: false,
    mode: "idle",
    started: null,
    finished: null,
    error: null,
    run_id: "demo10_crystal_init"
  },
  history: [2650, 2580, 2510, 2474],
  generations_available: [0],
  records: [
    { generation: 0, best_cost: 2474, gap: 0.0, improvement: 0, operator: "Initialization", execution_time: 0.015, diversity: 0.85, memory_trust: 1.0, no_improvement_count: 0 }
  ],
  route: [
    { city: 0, x: 20, y: 20 },
    { city: 1, x: 35, y: 65 },
    { city: 2, x: 60, y: 80 },
    { city: 3, x: 80, y: 55 },
    { city: 4, x: 90, y: 30 },
    { city: 5, x: 75, y: 15 },
    { city: 6, x: 50, y: 35 },
    { city: 7, x: 40, y: 10 },
    { city: 8, x: 15, y: 40 },
    { city: 9, x: 10, y: 70 }
  ],
  initial: {
    greedy_cost: 2650,
    greedy_time: 0.002,
    two_opt_local_cost: 2510,
    two_opt_local_time: 0.004,
    two_opt_refined_cost: 2474,
    two_opt_refined_time: 0.008,
    population_time: 0.012
  },
  operator_activity: {
    plays: { "adaptive_2opt": 12, "weak_edge": 8, "or_opt": 6 },
    credit: { "adaptive_2opt": 15.2, "weak_edge": 9.4, "or_opt": 4.1 }
  }
};

// Initialization
async function init() {
  startElapsedTimer();
  // 1. Immediately render crystal preview state so UI is instantly styled and populated
  try {
    registerCoordinates(DEFAULT_PREVIEW_STATE.route);
    render(DEFAULT_PREVIEW_STATE);
    renderRouteCanvas(DEFAULT_PREVIEW_STATE.route);
  } catch (err) {
    console.warn("Initial preview render notice:", err);
  }

  // 2. Fetch live state from backend API
  try {
    const s = await api("/api/state");
    render(s);
  } catch (e) {
    console.warn("Backend live state sync notice:", e.message);
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
