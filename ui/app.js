// PathSnap evaluator UI (runs on Laptop 2). Talks only to the API; never sees reference routes.
(function () {
  const $ = (id) => document.getElementById(id);
  const state = { api: "", trips: [], selected: null, status: {}, history: [] };

  // ---------- API address (remembered per browser; ?api= overrides) ----------
  const qs = new URLSearchParams(location.search).get("api");
  let saved = "";
  try { saved = localStorage.getItem("pathsnap-api") || ""; } catch (_) {}
  $("api-url").value = qs || saved || `http://${location.hostname || "localhost"}:8000`;

  function apiBase() { return $("api-url").value.trim().replace(/\/+$/, ""); }

  function setStamp(el, tone, text, press) {
    el.dataset.tone = tone; el.textContent = text;
    if (press) { el.classList.remove("vx-press"); void el.offsetWidth; el.classList.add("vx-press"); }
  }
  function showError(msg) { const b = $("api-error"); b.textContent = msg; b.hidden = !msg; }

  async function connect() {
    state.api = apiBase();
    try { localStorage.setItem("pathsnap-api", state.api); } catch (_) {}
    setStamp($("api-stamp"), "running", "Connecting");
    showError("");
    try {
      const h = await fetch(state.api + "/health").then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); });
      const t = await fetch(state.api + "/trips").then((r) => r.json());
      state.trips = t.trips;
      setStamp($("api-stamp"), "done", `API online · ${h.model}`, true);
      renderTrips();
    } catch (e) {
      setStamp($("api-stamp"), "failed", "API unreachable", true);
      showError(`Could not reach ${state.api}. Check that the API is running on Laptop 1 and both laptops are on the same network.`);
      state.trips = []; renderTrips();
    }
  }

  // ---------- trip list ----------
  function tripStamp(id) {
    const s = state.status[id];
    if (!s) return '<span class="vx-stamp" data-tone="skipped">Not run</span>';
    if (s === "running") return '<span class="vx-stamp" data-tone="running">Running</span>';
    if (s === "failed") return '<span class="vx-stamp" data-tone="failed">Failed</span>';
    return s >= 0.95 ? '<span class="vx-stamp" data-tone="strong">Matched</span>'
      : s >= 0.8 ? '<span class="vx-stamp" data-tone="done">Matched</span>'
      : '<span class="vx-stamp" data-tone="possible">Partial</span>';
  }
  function renderTrips() {
    $("trip-n").textContent = state.trips.length;
    $("trip-count").textContent = state.trips.length;
    const box = $("trip-list");
    if (!state.trips.length) {
      box.innerHTML = '<div class="empty-note"><b>No trips loaded</b><span>Connect to the API to list the evaluation trips.</span></div>';
      return;
    }
    box.innerHTML = state.trips.map((t) => `
      <button type="button" class="trip-card" data-id="${t.trip_id}" aria-pressed="${state.selected === t.trip_id}">
        <span><span class="tid">${t.trip_id}</span><br><span class="meta">${t.n_points.toLocaleString()} GPS points</span></span>
        ${tripStamp(t.trip_id)}
      </button>`).join("");
    box.querySelectorAll(".trip-card").forEach((b) => b.addEventListener("click", () => select(b.dataset.id)));
  }
  function select(id) {
    state.selected = id;
    $("run-btn").disabled = false;
    $("callout-title").textContent = `Trip ${id} selected`;
    document.querySelector(".callout .step").textContent = "Step 2";
    renderTrips();
  }

  // ---------- map ----------
  const map = L.map("map", { preferCanvas: true, zoomControl: true }).setView([20, 0], 2);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);
  const gpsLayer = L.layerGroup().addTo(map);
  const routeLayer = L.layerGroup().addTo(map);
  L.control.layers(null, { "Raw GPS": gpsLayer, "Predicted route": routeLayer }, { collapsed: false }).addTo(map);
  const css = getComputedStyle(document.documentElement);
  const VIOLET = css.getPropertyValue("--vx-stamp").trim() || "#5b3fa8";
  const INK3 = css.getPropertyValue("--vx-ink-3").trim() || "#5c6474";

  function drawResult(r) {
    gpsLayer.clearLayers(); routeLayer.clearLayers();
    r.gps.forEach((p) => L.circleMarker(p, { radius: 2.5, color: INK3, weight: 0, fillOpacity: 0.7 }).addTo(gpsLayer));
    r.predicted_route.forEach((seg) => L.polyline(seg, { color: VIOLET, weight: 4, opacity: 0.9 }).addTo(routeLayer));
    if (r.gps.length) {
      L.circleMarker(r.gps[0], { radius: 6, color: "#1c2433", fillColor: "#fff", fillOpacity: 1, weight: 2 })
        .bindTooltip("Start").addTo(gpsLayer);
      L.circleMarker(r.gps[r.gps.length - 1], { radius: 6, color: "#fff", fillColor: "#1c2433", fillOpacity: 1, weight: 2 })
        .bindTooltip("End").addTo(gpsLayer);
      map.fitBounds(L.latLngBounds(r.gps), { padding: [24, 24] });
    }
  }

  // ---------- results ----------
  const pct = (x) => (100 * x).toFixed(1) + "%";
  const tone = (x) => (x >= 0.9 ? "good" : x >= 0.75 ? "warn" : "bad");
  function setNum(id, v) { const el = $(id); el.textContent = pct(v); el.className = "num " + tone(v); }

  function field(label, value, mono) {
    return `<div class="field"><div class="fl">${label}</div><div class="fv${mono ? " mono" : ""}">${value}</div></div>`;
  }

  function showResult(r) {
    const m = r.metrics;
    $("result-trip").textContent = r.trip_id;
    setNum("s-top1", m.top1_accuracy_hmm); setNum("s-f1", m.route_f1); setNum("s-cont", m.route_continuity_hmm);
    const st = $("result-stamp");
    m.top1_accuracy_hmm >= 0.95 ? setStamp(st, "strong", "Matched", true)
      : m.top1_accuracy_hmm >= 0.8 ? setStamp(st, "done", "Matched", true) : setStamp(st, "possible", "Partial match", true);
    $("fields").innerHTML = [
      field("Trip ID", r.trip_id, true),
      field("GPS points", r.n_points.toLocaleString()),
      field("Candidate rows", r.n_candidates.toLocaleString()),
      field("Predicted arcs", r.predicted_arc_count.toLocaleString()),
      field("Predicted length", r.predicted_route_km.toFixed(2) + " km"),
      field("Route precision", pct(m.route_precision)),
      field("Route recall", pct(m.route_recall)),
      field("Points with candidates", pct(m.points_with_candidates)),
      field("HMM breaks", m.hmm_breaks),
    ].join("");
    $("cmp-body").innerHTML = `
      <tr><td>Top-1 accuracy</td><td>${pct(m.top1_accuracy_classifier_only)}</td><td><b>${pct(m.top1_accuracy_hmm)}</b></td></tr>
      <tr><td>Route continuity</td><td>${pct(m.route_continuity_classifier_only)}</td><td><b>${pct(m.route_continuity_hmm)}</b></td></tr>`;
    const names = { load_ms: "Load trip + build network", candidates_ms: "Candidate generation + features",
      classifier_ms: "Classifier scoring", viterbi_ms: "HMM / Viterbi + route assembly",
      total_inference_ms: "Total inference", scoring_ms: "Server-side scoring" };
    $("time-body").innerHTML = Object.entries(r.timings).map(([k, v]) =>
      `<tr><td>${names[k] || k}</td><td class="mono">${v.toFixed(1)}</td></tr>`).join("");
    drawResult(r);
  }

  async function run() {
    const id = state.selected; if (!id) return;
    const btn = $("run-btn");
    btn.disabled = true;
    $("run-icon").outerHTML = '<svg viewBox="0 0 24 24" id="run-icon" class="spin" style="fill:none;stroke:currentColor"><path d="M21 12a9 9 0 1 1-6.2-8.6"/></svg>';
    state.status[id] = "running"; renderTrips();
    setStamp($("result-stamp"), "running", "Running");
    showError("");
    try {
      const res = await fetch(state.api + "/predict", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ trip_id: id }),
      });
      if (!res.ok) throw new Error(`API returned ${res.status}`);
      const r = await res.json();
      state.status[id] = r.metrics.top1_accuracy_hmm;
      showResult(r);
      state.history.unshift({ when: new Date().toLocaleTimeString(), r });
      renderHistory();
    } catch (e) {
      state.status[id] = "failed";
      setStamp($("result-stamp"), "failed", "Failed", true);
      showError(`Prediction failed for trip ${id}: ${e.message}`);
    } finally {
      $("run-icon").outerHTML = '<svg viewBox="0 0 24 24" id="run-icon"><path d="M6 4l14 8-14 8z"/></svg>';
      btn.disabled = false; renderTrips();
    }
  }

  function renderHistory() {
    $("run-count").textContent = state.history.length;
    $("run-count").className = "count" + (state.history.length ? "" : " muted");
    if (!state.history.length) return;
    $("hist-body").innerHTML = state.history.map(({ when, r }) => {
      const m = r.metrics;
      return `<tr><td class="mono">${when}</td><td class="mono">${r.trip_id}</td><td>${r.n_points}</td>
        <td>${pct(m.top1_accuracy_hmm)}</td><td>${pct(m.top1_accuracy_classifier_only)}</td><td>${pct(m.route_f1)}</td>
        <td>${pct(m.route_continuity_hmm)}</td><td class="mono">${r.timings.total_inference_ms.toFixed(0)}</td>
        <td>${tripStamp(r.trip_id)}</td></tr>`;
    }).join("");
  }

  // ---------- navigation ----------
  const titles = { evaluate: "Evaluate a trip", history: "Run history", about: "How it works" };
  function route() {
    const page = (location.hash || "#evaluate").slice(1);
    document.querySelectorAll("[data-view]").forEach((s) => (s.hidden = s.dataset.view !== page));
    document.querySelectorAll(".rail-item").forEach((a) => a.classList.toggle("active", a.dataset.page === page));
    $("crumb").textContent = titles[page] || "";
    document.querySelector(".shell").classList.remove("drawer-open"); $("scrim").hidden = true;
    if (page === "evaluate") setTimeout(() => map.invalidateSize(), 50);
  }
  window.addEventListener("hashchange", route);
  $("menu-btn").addEventListener("click", () => {
    const shell = document.querySelector(".shell");
    if (matchMedia("(min-width: 1024px)").matches) {
      shell.classList.toggle("rail-hidden");
      setTimeout(() => map.invalidateSize(), 50);
    } else {
      const open = shell.classList.toggle("drawer-open"); $("scrim").hidden = !open;
    }
  });
  $("scrim").addEventListener("click", () => { document.querySelector(".shell").classList.remove("drawer-open"); $("scrim").hidden = true; });
  $("api-connect").addEventListener("click", connect);
  $("api-url").addEventListener("keydown", (e) => { if (e.key === "Enter") connect(); });
  $("run-btn").addEventListener("click", run);

  route();
  connect();
})();
