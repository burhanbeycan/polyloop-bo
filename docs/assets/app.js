// PolyLoop‑BO interactive GitHub Pages demo
// Runs a small ridge-regression surrogate in the browser (no server).

let MODEL = null;
let EXPERIMENTS = [];
let debounceTimer = null;



function guessGitHubContext(){
  try{
    const host = window.location.hostname || "";
    const path = (window.location.pathname || "/").split("/").filter(Boolean);
    if (host.endsWith("github.io") && path.length >= 1){
      const user = host.split(".")[0];
      const repo = path[0];
      return {user, repo};
    }
  }catch(e){}
  return null;
}

function updateRepoLink(fallbackRepo){
  const a = document.getElementById("repoLink");
  if (!a) return;
  const ctx = guessGitHubContext();
  if (ctx){
    const url = `https://github.com/${ctx.user}/${ctx.repo}`;
    a.href = url;
    a.textContent = url;
  }else if (fallbackRepo){
    a.href = `https://github.com/<your-username>/${fallbackRepo}`;
  }
}

function fmt(x, digits=2){
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x).toFixed(digits);
}

function clamp(x, lo, hi){
  return Math.min(Math.max(x, lo), hi);
}

function dot(a,b){
  let s = 0.0;
  for (let i=0;i<a.length;i++) s += a[i]*b[i];
  return s;
}

async function loadJSON(path){
  const r = await fetch(path);
  if (!r.ok) throw new Error(`Failed to load ${path}: ${r.status}`);
  return await r.json();
}

function stepForFeature(name){
  const steps = {
    "gelatin_wt_pct": 0.1,
    "wpu_wt_pct": 0.1,
    "pei_wt_pct": 0.05,
    "peox_wt_pct": 0.05,
    "allantoin_wt_pct": 0.05,
    "hyaluronic_acid_wt_pct": 0.02,
    "voltage_kV": 0.1,
    "flow_rate_mL_h": 0.01,
    "tip_to_collector_cm": 0.1,
    "humidity_pct": 0.5
  };
  return steps[name] ?? 0.1;
}

function setSlider(id, min, max, value){
  const el = document.getElementById(id);
  if (!el) return;
  el.min = String(min);
  el.max = String(max);
  el.step = String(stepForFeature(id));
  el.value = String(value);
}

function updateValLabel(id){
  const el = document.getElementById(id);
  const out = document.getElementById(`${id}_val`);
  if (!el || !out) return;
  const v = parseFloat(el.value);
  const digits = (stepForFeature(id) < 0.1) ? 2 : 1;
  out.textContent = fmt(v, digits);
}

function getInputVector(){
  const x = [];
  for (const f of MODEL.feature_names){
    const el = document.getElementById(f);
    x.push(parseFloat(el.value));
  }
  return x;
}

function predictFromVector(x){
  // standardise
  const z = x.map((v,i) => (v - MODEL.scaler.mean[i]) / MODEL.scaler.scale[i]);
  const out = {};
  for (const t of MODEL.targets){
    const m = MODEL.models[t];
    let y = m.intercept + dot(m.coef, z);
    const r = MODEL.output_ranges?.[t];
    if (r){
      y = clamp(y, r.min, r.max);
    }
    out[t] = y;
  }
  return out;
}

function renderPredictions(pred){
  const mapping = {
    "fiber_diameter_nm": {digits: 0},
    "bead_index": {digits: 3},
    "contact_angle_deg": {digits: 1},
    "zone_inhibition_mm": {digits: 2},
    "cell_viability_pct": {digits: 1}
  };
  for (const t of MODEL.targets){
    const el = document.getElementById(`out_${t}`);
    if (!el) continue;
    const d = mapping[t]?.digits ?? 2;
    el.textContent = fmt(pred[t], d);
  }
}

function median(arr){
  const a = [...arr].sort((x,y) => x-y);
  if (a.length === 0) return NaN;
  const mid = Math.floor(a.length/2);
  return (a.length % 2) ? a[mid] : 0.5*(a[mid-1]+a[mid]);
}

function setStats(){
  document.getElementById("statN").textContent = String(EXPERIMENTS.length);
  const r2vals = Object.values(MODEL.r2_test ?? {});
  const med = median(r2vals);
  document.getElementById("statR2").textContent = Number.isFinite(med) ? fmt(med, 2) : "—";
}

function updateModelInfo(){
  const lines = [];
  lines.push(`<b>${MODEL.name}</b>`);
  if (MODEL.note) lines.push(MODEL.note);
  if (MODEL.r2_test){
    const parts = [];
    for (const [k,v] of Object.entries(MODEL.r2_test)){
      parts.push(`${k}: ${fmt(v, 2)}`);
    }
    lines.push(`Test R² — ${parts.join(" · ")}`);
  }
  document.getElementById("modelInfo").innerHTML = lines.join("<br/>");
}

function predictAndRender(){
  const x = getInputVector();
  const pred = predictFromVector(x);
  renderPredictions(pred);
}

function predictDebounced(){
  if (debounceTimer) clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => predictAndRender(), 120);
}

function objectiveScore(pred){
  // Maximise: zone_inhibition, viability; minimise: bead_index, diameter
  function norm(v, lo, hi){
    return (v - lo) / (hi - lo + 1e-9);
  }
  const r = MODEL.output_ranges;
  const z = norm(pred.zone_inhibition_mm, r.zone_inhibition_mm.min, r.zone_inhibition_mm.max);
  const v = norm(pred.cell_viability_pct, r.cell_viability_pct.min, r.cell_viability_pct.max);
  const b = norm(pred.bead_index, r.bead_index.min, r.bead_index.max);
  const d = norm(pred.fiber_diameter_nm, r.fiber_diameter_nm.min, r.fiber_diameter_nm.max);
  const score = 0.40*z + 0.35*v + 0.15*(1-b) + 0.10*(1-d);
  return score;
}

function suggestExperiment(){
  const best = {score: -1e9, x: null, pred: null};

  for (let i=0;i<450;i++){
    const x = [];
    for (const f of MODEL.feature_names){
      const r = MODEL.input_ranges[f];
      const v = r.min + Math.random()*(r.max - r.min);
      x.push(v);
    }
    const pred = predictFromVector(x);

    // soft constraint: aim for decent viability
    if (pred.cell_viability_pct < 45) continue;

    const s = objectiveScore(pred);
    if (s > best.score){
      best.score = s;
      best.x = x;
      best.pred = pred;
    }
  }

  if (!best.x){
    alert("No feasible candidate found. Try again.");
    return;
  }

  // set sliders
  MODEL.feature_names.forEach((f,idx) => {
    const el = document.getElementById(f);
    el.value = String(best.x[idx]);
    updateValLabel(f);
  });
  renderPredictions(best.pred);

  // highlight in model info
  const msg = `Suggested candidate score: ${fmt(best.score, 3)} (heuristic multi-objective)`;
  const mi = document.getElementById("modelInfo");
  mi.innerHTML = mi.innerHTML + `<br/><span style="color:var(--accent2)">${msg}</span>`;
}

function renderTable(rows){
  const tb = document.querySelector("#expTable tbody");
  tb.innerHTML = "";
  const show = rows.slice(0, 120);
  for (const r of show){
    const tr = document.createElement("tr");
    const cells = [
      r.experiment_id,
      fmt(r.gelatin_wt_pct, 2),
      fmt(r.wpu_wt_pct, 2),
      fmt(r.pei_wt_pct, 2),
      fmt(r.peox_wt_pct, 2),
      fmt(r.allantoin_wt_pct, 2),
      fmt(r.hyaluronic_acid_wt_pct, 2),
      fmt(r.voltage_kV, 2),
      fmt(r.flow_rate_mL_h, 3),
      fmt(r.tip_to_collector_cm, 2),
      fmt(r.humidity_pct, 1),
      fmt(r.fiber_diameter_nm, 0),
      fmt(r.bead_index, 3),
      fmt(r.contact_angle_deg, 1),
      fmt(r.zone_inhibition_mm, 2),
      fmt(r.cell_viability_pct, 1),
    ];
    for (const c of cells){
      const td = document.createElement("td");
      td.textContent = String(c);
      tr.appendChild(td);
    }
    tb.appendChild(tr);
  }
}

function initControls(){
  // slider ranges from MODEL
  for (const f of MODEL.feature_names){
    const r = MODEL.input_ranges[f];
    const mean = MODEL.scaler.mean[MODEL.feature_names.indexOf(f)];
    const v0 = clamp(mean, r.min, r.max);
    setSlider(f, r.min, r.max, v0);
    updateValLabel(f);

    const el = document.getElementById(f);
    el.addEventListener("input", () => { updateValLabel(f); predictDebounced(); });
  }

  document.getElementById("predictBtn").addEventListener("click", () => predictAndRender());
  document.getElementById("suggestBtn").addEventListener("click", () => suggestExperiment());
  document.getElementById("resetBtn").addEventListener("click", () => {
    initControls();
    predictAndRender();
    updateModelInfo();
  });

  document.getElementById("filterBtn").addEventListener("click", () => {
    const q = (document.getElementById("filterInput").value || "").trim().toLowerCase();
    if (!q){ renderTable(EXPERIMENTS); return; }
    const rows = EXPERIMENTS.filter(r => String(r.experiment_id).toLowerCase().includes(q));
    renderTable(rows);
  });

  document.getElementById("clearBtn").addEventListener("click", () => {
    document.getElementById("filterInput").value = "";
    renderTable(EXPERIMENTS);
  });
}

async function main(){
  try{
    MODEL = await loadJSON("assets/model.json");
    const exp = await loadJSON("assets/experiments.json");
    EXPERIMENTS = exp.experiments ?? [];
    setStats();
    updateModelInfo();
    initControls();
    predictAndRender();
    renderTable(EXPERIMENTS);
    updateRepoLink("polyloop-bo");

    // repo link placeholder remains unless you set it manually
  }catch(err){
    console.error(err);
    const mi = document.getElementById("modelInfo");
    mi.textContent = "Failed to load model/assets. Check console.";
  }
}

main();
