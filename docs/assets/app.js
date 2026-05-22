const data = window.BIOMASTER_DATA || {};
const candidates = data.candidates || [];
const palette = ["#155fa8", "#148f8b", "#5f9648", "#d87832", "#8a5a9e", "#52606d"];

let activeFilter = "all";
let activeStructure = 0;
let viewer = null;
let viewerSpinning = false;

function formatScore(value, digits = 6) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "NA";
  return Number(value).toFixed(digits);
}

function formatInt(value) {
  return Number(value || 0).toLocaleString("en-US");
}

function formatPct(value) {
  return `${Number(value || 0).toFixed(2)}%`;
}

function statusLabel(status) {
  return status === "completed" ? "completed" : "missing output";
}

function setMetricValues() {
  const metrics = data.metrics || {};
  document.querySelectorAll("[data-metric]").forEach((node) => {
    const key = node.dataset.metric;
    const value = metrics[key];
    if (value === undefined) return;
    node.textContent = key.toLowerCase().includes("pct") ? formatPct(value) : formatInt(value);
  });
  const queueState = document.getElementById("queueState");
  if (queueState) queueState.textContent = "in progress";
}

function renderCandidates() {
  const tbody = document.getElementById("candidateRows");
  const searchInput = document.getElementById("candidateSearch");
  if (!tbody || !searchInput) return;

  const search = searchInput.value.trim().toLowerCase();
  const rows = candidates.filter((candidate) => {
    const matchesFilter = activeFilter === "all" || candidate.status === activeFilter;
    const haystack = `${candidate.drug} ${candidate.target} ${candidate.protein} ${candidate.pairId}`.toLowerCase();
    return matchesFilter && haystack.includes(search);
  });

  tbody.innerHTML = rows
    .map(
      (candidate) => `
        <tr>
          <td><span class="rank">#${candidate.rank}</span></td>
          <td>${candidate.drug}<br><small>${candidate.drugId}</small></td>
          <td><strong>${candidate.target}</strong><br><small>${candidate.protein}</small></td>
          <td>${formatScore(candidate.diseasePriority)}</td>
          <td>${formatScore(candidate.diffdock, 2)}</td>
          <td>${formatScore(candidate.consensus)}</td>
          <td><span class="status-pill ${candidate.status}">${statusLabel(candidate.status)}</span></td>
        </tr>
      `,
    )
    .join("");
}

function setupCandidateControls() {
  document.querySelectorAll("[data-filter]").forEach((button) => {
    button.addEventListener("click", () => {
      activeFilter = button.dataset.filter;
      document.querySelectorAll("[data-filter]").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      renderCandidates();
    });
  });

  const searchInput = document.getElementById("candidateSearch");
  if (searchInput) searchInput.addEventListener("input", renderCandidates);
}

function clearCanvas(canvas) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  return ctx;
}

function drawBarChart(canvas, rows) {
  if (!canvas || !rows?.length) return;
  const ctx = clearCanvas(canvas);
  const max = Math.max(...rows.map((row) => row.value));
  const left = 72;
  const top = 26;
  const barGap = 12;
  const barHeight = 18;
  ctx.font = "13px Inter, sans-serif";
  ctx.textBaseline = "middle";

  rows.forEach((row, index) => {
    const y = top + index * (barHeight + barGap);
    const width = ((canvas.width - left - 84) * row.value) / max;
    ctx.fillStyle = "#657083";
    ctx.fillText(row.label, 10, y + barHeight / 2);
    ctx.fillStyle = palette[index % palette.length];
    ctx.fillRect(left, y, width, barHeight);
    ctx.fillStyle = "#162131";
    ctx.fillText(String(row.value), left + width + 10, y + barHeight / 2);
  });
}

function drawDonutChart(canvas, rows) {
  if (!canvas || !rows?.length) return;
  const ctx = clearCanvas(canvas);
  const total = rows.reduce((sum, row) => sum + row.value, 0);
  const cx = 150;
  const cy = 136;
  const outer = 88;
  const inner = 52;
  let start = -Math.PI / 2;

  rows.forEach((row, index) => {
    const angle = (row.value / total) * Math.PI * 2;
    ctx.beginPath();
    ctx.arc(cx, cy, outer, start, start + angle);
    ctx.arc(cx, cy, inner, start + angle, start, true);
    ctx.closePath();
    ctx.fillStyle = palette[index % palette.length];
    ctx.fill();
    start += angle;
  });

  ctx.fillStyle = "#162131";
  ctx.font = "700 22px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(formatInt(total), cx, cy - 2);
  ctx.font = "12px Inter, sans-serif";
  ctx.fillStyle = "#657083";
  ctx.fillText("pairs", cx, cy + 18);
  ctx.textAlign = "left";

  rows.forEach((row, index) => {
    const y = 248 + index * 18;
    ctx.fillStyle = palette[index % palette.length];
    ctx.fillRect(20, y - 7, 10, 10);
    ctx.fillStyle = "#405069";
    ctx.font = "12px Inter, sans-serif";
    ctx.fillText(`${row.label}: ${formatInt(row.value)}`, 38, y - 2);
  });
}

function renderStackedBars(containerId, rows) {
  const container = document.getElementById(containerId);
  if (!container || !rows?.length) return;
  const total = rows.reduce((sum, row) => sum + row.value, 0);
  container.innerHTML = rows
    .map((row, index) => {
      const pct = total ? (row.value / total) * 100 : 0;
      return `
        <div class="stack-row">
          <div class="stack-label"><span>${row.label}</span><strong>${formatInt(row.value)} · ${pct.toFixed(1)}%</strong></div>
          <div class="stack-track"><i style="--w:${Math.max(pct, 1)}%; --c:${palette[index % palette.length]}"></i></div>
        </div>
      `;
    })
    .join("");
}

function renderCharts() {
  const charts = data.charts || {};
  drawBarChart(document.getElementById("targetChart"), charts.topTargets);
  drawDonutChart(document.getElementById("evidenceChart"), charts.evidenceCoverage);
  renderStackedBars("statusBars", charts.structuralStatus);
  renderStackedBars("layerBars", [
    ...(charts.txgnnStatus || []).map((row) => ({ label: `TxGNN ${row.label}`, value: row.value })),
    ...(charts.receptorStatus || []).map((row) => ({ label: row.label, value: row.value })),
  ]);
}

function parseSdf2D(sdf) {
  const lines = sdf.split(/\r?\n/);
  const counts = lines[3] || "";
  const atomCount = Number.parseInt(counts.slice(0, 3), 10);
  const bondCount = Number.parseInt(counts.slice(3, 6), 10);
  if (!atomCount || !bondCount) return null;
  const atoms = lines.slice(4, 4 + atomCount).map((line) => ({
    x: Number.parseFloat(line.slice(0, 10)),
    y: Number.parseFloat(line.slice(10, 20)),
    element: line.slice(31, 34).trim(),
  }));
  const bonds = lines.slice(4 + atomCount, 4 + atomCount + bondCount).map((line) => ({
    a: Number.parseInt(line.slice(0, 3), 10) - 1,
    b: Number.parseInt(line.slice(3, 6), 10) - 1,
  }));
  return { atoms, bonds };
}

function drawLigandSketch(sdf) {
  const canvas = document.getElementById("ligandSketch");
  if (!canvas) return;
  const ctx = clearCanvas(canvas);
  const parsed = parseSdf2D(sdf);
  if (!parsed) {
    ctx.fillStyle = "#657083";
    ctx.fillText("Ligand sketch unavailable", 20, 30);
    return;
  }

  const xs = parsed.atoms.map((atom) => atom.x);
  const ys = parsed.atoms.map((atom) => atom.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const scale = Math.min(320 / Math.max(maxX - minX, 1), 200 / Math.max(maxY - minY, 1));
  const tx = 50 - minX * scale + (320 - (maxX - minX) * scale) / 2;
  const ty = 30 - minY * scale + (200 - (maxY - minY) * scale) / 2;

  const project = (atom) => ({ x: atom.x * scale + tx, y: canvas.height - (atom.y * scale + ty) });
  ctx.lineCap = "round";
  ctx.strokeStyle = "#52606d";
  ctx.lineWidth = 2;
  parsed.bonds.forEach((bond) => {
    const a = project(parsed.atoms[bond.a]);
    const b = project(parsed.atoms[bond.b]);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  });

  parsed.atoms.forEach((atom) => {
    const p = project(atom);
    const isCarbon = atom.element === "C";
    ctx.beginPath();
    ctx.arc(p.x, p.y, isCarbon ? 3 : 5, 0, Math.PI * 2);
    ctx.fillStyle = isCarbon ? "#148f8b" : "#d87832";
    ctx.fill();
    if (!isCarbon) {
      ctx.fillStyle = "#162131";
      ctx.font = "10px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(atom.element, p.x, p.y - 8);
    }
  });
  ctx.textAlign = "left";
}

async function readText(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to load ${url}`);
  return response.text();
}

function setStructureMeta(sample) {
  document.getElementById("structureRank").textContent = `Rank #${sample.rank}`;
  document.getElementById("structureTitle").textContent = `${sample.drug} · ${sample.target}`;
  document.getElementById("structurePair").textContent = sample.pairId;
  document.getElementById("structureConfidence").textContent = formatScore(sample.confidence, 2);
  document.getElementById("structureConsensus").textContent = formatScore(sample.consensus);
  document.getElementById("structureReceptor").textContent = `${sample.protein} · ${sample.receptorStatus}`;
}

async function loadStructure(index) {
  const sample = data.structureSamples?.[index];
  const status = document.getElementById("viewerStatus");
  if (!sample || !status) return;
  activeStructure = index;
  setStructureMeta(sample);
  document.querySelectorAll("#structureTabs button").forEach((button, idx) => {
    button.classList.toggle("active", idx === index);
  });

  status.textContent = "Loading local PDB/SDF assets...";
  try {
    const [receptorText, ligandText] = await Promise.all([readText(sample.receptorUrl), readText(sample.ligandUrl)]);
    drawLigandSketch(ligandText);
    if (!window.$3Dmol) {
      status.textContent = "3Dmol is unavailable; showing ligand sketch only.";
      return;
    }
    const element = document.getElementById("structureViewer");
    element.innerHTML = "";
    viewer = window.$3Dmol.createViewer(element, { backgroundColor: "#f7fafc" });
    const receptor = viewer.addModel(receptorText, "pdb");
    receptor.setStyle({}, { cartoon: { color: "spectrum", opacity: 0.72 } });
    const ligand = viewer.addModel(ligandText, "sdf");
    ligand.setStyle({}, { stick: { radius: 0.22, colorscheme: "greenCarbon" } });
    viewer.zoomTo();
    viewer.render();
    viewer.spin(viewerSpinning);
    status.textContent = "Loaded receptor cartoon and DiffDock rank-1 ligand.";
  } catch (error) {
    status.textContent = error.message;
  }
}

function setupStructures() {
  const tabs = document.getElementById("structureTabs");
  if (!tabs || !data.structureSamples?.length) return;
  tabs.innerHTML = data.structureSamples
    .map(
      (sample, index) => `
        <button type="button" class="${index === 0 ? "active" : ""}" data-structure="${index}">
          #${sample.rank} ${sample.drug}<br><small>${sample.target} · ${sample.protein}</small>
        </button>
      `,
    )
    .join("");
  tabs.querySelectorAll("[data-structure]").forEach((button) => {
    button.addEventListener("click", () => loadStructure(Number(button.dataset.structure)));
  });
  document.getElementById("toggleSpin")?.addEventListener("click", () => {
    viewerSpinning = !viewerSpinning;
    if (viewer) viewer.spin(viewerSpinning);
  });
  document.getElementById("resetView")?.addEventListener("click", () => {
    if (viewer) {
      viewer.zoomTo();
      viewer.render();
    } else {
      loadStructure(activeStructure);
    }
  });
  loadStructure(0);
}

setMetricValues();
setupCandidateControls();
renderCandidates();
renderCharts();
setupStructures();
