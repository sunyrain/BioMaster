const data = window.BIOMASTER_DATA || {};
const candidates = data.candidates || [];
const palette = ["#155fa8", "#148f8b", "#5f9648", "#d87832", "#8a5a9e", "#52606d"];

let activeFilter = "completed";
let activeDirection = "all";
let activeCandidateIndex = null;

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => (
    {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[char]
  ));
}

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
  if (status === "completed") return "completed";
  if (status === "not_yet_run") return "not yet run";
  return "missing output";
}

function statusLabelZh(status) {
  if (status === "completed") return "结构完成";
  if (status === "not_yet_run") return "尚未运行";
  return "缺失输出";
}

function tierClass(value) {
  const tier = String(value || "").trim().charAt(0).toLowerCase();
  return ["a", "b", "c", "d"].includes(tier) ? tier : "c";
}

function scoreSourceLabel(candidate) {
  if (!candidate || candidate.status !== "completed") return "待结构补跑";
  return candidate.scoreSourceLabelZh || candidate.scoreSourceLabel || "DiffDock 来源未标注";
}

function scoreSourceClass(candidate) {
  if (!candidate || candidate.status !== "completed") return "pending";
  return candidate.scoreSource === "priority_rerun" ? "recovered" : "primary";
}

function renderPath(pathText) {
  if (!pathText) return "";
  return String(pathText)
    .split("→")
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 7)
    .map((item) => `<span>${escapeHtml(item)}</span>`)
    .join("");
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
  if (queueState) {
    const done = Number(metrics.fullRowsScored || metrics.structureCompleted || metrics.top1000Completed || 0);
    const total = Number(metrics.structureCandidates || metrics.fullRowsTotal || 0);
    queueState.textContent = total && done >= total ? "complete" : "in progress";
  }
}

function renderDiseaseDirections() {
  const container = document.getElementById("diseaseDirectionCards");
  if (!container || !data.diseaseDirections?.length) return;
  container.innerHTML = data.diseaseDirections
    .map((direction) => {
      const success = Number(direction.successRatePct || 0);
      const median = direction.medianDiffDock === null || direction.medianDiffDock === undefined ? "NA" : formatScore(direction.medianDiffDock, 2);
      return `
        <article class="direction-card" data-direction-card="${direction.direction}">
          <div class="direction-card-head">
            <div>
              <span>${direction.label}</span>
              <h3>${direction.labelZh}</h3>
            </div>
            <strong>${success.toFixed(1)}%</strong>
          </div>
          <p>${direction.summaryZh || ""}</p>
          <dl>
            <div><dt>ready pairs</dt><dd>${formatInt(direction.preparedPairs)}</dd></div>
            <div><dt>completed</dt><dd>${formatInt(direction.completed)}</dd></div>
            <div><dt>missing</dt><dd>${formatInt(direction.missing)}</dd></div>
            <div><dt>median confidence</dt><dd>${median}</dd></div>
          </dl>
        </article>
      `;
    })
    .join("");
}

function setupDirectionControls() {
  const container = document.getElementById("directionFilters");
  if (!container || !data.diseaseDirections?.length) return;
  container.innerHTML = [
    `<button class="active" type="button" data-direction="all">全部方向</button>`,
    ...data.diseaseDirections.map(
      (direction) => `<button type="button" data-direction="${direction.direction}">${direction.labelZh}</button>`,
    ),
  ].join("");
  container.querySelectorAll("[data-direction]").forEach((button) => {
    button.addEventListener("click", () => {
      activeDirection = button.dataset.direction;
      container.querySelectorAll("[data-direction]").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      renderCandidates();
    });
  });
}

function renderCandidates() {
  const tbody = document.getElementById("candidateRows");
  const searchInput = document.getElementById("candidateSearch");
  if (!tbody || !searchInput) return;

  const search = searchInput.value.trim().toLowerCase();
  const rows = candidates
    .map((candidate, index) => ({ candidate, index }))
    .filter(({ candidate }) => {
      const matchesFilter = activeFilter === "all" || candidate.status === activeFilter;
      const matchesDirection = activeDirection === "all" || candidate.direction === activeDirection;
      const haystack = (
        `${candidate.drug} ${candidate.target} ${candidate.protein} ${candidate.pairId} ${candidate.directionLabelZh || ""} ` +
        `${candidate.credibilityTierZh || ""} ${candidate.repurposingPostureZh || ""} ${candidate.evidencePathZh || ""} ` +
        `${scoreSourceLabel(candidate)}`
      ).toLowerCase();
      return matchesFilter && matchesDirection && haystack.includes(search);
    });

  const countNode = document.getElementById("candidateCount");
  if (countNode) countNode.textContent = `${formatInt(rows.length)} / ${formatInt(candidates.length)}`;

  if (!rows.length) {
    activeCandidateIndex = null;
    tbody.innerHTML = `<tr><td class="candidate-empty" colspan="4">没有匹配当前筛选条件的候选。</td></tr>`;
    renderCandidateDetail(null);
    return;
  }

  if (activeCandidateIndex === null || !rows.some((row) => row.index === activeCandidateIndex)) {
    activeCandidateIndex = rows[0].index;
  }

  const primaryScoreLabel = data.labels?.primaryScore || "Affinity";
  tbody.innerHTML = rows
    .map(
      ({ candidate, index }) => {
        const primaryScore = candidate.affinityScore ?? candidate.diseasePriority;
        const directionScore = candidate.directionScore ?? primaryScore;
        const selected = index === activeCandidateIndex ? "selected" : "";
        return `
        <tr class="${selected}" data-candidate-index="${index}" tabindex="0">
          <td>
            <span class="rank">#${escapeHtml(candidate.rank)}</span>
            <small>${escapeHtml(candidate.directionLabelZh || candidate.directionLabel || "")}</small>
          </td>
          <td class="candidate-identity">
            <strong>${escapeHtml(candidate.drug)}</strong>
            <small>${escapeHtml(candidate.target)} · ${escapeHtml(candidate.protein)}</small>
            <small>${formatInt(candidate.representedPairCount || 1)} represented records</small>
          </td>
          <td>
            <div class="score-stack">
              <span><b>${escapeHtml(primaryScoreLabel)}</b>${formatScore(directionScore, 3)}</span>
              <span><b>Affinity</b>${formatScore(candidate.affinityScore ?? primaryScore, 3)}</span>
              <span><b>Docking</b>${formatScore(candidate.diffdock, 2)}</span>
            </div>
          </td>
          <td class="candidate-evidence-compact">
            <span class="tier-pill ${tierClass(candidate.credibilityTier)}">${escapeHtml(candidate.credibilityTierZh || "C｜待补证据")}</span>
            <span class="status-pill ${escapeHtml(candidate.status)}">${escapeHtml(statusLabelZh(candidate.status))}</span>
            <span class="source-pill ${scoreSourceClass(candidate)}">${escapeHtml(scoreSourceLabel(candidate))}</span>
            <small>${escapeHtml(candidate.repurposingPostureZh || candidate.categoryZh || "")}</small>
          </td>
        </tr>
      `;
      },
    )
    .join("");

  tbody.querySelectorAll("[data-candidate-index]").forEach((row) => {
    const activate = () => {
      activateCandidate(Number(row.dataset.candidateIndex));
    };
    row.addEventListener("click", activate);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        activate();
      }
    });
  });
  renderCandidateDetail(candidates[activeCandidateIndex]);
}

function activateCandidate(index) {
  activeCandidateIndex = index;
  document.querySelectorAll("#candidateRows [data-candidate-index]").forEach((row) => {
    row.classList.toggle("selected", Number(row.dataset.candidateIndex) === activeCandidateIndex);
  });
  renderCandidateDetail(candidates[activeCandidateIndex]);
}

function renderCandidateDetail(candidate) {
  const container = document.getElementById("candidateDetail");
  if (!container) return;
  if (!candidate) {
    container.innerHTML = `
      <div class="candidate-detail-empty">
        <h3>暂无候选</h3>
        <p>调整疾病方向、结构状态或搜索关键词后查看详情。</p>
      </div>
    `;
    return;
  }

  const primaryScoreLabel = data.labels?.primaryScore || "Affinity";
  const primaryScore = candidate.affinityScore ?? candidate.diseasePriority;
  const directionScore = candidate.directionScore ?? primaryScore;
  container.innerHTML = `
    <div class="candidate-detail-header">
      <span class="rank">#${escapeHtml(candidate.rank)} · ${escapeHtml(candidate.directionLabelZh || candidate.directionLabel || "")}</span>
      <h3>${escapeHtml(candidate.drug)}</h3>
      <p>${escapeHtml(candidate.target)} · ${escapeHtml(candidate.proteinName || candidate.protein || "")}</p>
      <div class="candidate-detail-tags">
        <span class="tier-pill ${tierClass(candidate.credibilityTier)}">${escapeHtml(candidate.credibilityTierZh || "C｜待补证据")}</span>
        <span class="status-pill ${escapeHtml(candidate.status)}">${escapeHtml(statusLabelZh(candidate.status))}</span>
        <span class="source-pill ${scoreSourceClass(candidate)}">${escapeHtml(scoreSourceLabel(candidate))}</span>
      </div>
    </div>
    <div class="detail-score-grid">
      <div><span>${escapeHtml(primaryScoreLabel)}</span><strong>${formatScore(directionScore, 3)}</strong></div>
      <div><span>Affinity</span><strong>${formatScore(candidate.affinityScore ?? primaryScore, 3)}</strong></div>
      <div><span>DiffDock</span><strong>${formatScore(candidate.diffdock, 2)}</strong></div>
      <div><span>Credibility</span><strong>${formatScore(candidate.credibilityScore, 1)}</strong></div>
    </div>
    <section class="candidate-detail-block">
      <h4>证据路径</h4>
      <div class="evidence-path">${renderPath(candidate.evidencePathZh)}</div>
    </section>
    <section class="candidate-detail-block">
      <h4>中文候选合理性解释</h4>
      <p>${escapeHtml(candidate.rationaleZh || candidate.evidenceSummaryZh || "")}</p>
    </section>
    <section class="candidate-detail-block">
      <h4>下一步验证</h4>
      <p>${escapeHtml(candidate.nextStepZh || "")}</p>
      <p class="validation-gates">${escapeHtml(candidate.validationGatesZh || "")}</p>
    </section>
    <dl class="candidate-detail-meta">
      <div><dt>Pair ID</dt><dd>${escapeHtml(candidate.pairId || "")}</dd></div>
      <div><dt>Drug ID</dt><dd>${escapeHtml(candidate.drugId || "")}</dd></div>
      <div><dt>Accession</dt><dd>${escapeHtml(candidate.protein || "")}</dd></div>
      <div><dt>Therapeutic area</dt><dd>${escapeHtml(candidate.therapeuticArea || "NA")}</dd></div>
      <div><dt>Indication</dt><dd>${escapeHtml(candidate.indication || "NA")}</dd></div>
      <div><dt>Score source</dt><dd>${escapeHtml(scoreSourceLabel(candidate))}</dd></div>
      <div><dt>Represented records</dt><dd>${formatInt(candidate.representedPairCount || 1)}</dd></div>
    </dl>
  `;
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
  renderStackedBars("credibilityBars", charts.credibilityTiers);
  renderStackedBars("postureBars", charts.validationPostures);
  renderStackedBars("layerBars", [
    ...(charts.txgnnStatus || []).map((row) => ({ label: row.label, value: row.value })),
    ...(charts.receptorStatus || []).map((row) => ({ label: row.label, value: row.value })),
  ]);
}

setMetricValues();
renderDiseaseDirections();
setupDirectionControls();
setupCandidateControls();
renderCandidates();
renderCharts();
