const data = window.BIOMASTER_DATA || {};
const computeStatus = window.BIOMASTER_COMPUTE_STATUS || {};
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

function formatMaybePct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "NA";
  return `${Number(value).toFixed(digits)}%`;
}

function formatMaybeInt(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "NA";
  return formatInt(value);
}

function formatDurationHours(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "NA";
  const hours = Number(value);
  if (hours >= 48) return `${(hours / 24).toFixed(1)} days`;
  return `${hours.toFixed(1)} h`;
}

function formatUtc(value) {
  if (!value) return "NA";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toISOString().replace("T", " ").slice(0, 16) + " UTC";
}

function labelToTitle(value) {
  return String(value || "")
    .replace(/^[A-D]_/, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function buildProgressRow(label, value, max, note, color) {
  const pct = max ? (Number(value || 0) / Number(max)) * 100 : Number(value || 0);
  return `
    <div class="pipeline-step">
      <div class="pipeline-step-head">
        <span>${escapeHtml(label)}</span>
        <strong>${max ? `${formatInt(value)} / ${formatInt(max)}` : escapeHtml(String(value ?? "NA"))}</strong>
      </div>
      <div class="stack-track"><i style="--w:${Math.max(1, Math.min(pct, 100)).toFixed(2)}%; --c:${color || "#155fa8"}"></i></div>
      <small>${escapeHtml(note || "")}</small>
    </div>
  `;
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

function renderDelimitedTags(value) {
  if (!value) return "";
  return String(value)
    .split(/[;；]/)
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, 8)
    .map((item) => `<span>${escapeHtml(item)}</span>`)
    .join("");
}

function splitEvidenceItems(value, maxItems = 8) {
  if (!value) return [];
  return String(value)
    .split(/[;；。]/)
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, maxItems);
}

function evidenceChipClass(value) {
  const text = String(value || "");
  if (/Open Targets/i.test(text)) return "open-targets";
  if (/TxGNN/i.test(text)) return "txgnn";
  if (/FDA|治疗领域|适应症/i.test(text)) return "fda";
  if (/DiffDock|结构|姿态/i.test(text)) return "structure";
  if (/ConPLex|亲和/i.test(text)) return "affinity";
  return "context";
}

function renderEvidenceChips(value, maxItems = 8) {
  const items = splitEvidenceItems(value, maxItems);
  if (!items.length) return `<span class="evidence-chip muted">证据待补充</span>`;
  return items
    .map((item) => `<span class="evidence-chip ${evidenceChipClass(item)}">${escapeHtml(item)}</span>`)
    .join("");
}

function renderMiniStats(rows) {
  return rows
    .map(
      (row) => `
        <div class="mini-stat">
          <small>${escapeHtml(row.label)}</small>
          <strong>${escapeHtml(row.value)}</strong>
          ${row.note ? `<em>${escapeHtml(row.note)}</em>` : ""}
        </div>
      `,
    )
    .join("");
}

function renderSentenceList(value, maxItems = 3) {
  const sentences = splitSentences(value, maxItems);
  if (!sentences.length) return "";
  return `
    <ul class="sentence-list">
      ${sentences.map((sentence) => `<li>${escapeHtml(sentence)}</li>`).join("")}
    </ul>
  `;
}

function splitSentences(value, maxItems = 8) {
  if (!value) return [];
  return (String(value).replace(/\s+/g, " ").match(/[^。！？!?]+[。！？!?]?/g) || [])
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, maxItems);
}

function evidenceSentenceLabel(sentence, index) {
  if (/疾病方向|方向分数|ConPLex|亲和|模型/.test(sentence)) return "模型信号";
  if (/支持证据|Open Targets|TxGNN|FDA|ICD|治疗领域|适应症/.test(sentence)) return "证据来源";
  if (/DiffDock|confidence|rank-1|结构|姿态|结合/.test(sentence)) return "结构解释";
  if (/代表|回填|UniProt|记录|序列/.test(sentence)) return "代表复用";
  return `解释 ${index + 1}`;
}

function renderRationaleCards(candidate) {
  const fullText = candidate?.rationaleZh || candidate?.evidenceSummaryZh || "";
  const allSentences = splitSentences(fullText, 12);
  const sentences = allSentences.slice(0, 5);
  if (!sentences.length) {
    return `<div class="detail-empty-note">暂无结构化候选解释；请结合分数、证据路径和原始候选表审阅。</div>`;
  }
  const cards = sentences
    .map(
      (sentence, index) => `
        <article class="rationale-point">
          <span>${escapeHtml(evidenceSentenceLabel(sentence, index))}</span>
          <p>${escapeHtml(sentence)}</p>
        </article>
      `,
    )
    .join("");
  const remainder = allSentences.slice(5);
  if (!remainder.length) return cards;
  return `
    ${cards}
    <details class="rationale-more">
      <summary>查看完整解释</summary>
      <div>${remainder.map((sentence) => `<p>${escapeHtml(sentence)}</p>`).join("")}</div>
    </details>
  `;
}

function firstSentences(value, count = 2) {
  const sentences = splitSentences(value, count);
  return sentences.length ? sentences.join("") : String(value || "");
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

function renderOverviewStatus(full) {
  const heroState = document.getElementById("heroFullRunState");
  if (heroState) {
    const progress = Number(full.completedJobPct ?? full.scoredRowPct ?? 0);
    heroState.textContent = full.status
      ? `${full.status} · ${formatMaybePct(progress, 1)}`
      : "not available";
  }
}

function renderComputeSnapshot(full, standard, rescue, multiRescue, summary) {
  const strip = document.getElementById("computeSnapshotStrip");
  if (!strip) return;
  const progress = Number(full.completedJobPct ?? full.scoredRowPct ?? 0);
  const rows = [
    {
      label: "Full DiffDock",
      value: `${formatMaybePct(progress, 2)} complete`,
      note: `${formatMaybeInt(full.completedJobs)} / ${formatMaybeInt(full.totalJobs)} jobs; ETA ${formatDurationHours(full.etaHours)}`,
    },
    {
      label: "GPU state",
      value: `${formatMaybeInt(full.busyGpuCount)} / ${formatMaybeInt(full.gpuCount)} busy`,
      note: `${formatMaybeInt((full.activeLocks || []).length)} active locks; transient low util can occur inside a running chunk`,
    },
    {
      label: "Standard validation",
      value: standard.status || "unknown",
      note: standard.interpretationZh || "PoseBusters/ProLIF validation status is read from local outputs",
    },
    {
      label: "Rescue queue",
      value: `${formatMaybeInt(rescue.queuedRows)} rows`,
      note: `${formatMaybeInt(rescue.jobs)} single-ligand jobs; latest multi-ligand audit recommends ${formatMaybeInt(multiRescue.latestAuditRescueRecommendedLigands)} ligands / ${formatMaybeInt(multiRescue.latestAuditRescueRecommendedRows)} rows`,
    },
  ];
  strip.innerHTML = rows
    .map(
      (row) => `
        <article>
          <span>${escapeHtml(row.label)}</span>
          <strong>${escapeHtml(row.value)}</strong>
          <p>${escapeHtml(row.note)}</p>
        </article>
      `,
    )
    .join("");
}

function renderComputeStatus() {
  if (!computeStatus || !Object.keys(computeStatus).length) return;
  const active = computeStatus.active || {};
  const full = active.fullDiffdock || {};
  const standard = active.standardPoseFull3921 || {};
  const rescue = active.ligandRescue || {};
  const multiRescue = active.multiLigandRescue || {};
  const summary = computeStatus.summary || {};
  renderOverviewStatus(full);
  renderComputeSnapshot(full, standard, rescue, multiRescue, summary);

  const gpuOccupancy = document.getElementById("gpuOccupancy");
  if (gpuOccupancy) {
    const locks = full.activeLocks?.length || 0;
    gpuOccupancy.textContent =
      `${formatMaybeInt(full.busyGpuCount)} / ${formatMaybeInt(full.gpuCount)} telemetry · ${formatMaybeInt(locks)} locks`;
  }

  const completedModuleCount = document.getElementById("completedModuleCount");
  if (completedModuleCount) completedModuleCount.textContent = formatMaybeInt(summary.completedModuleCount);

  const wave1Rows = document.getElementById("wave1Rows");
  if (wave1Rows) wave1Rows.textContent = formatMaybeInt(summary.wave1Rows);

  const updated = document.getElementById("diffdockUpdatedAt");
  if (updated) updated.textContent = `updated ${formatUtc(full.createdUtc || computeStatus.sourceUpdatedUtc)}`;

  const progress = Number(full.completedJobPct ?? full.scoredRowPct ?? 0);
  const progressBar = document.getElementById("fullDiffdockProgressBar");
  if (progressBar) progressBar.style.width = `${Math.max(0, Math.min(progress, 100)).toFixed(2)}%`;

  const progressText = document.getElementById("fullDiffdockProgressText");
  if (progressText) progressText.textContent = formatMaybePct(progress, 2);

  const progressRows = document.getElementById("fullDiffdockProgressRows");
  if (progressRows) {
    progressRows.textContent =
      `${formatMaybeInt(full.completedJobs)} / ${formatMaybeInt(full.totalJobs)} jobs · ` +
      `${formatMaybeInt(full.scoredRows)} / ${formatMaybeInt(full.totalRows)} rows · ETA ${formatDurationHours(full.etaHours)}`;
  }

  const interpretation = document.getElementById("fullDiffdockInterpretation");
  if (interpretation) interpretation.textContent = full.interpretationZh || "";

  const chunkRail = document.getElementById("activeChunkRail");
  if (chunkRail) {
    const chunks = full.activeDetails?.length
      ? full.activeDetails
      : (full.activeLocks || []).map((jobId) => ({ jobId, inFlightRank1SdfCount: null }));
    chunkRail.innerHTML = chunks.length
      ? chunks
          .map(
            (chunk) => `
              <div class="active-chunk-card">
                <span>chunk ${String(chunk.jobId).padStart(5, "0")}</span>
                <strong>${formatMaybeInt(chunk.inFlightRank1SdfCount)}</strong>
                <small>in-flight rank1 SDF</small>
              </div>
            `,
          )
          .join("")
      : `<div class="compute-empty">No active chunk lock detected.</div>`;
  }

  const activeCards = document.getElementById("activeComputeCards");
  if (activeCards) {
    const standardDone = standard.status === "completed";
    activeCards.innerHTML = [
      {
        title: "Full DiffDock",
        status: full.status || "running",
        value: `${formatMaybeInt(full.completedOutputs)} outputs`,
        note: `${formatMaybeInt(full.missingOutputsInScoredJobs)} technical missing in scored chunks`,
      },
      {
        title: "Standard PoseBusters/ProLIF",
        status: standard.status || "unknown",
        value: standardDone ? "summary ready" : "running",
        note: standard.interpretationZh || "full3921 standard validation state",
      },
      {
        title: "CHEMBL3039504 rescue",
        status: rescue.status || "queued",
        value: `${formatMaybeInt(rescue.queuedRows)} rows`,
        note: `${formatMaybeInt(rescue.jobs)} jobs · starts after main queue and GPU idle`,
      },
      {
        title: "Multi-ligand rescue audit",
        status: multiRescue.status || "queued",
        value: `${formatMaybeInt(multiRescue.latestAuditRescueRecommendedLigands)} ligands`,
        note: `${formatMaybeInt(multiRescue.latestAuditRescueRecommendedRows)} latest audit rows; current prebuilt queue has ${formatMaybeInt(multiRescue.queuedLigands)} ligands / ${formatMaybeInt(multiRescue.queuedRows)} rows`,
      },
      {
        title: "Artifact manifest",
        status: "refreshing",
        value: `${formatMaybeInt(summary.artifactManifestCount)} files`,
        note: `${formatMaybeInt(summary.sourceScriptCount)} source scripts attributed`,
      },
    ]
      .map(
        (card) => `
          <article class="compute-mini-card">
            <span>${escapeHtml(card.status)}</span>
            <h3>${escapeHtml(card.title)}</h3>
            <strong>${escapeHtml(card.value)}</strong>
            <p>${escapeHtml(card.note)}</p>
          </article>
        `,
      )
      .join("");
  }

  renderGpuTelemetry(full);
}

function renderStatusReport() {
  const runPanel = document.getElementById("statusRunPanel");
  const throughputGrid = document.getElementById("statusThroughputGrid");
  const progressBar = document.getElementById("statusFullDiffdockProgressBar");
  const chunkRail = document.getElementById("statusActiveChunks");
  const gpuGrid = document.getElementById("statusGpuGrid");
  const footerSnapshot = document.getElementById("statusFooterSnapshot");
  if (!runPanel && !throughputGrid && !progressBar && !chunkRail && !gpuGrid && !footerSnapshot) return;

  const active = computeStatus.active || {};
  const full = active.fullDiffdock || {};
  const standard = active.standardPoseFull3921 || {};
  const rescue = active.ligandRescue || {};
  const multiRescue = active.multiLigandRescue || {};
  const summary = computeStatus.summary || {};
  const updated = formatUtc(full.createdUtc || computeStatus.sourceUpdatedUtc || computeStatus.updatedUtc);
  const activeLocks = full.activeLocks || [];

  if (runPanel) {
    runPanel.innerHTML = [
      ["Snapshot status", "running + completed layer", `snapshot ${updated}`],
      [
        "CUDA devices",
        `${formatMaybeInt(full.busyGpuCount)} / ${formatMaybeInt(full.gpuCount)} busy`,
        `${formatMaybeInt(activeLocks.length)} active DiffDock chunk locks`,
      ],
      [
        "Active full jobs",
        activeLocks.length ? activeLocks.map((jobId) => `chunk ${String(jobId).padStart(5, "0")}`).join(", ") : "none",
        `full expansion status: ${full.status || "unknown"}`,
      ],
    ]
      .map(
        ([label, value, note]) => `
          <div>
            <span>${escapeHtml(label)}</span>
            <strong>${escapeHtml(value)}</strong>
            <small>${escapeHtml(note)}</small>
          </div>
        `,
      )
      .join("");
  }

  if (throughputGrid) {
    throughputGrid.innerHTML = [
      [
        "Full DiffDock jobs",
        `${formatMaybeInt(full.completedJobs)} / ${formatMaybeInt(full.totalJobs)}`,
        `${formatMaybePct(full.completedJobPct, 2)} completed`,
      ],
      [
        "Full rows scored",
        `${formatMaybeInt(full.scoredRows)} / ${formatMaybeInt(full.totalRows)}`,
        `${formatMaybePct(full.scoredRowPct, 2)} row progress`,
      ],
      [
        "Completed outputs",
        formatMaybeInt(full.completedOutputs),
        `${formatMaybeInt(full.missingOutputsInScoredJobs)} technical missing in scored jobs`,
      ],
      [
        "ETA",
        `~${formatDurationHours(full.etaHours)}`,
        `estimated finish ${formatUtc(full.estimatedFinishUtc)}`,
      ],
      [
        "Standard validation",
        standard.status || "unknown",
        standard.interpretationZh || "full3921 standard PoseBusters/ProLIF state",
      ],
      [
        "Ligand rescue queue",
        `${formatMaybeInt(rescue.queuedRows)} rows`,
        `${formatMaybeInt(rescue.jobs)} jobs; watcher ready: ${rescue.watcherReady ? "yes" : "no"}`,
      ],
      [
        "Multi-ligand rescue audit",
        `${formatMaybeInt(multiRescue.latestAuditRescueRecommendedLigands)} ligands`,
        `${formatMaybeInt(multiRescue.latestAuditRescueRecommendedRows)} latest-audit rows; prebuilt queue ${formatMaybeInt(multiRescue.queuedLigands)} ligands / ${formatMaybeInt(multiRescue.queuedRows)} rows`,
      ],
      [
        "SOTA modules",
        formatMaybeInt(summary.completedModuleCount),
        `${formatMaybeInt(summary.artifactManifestCount)} indexed artifacts`,
      ],
      [
        "Validation panel",
        `${formatMaybeInt(summary.wave1Rows)} wave-1 rows`,
        `${formatMaybeInt(summary.experimentReadyRows)} experiment-ready candidates`,
      ],
    ]
      .map(
        ([label, value, note]) => `
          <article>
            <span>${escapeHtml(label)}</span>
            <strong>${escapeHtml(value)}</strong>
            <small>${escapeHtml(note)}</small>
          </article>
        `,
      )
      .join("");
  }

  if (progressBar) {
    const progress = Number(full.completedJobPct ?? full.scoredRowPct ?? 0);
    progressBar.style.width = `${Math.max(0, Math.min(progress, 100)).toFixed(2)}%`;
  }

  if (chunkRail) {
    const chunks = full.activeDetails?.length
      ? full.activeDetails
      : activeLocks.map((jobId) => ({ jobId, inFlightRank1SdfCount: null }));
    chunkRail.innerHTML = chunks.length
      ? chunks
          .map(
            (chunk) => `
              <div class="active-chunk-card">
                <span>chunk ${String(chunk.jobId).padStart(5, "0")}</span>
                <strong>${formatMaybeInt(chunk.inFlightRank1SdfCount)}</strong>
                <small>in-flight rank1 SDF</small>
              </div>
            `,
          )
          .join("")
      : `<div class="compute-empty">No active full DiffDock lock detected.</div>`;
  }

  if (gpuGrid) renderGpuTelemetryInto(gpuGrid, full?.gpus || []);

  if (footerSnapshot) footerSnapshot.textContent = `Snapshot: ${updated}`;
}

function renderSotaOverview() {
  if (!computeStatus || !Object.keys(computeStatus).length) return;
  const evidence = document.getElementById("evidenceLayerBars");
  if (evidence) {
    evidence.innerHTML = (computeStatus.evidenceLayers || [])
      .map((row, index) => {
        const value = Number(row.value || 0);
        return `
          <div class="evidence-layer-row">
            <div class="evidence-layer-head">
              <span>${escapeHtml(row.label)}</span>
              <strong>${formatMaybePct(row.value, 1)}</strong>
            </div>
            <div class="stack-track"><i style="--w:${Math.max(1, Math.min(value, 100))}%; --c:${palette[index % palette.length]}"></i></div>
            <small>${escapeHtml(row.note || "")}</small>
          </div>
        `;
      })
      .join("");
  }

  renderStackedBars("sotaReadyTierBars", computeStatus.sotaReady?.tierCounts);

  const benchmark = computeStatus.benchmarks || {};
  const benchmarkMetrics = document.getElementById("benchmarkMetrics");
  if (benchmarkMetrics) {
    const rows = [
      ["Known pair Recall@100k", formatMaybePct(benchmark.knownPairRecallAt100000Pct, 2), `${Number(benchmark.knownPairEnrichmentAt100000 || 0).toFixed(2)}x enrichment`],
      ["Final Recall@100", formatMaybePct(benchmark.finalPriorityValidationRecallAt100Pct, 2), `${formatMaybePct(benchmark.finalPriorityValidationPrecisionAt100Pct, 2)} precision`],
      ["Top100 observed hits", formatMaybeInt(benchmark.significanceTop100ObservedHits), `expected ${Number(benchmark.significanceTop100GlobalExpectedHits || 0).toFixed(2)}`],
      ["Top100 multi-evidence", formatMaybePct(benchmark.concordanceTop100MultiEvidencePct, 1), `global ${formatMaybePct(benchmark.concordanceMultiEvidencePct, 1)}`],
    ];
    benchmarkMetrics.innerHTML = rows
      .map(
        ([label, value, note]) => `
          <div>
            <span>${escapeHtml(label)}</span>
            <strong>${escapeHtml(value)}</strong>
            <small>${escapeHtml(note)}</small>
          </div>
        `,
      )
      .join("");
  }

  const structural = document.getElementById("structuralModelRows");
  if (structural) {
    structural.innerHTML = (computeStatus.structuralModels || [])
      .map((row, index) => {
        const value = Number(row.pct ?? 0);
        return `
          <div class="structural-model-row">
            <div>
              <strong>${escapeHtml(row.label)}</strong>
              <small>${escapeHtml(row.note || "")}</small>
            </div>
            <span>${formatMaybeInt(row.completed)} / ${formatMaybeInt(row.total)}</span>
            <div class="stack-track"><i style="--w:${Math.max(1, Math.min(value, 100))}%; --c:${palette[index % palette.length]}"></i></div>
            <b>${formatMaybePct(row.pct, 1)}</b>
          </div>
        `;
      })
      .join("");
  }

  renderStackedBars(
    "finalPriorityBars",
    (computeStatus.finalPriority?.tierCounts || []).map((row) => ({ label: `Tier ${row.label}`, value: row.value })),
  );
  renderStackedBars(
    "sotaActionBars",
    (computeStatus.sotaReady?.actionCounts || []).map((row) => ({ label: labelToTitle(row.label), value: row.value })),
  );

  const dependencyChecklist = document.getElementById("dependencyChecklist");
  if (dependencyChecklist) {
    const deps = computeStatus.dependencies?.blockers || [];
    dependencyChecklist.innerHTML = deps.length
      ? deps
          .map(
            (dep) => `
              <article class="dependency-check-row">
                <span>${escapeHtml(dep.status)}</span>
                <div>
                  <strong>${escapeHtml(dep.label)}</strong>
                  <p>${escapeHtml(dep.detailZh)}</p>
                </div>
              </article>
            `,
          )
          .join("")
      : `<div class="compute-empty">No external dependency blocker recorded.</div>`;
  }
}

function renderValidationPlan() {
  if (!computeStatus || !Object.keys(computeStatus).length) return;
  const panel = computeStatus.validationPanel || {};
  const metrics = document.getElementById("validationMetrics");
  if (metrics) {
    const rows = [
      ["Experiment-ready", panel.experimentReadyRows, "can move to assay planning"],
      ["Expert-review-ready", panel.reviewReadyRows, "requires specialist review"],
      ["Novel ready/review", panel.novelReadyRows, "new-pair or extension hypotheses"],
      ["Positive controls", panel.positiveControlReadyRows, "known mechanism anchors"],
      ["Balanced panel", panel.balancedPanelRows, `${formatMaybeInt(panel.balancedPanelDirections)} disease directions`],
      ["Diversity", panel.balancedUniqueDrugs, `${formatMaybeInt(panel.balancedUniqueTargets)} targets · ${formatMaybeInt(panel.balancedUniqueScaffolds)} scaffolds`],
    ];
    metrics.innerHTML = rows
      .map(
        ([label, value, note]) => `
          <div>
            <span>${escapeHtml(label)}</span>
            <strong>${formatMaybeInt(value)}</strong>
            <small>${escapeHtml(note)}</small>
          </div>
        `,
      )
      .join("");
  }

  renderStackedBars("assayModalityBars", panel.assayCounts);

  const deps = document.getElementById("dependencyCards");
  if (deps) {
    deps.innerHTML = (computeStatus.dependencies?.blockers || [])
      .map(
        (dep) => `
          <article class="dependency-card">
            <span>${escapeHtml(dep.status)}</span>
            <h3>${escapeHtml(dep.label)}</h3>
            <p>${escapeHtml(dep.detailZh)}</p>
          </article>
        `,
      )
      .join("");
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

function getFilteredCandidateRows() {
  const searchInput = document.getElementById("candidateSearch");
  const search = searchInput ? searchInput.value.trim().toLowerCase() : "";
  return candidates
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
}

function renderCandidates() {
  const tbody = document.getElementById("candidateRows");
  const searchInput = document.getElementById("candidateSearch");
  if (!tbody || !searchInput) return;

  const rows = getFilteredCandidateRows();

  const countNode = document.getElementById("candidateCount");
  if (countNode) countNode.textContent = `${formatInt(rows.length)} / ${formatInt(candidates.length)}`;

  if (!rows.length) {
    activeCandidateIndex = null;
    tbody.innerHTML = `<tr><td class="candidate-empty" colspan="4">没有匹配当前筛选条件的候选。</td></tr>`;
    renderCandidateDetail(null);
    updateCandidateSummary([]);
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
  updateCandidateSummary(rows);
}

function activateCandidate(index) {
  activeCandidateIndex = index;
  document.querySelectorAll("#candidateRows [data-candidate-index]").forEach((row) => {
    row.classList.toggle("selected", Number(row.dataset.candidateIndex) === activeCandidateIndex);
  });
  renderCandidateDetail(candidates[activeCandidateIndex]);
  updateCandidateSummary(getFilteredCandidateRows());
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
      <div class="evidence-path">${renderPath(candidate.evidencePathZh) || `<span>等待证据路径补充</span>`}</div>
      <div class="evidence-chip-grid">${renderEvidenceChips(candidate.evidenceSummaryZh || candidate.evidenceSummary, 8)}</div>
    </section>
    <section class="candidate-detail-block detail-text-card">
      <h4>中文候选合理性解释</h4>
      <div class="rationale-point-list">${renderRationaleCards(candidate)}</div>
    </section>
    <section class="candidate-detail-block detail-validation-card">
      <h4>下一步验证</h4>
      <div class="detail-next-step">${escapeHtml(candidate.nextStepZh || "结合疾病背景、结构姿态、已知靶点关系和实验可操作性进入专家审阅。")}</div>
      <div class="detail-tag-list">${renderDelimitedTags(candidate.validationGatesZh) || `<span>文献核查</span><span>结构复核</span><span>实验可行性</span>`}</div>
    </section>
    <h4 class="detail-meta-title">技术口径</h4>
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

function renderExecutiveSummary() {
  const metrics = data.metrics || {};
  const summary = computeStatus.summary || {};
  const full = computeStatus.active?.fullDiffdock || {};
  const validation = computeStatus.validationPanel || {};
  const benchmarks = computeStatus.benchmarks || {};
  const finalPriority = computeStatus.finalPriority || {};

  const snapshot = document.getElementById("snapshotUpdatedAt");
  if (snapshot) snapshot.textContent = formatUtc(computeStatus.updatedUtc || data.updated);

  const executiveCards = document.getElementById("executiveCards");
  if (executiveCards) {
    const cards = [
      {
        label: "Primary screen",
        value: formatInt(metrics.pairs),
        title: "药物-蛋白亲和矩阵已完成",
        note: `${formatInt(metrics.drugs)} drugs × ${formatInt(metrics.targets)} proteins; ${formatInt(metrics.uniqueSequences)} unique sequences reused.`,
      },
      {
        label: "Structure audit",
        value: formatMaybePct(metrics.fullOutputRatePct, 2),
        title: "疾病方向结构增强基本闭合",
        note: `${formatInt(metrics.structureCompleted)} / ${formatInt(metrics.structureCandidates)} DiffDock-ready representatives have rank-1 poses.`,
      },
      {
        label: "SOTA integration",
        value: formatMaybeInt(summary.completedModuleCount),
        title: "多证据模型层已接入",
        note: `A/B review-ready SOTA candidates: ${formatInt((computeStatus.sotaReady?.tierCounts || []).filter((row) => /^A_|^B_/.test(row.label)).reduce((sum, row) => sum + row.value, 0))}.`,
      },
      {
        label: "Validation",
        value: formatMaybeInt(validation.experimentReadyRows),
        title: "实验优先级队列已形成",
        note: `${formatMaybeInt(validation.balancedPanelRows)} balanced-panel rows; ${formatMaybeInt(summary.wave1Rows)} wave-1 rows.`,
      },
    ];
    executiveCards.innerHTML = cards
      .map(
        (card) => `
          <article class="executive-card">
            <span>${escapeHtml(card.label)}</span>
            <strong>${escapeHtml(card.value)}</strong>
            <h3>${escapeHtml(card.title)}</h3>
            <p>${escapeHtml(card.note)}</p>
          </article>
        `,
      )
      .join("");
  }

  const runway = document.getElementById("pipelineRunway");
  if (runway) {
    runway.innerHTML = [
      buildProgressRow("Input library", metrics.drugs, metrics.drugs, `${formatInt(metrics.targets)} druggable protein records`, palette[0]),
      buildProgressRow("Affinity screen", metrics.pairs, metrics.pairs, "ConPLex drug-target score matrix complete", palette[1]),
      buildProgressRow("Disease Top sets", metrics.structureCandidates, metrics.topCandidates, "sequence-level structural representative reduction", palette[2]),
      buildProgressRow("Disease DiffDock", metrics.structureCompleted, metrics.structureCandidates, "final output rate after rescue and reuse", palette[3]),
      buildProgressRow("Full expansion", full.completedJobs, full.totalJobs, `ETA ${formatDurationHours(full.etaHours)}; active chunks ${(full.activeLocks || []).join(", ") || "NA"}`, palette[4]),
    ].join("");
  }

  const decisionCards = document.getElementById("decisionCards");
  if (decisionCards) {
    const topKnown = computeStatus.sotaReady?.top100KnownRows ?? benchmarks.finalPriorityValidationPrecisionAt100Pct;
    const rows = [
      ["结果用途", "候选优先级与专家审阅短名单，不是药效或临床结论。"],
      ["已知召回", `Top100 中可见已知机制/靶点召回信号，当前 SOTA-ready Top100 known rows 为 ${formatMaybeInt(topKnown)}。`],
      ["新用途线索", `${formatMaybeInt(computeStatus.sotaReady?.top100NovelRows)} 个 Top100 候选属于 novel 或 extension 讨论对象。`],
      ["审阅队列", `最终优先级候选 ${formatMaybeInt(finalPriority.candidateRows || summary.candidateRows)} 行，后续按机制、疾病上下文、结构和安全性分轨。`],
    ];
    decisionCards.innerHTML = rows
      .map(
        ([title, note]) => `
          <div class="decision-card">
            <strong>${escapeHtml(title)}</strong>
            <p>${escapeHtml(note)}</p>
          </div>
        `,
      )
      .join("");
  }
}

function renderGpuTelemetry(full) {
  const grid = document.getElementById("gpuTelemetryGrid");
  if (!grid) return;
  renderGpuTelemetryInto(grid, full?.gpus || []);
}

function renderGpuTelemetryInto(grid, gpus) {
  if (!gpus.length) {
    grid.innerHTML = "";
    return;
  }
  grid.innerHTML = gpus
    .map((gpu) => {
      const memoryPct = gpu.memoryTotalMb ? (Number(gpu.memoryUsedMb || 0) / Number(gpu.memoryTotalMb)) * 100 : 0;
      const active = Number(gpu.memoryUsedMb || 0) > 512 || Number(gpu.utilizationPct || 0) > 0;
      return `
        <article class="gpu-card ${active ? "active" : "idle"}">
          <div class="gpu-card-head">
            <span>GPU ${escapeHtml(gpu.index)}</span>
            <strong>${active ? "active" : "telemetry idle"}</strong>
          </div>
          <div class="gpu-meter">
            <span>memory</span>
            <div class="stack-track"><i style="--w:${Math.max(1, Math.min(memoryPct, 100)).toFixed(1)}%; --c:${active ? "#148f8b" : "#9aa8b7"}"></i></div>
            <b>${formatMaybeInt(gpu.memoryUsedMb)} / ${formatMaybeInt(gpu.memoryTotalMb)} MB</b>
          </div>
          <div class="gpu-meta">
            <span>${formatMaybeInt(gpu.utilizationPct)}% util</span>
            <span>${formatMaybeInt(gpu.temperatureC)} C</span>
            <span>${Number(gpu.powerW || 0).toFixed(1)} W</span>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderCandidateReadingRail(visible, activeCandidate) {
  const rail = document.getElementById("candidateReadingRail");
  if (!rail) return;
  const completed = visible.filter((candidate) => candidate.status === "completed").length;
  const abTier = visible.filter((candidate) => ["a", "b"].includes(tierClass(candidate.credibilityTier))).length;
  const knownLike = visible.filter((candidate) => /已知|正控|multi|强候选/i.test(
    `${candidate.repurposingPostureZh || ""} ${candidate.category || ""} ${candidate.categoryZh || ""}`,
  )).length;
  const directionLabel =
    activeDirection === "all"
      ? "全部疾病方向"
      : (data.diseaseDirections || []).find((item) => item.direction === activeDirection)?.labelZh || activeDirection;
  const selectedTitle = activeCandidate
    ? `${activeCandidate.drug} → ${activeCandidate.target}`
    : "暂无选中候选";
  const selectedNote = activeCandidate
    ? firstSentences(activeCandidate.rationaleZh || activeCandidate.evidenceSummaryZh || activeCandidate.repurposingPostureZh || "", 2)
    : "调整筛选条件后查看候选解释。";
  const cards = [
    {
      label: "当前视图",
      title: `${directionLabel} · ${formatInt(visible.length)} rows`,
      kind: "scope",
      stats: [
        { label: "结构完成", value: formatInt(completed) },
        { label: "A/B 级", value: formatInt(abTier) },
      ],
      note: "筛选条件会同步刷新下方候选表和右侧详情。",
    },
    {
      label: "证据类型",
      title: `${formatInt(knownLike)} 个正控/强证据线索`,
      kind: "evidence",
      note: "这类结果主要说明流程能找回药理上可解释的信号，也可作为后续实验或文献审阅的阳性参照。",
    },
    {
      label: "当前候选",
      title: selectedTitle,
      kind: "selected",
      note: selectedNote,
      evidence: activeCandidate ? renderEvidenceChips(activeCandidate.evidenceSummaryZh || activeCandidate.evidenceSummary, 5) : "",
      wide: true,
    },
    {
      label: "阅读口径",
      title: "先看疾病证据，再看亲和和结构",
      kind: "rule",
      note: "DiffDock confidence 用于判断姿态是否值得审阅；负数不是结合自由能，missing output 是技术缺失。",
    },
  ];
  rail.innerHTML = cards
    .map(
      (card) => `
        <article class="rail-card ${escapeHtml(card.kind || "")} ${card.wide ? "wide" : ""}">
          <span class="rail-label">${escapeHtml(card.label)}</span>
          <h3>${escapeHtml(card.title)}</h3>
          ${card.stats ? `<div class="mini-stat-grid compact">${renderMiniStats(card.stats)}</div>` : ""}
          ${card.evidence ? `<div class="evidence-chip-grid compact">${card.evidence}</div>` : ""}
          ${renderSentenceList(card.note, 3) || `<p>${escapeHtml(card.note)}</p>`}
        </article>
      `,
    )
    .join("");
}

function updateCandidateSummary(rows) {
  const filterSummary = document.getElementById("candidateFilterSummary");
  const evidenceSummary = document.getElementById("candidateEvidenceSummary");
  const rail = document.getElementById("candidateReadingRail");
  if (!filterSummary && !evidenceSummary && !rail) return;

  const visible = rows.map((row) => row.candidate);
  const completed = visible.filter((candidate) => candidate.status === "completed").length;
  const missing = visible.filter((candidate) => candidate.status === "missing_output").length;
  const abTier = visible.filter((candidate) => ["a", "b"].includes(tierClass(candidate.credibilityTier))).length;
  const uniqueDrugs = new Set(visible.map((candidate) => candidate.drugId || candidate.drug).filter(Boolean)).size;
  const uniqueTargets = new Set(visible.map((candidate) => candidate.protein || candidate.target).filter(Boolean)).size;
  const activeCandidate = candidates[activeCandidateIndex] || visible[0] || null;
  renderCandidateReadingRail(visible, activeCandidate);

  if (filterSummary) {
    filterSummary.innerHTML = `
      <span>当前筛选</span>
      <h3>${formatInt(visible.length)} rows · ${formatInt(uniqueDrugs)} drugs · ${formatInt(uniqueTargets)} targets</h3>
      <div class="mini-stat-grid">
        ${renderMiniStats([
          { label: "结构完成", value: formatInt(completed), note: "DiffDock rank-1 姿态" },
          { label: "缺失输出", value: formatInt(missing), note: "进入技术审计" },
          { label: "A/B 级", value: formatInt(abTier), note: "优先专家审阅" },
        ])}
      </div>
    `;
  }

  if (evidenceSummary) {
    evidenceSummary.innerHTML = activeCandidate
      ? `
        <span>当前选中候选</span>
        <h3>${escapeHtml(activeCandidate.drug)} → ${escapeHtml(activeCandidate.target)}</h3>
        <div class="candidate-summary-tags">
          <span class="tier-pill ${tierClass(activeCandidate.credibilityTier)}">${escapeHtml(activeCandidate.credibilityTierZh || "C｜待补证据")}</span>
          <span class="status-pill ${escapeHtml(activeCandidate.status)}">${escapeHtml(statusLabelZh(activeCandidate.status))}</span>
          <span class="source-pill ${scoreSourceClass(activeCandidate)}">${escapeHtml(scoreSourceLabel(activeCandidate))}</span>
        </div>
        <div class="evidence-chip-grid compact">${renderEvidenceChips(activeCandidate.evidenceSummaryZh || activeCandidate.evidenceSummary, 5)}</div>
        <p class="candidate-summary-note">${escapeHtml(activeCandidate.repurposingPostureZh || activeCandidate.categoryZh || "候选需要结合疾病证据和结构证据审阅。")}</p>
      `
      : `
        <span>当前选中候选</span>
        <h3>暂无候选</h3>
        <p>调整疾病方向、结构状态或搜索关键词后查看候选摘要。</p>
      `;
  }
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

function renderQualityAudit() {
  const audit = data.qualityAudit || {};
  const narrative = document.getElementById("qualityAuditNarrative");
  if (narrative) narrative.textContent = audit.interpretationZh || "";

  const metrics = document.getElementById("qualityAuditMetrics");
  if (metrics) {
    const cards = [
      ["最终完成率", formatPct(audit.outputRatePct), `${formatInt(audit.completed)} / ${formatInt(audit.total)} representatives`],
      ["救援/复用完成", formatInt(audit.rerunRecovered), `${formatPct(audit.recoveryRatePct)} of primary missing`],
      ["剩余缺失", formatInt(audit.missing), `${formatPct(audit.remainingMissingPct)} of docking set`],
      ["原始缺失", formatInt(audit.primaryMissing), "before rescue and cross-direction reuse"],
    ];
    metrics.innerHTML = cards
      .map(
        ([label, value, note]) => `
          <article class="audit-metric-card">
            <span>${escapeHtml(label)}</span>
            <strong>${escapeHtml(value)}</strong>
            <small>${escapeHtml(note)}</small>
          </article>
        `,
      )
      .join("");
  }

  const directionRows = document.getElementById("qualityDirectionRows");
  if (directionRows && audit.directionMissing?.length) {
    directionRows.innerHTML = audit.directionMissing
      .map(
        (row) => `
          <tr>
            <td>${escapeHtml(row.labelZh || row.label)}</td>
            <td>${formatInt(row.completed)}</td>
            <td>${formatInt(row.missing)}</td>
            <td>${formatPct(row.missingRatePct)}</td>
          </tr>
        `,
      )
      .join("");
  }

  const failureModes = document.getElementById("qualityFailureModes");
  if (failureModes) {
    renderStackedBars(
      "qualityFailureModes",
      (audit.failureModes || []).map((row) => ({ label: row.label, value: row.value })),
    );
  }

  const receptorStatuses = document.getElementById("qualityReceptorStatuses");
  if (receptorStatuses) {
    renderStackedBars(
      "qualityReceptorStatuses",
      (audit.receptorStatuses || []).map((row) => ({ label: row.label, value: row.value })),
    );
  }

  const actionList = document.getElementById("qualityActions");
  if (actionList && audit.actions?.length) {
    actionList.innerHTML = audit.actions
      .map(
        (action, index) => `
          <article class="audit-action-card">
            <span>${String(index + 1).padStart(2, "0")}</span>
            <h3>${escapeHtml(action.titleZh)}</h3>
            <p>${escapeHtml(action.bodyZh)}</p>
          </article>
        `,
      )
      .join("");
  }

  const missingRows = document.getElementById("qualityMissingRows");
  if (missingRows && audit.topMissingExamples?.length) {
    missingRows.innerHTML = audit.topMissingExamples
      .map(
        (candidate) => `
          <tr>
            <td>
              <span class="rank">#${escapeHtml(candidate.rank)}</span>
              <small>${escapeHtml(candidate.directionLabelZh || "")}</small>
            </td>
            <td>
              <strong>${escapeHtml(candidate.drug)}</strong>
              <small>${escapeHtml(candidate.target)} · ${escapeHtml(candidate.protein)}</small>
            </td>
            <td>
              <span>${formatScore(candidate.directionScore, 3)}</span>
              <small>Affinity ${formatScore(candidate.affinityScore, 3)}</small>
            </td>
            <td>
              <span>${escapeHtml(candidate.diffdockError || "missing_output")}</span>
              <small>${escapeHtml(candidate.receptorStatus || "NA")}</small>
            </td>
          </tr>
        `,
      )
      .join("");
  }
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

renderExecutiveSummary();
setMetricValues();
renderComputeStatus();
renderStatusReport();
renderDiseaseDirections();
setupDirectionControls();
setupCandidateControls();
renderCandidates();
renderQualityAudit();
renderCharts();
renderSotaOverview();
renderValidationPlan();
