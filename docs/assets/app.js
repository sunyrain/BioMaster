const candidates = [
  {
    rank: 1,
    drug: "Afatinib Dimaleate",
    target: "EGFR",
    protein: "P00533",
    stage5: 0.926858,
    diffdock: "-0.46",
    stage6: 0.917979,
    status: "completed",
  },
  {
    rank: 2,
    drug: "Dacomitinib",
    target: "EGFR",
    protein: "P00533",
    stage5: 0.889966,
    diffdock: "-1.12",
    stage6: 0.874261,
    status: "completed",
  },
  {
    rank: 3,
    drug: "Cabozantinib S-Malate",
    target: "KIT",
    protein: "P10721",
    stage5: 0.864188,
    diffdock: "NA",
    stage6: 0.864188,
    status: "missing_output",
  },
  {
    rank: 4,
    drug: "Momelotinib Dihydrochloride",
    target: "EGFR",
    protein: "P00533",
    stage5: 0.861371,
    diffdock: "-0.75",
    stage6: 0.856885,
    status: "completed",
  },
  {
    rank: 5,
    drug: "Repotrectinib",
    target: "JAK1",
    protein: "P23458",
    stage5: 0.860247,
    diffdock: "-1.02",
    stage6: 0.850873,
    status: "completed",
  },
  {
    rank: 6,
    drug: "Pazopanib Hydrochloride",
    target: "KIT",
    protein: "P10721",
    stage5: 0.901546,
    diffdock: "-3.00",
    stage6: 0.848898,
    status: "completed",
  },
  {
    rank: 7,
    drug: "Osimertinib Mesylate",
    target: "EGFR",
    protein: "P00533",
    stage5: 0.86767,
    diffdock: "-1.98",
    stage6: 0.839205,
    status: "completed",
  },
  {
    rank: 8,
    drug: "Trilaciclib Dihydrochloride",
    target: "JAK1",
    protein: "P23458",
    stage5: 0.797277,
    diffdock: "0.60",
    stage6: 0.827686,
    status: "completed",
  },
  {
    rank: 9,
    drug: "Neratinib Maleate",
    target: "EGFR",
    protein: "P00533",
    stage5: 0.841782,
    diffdock: "-2.44",
    stage6: 0.808585,
    status: "completed",
  },
  {
    rank: 10,
    drug: "Tucatinib",
    target: "EGFR",
    protein: "P00533",
    stage5: 0.845578,
    diffdock: "-2.70",
    stage6: 0.806943,
    status: "completed",
  },
  {
    rank: 11,
    drug: "Lazertinib Mesylate",
    target: "EGFR",
    protein: "P00533",
    stage5: 0.860511,
    diffdock: "-3.72",
    stage6: 0.800536,
    status: "completed",
  },
  {
    rank: 12,
    drug: "Palbociclib",
    target: "JAK1",
    protein: "P23458",
    stage5: 0.801875,
    diffdock: "-1.16",
    stage6: 0.798635,
    status: "completed",
  },
  {
    rank: 13,
    drug: "Ribociclib Succinate",
    target: "EGFR",
    protein: "P00533",
    stage5: 0.779332,
    diffdock: "-0.20",
    stage6: 0.797451,
    status: "completed",
  },
  {
    rank: 14,
    drug: "Sorafenib Tosylate",
    target: "EGFR",
    protein: "P00533",
    stage5: 0.813223,
    diffdock: "-1.89",
    stage6: 0.79461,
    status: "completed",
  },
  {
    rank: 15,
    drug: "Bosutinib Monohydrate",
    target: "CDH11",
    protein: "P55287",
    stage5: 0.790755,
    diffdock: "NA",
    stage6: 0.790755,
    status: "missing_output",
  },
  {
    rank: 16,
    drug: "Pazopanib Hydrochloride",
    target: "EGFR",
    protein: "P00533",
    stage5: 0.834875,
    diffdock: "-3.21",
    stage6: 0.788296,
    status: "completed",
  },
  {
    rank: 17,
    drug: "Rucaparib Camsylate",
    target: "JAK1",
    protein: "P23458",
    stage5: 0.765392,
    diffdock: "-0.23",
    stage6: 0.78504,
    status: "completed",
  },
  {
    rank: 18,
    drug: "Cabozantinib S-Malate",
    target: "EGFR",
    protein: "P00533",
    stage5: 0.784071,
    diffdock: "NA",
    stage6: 0.784071,
    status: "missing_output",
  },
  {
    rank: 19,
    drug: "Zongertinib",
    target: "EGFR",
    protein: "P00533",
    stage5: 0.825636,
    diffdock: "-3.02",
    stage6: 0.784,
    status: "completed",
  },
  {
    rank: 20,
    drug: "Larotrectinib Sulfate",
    target: "JAK1",
    protein: "P23458",
    stage5: 0.778848,
    diffdock: "-1.25",
    stage6: 0.777377,
    status: "completed",
  },
];

let activeFilter = "all";

function formatScore(value) {
  return Number(value).toFixed(6);
}

function statusLabel(status) {
  return status === "completed" ? "completed" : "missing output";
}

function renderCandidates() {
  const tbody = document.getElementById("candidateRows");
  const search = document.getElementById("candidateSearch").value.trim().toLowerCase();
  const rows = candidates.filter((candidate) => {
    const matchesFilter = activeFilter === "all" || candidate.status === activeFilter;
    const haystack = `${candidate.drug} ${candidate.target} ${candidate.protein}`.toLowerCase();
    return matchesFilter && haystack.includes(search);
  });

  tbody.innerHTML = rows
    .map(
      (candidate) => `
        <tr>
          <td><span class="rank">#${candidate.rank}</span></td>
          <td>${candidate.drug}</td>
          <td><strong>${candidate.target}</strong><br><small>${candidate.protein}</small></td>
          <td>${formatScore(candidate.stage5)}</td>
          <td>${candidate.diffdock}</td>
          <td>${formatScore(candidate.stage6)}</td>
          <td><span class="status-pill ${candidate.status}">${statusLabel(candidate.status)}</span></td>
        </tr>
      `,
    )
    .join("");
}

document.querySelectorAll("[data-filter]").forEach((button) => {
  button.addEventListener("click", () => {
    activeFilter = button.dataset.filter;
    document.querySelectorAll("[data-filter]").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    renderCandidates();
  });
});

document.getElementById("candidateSearch").addEventListener("input", renderCandidates);

renderCandidates();
