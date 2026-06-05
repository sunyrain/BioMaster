#!/usr/bin/env python3
"""Build the final practical gate before wet-lab purchase and assay setup."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WETLAB_DIR = Path("outputs/sota_validation/wetlab_validation_package")
INPUT_NAME = "wetlab_pre_purchase_focus_package.csv"


MANUAL_GATES = [
    "compound_form_purity_vendor",
    "solubility_storage_vehicle",
    "clinical_free_exposure",
    "target_expression_in_model",
    "direction_of_modulation",
    "exact_novelty_pubmed_clinicaltrials",
    "primary_orthogonal_counterscreen_ready",
    "assay_interference_risk",
]


TARGET_NOTES: dict[str, dict[str, str]] = {
    "EGFR": {
        "direction": "inhibition should suppress EGFR pathway signaling; validate pEGFR/pERK or pAKT.",
        "manual": "Use EGFR-positive and EGFR-low cells; confirm kinase inhibition rather than broad cytotoxicity.",
    },
    "KIT": {
        "direction": "inhibition should suppress KIT signaling in a KIT-expressing model.",
        "manual": "Check direct KIT engagement and kinase-family selectivity; avoid interpreting EGFR carryover as KIT biology.",
    },
    "ADORA1": {
        "direction": "clarify whether the drug behaves as agonist, antagonist, or indirect modulator in the ADORA1 assay.",
        "manual": "Separate receptor signaling from nucleoside-metabolism or cytotoxic effects.",
    },
    "F2R": {
        "direction": "clarify PAR1 agonist/antagonist behavior using thrombin or PAR1 activating peptide stimulation.",
        "manual": "Counterscreen 5-HT3 biology for palonosetron-like compounds and require receptor-specific rescue/competition.",
    },
    "CHRM3": {
        "direction": "antagonist activity should reduce acetylcholine/carbachol-induced CHRM3 signaling.",
        "manual": "Use muscarinic subtype counterscreens because subtype spillover is common.",
    },
    "GLP1R": {
        "direction": "clarify GLP1R agonist/antagonist mode through cAMP or beta-arrestin response.",
        "manual": "For ramipril-like hypotheses, separate receptor signal from ACE-pathway and metabolite effects.",
    },
    "ADRA1A": {
        "direction": "antagonist activity should reduce phenylephrine/norepinephrine-induced alpha-1A signaling.",
        "manual": "Counterscreen dopaminergic pharmacology for amisulpride-like compounds.",
    },
    "EDNRA": {
        "direction": "antagonist activity should reduce endothelin-1-induced EDNRA signaling.",
        "manual": "Separate EDNRA engagement from cholinesterase-related pharmacology for donepezil-like compounds.",
    },
    "OXTR": {
        "direction": "clarify OXTR agonist/antagonist behavior through calcium/IP1/beta-arrestin response.",
        "manual": "Check permeability and opioid receptor counterscreens for methylnaltrexone-like compounds.",
    },
    "CRHR1": {
        "direction": "antagonist activity should reduce CRH/urocortin-induced CRHR1 cAMP signaling.",
        "manual": "Counterscreen CRHR2 and canonical opioid pharmacology where relevant.",
    },
}


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def number(value: Any) -> float | None:
    if value in (None, "", "NA"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def compact(value: Any, limit: int = 180) -> str:
    content = " ".join(text(value).split())
    if len(content) <= limit:
        return content
    return content[: limit - 1].rstrip() + "..."


def truthy(value: Any) -> bool:
    return text(value).lower() in {"1", "1.0", "true", "yes", "y"}


def tier_score(value: Any, default: float = 50.0) -> float:
    content = text(value)
    if content.startswith(("A", "A_")):
        return 100.0
    if content.startswith(("B", "B_")):
        return 82.0
    if content.startswith(("C", "C_")):
        return 45.0
    if content.startswith(("D", "D_")):
        return 10.0
    return default


def score_0_100(value: Any) -> float:
    parsed = number(value)
    if parsed is None:
        return 0.0
    if -1.5 <= parsed <= 1.5:
        parsed *= 100.0
    return max(0.0, min(100.0, parsed))


def flag_lookup(evidence_flags: str, key: str) -> bool:
    for item in evidence_flags.split(";"):
        if ":" not in item:
            continue
        name, value = item.strip().split(":", 1)
        if name.strip() == key:
            return value.strip().upper().startswith("Y")
    return False


def automatic_gate_flags(row: pd.Series) -> dict[str, bool]:
    evidence_flags = text(row.get("evidenceFlags"))
    pass_count = number(row.get("passCount")) or 0
    hard_holds = text(row.get("hardHolds"))
    soft_holds = text(row.get("softHolds"))
    direction = text(row.get("direction"))
    depmap = text(row.get("depmapDependencyTier"))
    return {
        "multi_evidence": pass_count >= 9 and not hard_holds,
        "model_affinity": flag_lookup(evidence_flags, "model") or score_0_100(row.get("affinityScore")) >= 50,
        "disease_context": flag_lookup(evidence_flags, "disease") and not ("disease evidence is weak" in soft_holds),
        "pathway_context": flag_lookup(evidence_flags, "pathway"),
        "signature_reversal": flag_lookup(evidence_flags, "signature") and not ("CMap/LINCS reversal not supportive" in soft_holds),
        "tissue_or_model_context": flag_lookup(evidence_flags, "tissue")
        and (direction != "oncology" or depmap.startswith(("A_", "B_"))),
        "structure_interpretable": flag_lookup(evidence_flags, "structure"),
        "admet_acceptable": flag_lookup(evidence_flags, "admet"),
        "assay_plan_ready": flag_lookup(evidence_flags, "assay"),
        "diffdock_available": flag_lookup(evidence_flags, "diffdock"),
    }


def manual_gate_status(row: pd.Series) -> dict[str, str]:
    role = text(row.get("validationRole"))
    target = text(row.get("target"))
    action = text(row.get("purchaseAction"))
    notes = TARGET_NOTES.get(target, {})
    status = {
        "compound_form_purity_vendor": "manual_required",
        "solubility_storage_vehicle": "manual_required",
        "clinical_free_exposure": "manual_required_high_priority",
        "target_expression_in_model": "manual_required_high_priority",
        "direction_of_modulation": "manual_required_high_priority",
        "exact_novelty_pubmed_clinicaltrials": "manual_required",
        "primary_orthogonal_counterscreen_ready": "manual_required",
        "assay_interference_risk": "manual_required",
    }
    if role == "positive_control":
        status["exact_novelty_pubmed_clinicaltrials"] = "known_control_expected"
    if action == "hold_before_purchase":
        for key in status:
            status[key] = "hold_until_auto_or_expert_issue_resolved"
    if not notes:
        status["direction_of_modulation"] = "manual_required_target_specific"
    return status


def gate_score(row: pd.Series, flags: dict[str, bool]) -> float:
    auto_pass = sum(1 for passed in flags.values() if passed)
    disease = max(
        score_0_100(row.get("openTargetsScore")),
        score_0_100(row.get("txgnnScore")),
        score_0_100(row.get("kgEvidenceScore")),
    )
    pre_purchase = number(row.get("prePurchaseFocusScore")) or 0.0
    structure = 0.5 * tier_score(row.get("structureConfidenceTier")) + 0.5 * tier_score(
        row.get("standardPoseValidationTier"), 70.0
    )
    assay = 100.0 if flags["assay_plan_ready"] else 35.0
    admet = tier_score(row.get("admetTier"))
    novelty = {
        "novel_repurposing_candidate": 100.0,
        "mechanism_extension": 86.0,
        "positive_control": 64.0,
    }.get(text(row.get("validationRole")), 58.0)
    score = (
        0.24 * pre_purchase
        + 0.18 * auto_pass * 10.0
        + 0.14 * disease
        + 0.12 * structure
        + 0.10 * admet
        + 0.08 * assay
        + 0.08 * tier_score(row.get("cmapReversalTier"), 40.0)
        + 0.06 * novelty
    )
    if text(row.get("hardHolds")):
        score -= 22.0
    if text(row.get("purchaseAction")) == "expert_review_before_purchase":
        score -= 6.0
    if text(row.get("purchaseAction")) == "hold_before_purchase":
        score -= 20.0
    if text(row.get("validationRole")) == "positive_control":
        score -= 4.0
    return round(max(0.0, min(100.0, score)), 4)


def gate_tier(row: pd.Series, flags: dict[str, bool], score: float) -> str:
    auto_pass = sum(1 for passed in flags.values() if passed)
    action = text(row.get("purchaseAction"))
    source_tier = text(row.get("prePurchaseTier"))
    if source_tier == "batch_1_core_6" and score >= 86 and auto_pass >= 9:
        return "go_core_6_after_manual_confirmation"
    if source_tier == "batch_1_extension_12" and score >= 78 and auto_pass >= 8 and action != "hold_before_purchase":
        return "go_extension_12_after_manual_confirmation"
    if source_tier == "backup_24" and score >= 74 and auto_pass >= 8 and action != "hold_before_purchase":
        return "backup_replacement_after_manual_confirmation"
    if action == "expert_review_before_purchase":
        return "expert_review_before_purchase"
    return "hold_before_purchase"


def automatic_summary(flags: dict[str, bool]) -> str:
    failed = [key for key, passed in flags.items() if not passed]
    if not failed:
        return "all automatic evidence gates passed"
    return "failed automatic gates: " + ", ".join(failed)


def manual_summary(row: pd.Series, manual: dict[str, str]) -> str:
    target = text(row.get("target"))
    notes = TARGET_NOTES.get(target, {})
    high = [key for key, value in manual.items() if "high_priority" in value]
    base = "high-priority manual checks: " + ", ".join(high)
    if notes:
        base += f"; modulation note: {notes['direction']}; assay risk: {notes['manual']}"
    return base


def recommendation(row: pd.Series, tier: str) -> str:
    role = text(row.get("validationRole"))
    if tier.startswith("go_core_6"):
        return "Run first if manual checks pass; use this set to calibrate assay behavior and test discovery value."
    if tier.startswith("go_extension_12"):
        return "Run after core 6 setup is confirmed, or include immediately if platform capacity is available."
    if tier.startswith("backup"):
        return "Keep as replacement if a core/extension candidate fails vendor, exposure, solubility, or model checks."
    if tier == "expert_review_before_purchase":
        return "Discuss with disease and assay experts before purchase; current evidence is not enough for first-batch spending."
    if role == "positive_control":
        return "Hold only if an equivalent platform control is already available."
    return "Do not purchase before resolving automatic or manual gate issues."


def build_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        flags = automatic_gate_flags(row)
        manual = manual_gate_status(row)
        score = gate_score(row, flags)
        tier = gate_tier(row, flags, score)
        target = text(row.get("target"))
        notes = TARGET_NOTES.get(target, {})
        payload = row.to_dict()
        payload.update(
            {
                "finalGateScore": score,
                "finalGateTier": tier,
                "automaticGatePassCount": sum(1 for passed in flags.values() if passed),
                "automaticGateStatus": "; ".join(f"{key}:{'Y' if value else 'N'}" for key, value in flags.items()),
                "automaticGateSummaryZh": automatic_summary(flags),
                "manualGateStatus": "; ".join(f"{key}:{value}" for key, value in manual.items()),
                "manualGateSummaryZh": manual_summary(row, manual),
                "mechanismDirectionCheck": notes.get(
                    "direction",
                    "clarify target-specific direction of modulation before disease phenotype interpretation.",
                ),
                "mainAssayRisk": notes.get(
                    "manual",
                    "confirm that the observed phenotype is explained by direct target engagement, not nonspecific activity.",
                ),
                "finalGateRecommendationZh": recommendation(row, tier),
            }
        )
        rows.append(payload)

    order = {
        "go_core_6_after_manual_confirmation": 0,
        "go_extension_12_after_manual_confirmation": 1,
        "backup_replacement_after_manual_confirmation": 2,
        "expert_review_before_purchase": 3,
        "hold_before_purchase": 4,
    }
    rows.sort(
        key=lambda item: (
            order.get(text(item.get("finalGateTier")), 9),
            -float(number(item.get("finalGateScore")) or 0.0),
            text(item.get("drug")),
            text(item.get("target")),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["finalGateRank"] = rank
    return rows


def table_md(rows: list[dict[str, Any]], tier: str, limit: int) -> str:
    block = [row for row in rows if text(row.get("finalGateTier")) == tier][:limit]
    headers = ["Rank", "Direction", "Drug", "Target", "Role", "Score", "Auto", "Manual focus"]
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in block:
        values = [
            row.get("finalGateRank"),
            row.get("directionLabelZh") or row.get("direction"),
            row.get("drug"),
            f"{row.get('target')} ({row.get('protein')})",
            row.get("validationRoleZh"),
            f"{number(row.get('finalGateScore')) or 0:.1f}",
            f"{row.get('automaticGatePassCount')}/10",
            compact(row.get("mechanismDirectionCheck"), 120),
        ]
        lines.append("|" + "|".join(str(value).replace("|", "/") for value in values) + "|")
    if len(lines) == 2:
        lines.append("|none|none|none|none|none|none|none|none|")
    return "\n".join(lines)


def build_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    counts = Counter(text(row.get("finalGateTier")) for row in rows)
    action_counts = Counter(text(row.get("purchaseAction")) for row in rows)
    core = counts.get("go_core_6_after_manual_confirmation", 0)
    extension = counts.get("go_extension_12_after_manual_confirmation", 0)
    backup = counts.get("backup_replacement_after_manual_confirmation", 0)
    return f"""# Final Pre-Experiment Gate

Generated UTC: {summary['createdUtc']}

## Purpose

This package is the last computational focusing step before purchasing compounds or opening wet-lab assays. It does not claim biological validation. It identifies which candidates are worth manual confirmation first, and which ones should wait until vendor, exposure, mechanism-direction, assay-interference, and model-expression checks are resolved.

## Current Decision

- Input purchase/assay candidates: {summary['inputRows']}
- Core first-batch candidates after automatic gating: {core}
- Extension candidates after automatic gating: {extension}
- Backup replacement candidates: {backup}
- Expert review before purchase: {counts.get('expert_review_before_purchase', 0)}
- Hold before purchase: {counts.get('hold_before_purchase', 0)}

The practical recommendation is to start with the core 6 only after all manual checks pass. Expand to 12 only if compound handling, disease model, primary assay, orthogonal assay, and counterscreens are ready.

## Automatic Gates

The automatic gate uses evidence already present in the project outputs: affinity/model support, disease context, pathway context, CMap/LINCS reversal, tissue or DepMap context, interpretable structure, ADMET, assay plan, DiffDock availability, and multi-evidence consistency. These gates can prioritize candidates, but they cannot replace the manual checks below.

## Manual Checks Before Spending

- Exact compound form, purity, vendor, delivery time, storage, and vehicle compatibility.
- Clinically achievable free concentration versus planned assay concentration range.
- Target expression in the exact disease model or engineered assay model.
- Direction of modulation: inhibitor/agonist/antagonist must match the disease hypothesis.
- PubMed, ClinicalTrials, and label review for the exact drug-target-disease hypothesis.
- Primary assay, orthogonal target-engagement assay, counterscreen, and cytotoxicity readout must all be executable.
- Assay interference risk: PAINS-like behavior, aggregation, fluorescence/luminescence interference, nonspecific cytotoxicity, transporter/permeability artifacts, and canonical off-target pharmacology.

## Core 6

{table_md(rows, 'go_core_6_after_manual_confirmation', 6)}

## Extension 12

{table_md(rows, 'go_extension_12_after_manual_confirmation', 12)}

## Backup Replacements

{table_md(rows, 'backup_replacement_after_manual_confirmation', 18)}

## Interpretation

The highest-value experimental set is not the largest set. It is the smallest set that contains assay calibration controls, mechanism-extension hypotheses, and novel repurposing candidates while preserving disease relevance, structural interpretability, safety feasibility, and executable orthogonal validation. A disease-cell phenotype alone should not be treated as validation unless direct target engagement and counterscreens agree.

## Outputs

- `wetlab_final_pre_experiment_gate.csv`
- `wetlab_final_pre_experiment_gate_top12.csv`
- `wetlab_final_pre_experiment_gate_summary.json`
"""


def build_summary(rows: list[dict[str, Any]], created: str) -> dict[str, Any]:
    tier_counts = Counter(text(row.get("finalGateTier")) for row in rows)
    role_counts = Counter(text(row.get("validationRole")) for row in rows if text(row.get("finalGateTier")).startswith("go_"))
    direction_counts = Counter(text(row.get("direction")) for row in rows if text(row.get("finalGateTier")).startswith("go_"))
    action_counts = Counter(text(row.get("purchaseAction")) for row in rows)
    return {
        "createdUtc": created,
        "inputRows": len(rows),
        "finalGateTierCounts": dict(tier_counts),
        "goCandidateRows": sum(v for k, v in tier_counts.items() if k.startswith("go_")),
        "backupRows": tier_counts.get("backup_replacement_after_manual_confirmation", 0),
        "expertReviewRows": tier_counts.get("expert_review_before_purchase", 0),
        "holdRows": tier_counts.get("hold_before_purchase", 0),
        "goRoleCounts": dict(role_counts),
        "goDirectionCounts": dict(direction_counts),
        "purchaseActionCounts": dict(action_counts),
        "manualGateCount": len(MANUAL_GATES),
        "manualGates": MANUAL_GATES,
        "outputs": {
            "finalGateCsv": str(WETLAB_DIR / "wetlab_final_pre_experiment_gate.csv"),
            "top12Csv": str(WETLAB_DIR / "wetlab_final_pre_experiment_gate_top12.csv"),
            "summaryJson": str(WETLAB_DIR / "wetlab_final_pre_experiment_gate_summary.json"),
            "markdown": str(WETLAB_DIR / "WETLAB_FINAL_PRE_EXPERIMENT_GATE.md"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()

    root = args.root.resolve()
    in_path = root / WETLAB_DIR / INPUT_NAME
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_dir = root / WETLAB_DIR
    df = pd.read_csv(in_path)
    rows = build_rows(df)
    created = now_utc()
    summary = build_summary(rows, created)

    out_csv = out_dir / "wetlab_final_pre_experiment_gate.csv"
    top12_csv = out_dir / "wetlab_final_pre_experiment_gate_top12.csv"
    out_json = out_dir / "wetlab_final_pre_experiment_gate_summary.json"
    out_md = out_dir / "WETLAB_FINAL_PRE_EXPERIMENT_GATE.md"

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    top12 = [
        row
        for row in rows
        if text(row.get("finalGateTier"))
        in {"go_core_6_after_manual_confirmation", "go_extension_12_after_manual_confirmation"}
    ][:12]
    pd.DataFrame(top12).to_csv(top12_csv, index=False)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(build_markdown(summary, rows), encoding="utf-8")

    print(
        json.dumps(
            {
                "out_csv": str(out_csv.relative_to(root)),
                "out_top12": str(top12_csv.relative_to(root)),
                "out_json": str(out_json.relative_to(root)),
                "out_md": str(out_md.relative_to(root)),
                "tierCounts": summary["finalGateTierCounts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
