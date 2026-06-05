from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


KINASE_MARKERS = ("kinase", "jak", "egfr", "kit", "src", "abl", "mapk", "raf", "trk", "flt", "fgfr", "pdgfr")
RECEPTOR_MARKERS = ("receptor", "membrane receptor")
TRANSPORTER_MARKERS = ("transporter", "solute carrier", "abc")
ION_CHANNEL_MARKERS = ("ion channel", "channel")


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


def num(value: Any, default: float = 0.0) -> float:
    parsed = number(value)
    return default if parsed is None else float(parsed)


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def pct(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) * 100 if denominator else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tier_score(value: Any, mapping: dict[str, float], default: float = 50.0) -> float:
    text = str(value or "")
    for prefix, score in mapping.items():
        if text.startswith(prefix) or text == prefix:
            return score
    return default


def novelty_group(row: pd.Series) -> str:
    action = str(row.get("sotaReadyAction", ""))
    novelty = str(row.get("auditNoveltyClass") or row.get("noveltyClass") or "")
    if truthy(row.get("knownDrugTargetPair")) or "known_benchmark" in novelty or "known_kg_drug_target" in novelty:
        return "positive_control_known_mechanism"
    if action == "mechanism_extension_repurposing" or "known_disease_use_new_target" in novelty:
        return "mechanism_extension_repurposing"
    if action == "novel_pair_expert_review" or "new_pair" in novelty or "model_priority" in novelty:
        return "novel_pair_or_new_target"
    if "safety" in novelty or action == "safety_or_contraindication_review":
        return "safety_or_contraindication_context"
    return "secondary_or_context_review"


def assay_modality(row: pd.Series) -> tuple[str, str]:
    target_class = str(row.get("catalogTarget_Class", "")).lower()
    target = str(row.get("target", "")).lower()
    protein_name = str(row.get("proteinName", "")).lower()
    text = " ".join([target_class, target, protein_name])
    if any(marker in text for marker in KINASE_MARKERS):
        return "biochemical_kinase_or_enzyme_assay", "Purified target activity assay plus orthogonal cell signaling readout."
    if any(marker in text for marker in ION_CHANNEL_MARKERS):
        return "electrophysiology_or_channel_function_assay", "Patch-clamp or channel-function assay followed by disease-relevant cellular phenotype."
    if any(marker in text for marker in TRANSPORTER_MARKERS):
        return "transporter_activity_assay", "Transporter uptake/efflux assay plus exposure and interaction review."
    if any(marker in text for marker in RECEPTOR_MARKERS):
        return "cell_based_receptor_function_assay", "Receptor pathway reporter or ligand-response assay in disease-relevant cells."
    if "transcription factor" in text:
        return "cellular_reporter_or_target_engagement_assay", "Reporter or target-engagement assay; direct biochemical assay may be difficult."
    return "target_engagement_and_cell_phenotype_assay", "Target engagement assay plus disease-relevant cell phenotype."


def validation_gate(row: pd.Series, novelty: str) -> tuple[str, str]:
    if novelty == "positive_control_known_mechanism":
        return "positive_control", "Use first to calibrate assay, score interpretation, and expected known-biology recovery."
    if novelty == "mechanism_extension_repurposing":
        return "mechanism_extension", "Review indication literature and test whether target engagement explains disease-direction effect."
    if novelty == "novel_pair_or_new_target":
        return "novel_candidate", "Prioritize orthogonal validation: target engagement, disease cell phenotype, and literature contradiction review."
    if novelty == "safety_or_contraindication_context":
        return "risk_review", "Do not advance before label, contraindication, directionality, and disease safety context review."
    return "secondary_review", "Retain as secondary candidate until evidence gap or context issue is resolved."


def component_scores(row: pd.Series) -> dict[str, float]:
    model = num(row.get("sotaReadyScore"), num(row.get("finalPriorityScore"), 0.0))
    evidence = num(row.get("evidenceReadinessScore"), 0.0)
    structure = max(
        tier_score(
            row.get("structureConfidenceTier"),
            {
                "A_": 96.0,
                "B_": 84.0,
                "C_": 55.0,
                "D_": 15.0,
                "not_applicable": 20.0,
            },
            45.0,
        ),
        tier_score(
            row.get("poseInterpretabilityTier"),
            {
                "A_": 100.0,
                "B_": 86.0,
                "C_": 48.0,
                "D_": 10.0,
            },
            45.0,
        )
        * 0.96,
    )
    target = tier_score(
        row.get("targetDruggabilityTier"),
        {
            "A_": 100.0,
            "B_": 84.0,
            "C_": 58.0,
            "D_": 25.0,
        },
        50.0,
    )
    chemotype = num(row.get("chemotypeReadinessScore"), 70.0)
    direction = num(row.get("directionReadinessScore"), 70.0)
    admet = tier_score(row.get("admetTier"), {"A": 100.0, "B": 82.0, "C": 48.0, "D": 18.0}, 45.0)
    novelty = novelty_score(row)
    return {
        "modelScoreComponent": max(0.0, min(100.0, model)),
        "evidenceScoreComponent": max(0.0, min(100.0, evidence)),
        "structureScoreComponent": max(0.0, min(100.0, structure)),
        "targetScoreComponent": max(0.0, min(100.0, target)),
        "chemotypeScoreComponent": max(0.0, min(100.0, chemotype)),
        "directionScoreComponent": max(0.0, min(100.0, direction)),
        "admetScoreComponent": max(0.0, min(100.0, admet)),
        "noveltyScoreComponent": max(0.0, min(100.0, novelty)),
    }


def novelty_score(row: pd.Series) -> float:
    group = novelty_group(row)
    if group == "novel_pair_or_new_target":
        return 96.0
    if group == "mechanism_extension_repurposing":
        return 88.0
    if group == "positive_control_known_mechanism":
        return 74.0
    if group == "safety_or_contraindication_context":
        return 20.0
    return 55.0


def risk_penalty(row: pd.Series) -> float:
    penalty = 0.0
    if str(row.get("hardFlags", "")).strip().lower() not in {"", "none", "nan"}:
        penalty += 18.0
    if not truthy(row.get("riskCleanFlag")):
        penalty += 6.0
    if str(row.get("admetTier", "")) == "D":
        penalty += 14.0
    elif str(row.get("admetTier", "")) == "C":
        penalty += 6.0
    if truthy(row.get("contraindicationFlag")) or truthy(row.get("hasContraindicationDiseaseEdge")):
        penalty += 14.0
    if str(row.get("poseInterpretabilityTier", "")).startswith("D_"):
        penalty += 12.0
    elif str(row.get("poseInterpretabilityTier", "")).startswith("C_"):
        penalty += 5.0
    if str(row.get("targetDruggabilityTier", "")).startswith("C_"):
        penalty += 4.0
    if truthy(row.get("singleEvidenceDominatedFlag")):
        penalty += 8.0
    if str(row.get("sotaReadyAction", "")) == "safety_or_contraindication_review":
        penalty += 10.0
    return penalty


def validation_score(row: pd.Series) -> tuple[float, dict[str, float], float]:
    comps = component_scores(row)
    penalty = risk_penalty(row)
    score = (
        0.16 * comps["modelScoreComponent"]
        + 0.18 * comps["evidenceScoreComponent"]
        + 0.17 * comps["structureScoreComponent"]
        + 0.16 * comps["targetScoreComponent"]
        + 0.09 * comps["chemotypeScoreComponent"]
        + 0.10 * comps["directionScoreComponent"]
        + 0.08 * comps["admetScoreComponent"]
        + 0.06 * comps["noveltyScoreComponent"]
        - penalty
    )
    return round(max(0.0, min(100.0, score)), 4), comps, round(penalty, 4)


def validation_tier(row: pd.Series, score: float) -> str:
    action = str(row.get("sotaReadyAction", ""))
    if action == "safety_or_contraindication_review" or truthy(row.get("contraindicationFlag")):
        return "D_risk_hold"
    if str(row.get("poseInterpretabilityTier", "")).startswith("D_"):
        return "D_structure_hold"
    if score >= 84 and str(row.get("poseInterpretabilityTier", "")).startswith(("A_", "B_")):
        return "A_experiment_ready"
    if score >= 72:
        return "B_expert_review_ready"
    if score >= 58:
        return "C_secondary_or_context_review"
    return "D_low_priority"


def select_balanced(
    rows: list[dict[str, Any]],
    limit: int,
    per_direction_min: int,
    max_per_drug: int,
    max_per_target: int,
    max_per_scaffold: int,
    allowed_tiers: tuple[str, ...],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_keys: set[tuple[str, str]] = set()
    drug_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    scaffold_counts: Counter[str] = Counter()

    def can_add(row: dict[str, Any], drug_cap: int, target_cap: int, scaffold_cap: int) -> bool:
        key = (str(row.get("direction")), str(row.get("pairId")))
        if key in used_keys:
            return False
        if not str(row.get("validationTier", "")).startswith(allowed_tiers):
            return False
        if drug_counts[str(row.get("drugId") or row.get("drug"))] >= drug_cap:
            return False
        if target_counts[str(row.get("protein") or row.get("target"))] >= target_cap:
            return False
        scaffold = str(row.get("murckoScaffold") or row.get("chemotypeClusterId") or "")
        if scaffold and scaffold_counts[scaffold] >= scaffold_cap:
            return False
        return True

    def add(row: dict[str, Any], selection_pass: str) -> None:
        copied = dict(row)
        copied["balancedSelectionPass"] = selection_pass
        selected.append(copied)
        used_keys.add((str(row.get("direction")), str(row.get("pairId"))))
        drug_counts[str(row.get("drugId") or row.get("drug"))] += 1
        target_counts[str(row.get("protein") or row.get("target"))] += 1
        scaffold = str(row.get("murckoScaffold") or row.get("chemotypeClusterId") or "")
        if scaffold:
            scaffold_counts[scaffold] += 1

    by_direction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_direction[str(row.get("direction"))].append(row)

    for direction in sorted(by_direction):
        direction_added = 0
        for row in by_direction[direction]:
            if direction_added >= per_direction_min or len(selected) >= limit:
                break
            if can_add(row, max_per_drug, max_per_target, max_per_scaffold):
                add(row, "strict_direction_minimum")
                direction_added += 1

    passes = [
        ("strict_diversity", max_per_drug, max_per_target, max_per_scaffold),
        ("relaxed_drug_target_scaffold_caps", max(max_per_drug * 2, max_per_drug + 3), max(max_per_target * 2, max_per_target + 3), max(max_per_scaffold * 2, max_per_scaffold + 4)),
        ("score_priority_fill", 999999, 999999, 999999),
    ]
    for selection_pass, drug_cap, target_cap, scaffold_cap in passes:
        for row in rows:
            if len(selected) >= limit:
                break
            if can_add(row, drug_cap, target_cap, scaffold_cap):
                add(row, selection_pass)
        if len(selected) >= limit:
            break
    return selected


def row_to_panel(row: pd.Series) -> dict[str, Any]:
    score, comps, penalty = validation_score(row)
    novelty = novelty_group(row)
    gate, gate_note = validation_gate(row, novelty)
    assay, assay_note = assay_modality(row)
    panel_row = {
        "validationRankGlobal": "",
        "validationRankWithinDirection": "",
        "validationScore": score,
        "validationTier": "",
        "validationGate": gate,
        "validationGateNote": gate_note,
        "assayModality": assay,
        "assayRationale": assay_note,
        "noveltyGroup": novelty,
        "validationRiskPenalty": penalty,
    }
    panel_row.update(comps)
    keep_cols = [
        "sotaReadyRankGlobal",
        "sotaReadyRankWithinDirection",
        "finalRankGlobal",
        "direction",
        "directionLabelZhFinal",
        "pairId",
        "drugId",
        "drug",
        "target",
        "protein",
        "proteinName",
        "sotaReadyScore",
        "sotaReadyTier",
        "sotaReadyAction",
        "auditNoveltyClass",
        "noveltyClass",
        "knownDrugTargetPair",
        "strictNovelPairFlag",
        "mechanismExtensionFlag",
        "evidenceConcordanceTier",
        "evidenceSupportCount",
        "strongEvidenceCount",
        "riskCleanFlag",
        "structureConfidenceTier",
        "poseInterpretabilityTier",
        "computedPocketResidues5A",
        "computedPocketMeanPlddt5A",
        "computedMinHeavyAtomDistance",
        "contactClassCounts",
        "topContactResidues",
        "targetDruggabilityTier",
        "catalogMax_Clinical_Phase",
        "catalogTarget_Class",
        "catalogDruggable_Modalities",
        "murckoScaffold",
        "chemotypeClusterId",
        "admetTier",
        "poseAuditStatus",
        "poseAuditReason",
        "openTargetsScore",
        "integratedTxgnnScore",
        "diffdock",
        "hardFlags",
        "softFlags",
        "kgExplanationZh",
        "evidenceSummaryZh",
        "validationGatesZh",
        "discussionSummary",
        "therapeuticArea",
        "indication",
        "targetDiseaseExamples",
        "qed",
        "logP",
        "tpsa",
        "molecularWeight",
    ]
    for col in keep_cols:
        if col in row.index:
            panel_row[col] = row.get(col)
    panel_row["validationSummary"] = validation_summary(panel_row)
    return panel_row


def validation_summary(row: dict[str, Any]) -> str:
    return (
        f"{row.get('drug')} - {row.get('target')} ({row.get('direction')}) is {row.get('validationTier')} "
        f"with {row.get('noveltyGroup')}; assay: {row.get('assayModality')}; "
        f"structure: {row.get('poseInterpretabilityTier')}; target: {row.get('targetDruggabilityTier')}; "
        f"risk penalty {row.get('validationRiskPenalty')}."
    )


def summarize(panel_rows: list[dict[str, Any]], balanced: list[dict[str, Any]], novel: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    tier_counts = Counter(str(row.get("validationTier")) for row in panel_rows)
    gate_counts = Counter(str(row.get("validationGate")) for row in panel_rows)
    assay_counts = Counter(str(row.get("assayModality")) for row in panel_rows)
    novelty_counts = Counter(str(row.get("noveltyGroup")) for row in panel_rows)
    direction_rows: list[dict[str, Any]] = []
    assay_rows: list[dict[str, Any]] = []
    by_direction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in panel_rows:
        by_direction[str(row.get("direction"))].append(row)
    for direction, group in sorted(by_direction.items()):
        direction_rows.append(
            {
                "direction": direction,
                "rows": len(group),
                "tierA": sum(1 for row in group if str(row.get("validationTier")).startswith("A_")),
                "tierB": sum(1 for row in group if str(row.get("validationTier")).startswith("B_")),
                "novelRows": sum(1 for row in group if row.get("noveltyGroup") == "novel_pair_or_new_target"),
                "experimentReadyNovelRows": sum(
                    1
                    for row in group
                    if str(row.get("validationTier")).startswith(("A_", "B_"))
                    and row.get("noveltyGroup") == "novel_pair_or_new_target"
                ),
                "balancedPanelRows": sum(1 for row in balanced if row.get("direction") == direction),
                "medianValidationScore": round(
                    float(pd.Series([num(row.get("validationScore")) for row in group]).median()), 4
                ),
            }
        )
    for assay, count in sorted(assay_counts.items()):
        assay_rows.append(
            {
                "assayModality": assay,
                "rows": count,
                "experimentReadyRows": sum(
                    1 for row in panel_rows if row.get("assayModality") == assay and str(row.get("validationTier")).startswith(("A_", "B_"))
                ),
                "novelRows": sum(1 for row in panel_rows if row.get("assayModality") == assay and row.get("noveltyGroup") == "novel_pair_or_new_target"),
            }
        )
    top100 = [row for row in panel_rows if int(num(row.get("validationRankGlobal"), 999999)) <= 100]
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidateRows": len(panel_rows),
        "validationTierCounts": dict(tier_counts),
        "validationGateCounts": dict(gate_counts),
        "assayModalityCounts": dict(assay_counts),
        "noveltyGroupCounts": dict(novelty_counts),
        "experimentReadyRows": tier_counts.get("A_experiment_ready", 0),
        "reviewReadyRows": tier_counts.get("B_expert_review_ready", 0),
        "experimentOrReviewReadyRows": tier_counts.get("A_experiment_ready", 0) + tier_counts.get("B_expert_review_ready", 0),
        "novelExperimentOrReviewReadyRows": sum(
            1
            for row in panel_rows
            if str(row.get("validationTier")).startswith(("A_", "B_"))
            and row.get("noveltyGroup") == "novel_pair_or_new_target"
        ),
        "positiveControlReadyRows": sum(
            1
            for row in panel_rows
            if str(row.get("validationTier")).startswith(("A_", "B_"))
            and row.get("noveltyGroup") == "positive_control_known_mechanism"
        ),
        "balancedPanelRows": len(balanced),
        "balancedPanelDirections": len({row.get("direction") for row in balanced}),
        "balancedPanelUniqueDrugs": len({row.get("drugId") or row.get("drug") for row in balanced}),
        "balancedPanelUniqueTargets": len({row.get("protein") or row.get("target") for row in balanced}),
        "balancedPanelUniqueScaffolds": len({row.get("murckoScaffold") for row in balanced if row.get("murckoScaffold")}),
        "novelValidationPanelRows": len(novel),
        "top100ExperimentOrReviewReadyRows": sum(1 for row in top100 if str(row.get("validationTier")).startswith(("A_", "B_"))),
        "top100NovelRows": sum(1 for row in top100 if row.get("noveltyGroup") == "novel_pair_or_new_target"),
        "methodNote": (
            "Validation score is a transparent triage score for assay planning. It combines model priority, "
            "evidence concordance, residue-contact interpretability, target druggability, chemotype readiness, "
            "direction specificity, ADMET tier, novelty, and explicit risk penalties. It is not a biological proof."
        ),
    }
    return summary, direction_rows, assay_rows


def build_panel(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    source = root / args.source
    df = pd.read_csv(source).fillna("")
    panel_rows = [row_to_panel(row) for _, row in df.iterrows()]
    for row in panel_rows:
        row["validationTier"] = validation_tier(pd.Series(row), float(row["validationScore"]))
        row["validationSummary"] = validation_summary(row)
    panel_rows = sorted(
        panel_rows,
        key=lambda row: (
            -num(row.get("validationScore")),
            num(row.get("sotaReadyRankGlobal"), 999999),
            str(row.get("direction")),
            str(row.get("pairId")),
        ),
    )
    direction_counts: Counter[str] = Counter()
    for idx, row in enumerate(panel_rows, 1):
        direction = str(row.get("direction"))
        direction_counts[direction] += 1
        row["validationRankGlobal"] = idx
        row["validationRankWithinDirection"] = direction_counts[direction]

    balanced = select_balanced(
        panel_rows,
        args.balanced_limit,
        args.per_direction_min,
        args.max_per_drug,
        args.max_per_target,
        args.max_per_scaffold,
        ("A_", "B_"),
    )
    novel = [
        row
        for row in panel_rows
        if str(row.get("validationTier")).startswith(("A_", "B_"))
        and row.get("noveltyGroup") == "novel_pair_or_new_target"
    ][: args.novel_limit]
    positive = [
        row
        for row in panel_rows
        if str(row.get("validationTier")).startswith(("A_", "B_"))
        and row.get("noveltyGroup") == "positive_control_known_mechanism"
    ][: args.positive_control_limit]
    risk = [
        row
        for row in panel_rows
        if str(row.get("validationTier")).startswith("D_")
        or row.get("validationGate") == "risk_review"
    ][: args.review_limit]

    summary, direction_rows, assay_rows = summarize(panel_rows, balanced, novel)
    out_dir = root / args.out_dir
    final_dir = root / "outputs/sota_validation/final_prioritization"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "experimental_validation_panel.csv", panel_rows)
    write_csv(out_dir / "experimental_validation_direction_summary.csv", direction_rows)
    write_csv(out_dir / "experimental_validation_assay_summary.csv", assay_rows)
    write_csv(final_dir / "final_priority_experimental_validation_panel.csv", panel_rows)
    write_csv(final_dir / "final_priority_experimental_validation_balanced_shortlist.csv", balanced)
    write_csv(final_dir / "final_priority_experimental_validation_novel_shortlist.csv", novel)
    write_csv(final_dir / "final_priority_experimental_validation_positive_controls.csv", positive)
    write_csv(final_dir / "final_priority_experimental_validation_risk_review.csv", risk)
    write_json(out_dir / "experimental_validation_panel_summary.json", summary)
    write_json(final_dir / "final_priority_experimental_validation_panel_summary.json", summary)
    (final_dir / "FINAL_PRIORITY_EXPERIMENTAL_VALIDATION_PANEL.md").write_text(
        markdown(summary, direction_rows, assay_rows, balanced, novel, positive),
        encoding="utf-8",
    )
    return {
        "summary": summary,
        "panel_path": "outputs/sota_validation/experimental_validation/experimental_validation_panel.csv",
        "balanced_path": "outputs/sota_validation/final_prioritization/final_priority_experimental_validation_balanced_shortlist.csv",
        "novel_path": "outputs/sota_validation/final_prioritization/final_priority_experimental_validation_novel_shortlist.csv",
    }


def markdown(
    summary: dict[str, Any],
    direction_rows: list[dict[str, Any]],
    assay_rows: list[dict[str, Any]],
    balanced: list[dict[str, Any]],
    novel: list[dict[str, Any]],
    positive: list[dict[str, Any]],
) -> str:
    lines = [
        "# Final-Priority Experimental Validation Panel",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "## Purpose",
        "",
        "This panel converts the final multi-evidence ranking into assay-planning queues for expert validation.",
        "",
        "## Headline Metrics",
        "",
        f"- Candidate rows: {summary['candidateRows']}",
        f"- Experiment-ready rows: {summary['experimentReadyRows']}",
        f"- Experiment-or-review-ready rows: {summary['experimentOrReviewReadyRows']}",
        f"- Novel experiment-or-review-ready rows: {summary['novelExperimentOrReviewReadyRows']}",
        f"- Positive-control ready rows: {summary['positiveControlReadyRows']}",
        f"- Balanced shortlist rows: {summary['balancedPanelRows']} across {summary['balancedPanelDirections']} directions, "
        f"{summary['balancedPanelUniqueDrugs']} drugs, {summary['balancedPanelUniqueTargets']} targets, "
        f"{summary['balancedPanelUniqueScaffolds']} scaffolds",
        f"- Novel validation shortlist rows: {summary['novelValidationPanelRows']}",
        f"- Validation tier counts: {summary['validationTierCounts']}",
        f"- Assay modality counts: {summary['assayModalityCounts']}",
        "",
        "## Direction Summary",
        "",
    ]
    for row in direction_rows:
        lines.append(
            f"- {row['direction']}: rows {row['rows']}; A/B {row['tierA'] + row['tierB']}; "
            f"novel A/B {row['experimentReadyNovelRows']}; balanced panel rows {row['balancedPanelRows']}."
        )
    lines.extend(["", "## Assay Summary", ""])
    for row in assay_rows:
        lines.append(
            f"- {row['assayModality']}: rows {row['rows']}; experiment/review-ready {row['experimentReadyRows']}; novel {row['novelRows']}."
        )
    lines.extend(["", "## Representative Balanced Shortlist", ""])
    for row in balanced[:15]:
        lines.append(
            f"- Rank {row.get('validationRankGlobal')}: {row.get('drug')} - {row.get('target')} "
            f"({row.get('direction')}), {row.get('validationTier')}, assay {row.get('assayModality')}."
        )
    lines.extend(["", "## Representative Novel Shortlist", ""])
    for row in novel[:12]:
        lines.append(
            f"- Rank {row.get('validationRankGlobal')}: {row.get('drug')} - {row.get('target')} "
            f"({row.get('direction')}), score {row.get('validationScore')}, {row.get('poseInterpretabilityTier')}."
        )
    lines.extend(["", "## Positive Controls", ""])
    for row in positive[:10]:
        lines.append(
            f"- Rank {row.get('validationRankGlobal')}: {row.get('drug')} - {row.get('target')} "
            f"({row.get('direction')}), {row.get('assayModality')}."
        )
    lines.extend(["", "## Method Note", "", summary["methodNote"], ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build final experimental validation planning panel.")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--source",
        default="outputs/sota_validation/final_prioritization/final_priority_pose_interpretability_augmented_table.csv",
    )
    parser.add_argument("--out-dir", default="outputs/sota_validation/experimental_validation")
    parser.add_argument("--balanced-limit", type=int, default=300)
    parser.add_argument("--per-direction-min", type=int, default=30)
    parser.add_argument("--max-per-drug", type=int, default=6)
    parser.add_argument("--max-per-target", type=int, default=5)
    parser.add_argument("--max-per-scaffold", type=int, default=8)
    parser.add_argument("--novel-limit", type=int, default=300)
    parser.add_argument("--positive-control-limit", type=int, default=120)
    parser.add_argument("--review-limit", type=int, default=500)
    args = parser.parse_args()

    result = build_panel(Path(args.root).resolve(), args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
