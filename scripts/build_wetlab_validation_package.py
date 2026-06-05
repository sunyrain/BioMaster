#!/usr/bin/env python3
"""Build a wet-lab validation recommendation package from final priority tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRIORITY_DIR = Path("outputs/sota_validation/final_prioritization")
OUT_DIR = Path("outputs/sota_validation/wetlab_validation_package")


ASSAY_GUIDE: dict[str, dict[str, str]] = {
    "biochemical_kinase_or_enzyme_assay": {
        "primary": "Purified biochemical IC50/Kd assay for target activity or binding.",
        "orthogonal": "Cell target-engagement readout such as CETSA/NanoBRET or pathway phospho-marker.",
        "counter": "Kinase/enzyme selectivity panel and inactive structural analog or vehicle control.",
    },
    "cell_based_receptor_function_assay": {
        "primary": "Cell receptor-function assay matched to signaling mode, such as cAMP, calcium flux, beta-arrestin, or reporter response.",
        "orthogonal": "Ligand competition, receptor internalization, or target knockdown/antagonist rescue.",
        "counter": "Parental-cell, unrelated receptor, and cytotoxicity counterscreens.",
    },
    "electrophysiology_or_channel_function_assay": {
        "primary": "Automated or manual patch-clamp/channel-function assay.",
        "orthogonal": "Independent ion-flux or disease-cell excitability readout.",
        "counter": "Channel selectivity panel, hERG liability review, and vehicle control.",
    },
    "transporter_activity_assay": {
        "primary": "Substrate uptake or efflux assay in transporter-expressing cells.",
        "orthogonal": "Transporter inhibitor control and concentration-dependent intracellular exposure check.",
        "counter": "Parental-cell, permeability, and cytotoxicity counterscreens.",
    },
    "cellular_reporter_or_target_engagement_assay": {
        "primary": "Reporter or target-engagement assay in a disease-relevant cell context.",
        "orthogonal": "Genetic perturbation rescue, CETSA/DARTS, or pathway marker validation.",
        "counter": "Reporter artifact, cytotoxicity, and unrelated-pathway counterscreens.",
    },
    "target_engagement_and_cell_phenotype_assay": {
        "primary": "Direct target-engagement assay followed by disease-relevant cell phenotype.",
        "orthogonal": "SPR/MST/thermal-shift or genetic target perturbation where feasible.",
        "counter": "Cytotoxicity, unrelated phenotype, and assay-interference counterscreens.",
    },
}


DIRECTION_MODELS: dict[str, str] = {
    "oncology": "Use target-expressing cancer cell lines selected by expression/DepMap context; pair target engagement with proliferation, apoptosis, and pathway-marker endpoints.",
    "infectious_disease": "Use pathogen or infected-host-cell models appropriate to the predicted mechanism; separate host-target rescue from direct pathogen inhibition and include host-cell viability.",
    "cardiovascular": "Use cardiomyocyte, endothelial, vascular smooth-muscle, platelet, or transporter models depending on target biology; include safety readouts where relevant.",
    "neurology_psychiatry": "Use receptor/channel or neuronal/glial/iPSC-derived models; prioritize functional signaling and cytotoxicity over generic viability alone.",
    "immunology_inflammation": "Use PBMC, macrophage, dendritic-cell, T-cell, or cytokine-release models matched to the predicted pathway; include basal and stimulated states.",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


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


def fmt_num(value: Any, digits: int = 2) -> str:
    parsed = number(value)
    return "" if parsed is None else f"{parsed:.{digits}f}"


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def compact(value: Any, limit: int = 220) -> str:
    content = " ".join(text(value).split())
    if len(content) <= limit:
        return content
    return content[: limit - 1].rstrip() + "..."


def load_table(root: Path, name: str) -> pd.DataFrame:
    path = root / PRIORITY_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def validation_role(row: pd.Series) -> str:
    group = text(row.get("expertNoveltyGroup")) or text(row.get("noveltyGroup"))
    if truthy(row.get("knownDrugTargetPair")) or "positive_control" in group:
        return "positive_control"
    if "mechanism_extension" in group:
        return "mechanism_extension"
    if "novel" in group:
        return "novel_repurposing_candidate"
    return "secondary_review"


def role_zh(role: str) -> str:
    return {
        "positive_control": "阳性对照/已知机制",
        "mechanism_extension": "机制延展",
        "novel_repurposing_candidate": "新用途或新靶点候选",
        "secondary_review": "二线专家复核候选",
    }.get(role, role)


def wetlab_priority(row: pd.Series, source: str) -> str:
    if source == "wave1_24":
        return "P1_wave1_direct_validation"
    role = validation_role(row)
    score = number(row.get("expertReviewScore")) or number(row.get("validationScore")) or 0.0
    admet = text(row.get("admetTier")) or text(row.get("sotaMlAdmetTier"))
    structure = text(row.get("structureConfidenceTier"))
    if role == "positive_control":
        return "P1_assay_calibration_control"
    if score >= 90 and not admet.startswith("D") and not structure.startswith("D_"):
        return "P1_wave1_or_wave2_validation"
    return "P2_expert_review_before_validation"


def gating_decision(row: pd.Series) -> str:
    role = validation_role(row)
    if role == "positive_control":
        return "First use as assay calibration. Expected result is recovery of known or literature-consistent target biology."
    if role == "mechanism_extension":
        return "Advance if target engagement and disease-cell phenotype are directionally consistent at non-toxic concentrations."
    if role == "novel_repurposing_candidate":
        return "Advance only with orthogonal confirmation: target engagement, disease phenotype, and safety/counterscreen consistency."
    return "Review mechanism, safety, and assay feasibility before wet-lab execution."


def stop_criteria(row: pd.Series) -> str:
    return (
        "Stop or deprioritize if activity appears only at cytotoxic concentrations, "
        "target engagement is absent in the orthogonal assay, disease phenotype is directionally inconsistent, "
        "or label/contraindication review creates an unacceptable translational risk."
    )


def concentration_plan(row: pd.Series) -> str:
    mw = number(row.get("molecularWeight"))
    if mw is not None and mw > 700:
        return "Start 8-point dose response with solubility check; avoid interpreting high-concentration activity without exposure confirmation."
    return "Start 8-10 point dose response, typically 0.01-30 uM or assay-appropriate range, with vehicle and cytotoxicity controls."


def assay_plan(row: pd.Series) -> dict[str, str]:
    modality = text(row.get("assayModality")) or "target_engagement_and_cell_phenotype_assay"
    guide = ASSAY_GUIDE.get(modality, ASSAY_GUIDE["target_engagement_and_cell_phenotype_assay"])
    direction = text(row.get("direction"))
    disease_model = DIRECTION_MODELS.get(direction, "Use disease-relevant cells and a target-specific functional readout before broad phenotype screening.")
    return {
        "primaryAssay": guide["primary"],
        "orthogonalAssay": guide["orthogonal"],
        "counterScreen": guide["counter"],
        "diseaseModelRecommendation": disease_model,
    }


def evidence_count(row: pd.Series) -> int:
    value = number(row.get("expertEvidenceSupportCount")) or number(row.get("evidenceSupportCount"))
    if value is not None:
        return int(round(value))
    fields = [
        "openTargetsScore",
        "integratedTxgnnScore",
        "kgEvidenceScore",
        "expertPathwayScore",
        "expertCmapScore",
        "expertTissueScore",
        "expertStructureScore",
        "expertAdmetScore",
        "expertDepmapScore",
    ]
    return sum(1 for field in fields if (number(row.get(field)) or 0.0) > 0)


def tier_value(value: Any, mapping: dict[str, float], default: float = 50.0) -> float:
    content = text(value)
    for prefix, score in mapping.items():
        if content.startswith(prefix) or content == prefix:
            return score
    return default


def assay_feasibility_score(row: pd.Series) -> float:
    modality = text(row.get("assayModality"))
    return {
        "biochemical_kinase_or_enzyme_assay": 96.0,
        "cell_based_receptor_function_assay": 90.0,
        "target_engagement_and_cell_phenotype_assay": 82.0,
        "transporter_activity_assay": 78.0,
        "cellular_reporter_or_target_engagement_assay": 76.0,
        "electrophysiology_or_channel_function_assay": 72.0,
    }.get(modality, 75.0)


def novelty_value_score(row: pd.Series) -> float:
    role = validation_role(row)
    return {
        "novel_repurposing_candidate": 100.0,
        "mechanism_extension": 86.0,
        "positive_control": 62.0,
        "secondary_review": 55.0,
    }.get(role, 55.0)


def focus_score(row: pd.Series) -> float:
    expert = number(row.get("expertReviewScore")) or number(row.get("validationScore")) or 0.0
    disease = max(
        number(row.get("openTargetsScore")) or 0.0,
        number(row.get("integratedTxgnnScore")) or 0.0,
        number(row.get("kgEvidenceScore")) or 0.0,
    )
    if disease <= 1.5:
        disease *= 100.0
    support = min(100.0, evidence_count(row) / 8.0 * 100.0)
    cmap = tier_value(
        row.get("cmapReversalTier"),
        {
            "A_": 100.0,
            "B_": 82.0,
            "C_": 45.0,
        },
        35.0,
    )
    structure = tier_value(
        row.get("structureConfidenceTier"),
        {
            "A_": 100.0,
            "B_": 86.0,
            "C_": 45.0,
            "D_": 10.0,
        },
        55.0,
    )
    standard_pose = tier_value(
        row.get("standardPoseValidationTier"),
        {
            "A_": 100.0,
            "B_": 86.0,
            "C_": 45.0,
        },
        72.0,
    )
    admet = tier_value(row.get("admetTier"), {"A": 100.0, "B": 86.0, "C": 48.0, "D": 10.0}, 50.0)
    ml_admet = tier_value(
        row.get("sotaMlAdmetTier"),
        {
            "A_": 100.0,
            "B_": 82.0,
            "C_": 52.0,
            "D_": 12.0,
        },
        65.0,
    )
    penalty = 0.0
    if text(row.get("poseAuditStatus")).lower() not in {"", "pass", "warning"}:
        penalty += 12.0
    if text(row.get("poseAuditStatus")).lower() == "warning":
        penalty += 3.0
    if truthy(row.get("contraindicationFlag")) or truthy(row.get("hasContraindicationDiseaseEdge")):
        penalty += 18.0
    score = (
        0.18 * expert
        + 0.14 * disease
        + 0.12 * support
        + 0.13 * cmap
        + 0.12 * structure
        + 0.07 * standard_pose
        + 0.10 * admet
        + 0.06 * ml_admet
        + 0.04 * assay_feasibility_score(row)
        + 0.04 * novelty_value_score(row)
        - penalty
    )
    return round(max(0.0, min(100.0, score)), 4)


def passes_focus_gate(row: pd.Series) -> bool:
    if not (text(row.get("admetTier")).startswith(("A", "B")) or text(row.get("sotaMlAdmetTier")).startswith(("A_", "B_"))):
        return False
    if not text(row.get("structureConfidenceTier")).startswith(("A_", "B_")):
        return False
    if text(row.get("cmapReversalTier")) and not text(row.get("cmapReversalTier")).startswith(("A_", "B_")):
        return False
    if truthy(row.get("contraindicationFlag")) or truthy(row.get("hasContraindicationDiseaseEdge")):
        return False
    if evidence_count(row) < 5:
        return False
    return True


def focus_reason(row: pd.Series) -> str:
    parts = [
        f"score {focus_score(row):.1f}",
        f"{role_zh(validation_role(row))}",
        f"{evidence_count(row)} evidence layers",
    ]
    cmap = text(row.get("cmapReversalTier"))
    if cmap:
        parts.append(cmap)
    structure = text(row.get("structureConfidenceTier"))
    if structure:
        parts.append(structure)
    admet = text(row.get("admetTier")) or text(row.get("sotaMlAdmetTier"))
    if admet:
        parts.append(f"ADMET {admet}")
    return "; ".join(parts)


def is_tier(value: Any, prefixes: tuple[str, ...]) -> bool:
    content = text(value)
    return bool(content) and content.startswith(prefixes)


def evidence_readiness(row: dict[str, Any]) -> dict[str, Any]:
    """Summarize hard evidence gates from the normalized wet-lab queue."""
    affinity = number(row.get("affinityScore")) or 0.0
    ot = number(row.get("openTargetsScore")) or 0.0
    txgnn = number(row.get("txgnnScore")) or 0.0
    kg = number(row.get("kgEvidenceScore")) or 0.0
    diffdock = number(row.get("diffdock"))
    support_count = int(number(row.get("evidenceSupportCount")) or 0)
    role = text(row.get("validationRole"))
    structure = text(row.get("structureConfidenceTier"))
    standard_pose = text(row.get("standardPoseValidationTier"))
    pose_status = text(row.get("poseAuditStatus")).lower()
    cmap = text(row.get("cmapReversalTier"))
    pathway = text(row.get("pathwayDiseaseContextTier"))
    gtex = text(row.get("gtexContextTier"))
    tissue = text(row.get("tissueContextTier"))
    depmap = text(row.get("depmapDependencyTier"))
    admet = text(row.get("admetTier"))
    sdf_path = text(row.get("confidenceSdfPath"))
    receptor_path = text(row.get("receptorPdbPath"))

    disease_supported = max(ot, txgnn, kg / 100.0 if kg > 1.5 else kg) >= 0.35
    pathway_supported = pathway.startswith(("A_", "B_"))
    signature_supported = (not cmap) or cmap.startswith(("A_", "B_"))
    tissue_supported = gtex.startswith(("A_", "B_")) or tissue.startswith(("A_", "B_"))
    oncology_depmap_ok = row.get("direction") != "oncology" or depmap.startswith(("A_", "B_"))
    structure_supported = structure.startswith(("A_", "B_")) and pose_status in {"", "pass", "warning"} and bool(sdf_path or receptor_path)
    admet_supported = admet.startswith(("A", "B"))
    model_supported = affinity >= 0.50 or role == "positive_control"
    assay_supported = bool(text(row.get("primaryAssay"))) and bool(text(row.get("orthogonalAssay"))) and bool(text(row.get("counterScreen")))
    diffdock_supported = diffdock is not None

    flags = {
        "model": model_supported,
        "disease": disease_supported,
        "pathway": pathway_supported,
        "signature": signature_supported,
        "tissue": tissue_supported,
        "depmap": oncology_depmap_ok,
        "structure": structure_supported,
        "admet": admet_supported,
        "assay": assay_supported,
        "diffdock": diffdock_supported,
    }
    pass_count = sum(1 for passed in flags.values() if passed)

    hard_holds: list[str] = []
    soft_holds: list[str] = []
    if not admet_supported:
        hard_holds.append("ADMET not in A/B tier")
    if not structure_supported:
        hard_holds.append("structure or pose support not yet sufficient")
    if not assay_supported:
        hard_holds.append("primary/orthogonal/counterscreen assay plan incomplete")
    if support_count < 5:
        hard_holds.append("fewer than five evidence layers")
    if not signature_supported:
        soft_holds.append("CMap/LINCS reversal not supportive")
    if not tissue_supported:
        soft_holds.append("target tissue context not yet supportive")
    if not oncology_depmap_ok:
        soft_holds.append("oncology candidate lacks recurrent DepMap support")
    if not disease_supported:
        soft_holds.append("disease evidence is weak")
    if not model_supported:
        soft_holds.append("affinity/model score below preferred threshold")

    if hard_holds:
        decision = "hold_before_purchase"
    elif pass_count >= 9 and not soft_holds:
        decision = "go_core"
    elif pass_count >= 8 and len(soft_holds) <= 1:
        decision = "go_if_budget_allows"
    elif pass_count >= 7:
        decision = "conditional_expert_review"
    else:
        decision = "deprioritize_for_wave1"

    if role == "positive_control" and decision.startswith("go"):
        experiment_tier = "calibration_control"
    elif decision == "go_core":
        experiment_tier = "wave1_core"
    elif decision == "go_if_budget_allows":
        experiment_tier = "wave1_extension"
    elif decision == "conditional_expert_review":
        experiment_tier = "manual_review"
    else:
        experiment_tier = "hold"

    return {
        "decision": decision,
        "experimentTier": experiment_tier,
        "passCount": pass_count,
        "evidenceFlags": "; ".join(f"{key}:{'Y' if value else 'N'}" for key, value in flags.items()),
        "hardHolds": "; ".join(hard_holds),
        "softHolds": "; ".join(soft_holds),
    }


def score_to_100(value: Any) -> float:
    parsed = number(value)
    if parsed is None:
        return 0.0
    if -1.5 <= parsed <= 1.5:
        return max(0.0, min(100.0, parsed * 100.0))
    return max(0.0, min(100.0, parsed))


def disease_signal_score(row: dict[str, Any] | pd.Series) -> float:
    values = [
        score_to_100(row.get("openTargetsScore")),
        score_to_100(row.get("txgnnScore") or row.get("integratedTxgnnScore")),
        score_to_100(row.get("kgEvidenceScore")),
    ]
    return max(values)


def diffusion_pose_score(row: dict[str, Any] | pd.Series) -> float:
    diffdock = number(row.get("diffdock"))
    if diffdock is None:
        return 35.0
    if diffdock <= -5:
        return 100.0
    if diffdock <= -4:
        return 90.0
    if diffdock <= -3:
        return 80.0
    if diffdock <= -2:
        return 68.0
    if diffdock <= -1:
        return 55.0
    return 38.0


def final_structure_score(row: dict[str, Any] | pd.Series) -> float:
    structure = tier_value(
        row.get("structureConfidenceTier"),
        {"A_": 100.0, "B_": 86.0, "C_": 50.0, "D_": 10.0},
        55.0,
    )
    pose = tier_value(
        row.get("standardPoseValidationTier"),
        {"A_": 100.0, "B_": 86.0, "C_": 50.0, "D_": 15.0},
        68.0,
    )
    return round(0.55 * structure + 0.25 * pose + 0.20 * diffusion_pose_score(row), 4)


def final_admet_score(row: dict[str, Any] | pd.Series) -> float:
    return tier_value(
        row.get("admetTier") or row.get("sotaMlAdmetTier"),
        {"A": 100.0, "A_": 100.0, "B": 86.0, "B_": 82.0, "C": 48.0, "C_": 52.0, "D": 10.0, "D_": 12.0},
        55.0,
    )


def final_focus_value_score(row: dict[str, Any]) -> float:
    model = score_to_100(row.get("affinityScore"))
    disease = disease_signal_score(row)
    support = min(100.0, (number(row.get("evidenceSupportCount")) or 0.0) / 8.0 * 100.0)
    gate = min(100.0, (number(row.get("passCount")) or 0.0) / 10.0 * 100.0)
    pathway = tier_value(row.get("pathwayDiseaseContextTier"), {"A_": 100.0, "B_": 82.0, "C_": 45.0}, 35.0)
    signature = tier_value(row.get("cmapReversalTier"), {"A_": 100.0, "B_": 82.0, "C_": 45.0}, 35.0)
    tissue = max(
        tier_value(row.get("gtexContextTier"), {"A_": 100.0, "B_": 82.0, "C_": 45.0}, 35.0),
        tier_value(row.get("tissueContextTier"), {"A_": 100.0, "B_": 82.0, "C_": 45.0}, 35.0),
    )
    depmap = tier_value(row.get("depmapDependencyTier"), {"A_": 100.0, "B_": 82.0, "C_": 45.0}, 70.0)
    structure = final_structure_score(row)
    admet = final_admet_score(row)
    assay = assay_feasibility_score(pd.Series(row))
    novelty = novelty_value_score(pd.Series(row))
    decision = text(row.get("decision"))
    role = text(row.get("validationRole"))

    score = (
        0.12 * model
        + 0.14 * disease
        + 0.10 * support
        + 0.09 * gate
        + 0.10 * pathway
        + 0.10 * signature
        + 0.08 * tissue
        + 0.05 * depmap
        + 0.10 * structure
        + 0.07 * admet
        + 0.03 * assay
        + 0.02 * novelty
    )
    if decision == "go_core":
        score += 4.0
    elif decision == "go_if_budget_allows":
        score += 1.5
    elif decision == "conditional_expert_review":
        score -= 4.0
    elif decision in {"hold_before_purchase", "deprioritize_for_wave1"}:
        score -= 18.0
    if role == "positive_control":
        score -= 3.0
    if text(row.get("hardHolds")):
        score -= 15.0
    if text(row.get("softHolds")):
        score -= min(8.0, 2.5 * len([item for item in text(row.get("softHolds")).split(";") if item.strip()]))
    return round(max(0.0, min(100.0, score)), 4)


def purchase_action(row: dict[str, Any]) -> str:
    decision = text(row.get("decision"))
    score = number(row.get("prePurchaseFocusScore")) or final_focus_value_score(row)
    if decision == "go_core" and score >= 82 and not text(row.get("hardHolds")):
        return "ready_for_final_manual_purchase_check"
    if decision in {"go_core", "go_if_budget_allows"} and score >= 76 and not text(row.get("hardHolds")):
        return "backup_after_purchase_feasibility_check"
    if decision == "conditional_expert_review":
        return "expert_review_before_purchase"
    return "hold_before_purchase"


def focus_rationale_zh(row: dict[str, Any]) -> str:
    parts = [
        f"综合聚焦分 {fmt_num(row.get('prePurchaseFocusScore'), 1)}",
        text(row.get("validationRoleZh")),
        f"{text(row.get('passCount')) or '0'} 个硬证据门通过",
        f"{text(row.get('evidenceSupportCount')) or '0'} 层证据支持",
    ]
    if text(row.get("cmapReversalTier")):
        parts.append(f"CMap {text(row.get('cmapReversalTier'))}")
    if text(row.get("structureConfidenceTier")):
        parts.append(f"结构 {text(row.get('structureConfidenceTier'))}")
    if text(row.get("admetTier")):
        parts.append(f"ADMET {text(row.get('admetTier'))}")
    holds = text(row.get("hardHolds")) or text(row.get("softHolds"))
    if holds:
        parts.append(f"需处理: {compact(holds, 90)}")
    return "；".join(part for part in parts if part)


def manual_purchase_checks_zh(row: dict[str, Any]) -> str:
    checks = [
        "确认 exact salt/form、纯度、供应商和到货周期",
        "确认 DMSO/水相溶解度和储存条件",
        "核对临床可达游离浓度与计划剂量范围",
        "确认目标蛋白在拟用疾病模型中表达",
        "确认 primary assay、orthogonal assay、counterscreen 均可执行",
        "PubMed/ClinicalTrials 核查该 drug-target-disease 是否已知",
    ]
    if text(row.get("direction")) == "oncology":
        checks.append("核对目标癌种细胞系的表达和 DepMap 依赖")
    if text(row.get("validationRole")) == "novel_repurposing_candidate":
        checks.append("优先设置靶点敲低/拮抗剂 rescue 或竞争结合验证")
    return "；".join(checks)


def pre_purchase_go_no_go_zh(row: dict[str, Any]) -> str:
    return (
        "进入实验的最低条件：化合物质量合格；非毒性浓度下出现剂量反应；primary target readout 为阳性；"
        "orthogonal target engagement 或 pathway readout 方向一致；疾病表型不由细胞毒性或 assay interference 解释。"
    )


def build_pre_purchase_focus_rows(
    decision_rows: list[dict[str, Any]],
    first_experiment_rows: list[dict[str, Any]],
    backup_limit: int = 24,
) -> list[dict[str, Any]]:
    first_keys = {
        (row["drug"].lower(), row["target"], row["direction"]): row
        for row in first_experiment_rows
    }
    scored: list[dict[str, Any]] = []
    for row in decision_rows:
        payload = {**row}
        payload["prePurchaseFocusScore"] = f"{final_focus_value_score(payload):.4f}"
        payload["purchaseAction"] = purchase_action(payload)
        payload["focusRationaleZh"] = focus_rationale_zh(payload)
        payload["manualPurchaseChecksZh"] = manual_purchase_checks_zh(payload)
        payload["prePurchaseGoNoGoZh"] = pre_purchase_go_no_go_zh(payload)
        scored.append(payload)

    action_order = {
        "ready_for_final_manual_purchase_check": 0,
        "backup_after_purchase_feasibility_check": 1,
        "expert_review_before_purchase": 2,
        "hold_before_purchase": 3,
    }
    scored.sort(
        key=lambda row: (
            action_order.get(text(row.get("purchaseAction")), 9),
            -float(number(row.get("prePurchaseFocusScore")) or 0.0),
            text(row.get("drug")),
            text(row.get("target")),
        )
    )

    selected_backup: set[tuple[str, str, str]] = set()
    drug_counts = Counter(
        row["drug"].lower()
        for row in first_experiment_rows
    )
    target_counts = Counter(row["target"] for row in first_experiment_rows)
    direction_counts = Counter(row["direction"] for row in first_experiment_rows)
    for row in scored:
        if len(selected_backup) >= max(0, backup_limit - len(first_experiment_rows)):
            break
        key = (row["drug"].lower(), row["target"], row["direction"])
        if key in first_keys:
            continue
        if row["purchaseAction"] not in {
            "ready_for_final_manual_purchase_check",
            "backup_after_purchase_feasibility_check",
        }:
            continue
        if drug_counts[row["drug"].lower()] >= 2 or target_counts[row["target"]] >= 2 or direction_counts[row["direction"]] >= 8:
            continue
        selected_backup.add(key)
        drug_counts[row["drug"].lower()] += 1
        target_counts[row["target"]] += 1
        direction_counts[row["direction"]] += 1

    final_rows: list[dict[str, Any]] = []
    for row in scored:
        key = (row["drug"].lower(), row["target"], row["direction"])
        payload = {**row}
        first_row = first_keys.get(key)
        if first_row:
            payload["prePurchaseTier"] = first_row["firstExperimentBatch"]
            payload["prePurchaseExperimentRank"] = first_row["firstExperimentRank"]
        elif key in selected_backup:
            payload["prePurchaseTier"] = "backup_24"
            payload["prePurchaseExperimentRank"] = ""
        elif payload["purchaseAction"] == "expert_review_before_purchase":
            payload["prePurchaseTier"] = "expert_review_hold"
            payload["prePurchaseExperimentRank"] = ""
        else:
            payload["prePurchaseTier"] = "hold"
            payload["prePurchaseExperimentRank"] = ""
        final_rows.append(payload)

    final_rows.sort(
        key=lambda row: (
            {
                "batch_1_core_6": 0,
                "batch_1_extension_12": 1,
                "backup_24": 2,
                "expert_review_hold": 3,
                "hold": 4,
            }.get(text(row.get("prePurchaseTier")), 9),
            int(number(row.get("prePurchaseExperimentRank")) or 999999),
            -float(number(row.get("prePurchaseFocusScore")) or 0.0),
            text(row.get("drug")),
            text(row.get("target")),
        )
    )
    for rank, row in enumerate(final_rows, start=1):
        row["prePurchaseFocusRank"] = rank
    return final_rows


def build_pre_experiment_decision_rows(purchase_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for row in purchase_rows:
        readiness = evidence_readiness(row)
        payload = {**row, **readiness}
        ranked.append(
            {
                **payload,
                "preExperimentScore": final_focus_value_score(payload),
            }
        )
    ranked.sort(
        key=lambda item: (
            {
                "go_core": 0,
                "go_if_budget_allows": 1,
                "conditional_expert_review": 2,
                "hold_before_purchase": 3,
                "deprioritize_for_wave1": 4,
            }.get(item["decision"], 9),
            -float(item["preExperimentScore"]),
            text(item.get("drug")),
            text(item.get("target")),
        )
    )
    for rank, row in enumerate(ranked, start=1):
        row["preExperimentRank"] = rank
    return ranked


def build_pre_experiment_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    counts = Counter(row["decision"] for row in rows)
    core = [row for row in rows if row["decision"] == "go_core"][:12]
    conditional = [row for row in rows if row["decision"] == "conditional_expert_review"][:12]

    def decision_table(block: list[dict[str, Any]]) -> str:
        headers = ["Rank", "Decision", "Direction", "Drug", "Target", "Role", "Pass", "Main hold"]
        lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
        for row in block:
            hold = text(row.get("hardHolds")) or text(row.get("softHolds")) or "none"
            values = [
                str(row["preExperimentRank"]),
                row["decision"],
                row["directionLabelZh"] or row["direction"],
                row["drug"],
                f"{row['target']} ({row['protein']})",
                row["validationRoleZh"],
                str(row["passCount"]),
                compact(hold, 90),
            ]
            lines.append("|" + "|".join(str(value).replace("|", "/") for value in values) + "|")
        return "\n".join(lines)

    return f"""# Pre-Experiment Decision Matrix

Generated UTC: {summary['createdUtc']}

## Purpose

This matrix narrows the current computational queue into candidates that are worth purchasing and testing first. It uses only evidence already present in the BioMaster outputs: affinity/model score, disease evidence, pathway/CREEDS context, CMap/LINCS reversal, GTEx/HPA tissue context, DepMap for oncology, ADMET, contraindication flags, structure/pose readiness, and assay feasibility.

## Decision Counts

- go_core: {counts.get('go_core', 0)}
- go_if_budget_allows: {counts.get('go_if_budget_allows', 0)}
- conditional_expert_review: {counts.get('conditional_expert_review', 0)}
- hold_before_purchase: {counts.get('hold_before_purchase', 0)}
- deprioritize_for_wave1: {counts.get('deprioritize_for_wave1', 0)}

## Strict Go Criteria

A candidate is placed in `go_core` only when it has multi-evidence support, an executable assay plan, acceptable ADMET, and interpretable structural output. Candidates with missing core evidence are not rejected biologically; they are held until the missing evidence is resolved.

## Highest-Priority Go Candidates

{decision_table(core)}

## Conditional Candidates For Expert Review

{decision_table(conditional)}

## Practical Use

For the first wet-lab batch, start with `go_core` candidates plus 2-3 positive controls. Add `go_if_budget_allows` only after compound availability, solubility, clinical exposure relevance, and assay/counterscreen availability are confirmed. Do not spend the first experiment on `hold_before_purchase` records unless an expert has a specific biological reason to override the gate.
"""


def select_first_experiment_panel(decision_rows: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    pool = [
        row
        for row in decision_rows
        if row["decision"] in {"go_core", "go_if_budget_allows"}
        and row["validationRole"] in {"positive_control", "mechanism_extension", "novel_repurposing_candidate"}
    ]
    pool.sort(
        key=lambda row: (
            0 if row["decision"] == "go_core" else 1,
            -float(row["preExperimentScore"]),
            text(row.get("drug")),
            text(row.get("target")),
        )
    )

    selected: list[dict[str, Any]] = []
    drug_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    direction_counts: Counter[str] = Counter()

    role_targets = [
        ("positive_control", 2),
        ("mechanism_extension", 2),
        ("novel_repurposing_candidate", 2),
        ("positive_control", 3),
        ("mechanism_extension", 4),
        ("novel_repurposing_candidate", 5),
    ]
    for role, required in role_targets:
        for row in pool:
            if len(selected) >= min(limit, 6) and required <= 2:
                break
            if len([item for item in selected if item["validationRole"] == role]) >= required:
                break
            if row in selected or row["validationRole"] != role:
                continue
            drug = text(row.get("drug")).lower()
            target = text(row.get("target"))
            direction = text(row.get("direction"))
            if drug_counts[drug] >= 2 or target_counts[target] >= 2 or direction_counts[direction] >= 6:
                continue
            selected.append(row)
            drug_counts[drug] += 1
            target_counts[target] += 1
            direction_counts[direction] += 1

    for row in pool:
        if len(selected) >= limit:
            break
        if row in selected:
            continue
        drug = text(row.get("drug")).lower()
        target = text(row.get("target"))
        direction = text(row.get("direction"))
        if drug_counts[drug] >= 2 or target_counts[target] >= 2 or direction_counts[direction] >= 6:
            continue
        selected.append(row)
        drug_counts[drug] += 1
        target_counts[target] += 1
        direction_counts[direction] += 1

    panel: list[dict[str, Any]] = []
    for rank, row in enumerate(selected[:limit], start=1):
        panel.append(
            {
                **row,
                "firstExperimentRank": rank,
                "firstExperimentBatch": "batch_1_core_6" if rank <= 6 else "batch_1_extension_12",
                "firstExperimentReason": (
                    f"{row['decision']}; {row['validationRoleZh']}; "
                    f"{row['passCount']} evidence gates; {row['assayModality']}"
                ),
            }
        )
    return panel


def compact_focus_table(rows: list[dict[str, Any]], tier: str, limit: int) -> str:
    block = [row for row in rows if row.get("prePurchaseTier") == tier][:limit]
    headers = ["Rank", "Direction", "Drug", "Target", "Role", "Focus", "Action", "Rationale"]
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in block:
        values = [
            row.get("prePurchaseFocusRank"),
            row.get("directionLabelZh") or row.get("direction"),
            row.get("drug"),
            f"{row.get('target')} ({row.get('protein')})",
            row.get("validationRoleZh"),
            fmt_num(row.get("prePurchaseFocusScore"), 1),
            row.get("purchaseAction"),
            compact(row.get("focusRationaleZh"), 120),
        ]
        lines.append("|" + "|".join(str(value).replace("|", "/") for value in values) + "|")
    return "\n".join(lines)


def build_pre_purchase_focus_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    counts = Counter(row["prePurchaseTier"] for row in rows)
    action_counts = Counter(row["purchaseAction"] for row in rows)
    return f"""# Final Pre-Purchase Focus Package

Generated UTC: {summary['createdUtc']}

## Purpose

This package is the last computational narrowing step before wet-lab purchasing. It separates assay calibration controls from mechanism-extension and higher-upside repurposing hypotheses, then applies a practical purchase gate: compound quality, assay feasibility, disease context, target engagement, ADMET, structural interpretability, and redundancy control.

## Tier Counts

- batch_1_core_6: {counts.get('batch_1_core_6', 0)}
- batch_1_extension_12: {counts.get('batch_1_extension_12', 0)}
- backup_24: {counts.get('backup_24', 0)}
- expert_review_hold: {counts.get('expert_review_hold', 0)}
- hold: {counts.get('hold', 0)}

## Purchase Action Counts

- ready_for_final_manual_purchase_check: {action_counts.get('ready_for_final_manual_purchase_check', 0)}
- backup_after_purchase_feasibility_check: {action_counts.get('backup_after_purchase_feasibility_check', 0)}
- expert_review_before_purchase: {action_counts.get('expert_review_before_purchase', 0)}
- hold_before_purchase: {action_counts.get('hold_before_purchase', 0)}

## Core 6

The first six candidates are intended to test the whole workflow with a balanced design: positive controls, mechanism-extension hypotheses, and novel repurposing hypotheses.

{compact_focus_table(rows, 'batch_1_core_6', 6)}

## Extension 12

These extend disease and mechanism coverage after compound handling, assay availability, and counterscreens are confirmed.

{compact_focus_table(rows, 'batch_1_extension_12', 12)}

## Backup 24

These are not first-order purchases, but they are the preferred replacements if one of the first 12 fails vendor, solubility, exposure, or assay checks.

{compact_focus_table(rows, 'backup_24', 24)}

## Manual Gate Before Spending

For every candidate marked ready or backup, confirm exact compound form, vendor availability, purity, storage, solubility, clinically relevant concentration range, disease-model target expression, assay/counterscreen availability, and literature novelty. A candidate should not be promoted from computational recommendation to biological claim unless target engagement, orthogonal readout, disease phenotype, dose response, and counterscreens agree.
"""


TARGET_EXECUTION_GUIDE: dict[str, dict[str, str]] = {
    "EGFR": {
        "targetBiology": "Receptor tyrosine kinase; first validate direct kinase inhibition, then cellular EGFR pathway suppression.",
        "primaryReadout": "EGFR kinase IC50/Kd or enzymatic activity inhibition; pair with pEGFR and downstream pERK/pAKT in EGFR-expressing cells.",
        "modelExamples": "EGFR-driven NSCLC or epithelial cancer lines such as PC9/HCC827 if available; include EGFR-low or EGFR-wild-type cells as specificity context.",
        "positiveControl": "Known EGFR inhibitor such as gefitinib, erlotinib, osimertinib, or dacomitinib, matched to assay availability.",
        "counterScreenDetail": "Parallel cytotoxicity, unrelated kinase panel, EGFR-low cells, and pathway-rescue or ligand-stimulation control.",
        "mainConfounder": "Apparent phenotype can be driven by broad kinase/cytotoxic effects rather than target-specific EGFR engagement.",
    },
    "KIT": {
        "targetBiology": "Type III receptor tyrosine kinase; test KIT biochemical activity and ligand/pathway-dependent cellular signaling.",
        "primaryReadout": "KIT kinase activity or binding assay, followed by cellular pKIT/pERK/pAKT suppression in KIT-expressing cells.",
        "modelExamples": "KIT-dependent GIST/mast-cell models if available; include KIT-low cells or unrelated RTK cells.",
        "positiveControl": "Imatinib, sunitinib, or another validated KIT inhibitor as platform control.",
        "counterScreenDetail": "Kinase selectivity panel, KIT-low cells, cytotoxicity, and pathway specificity against EGFR/PDGFR where feasible.",
        "mainConfounder": "Gefitinib may show kinase-family cross-reactivity; distinguish direct KIT engagement from EGFR-pathway carryover.",
    },
    "ADORA1": {
        "targetBiology": "Gi-coupled adenosine receptor; validate receptor signaling before interpreting disease-cell phenotype.",
        "primaryReadout": "ADORA1 functional assay such as inhibition of forskolin-stimulated cAMP, beta-arrestin, or ligand-competition binding.",
        "modelExamples": "ADORA1-overexpressing or endogenous ADORA1-positive cancer/immune/neuronal cells; include receptor-negative parental cells.",
        "positiveControl": "Adenosine/CPA/CCPA agonist or a validated ADORA1 antagonist, depending on assay direction.",
        "counterScreenDetail": "ADORA2A/ADORA2B/ADORA3 selectivity, receptor-negative cells, nucleoside-transporter context, and cytotoxicity.",
        "mainConfounder": "Purine analogs can cause nucleoside-metabolism or cytotoxic effects that mimic receptor-linked biology.",
    },
    "F2R": {
        "targetBiology": "PAR1/thrombin receptor GPCR; test receptor-function modulation and separate it from serotonin-receptor pharmacology.",
        "primaryReadout": "PAR1 calcium flux, IP1, or beta-arrestin response after thrombin or PAR1 activating peptide stimulation.",
        "modelExamples": "PAR1-positive tumor, endothelial, platelet-like, or engineered receptor cells; include receptor-low parental cells.",
        "positiveControl": "PAR1 activating peptide for agonist response and vorapaxar/atopaxar or another PAR1 antagonist for inhibition control.",
        "counterScreenDetail": "5-HT3 receptor counterscreen, cytotoxicity, receptor-negative cells, and thrombin-independent pathway checks.",
        "mainConfounder": "Palonosetron is an established 5-HT3 agent; any PAR1 signal needs strong receptor-specific orthogonal confirmation.",
    },
    "CHRM3": {
        "targetBiology": "Gq-coupled muscarinic receptor; useful as receptor-function calibration and antimuscarinic assay control.",
        "primaryReadout": "CHRM3 calcium flux or IP1 response to acetylcholine/carbachol and antagonist dose-response.",
        "modelExamples": "CHRM3-expressing engineered cells or endogenous epithelial/smooth-muscle models; include CHRM1/2/4/5 selectivity if feasible.",
        "positiveControl": "Acetylcholine/carbachol agonist and atropine or solifenacin antagonist control.",
        "counterScreenDetail": "Muscarinic subtype selectivity, parental-cell response, cytotoxicity, and assay-interference controls.",
        "mainConfounder": "Subtype selectivity can be weak in cell assays; use orthogonal ligand competition or subtype panel.",
    },
    "GLP1R": {
        "targetBiology": "Gs-coupled peptide GPCR; receptor engagement should be shown through cAMP or beta-arrestin before disease interpretation.",
        "primaryReadout": "GLP1R cAMP accumulation or beta-arrestin response with agonist/antagonist mode clarified.",
        "modelExamples": "GLP1R-expressing engineered cells, endocrine/metabolic cells, endothelial/cardiometabolic models if expression is confirmed.",
        "positiveControl": "GLP-1 or exendin-4 agonist and exendin(9-39) or another validated antagonist where available.",
        "counterScreenDetail": "ACE-pathway counterscreen for ramipril, GLP1R-negative cells, cytotoxicity, and peptide-receptor specificity.",
        "mainConfounder": "Ramipril is a prodrug/ACE-pathway drug; receptor signal must be separated from canonical ACE biology and metabolite effects.",
    },
    "ADRA1A": {
        "targetBiology": "Gq-coupled adrenergic receptor; validate receptor-function modulation and subtype selectivity.",
        "primaryReadout": "ADRA1A calcium flux or IP1 response to phenylephrine/norepinephrine with antagonist-mode dose-response.",
        "modelExamples": "ADRA1A-expressing engineered cells or neuronal/vascular models with confirmed expression.",
        "positiveControl": "Phenylephrine agonist and prazosin or another alpha-1 antagonist control.",
        "counterScreenDetail": "ADRA1B/ADRA1D, dopamine receptor, cytotoxicity, and parental-cell counterscreens.",
        "mainConfounder": "Amisulpride has dopaminergic pharmacology; alpha-1 signal needs clean receptor-selectivity evidence.",
    },
    "EDNRA": {
        "targetBiology": "Endothelin A receptor GPCR; useful for cardiovascular/vascular phenotype only after direct receptor readout.",
        "primaryReadout": "EDNRA calcium flux, IP1, or beta-arrestin response to endothelin-1 and antagonist-mode dose-response.",
        "modelExamples": "EDNRA-positive vascular smooth-muscle, endothelial, cardiomyocyte, or engineered receptor cells.",
        "positiveControl": "Endothelin-1 agonist and BQ-123, ambrisentan, or another endothelin receptor antagonist control.",
        "counterScreenDetail": "EDNRB selectivity, cholinesterase/pathway counterscreen for donepezil, cytotoxicity, and receptor-negative cells.",
        "mainConfounder": "Donepezil's canonical cholinesterase activity can confound cardiovascular phenotypes without direct EDNRA engagement.",
    },
    "OXTR": {
        "targetBiology": "Gq-coupled oxytocin receptor; validate receptor-function modulation and separate from opioid/peripheral effects.",
        "primaryReadout": "OXTR calcium flux, IP1, or beta-arrestin response to oxytocin with agonist/antagonist dose-response.",
        "modelExamples": "OXTR-expressing engineered, neuronal, endocrine, or smooth-muscle models after expression confirmation.",
        "positiveControl": "Oxytocin agonist and atosiban or another OXTR antagonist control.",
        "counterScreenDetail": "AVPR1A/AVPR1B/AVPR2 selectivity, opioid receptor counterscreen, cytotoxicity, and receptor-negative cells.",
        "mainConfounder": "Methylnaltrexone is quaternary/peripherally acting; cell permeability and opioid receptor effects must be checked.",
    },
    "CRHR1": {
        "targetBiology": "Gs-coupled corticotropin-releasing hormone receptor; validate cAMP signaling before disease-cell phenotype.",
        "primaryReadout": "CRHR1 cAMP response to CRH/urocortin and antagonist-mode dose-response.",
        "modelExamples": "CRHR1-expressing engineered, neuronal, endocrine, or stress-axis relevant cells with target expression confirmed.",
        "positiveControl": "CRH or urocortin agonist and antalarmin/CP-154526-like CRHR1 antagonist if available.",
        "counterScreenDetail": "CRHR2 selectivity, opioid receptor counterscreen, cytotoxicity, and receptor-negative cells.",
        "mainConfounder": "Methylnaltrexone's canonical opioid pharmacology and permeability can create false receptor-phenotype links.",
    },
}


def execution_profile(row: dict[str, Any]) -> dict[str, str]:
    guide = TARGET_EXECUTION_GUIDE.get(text(row.get("target")), {})
    modality = text(row.get("assayModality"))
    if not guide:
        guide = {
            "targetBiology": "Target-specific assay should be selected from the known receptor/enzyme biology before phenotype screening.",
            "primaryReadout": text(row.get("primaryAssay")),
            "modelExamples": text(row.get("diseaseModelRecommendation")),
            "positiveControl": "Use a literature-supported agonist, antagonist, inhibitor, or substrate matched to the target mechanism.",
            "counterScreenDetail": text(row.get("counterScreen")),
            "mainConfounder": "The main risk is a non-specific phenotype without direct target engagement.",
        }
    dose = text(row.get("concentrationPlan")) or "Run an 8-10 point dose response with solubility, vehicle, and cytotoxicity controls."
    replicate = (
        "Minimum n=3 biological replicates after pilot; run technical duplicates/triplicates where the platform supports it; "
        "repeat hits in an independent day/batch before promotion."
    )
    if modality == "biochemical_kinase_or_enzyme_assay":
        assay_sequence = (
            "1) biochemical target IC50/Kd; 2) cellular target-engagement or phospho-pathway readout; "
            "3) disease-cell phenotype only if target engagement is measurable."
        )
    else:
        assay_sequence = (
            "1) receptor/function dose-response; 2) orthogonal ligand competition, internalization, knockdown, or antagonist-rescue; "
            "3) disease-cell phenotype only after target signal is confirmed."
        )
    return {
        **guide,
        "doseResponsePlan": dose,
        "replicatePlan": replicate,
        "assaySequence": assay_sequence,
        "negativeControls": "DMSO/vehicle, no-ligand or no-stimulation condition, parental or target-low cells, and unrelated receptor/enzyme where feasible.",
        "goNoGoCriteria": (
            "Advance only if there is a reproducible dose response at non-toxic concentrations, direct target readout is positive, "
            "orthogonal target engagement agrees, and disease phenotype changes in the predicted direction."
        ),
        "stopCriteriaShort": (
            "Hold if activity appears only with cytotoxicity, no orthogonal target engagement, poor solubility/exposure, "
            "or a counterscreen explains the signal."
        ),
    }


def build_experiment_execution_protocol_rows(first_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in first_rows:
        profile = execution_profile(row)
        rows.append(
            {
                "executionRank": row["firstExperimentRank"],
                "executionBatch": row["firstExperimentBatch"],
                "direction": row["direction"],
                "directionLabelZh": row["directionLabelZh"],
                "drug": row["drug"],
                "target": row["target"],
                "protein": row["protein"],
                "proteinName": row["proteinName"],
                "validationRole": row["validationRole"],
                "validationRoleZh": row["validationRoleZh"],
                "decision": row["decision"],
                "preExperimentScore": row["preExperimentScore"],
                "assayModality": row["assayModality"],
                "targetBiology": profile["targetBiology"],
                "assaySequence": profile["assaySequence"],
                "primaryReadout": profile["primaryReadout"],
                "orthogonalValidation": row["orthogonalAssay"],
                "modelExamples": profile["modelExamples"],
                "positiveControl": profile["positiveControl"],
                "negativeControls": profile["negativeControls"],
                "counterScreenDetail": profile["counterScreenDetail"],
                "doseResponsePlan": profile["doseResponsePlan"],
                "replicatePlan": profile["replicatePlan"],
                "goNoGoCriteria": profile["goNoGoCriteria"],
                "stopCriteriaShort": profile["stopCriteriaShort"],
                "mainConfounder": profile["mainConfounder"],
                "prePurchaseManualChecks": manual_purchase_checks_zh(row),
                "expertQuestion": (
                    f"Does {row['drug']} produce target-specific {row['target']} engagement and a disease-relevant "
                    "phenotype that cannot be explained by toxicity, assay interference, or the drug's known canonical mechanism?"
                ),
            }
        )
    return rows


def build_experiment_execution_protocol_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    core = [row for row in rows if row["executionBatch"] == "batch_1_core_6"]
    extension = [row for row in rows if row["executionBatch"] == "batch_1_extension_12"]

    def table(block: list[dict[str, Any]]) -> str:
        headers = ["Rank", "Direction", "Drug", "Target", "Role", "Primary Readout", "Main Risk"]
        lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
        for row in block:
            values = [
                row["executionRank"],
                row["directionLabelZh"] or row["direction"],
                row["drug"],
                f"{row['target']} ({row['protein']})",
                row["validationRoleZh"],
                compact(row["primaryReadout"], 105),
                compact(row["mainConfounder"], 95),
            ]
            lines.append("|" + "|".join(str(value).replace("|", "/") for value in values) + "|")
        return "\n".join(lines)

    return f"""# Wet-Lab Experiment Execution Protocol

Generated UTC: {summary['createdUtc']}

## Purpose

This file converts the first 12 computationally prioritized candidates into an experiment-facing protocol sheet. It is intended for experimental planning and expert review, not as a substitute for platform-specific SOPs or safety review.

## Global Design

- Start with `batch_1_core_6` if capacity is limited; add `batch_1_extension_12` after procurement and assay feasibility checks.
- Every candidate needs direct target readout, orthogonal target engagement, disease-relevant phenotype, dose response, and counterscreens.
- Do not interpret a disease-cell phenotype alone as validation.
- Hold candidates when activity appears only at cytotoxic concentrations or when counterscreens explain the signal.

## Core 6

{table(core)}

## Extension 12

{table(extension)}

## Shared Execution Gate

For each candidate, first confirm exact compound form, purity, solubility, storage, vehicle compatibility, clinically relevant concentration range, target expression in the planned model, and assay/counterscreen availability. Promote only reproducible, non-toxic, dose-dependent signals with matching primary and orthogonal target evidence.
"""


def procurement_priority(row: dict[str, Any]) -> str:
    if text(row.get("executionBatch")) == "batch_1_core_6":
        return "order_first_after_manual_feasibility_check"
    return "order_after_core6_platform_check"


def clinical_exposure_check(row: dict[str, Any]) -> str:
    return (
        "Confirm human Cmax and preferably free Cmax from label/literature; planned top assay concentration should be "
        "justified against clinically reachable exposure or explicitly treated as mechanism-only."
    )


def compound_form_check(row: dict[str, Any]) -> str:
    drug = text(row.get("drug"))
    return (
        f"Confirm exact purchasable form for {drug}: salt/free base, hydrate, stereochemistry, purity, COA, "
        "lot identity, molecular weight used for dosing, and whether the assayed form matches the computational structure."
    )


def platform_assay_check(row: dict[str, Any]) -> str:
    modality = text(row.get("assayModality"))
    target = text(row.get("target"))
    if modality == "biochemical_kinase_or_enzyme_assay":
        return (
            f"Ask platform whether a validated {target} biochemical IC50/Kd or activity assay is available, "
            "including ATP/substrate concentration, reference inhibitor, and selectivity/cytotoxicity follow-up."
        )
    return (
        f"Ask platform whether a validated {target} receptor-function assay is available, including agonist/stimulation "
        "condition, antagonist/agonist mode, receptor expression system, parental-cell control, and cytotoxicity readout."
    )


def model_expression_check(row: dict[str, Any]) -> str:
    direction = text(row.get("direction"))
    target = text(row.get("target"))
    if direction == "oncology":
        return (
            f"Select disease model only after confirming {target} expression and, where relevant, DepMap dependency "
            "or pathway activity in the exact cancer cell line."
        )
    return f"Confirm {target} expression and pathway relevance in the exact cell or tissue model before phenotype testing."


def build_procurement_platform_checklist_rows(execution_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in execution_rows:
        rows.append(
            {
                "checkRank": row["executionRank"],
                "executionBatch": row["executionBatch"],
                "procurementPriority": procurement_priority(row),
                "direction": row["direction"],
                "directionLabelZh": row["directionLabelZh"],
                "drug": row["drug"],
                "target": row["target"],
                "protein": row["protein"],
                "proteinName": row["proteinName"],
                "validationRole": row["validationRole"],
                "validationRoleZh": row["validationRoleZh"],
                "assayModality": row["assayModality"],
                "compoundVendorStatus": "manual_confirm_before_order",
                "compoundFormCheck": compound_form_check(row),
                "purityStorageSolubilityCheck": (
                    "Require COA/purity, storage condition, freeze-thaw guidance, DMSO or aqueous solubility, "
                    "vehicle tolerance in assay, and visible precipitation check across the dose range."
                ),
                "clinicalExposureCheck": clinical_exposure_check(row),
                "platformAssayAvailabilityCheck": platform_assay_check(row),
                "targetExpressionModelCheck": model_expression_check(row),
                "primaryReadout": row["primaryReadout"],
                "orthogonalValidation": row["orthogonalValidation"],
                "positiveControl": row["positiveControl"],
                "negativeControls": row["negativeControls"],
                "counterScreenDetail": row["counterScreenDetail"],
                "expectedDeliverables": (
                    "8-10 point dose-response curve, raw and fitted IC50/EC50 if applicable, cytotoxicity/viability "
                    "counterreadout, positive/negative control performance, orthogonal target-engagement result, and "
                    "a short pass/hold interpretation."
                ),
                "preOrderGoNoGo": (
                    "Order only if compound form is available, assay platform can run the target readout and counterscreen, "
                    "planned concentrations are soluble and interpretable, and the chosen model expresses the target."
                ),
                "mainRiskToResolve": row["mainConfounder"],
                "manualConfirmationStatus": "pending",
                "ownerNotes": "",
            }
        )
    return rows


def build_procurement_platform_checklist_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    def table(block: list[dict[str, Any]]) -> str:
        headers = ["Rank", "Batch", "Drug", "Target", "Priority", "Assay Check", "Main Risk"]
        lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
        for row in block:
            values = [
                row["checkRank"],
                row["executionBatch"],
                row["drug"],
                f"{row['target']} ({row['protein']})",
                row["procurementPriority"],
                compact(row["platformAssayAvailabilityCheck"], 110),
                compact(row["mainRiskToResolve"], 90),
            ]
            lines.append("|" + "|".join(str(value).replace("|", "/") for value in values) + "|")
        return "\n".join(lines)

    return f"""# Wet-Lab Procurement And Platform Checklist

Generated UTC: {summary['createdUtc']}

## Purpose

This checklist is the final pre-spending gate for the first 12 candidates. It is designed for vendor, assay-platform, and PI review before ordering compounds or reserving assay capacity.

## How To Use

- Start with the core 6 and do not purchase the extension set until the core compounds, assays, and counterscreens are feasible.
- Treat every `manualConfirmationStatus` value as pending until compound form, solubility, clinical exposure relevance, assay availability, and model target expression are confirmed.
- Replace any failed core candidate with the highest-ranked backup from `wetlab_pre_purchase_focus_package.csv`.

## First 12 Checklist

{table(rows)}

## Minimum Confirmation Before Order

Each candidate should have a purchasable exact form, documented purity/COA, workable solubility and vehicle, clinically interpretable concentration range, target-specific primary assay, orthogonal target-engagement readout, positive and negative controls, cytotoxicity or assay-interference counterscreen, and confirmed target expression in the planned model.
"""


def build_first_experiment_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    role_counts = Counter(row["validationRole"] for row in rows)
    direction_counts = Counter(row["direction"] for row in rows)
    headers = ["Rank", "Batch", "Direction", "Drug", "Target", "Role", "Assay", "Reason"]
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        values = [
            str(row["firstExperimentRank"]),
            row["firstExperimentBatch"],
            row["directionLabelZh"] or row["direction"],
            row["drug"],
            f"{row['target']} ({row['protein']})",
            row["validationRoleZh"],
            row["assayModality"],
            compact(row["firstExperimentReason"], 120),
        ]
        lines.append("|" + "|".join(str(value).replace("|", "/") for value in values) + "|")

    return f"""# First Experiment Recommended Panel

Generated UTC: {summary['createdUtc']}

## Scope

This is the most concentrated wet-lab starting panel. It is not the largest candidate list. It deliberately mixes positive controls, mechanism-extension hypotheses, and higher-value repurposing candidates so that the first experiment can evaluate both assay calibration and discovery potential.

## Panel Composition

- Rows: {len(rows)}
- Role counts: {dict(role_counts)}
- Disease-direction counts: {dict(direction_counts)}
- Batch rule: run the first 6 if capacity is tight; expand to 12 after compound handling, assay availability, and counterscreens are confirmed.

## Recommended Panel

{chr(10).join(lines)}

## Execution Rule

Do not interpret a disease-cell phenotype alone as validation. Each candidate should pass compound quality, primary target readout, orthogonal target-engagement or pathway readout, cytotoxicity counterscreen, and disease-relevant phenotype before being promoted.
"""


def select_focused_rows(top50: pd.DataFrame, limit: int = 12) -> list[dict[str, Any]]:
    candidates = top50[top50.apply(passes_focus_gate, axis=1)].copy()
    candidates["_focus_score"] = candidates.apply(focus_score, axis=1)
    candidates["_role"] = candidates.apply(validation_role, axis=1)
    candidates = candidates.sort_values(["_focus_score", "expertReviewScore"], ascending=False)

    selected_indices: list[int] = []
    drug_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()

    role_minimums = {
        "positive_control": 1,
        "mechanism_extension": 3,
        "novel_repurposing_candidate": 5,
    }
    for role, required in role_minimums.items():
        for idx, row in candidates[candidates["_role"] == role].iterrows():
            if len(selected_indices) >= limit or role_counts[role] >= required:
                break
            drug = text(row.get("drug")).lower()
            target = text(row.get("target"))
            if drug_counts[drug] >= 1 or target_counts[target] >= 2:
                continue
            selected_indices.append(idx)
            drug_counts[drug] += 1
            target_counts[target] += 1
            role_counts[role] += 1

    for idx, row in candidates.iterrows():
        if len(selected_indices) >= limit:
            break
        if idx in selected_indices:
            continue
        drug = text(row.get("drug")).lower()
        target = text(row.get("target"))
        if drug_counts[drug] >= 1 or target_counts[target] >= 2:
            continue
        selected_indices.append(idx)
        drug_counts[drug] += 1
        target_counts[target] += 1
        role_counts[text(row.get("_role"))] += 1

    focused = candidates.loc[selected_indices].sort_values("_focus_score", ascending=False)
    rows: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(focused.iterrows(), start=1):
        normalized = normalized_row(row, rank, "focused_top12")
        normalized["focusRank"] = rank
        normalized["focusTier"] = "F1_core_6" if rank <= 6 else "F2_expansion_12"
        normalized["focusScore"] = f"{float(row['_focus_score']):.4f}"
        normalized["focusReason"] = focus_reason(row)
        normalized["preExperimentManualChecks"] = (
            "Confirm literature novelty, exact salt/form, vendor availability, solubility, clinical exposure/Cmax relevance, "
            "target expression in planned disease model, and assay/counterscreen availability."
        )
        return_cols = normalized
        rows.append(return_cols)
    return rows


def normalized_row(row: pd.Series, rank: int, source: str) -> dict[str, Any]:
    role = validation_role(row)
    assay = assay_plan(row)
    score = number(row.get("expertReviewScore")) or number(row.get("validationScore"))
    payload: dict[str, Any] = {
        "wetlabRank": rank,
        "wetlabPriority": wetlab_priority(row, source),
        "validationRole": role,
        "validationRoleZh": role_zh(role),
        "sourcePanel": source,
        "direction": text(row.get("direction")),
        "directionLabelZh": text(row.get("directionLabelZhFinal")),
        "drugId": text(row.get("drugId")),
        "drug": text(row.get("drug")),
        "target": text(row.get("target")),
        "protein": text(row.get("protein")),
        "proteinName": text(row.get("proteinName")),
        "knownDrugTargetPair": int(truthy(row.get("knownDrugTargetPair"))),
        "noveltyClass": text(row.get("noveltyClass")) or text(row.get("auditNoveltyClass")),
        "reviewTrack": text(row.get("reviewTrack")),
        "score": fmt_num(score, 4),
        "affinityScore": fmt_num(row.get("affinityScore"), 4),
        "openTargetsScore": fmt_num(row.get("openTargetsScore"), 4),
        "txgnnScore": fmt_num(row.get("integratedTxgnnScore"), 4),
        "kgEvidenceScore": fmt_num(row.get("kgEvidenceScore"), 4),
        "cmapReversalTier": text(row.get("cmapReversalTier")),
        "cmapBestRawReversal": fmt_num(row.get("cmapBestRawReversal"), 4),
        "pathwayDiseaseContextTier": text(row.get("pathwayDiseaseContextTier")),
        "gtexContextTier": text(row.get("gtexContextTier")),
        "tissueContextTier": text(row.get("tissueContextTier")),
        "depmapDependencyTier": text(row.get("depmapDependencyTier")),
        "admetTier": text(row.get("admetTier")) or text(row.get("sotaMlAdmetTier")),
        "mlAdmetRiskFlags": text(row.get("mlAdmetRiskFlags")),
        "structureConfidenceTier": text(row.get("structureConfidenceTier")),
        "standardPoseValidationTier": text(row.get("standardPoseValidationTier")),
        "poseAuditStatus": text(row.get("poseAuditStatus")),
        "diffdock": fmt_num(row.get("diffdock"), 4),
        "assayModality": text(row.get("assayModality")),
        "primaryAssay": assay["primaryAssay"],
        "orthogonalAssay": assay["orthogonalAssay"],
        "counterScreen": assay["counterScreen"],
        "diseaseModelRecommendation": assay["diseaseModelRecommendation"],
        "concentrationPlan": concentration_plan(row),
        "advanceCriteria": gating_decision(row),
        "stopCriteria": stop_criteria(row),
        "evidenceSupportCount": evidence_count(row),
        "rationaleZh": compact(row.get("expertRationaleZh") or row.get("validationSummary") or row.get("discussionSummary"), 260),
        "evidenceSummaryZh": compact(row.get("evidenceSummaryZh"), 260),
        "reviewGapsZh": compact(row.get("expertReviewGapsZh"), 220),
        "confidenceSdfPath": text(row.get("confidenceSdfPath")),
        "receptorPdbPath": text(row.get("receptorPdbPath")),
    }
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def direction_top10_rows(balanced: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    direction_order = [
        "oncology",
        "infectious_disease",
        "cardiovascular",
        "neurology_psychiatry",
        "immunology_inflammation",
    ]
    rank_columns = ["validationRankWithinDirection", "validationRankGlobal"]
    for direction in direction_order:
        block = balanced[balanced["direction"] == direction].copy()
        if block.empty:
            continue
        sort_cols = [col for col in rank_columns if col in block.columns]
        if sort_cols:
            block = block.sort_values(sort_cols)
        block["_pair"] = block["drug"].astype(str).str.lower().str.strip() + "|" + block["target"].astype(str)
        block = block.drop_duplicates("_pair", keep="first").head(10)
        for idx, (_, row) in enumerate(block.iterrows(), start=1):
            rows.append(normalized_row(row, idx, f"direction_top10_{direction}"))
    return rows


def markdown_table(rows: list[dict[str, Any]], limit: int = 24) -> str:
    headers = ["Rank", "Direction", "Drug", "Target", "Role", "Assay", "Why this is testable"]
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows[:limit]:
        values = [
            str(row["wetlabRank"]),
            row["directionLabelZh"] or row["direction"],
            row["drug"],
            f"{row['target']} ({row['protein']})",
            row["validationRoleZh"],
            row["assayModality"],
            compact(row["rationaleZh"] or row["evidenceSummaryZh"], 120),
        ]
        escaped = [str(value).replace("|", "/") for value in values]
        lines.append("|" + "|".join(escaped) + "|")
    return "\n".join(lines)


def build_focus_markdown(summary: dict[str, Any], focused_rows: list[dict[str, Any]]) -> str:
    table = markdown_table(focused_rows, 12)
    return f"""# Pre-Experiment Focusing Strategy

Generated UTC: {summary['createdUtc']}

## Recommendation

Before starting wet-lab work, reduce the candidate set to a decision-grade panel rather than testing the full computational Top list. The current focused panel contains {len(focused_rows)} candidates:

- Core 6: run first if budget or assay capacity is tight.
- Expansion 12: add after assays, compound handling, and counterscreens are ready.

## Focusing Funnel

1. Remove technical and translational holds: ADMET D, contraindication flags, structure D/missing when binding interpretation is required, poor assay feasibility, and unavailable compounds.
2. Require multi-evidence agreement: affinity/model support plus disease evidence, pathway/CREEDS, CMap/LINCS reversal, tissue or DepMap context, and interpretable structure.
3. Remove redundant hypotheses: keep only one salt/form, avoid too many candidates from the same drug, target, or pathway unless used deliberately as controls.
4. Separate validation roles: keep a small number of positive controls, several mechanism-extension candidates, and the strongest novel repurposing candidates.
5. Match the experiment to the biology: do target engagement first, then disease-cell phenotype, then counterscreens and safety readouts.

## Focused Candidates

{table}

## Why Not Simply Use The Highest Raw Score

Raw rank is useful for screening, but wet-lab value depends on whether the result can be tested cleanly. A candidate with a slightly lower computational score can be more valuable if it has a clear assay, a disease-relevant model, CMap reversal, acceptable ADMET, interpretable structure, and a non-redundant mechanism.

## Manual Checks Before Purchase

- PubMed and ClinicalTrials novelty check for the exact drug-target-disease hypothesis.
- Confirm compound salt form, purity, storage, solubility, and vehicle compatibility.
- Check clinically achievable free concentration against the planned assay concentration range.
- Confirm target expression and disease relevance in the exact planned cell line or model.
- Select one primary assay, one orthogonal target-engagement assay, and at least one counterscreen.
"""


def build_markdown(
    root: Path,
    summary: dict[str, Any],
    wave1_rows: list[dict[str, Any]],
    direction_rows: list[dict[str, Any]],
    focused_rows: list[dict[str, Any]],
    pre_purchase_focus_rows: list[dict[str, Any]],
) -> str:
    progress = read_json(root / "outputs/report_scale/diffdock_full_run/diffdock_full_progress_summary.json")
    full_jobs = f"{progress.get('completedJobs', 'NA')} / {progress.get('totalJobs', 'NA')}"
    full_pct = progress.get("scoredRowPct", "NA")
    eta = progress.get("estimatedFinishUtc", "NA")
    return f"""# Wet-Lab Validation Recommendation Package

Generated UTC: {summary['createdUtc']}

## Purpose

This package converts the current BioMaster multi-evidence ranking into an experimental validation queue. It should not be interpreted as proof of efficacy. The immediate goal is to test whether the computational pipeline enriches drug-target-disease hypotheses that show target engagement, directional disease biology, acceptable safety context, and interpretable structural support.

## Current Candidate Space

- Full experimental-validation candidate rows: {summary['candidateRows']}
- Experiment-ready rows: {summary['experimentReadyRows']}
- Expert-review-ready rows: {summary['reviewReadyRows']}
- Experiment or review ready rows: {summary['experimentOrReviewReadyRows']}
- Balanced shortlist rows: {summary['balancedShortlistRows']}
- Integrated expert Top50 rows: {summary['expertTop50Rows']}
- Recommended Wave1 rows: {summary['wave1Rows']}
- Disease directions represented in Wave1: {summary['wave1DirectionCounts']}

## Recommended First Wet-Lab Design

Use the focused 6-12 candidate panel if the goal is maximum value per experiment. Use the 24-candidate Wave1 panel if the goal is broader disease-direction coverage. If budget allows, expand to 48 candidates by taking the remaining highest-ranked records from the integrated expert Top50 and preserving disease-direction balance.

The first batch should mix three roles:

- Positive controls / known mechanisms: calibrate assay sensitivity and confirm that known biology can be recovered.
- Mechanism-extension candidates: test whether a known drug or drug family extends to a nearby target, pathway, or disease context.
- Novel repurposing candidates: test high-value hypotheses only after target engagement and counterscreens are built into the assay plan.

## Hard Exclusion Rules Before Ordering Compounds

Exclude or hold candidates with any of the following unless a domain expert explicitly overrides the decision:

- `hardFlags` other than `none`.
- ADMET tier D, severe ML-ADMET risk flags, strong contraindication, or unacceptable drug-drug interaction concern.
- Missing structural output if the planned experiment requires a binding pose or pocket interpretation.
- No feasible target assay, unavailable compound, poor solubility, or expected active concentration outside a realistic exposure window.
- Disease direction unsupported by tissue, pathway, CMap/CREEDS, or literature review.

## Assay Strategy

The experiment should be staged rather than run as a single broad viability screen:

1. Confirm compound quality, solubility, and usable concentration range.
2. Run the target-specific primary assay listed in the CSV package.
3. Run an orthogonal target-engagement or pathway assay.
4. Test a disease-relevant cellular phenotype only after target signal is measurable.
5. Apply counterscreens for cytotoxicity, assay interference, unrelated receptors/enzymes, or parental cells.
6. Advance only candidates with dose response, non-toxic activity, and agreement between target signal and disease phenotype.

## Current Full DiffDock State

Full DiffDock remains a running computation and is not yet the final closed structural layer:

- Main queue jobs: {full_jobs}
- Scored row percentage: {full_pct}%
- Estimated finish UTC: {eta}

Rows marked as missing output are technical DiffDock output failures, not biological negative calls. Final wet-lab selection should be refreshed after the full queue, ligand rescue, final merge, and structure-facing re-audits complete.

## Wave1 Candidate Panel

{markdown_table(wave1_rows, 24)}

## Most Focused Pre-Experiment Panel

{markdown_table(focused_rows, 12)}

## Final Pre-Purchase Core 6

This is the most concentrated first experiment set after applying practical purchase, assay, redundancy, ADMET, disease-context, and structural-readiness gates.

{compact_focus_table(pre_purchase_focus_rows, 'batch_1_core_6', 6)}

## Disease-Direction Top10 Panel

The file `wetlab_direction_top10.csv` provides up to ten candidates per disease direction from the balanced validation shortlist. This is the preferred source when the experimental team wants equal disease-area coverage rather than a purely global Top50 queue.

{markdown_table(direction_rows, 50)}

## Output Files

- `wetlab_wave1_24.csv`: first recommended wet-lab batch.
- `wetlab_expert_top50.csv`: expert-review expansion pool.
- `wetlab_direction_top10.csv`: disease-balanced Top10 per direction.
- `wetlab_purchase_and_assay_queue.csv`: combined ordering and assay-planning queue.
- `wetlab_pre_purchase_focus_package.csv`: final focused package before compound purchasing.
- `wetlab_platform_procurement_checklist_12.csv`: vendor/platform confirmation sheet for the first 12 candidates.
- `wetlab_validation_summary.json`: machine-readable counts and provenance.

## Decision Rule

A candidate should move from computational recommendation to biological claim only if it passes target engagement, orthogonal validation, disease-relevant phenotype, concentration-response, and safety/counterscreen checks. A high computational rank alone is not sufficient for a wet-lab conclusion.
"""


def build(root: Path) -> dict[str, Any]:
    out_dir = root / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    wave1 = load_table(root, "final_priority_integrated_expert_review_wave1_24.csv")
    top50 = load_table(root, "final_priority_integrated_expert_review_panel_top50.csv")
    balanced = load_table(root, "final_priority_experimental_validation_balanced_shortlist.csv")
    panel_summary = read_json(root / PRIORITY_DIR / "final_priority_experimental_validation_panel_summary.json")
    expert_summary = read_json(root / PRIORITY_DIR / "final_priority_integrated_expert_review_panel_summary.json")

    wave1_rows = [normalized_row(row, int(row.get("expertReviewRank", idx)), "wave1_24") for idx, (_, row) in enumerate(wave1.iterrows(), start=1)]
    top50_rows = [normalized_row(row, int(row.get("expertReviewRank", idx)), "expert_top50") for idx, (_, row) in enumerate(top50.iterrows(), start=1)]
    direction_rows = direction_top10_rows(balanced)
    focused_rows = select_focused_rows(top50, limit=12)

    seen: set[tuple[str, str, str]] = set()
    purchase_rows: list[dict[str, Any]] = []
    for row in focused_rows + wave1_rows + top50_rows + direction_rows:
        key = (row["drug"].lower(), row["target"], row["direction"])
        if key in seen:
            continue
        seen.add(key)
        purchase_rows.append({**row, "purchaseCheck": "verify vendor, salt form, purity, solubility, storage, and assay-compatible vehicle"})
    decision_rows = build_pre_experiment_decision_rows(purchase_rows)
    first_experiment_rows = select_first_experiment_panel(decision_rows, limit=12)
    pre_purchase_focus_rows = build_pre_purchase_focus_rows(decision_rows, first_experiment_rows, backup_limit=24)
    experiment_execution_rows = build_experiment_execution_protocol_rows(first_experiment_rows)
    procurement_platform_rows = build_procurement_platform_checklist_rows(experiment_execution_rows)

    write_csv(out_dir / "wetlab_wave1_24.csv", wave1_rows)
    write_csv(out_dir / "wetlab_expert_top50.csv", top50_rows)
    write_csv(out_dir / "wetlab_direction_top10.csv", direction_rows)
    write_csv(out_dir / "wetlab_focused_top12.csv", focused_rows)
    write_csv(out_dir / "wetlab_purchase_and_assay_queue.csv", purchase_rows)
    write_csv(out_dir / "wetlab_pre_experiment_decision_matrix.csv", decision_rows)
    write_csv(out_dir / "wetlab_first_experiment_panel_12.csv", first_experiment_rows)
    write_csv(out_dir / "wetlab_pre_purchase_focus_package.csv", pre_purchase_focus_rows)
    write_csv(out_dir / "wetlab_experiment_execution_protocol_12.csv", experiment_execution_rows)
    write_csv(out_dir / "wetlab_platform_procurement_checklist_12.csv", procurement_platform_rows)

    role_counts = Counter(row["validationRole"] for row in wave1_rows)
    direction_counts = Counter(row["direction"] for row in wave1_rows)
    assay_counts = Counter(row["assayModality"] for row in wave1_rows)
    decision_counts = Counter(row["decision"] for row in decision_rows)
    pre_purchase_tier_counts = Counter(row["prePurchaseTier"] for row in pre_purchase_focus_rows)
    pre_purchase_action_counts = Counter(row["purchaseAction"] for row in pre_purchase_focus_rows)
    summary = {
        "createdUtc": now_utc(),
        "candidateRows": int(panel_summary.get("candidateRows", 0)),
        "experimentReadyRows": int(panel_summary.get("experimentReadyRows", 0)),
        "reviewReadyRows": int(panel_summary.get("reviewReadyRows", 0)),
        "experimentOrReviewReadyRows": int(panel_summary.get("experimentOrReviewReadyRows", 0)),
        "balancedShortlistRows": int(panel_summary.get("balancedPanelRows", len(balanced))),
        "expertTop50Rows": int(expert_summary.get("panelRows", len(top50_rows))),
        "wave1Rows": int(expert_summary.get("wave1Rows", len(wave1_rows))),
        "wave1RoleCounts": dict(role_counts),
        "wave1DirectionCounts": dict(direction_counts),
        "wave1AssayCounts": dict(assay_counts),
        "directionTop10Rows": len(direction_rows),
        "focusedTop12Rows": len(focused_rows),
        "focusedCore6Rows": min(6, len(focused_rows)),
        "purchaseAndAssayQueueRows": len(purchase_rows),
        "preExperimentDecisionRows": len(decision_rows),
        "preExperimentDecisionCounts": dict(decision_counts),
        "firstExperimentPanelRows": len(first_experiment_rows),
        "firstExperimentPanelRoleCounts": dict(Counter(row["validationRole"] for row in first_experiment_rows)),
        "firstExperimentPanelDirectionCounts": dict(Counter(row["direction"] for row in first_experiment_rows)),
        "prePurchaseFocusRows": len(pre_purchase_focus_rows),
        "prePurchaseTierCounts": dict(pre_purchase_tier_counts),
        "prePurchaseActionCounts": dict(pre_purchase_action_counts),
        "experimentExecutionProtocolRows": len(experiment_execution_rows),
        "experimentExecutionProtocolCoreRows": sum(1 for row in experiment_execution_rows if row["executionBatch"] == "batch_1_core_6"),
        "experimentExecutionProtocolExtensionRows": sum(1 for row in experiment_execution_rows if row["executionBatch"] == "batch_1_extension_12"),
        "experimentExecutionProtocolAssayCounts": dict(Counter(row["assayModality"] for row in experiment_execution_rows)),
        "procurementPlatformChecklistRows": len(procurement_platform_rows),
        "procurementPlatformChecklistCoreRows": sum(1 for row in procurement_platform_rows if row["executionBatch"] == "batch_1_core_6"),
        "procurementPlatformChecklistExtensionRows": sum(1 for row in procurement_platform_rows if row["executionBatch"] == "batch_1_extension_12"),
        "procurementPlatformChecklistStatusCounts": dict(Counter(row["manualConfirmationStatus"] for row in procurement_platform_rows)),
        "sourceFiles": {
            "wave1": str(PRIORITY_DIR / "final_priority_integrated_expert_review_wave1_24.csv"),
            "top50": str(PRIORITY_DIR / "final_priority_integrated_expert_review_panel_top50.csv"),
            "balancedShortlist": str(PRIORITY_DIR / "final_priority_experimental_validation_balanced_shortlist.csv"),
            "panelSummary": str(PRIORITY_DIR / "final_priority_experimental_validation_panel_summary.json"),
            "expertSummary": str(PRIORITY_DIR / "final_priority_integrated_expert_review_panel_summary.json"),
        },
        "outputs": {
            "wave1": str(OUT_DIR / "wetlab_wave1_24.csv"),
            "top50": str(OUT_DIR / "wetlab_expert_top50.csv"),
            "directionTop10": str(OUT_DIR / "wetlab_direction_top10.csv"),
            "focusedTop12": str(OUT_DIR / "wetlab_focused_top12.csv"),
            "purchaseAndAssayQueue": str(OUT_DIR / "wetlab_purchase_and_assay_queue.csv"),
            "preExperimentDecisionMatrix": str(OUT_DIR / "wetlab_pre_experiment_decision_matrix.csv"),
            "firstExperimentPanel": str(OUT_DIR / "wetlab_first_experiment_panel_12.csv"),
            "prePurchaseFocusPackage": str(OUT_DIR / "wetlab_pre_purchase_focus_package.csv"),
            "experimentExecutionProtocol": str(OUT_DIR / "wetlab_experiment_execution_protocol_12.csv"),
            "procurementPlatformChecklist": str(OUT_DIR / "wetlab_platform_procurement_checklist_12.csv"),
            "markdown": str(OUT_DIR / "WETLAB_VALIDATION_RECOMMENDATION.md"),
            "focusMarkdown": str(OUT_DIR / "WETLAB_PRE_EXPERIMENT_FOCUSING_STRATEGY.md"),
            "decisionMarkdown": str(OUT_DIR / "WETLAB_PRE_EXPERIMENT_DECISION_MATRIX.md"),
            "firstExperimentMarkdown": str(OUT_DIR / "WETLAB_FIRST_EXPERIMENT_PANEL.md"),
            "prePurchaseFocusMarkdown": str(OUT_DIR / "WETLAB_PRE_PURCHASE_FOCUS_PACKAGE.md"),
            "experimentExecutionMarkdown": str(OUT_DIR / "WETLAB_EXPERIMENT_EXECUTION_PROTOCOL.md"),
            "procurementPlatformChecklistMarkdown": str(OUT_DIR / "WETLAB_PLATFORM_PROCUREMENT_CHECKLIST.md"),
            "summaryJson": str(OUT_DIR / "wetlab_validation_summary.json"),
        },
        "methodNote": (
            "Wet-lab priority is a validation-planning label. It integrates current expert-review rank, "
            "novelty role, disease evidence, pathway/CMap/tissue context, ADMET, and structure readiness. "
            "It is not a claim of biological efficacy."
        ),
    }
    write_json(out_dir / "wetlab_validation_summary.json", summary)
    (out_dir / "WETLAB_VALIDATION_RECOMMENDATION.md").write_text(
        build_markdown(root, summary, wave1_rows, direction_rows, focused_rows, pre_purchase_focus_rows),
        encoding="utf-8",
    )
    (out_dir / "WETLAB_PRE_EXPERIMENT_FOCUSING_STRATEGY.md").write_text(
        build_focus_markdown(summary, focused_rows),
        encoding="utf-8",
    )
    (out_dir / "WETLAB_PRE_EXPERIMENT_DECISION_MATRIX.md").write_text(
        build_pre_experiment_markdown(summary, decision_rows),
        encoding="utf-8",
    )
    (out_dir / "WETLAB_FIRST_EXPERIMENT_PANEL.md").write_text(
        build_first_experiment_markdown(summary, first_experiment_rows),
        encoding="utf-8",
    )
    (out_dir / "WETLAB_PRE_PURCHASE_FOCUS_PACKAGE.md").write_text(
        build_pre_purchase_focus_markdown(summary, pre_purchase_focus_rows),
        encoding="utf-8",
    )
    (out_dir / "WETLAB_EXPERIMENT_EXECUTION_PROTOCOL.md").write_text(
        build_experiment_execution_protocol_markdown(summary, experiment_execution_rows),
        encoding="utf-8",
    )
    (out_dir / "WETLAB_PLATFORM_PROCUREMENT_CHECKLIST.md").write_text(
        build_procurement_platform_checklist_markdown(summary, procurement_platform_rows),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    summary = build(args.root.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
