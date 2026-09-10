"""Read the frozen SPR64 design without implying experiment completion/release."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .explorer_annotations import _number, _rows, _text


DESIGN_DIR = "outputs/retargetmap_spr64_design_20260909"
MASTER_PATH = f"{DESIGN_DIR}/INTERNAL_MASTER_512.csv"
ROSTER_PATH = f"{DESIGN_DIR}/TARGET_ROSTER_64.csv"
CONTROL_PATH = f"{DESIGN_DIR}/CONTROL_EVIDENCE_64.csv"
SUMMARY_PATH = f"{DESIGN_DIR}/SUMMARY.json"
ROLE_LABELS = {
    "OUR_FROZEN_MODEL_HIGH": "模型高排名候选",
    "OUR_FROZEN_MODEL_INTERMEDIATE": "模型中排名候选",
    "OUR_FROZEN_MODEL_LOW_BACKGROUND": "模型低排名背景",
    "POSITIVE_CONTROL": "靶点质控参考对照",
}


def _truth(value: Any) -> bool | None:
    normalized = _text(value).lower()
    return True if normalized == "true" else False if normalized == "false" else None


def load_experiments(root: Path, drugs: dict, targets: dict) -> dict[str, Any]:
    """Attach exact-identity design rows; off-catalog controls remain target-only.

    Rank fields describe the frozen design's source versions and denominators,
    not the current explorer model. Reference pChEMBL values retain endpoints.
    """
    root = Path(root)
    from .explorer_comprehensive_spr import DESIGN, load_comprehensive
    if (root / DESIGN).exists():
        return load_comprehensive(root, drugs, targets)
    result = {
        "drugs": {key: {"experiments": []} for key in drugs},
        "targets": {key: {"experiments": []} for key in targets},
        "sources": [{"name": name, "path": relative, "available": (root / relative).is_file()}
                    for name, relative in [
                        ("SPR64 / 512 internal design · 2026-09-09", MASTER_PATH),
                        ("SPR64 target constructs and release roster", ROSTER_PATH),
                        ("SPR64 reference-control assay evidence", CONTROL_PATH),
                        ("SPR64 design summary", SUMMARY_PATH),
                    ]],
        "counts": {},
    }
    roster = {row["target_chembl_id"]: row for row in _rows(root / ROSTER_PATH)}
    controls = {row["pair_id"]: row for row in _rows(root / CONTROL_PATH)}
    rows = list(_rows(root / MASTER_PATH))
    role_counts: Counter = Counter()
    release_counts: Counter = Counter()
    seen_experiments: set[str] = set()
    outside_drugs: set[str] = set()
    outside_targets: set[str] = set()
    for row in rows:
        experiment_id = _text(row.get("experiment_id"))
        if not experiment_id or experiment_id in seen_experiments:
            raise ValueError("SPR design experiment_id must be present and unique")
        seen_experiments.add(experiment_id)
        # Never use drug_names, ChEMBL parent IDs, or connectivity-only InChIKeys here.
        drug_id, target_id = _text(row.get("ligand_inchikey")), _text(row.get("target_chembl_id"))
        pair_id = _text(row.get("pair_id") or row.get("pairId"))
        target_context = roster.get(target_id, {})
        control = controls.get(pair_id, {})
        merged = {**target_context, **control, **{k: v for k, v in row.items() if _text(v)}}
        role = _text(row.get("selection_role"))
        is_control = role == "POSITIVE_CONTROL" or row.get("row_type") == "REFERENCE_CONTROL"
        release_status = _text(row.get("release_status") or target_context.get("release_status")) or "RELEASE_STATUS_UNAVAILABLE"
        role_counts[role] += 1
        release_counts[release_status] += 1
        if drug_id not in drugs:
            outside_drugs.add(drug_id)
        if target_id not in targets:
            outside_targets.add(target_id)
        item = {
            "id": experiment_id, "name": f"{row.get('drug_names', drug_id)} → {row.get('gene_symbol', target_id)}",
            "experiment_id": experiment_id, "pair_id": pair_id, "design_id": "SPR64_512_20260909",
            "design_date": "2026-09-09", "source": "BioMaster SPR64 / 512 design · 2026-09-09",
            "source_path": MASTER_PATH, "drug_id": drug_id, "target_id": target_id,
            "drug_name": _text(row.get("drug_names")), "gene_symbol": _text(row.get("gene_symbol")),
            "drug_in_catalog": drug_id in drugs, "target_in_catalog": target_id in targets,
            "role": role, "role_label": ROLE_LABELS.get(role, role), "is_control": is_control,
            "experiment_arm": _text(row.get("experiment_arm")), "row_type": _text(row.get("row_type")),
            "logistics_batch": _number(row.get("logistics_batch")),
            "release_status": release_status,
            "status_label": "设计审核中 · 尚未放行" if "NOT_RELEASED" in release_status else release_status,
            "result_status": "NO_EXPERIMENTAL_RESULT_IN_DESIGN_ARTIFACT",
            "evidence": "计算实验设计及参考证据；不代表已完成实验、实测结合或预期命中率",
            "construct_recommendation": _text(merged.get("construct_recommendation")),
            "construct_status": _text(target_context.get("construct_status")),
            "special_system_review": _truth(target_context.get("special_system_review")),
            "endpoint_plan": _text(row.get("endpoint_plan")),
            "pair_review_status": _text(row.get("pair_review_status")),
            "novelty_status": _text(row.get("novelty_status")),
            "reference_control": _text(target_context.get("reference_control")),
            "control_status": _text(merged.get("control_status")),
            "rank_context": "冻结实验设计原始分数版本；不等于当前浏览器模型的排序",
            "ranks": {key: _number(row.get(key)) for key in [
                "retargetmap_rank_384", "dtiam_rank_384", "routed_rank_within_drug",
                "routed_candidate_target_count", "aux_rank_drug_within_target_720",
            ] if _number(row.get(key)) is not None},
            "review": {key: _text(row.get(key)) for key in [
                "support_note", "positive_reference_name", "positive_reference_chembl", "positive_reference_endpoints",
                "negative_reference_name", "negative_reference_chembl", "negative_reference_endpoints",
                "discovery_evidence_tier", "current_score_status", "scoring_run_label",
            ] if _text(row.get(key))},
            "exclusion_audit": {key: _truth(row.get(key)) for key in [
                "local_chembl_pair_found", "kirhub_pair_found", "known_moa_target_collision",
                "exact_pair_in_full_fit_training", "bindingdb_exact_pair_found", "gtopdb_exact_pair_found",
            ] if _truth(row.get(key)) is not None},
        }
        if is_control:
            item["control_evidence"] = {
                "chembl_id": _text(merged.get("control_chembl_id")),
                "category": _text(merged.get("control_evidence_category")),
                "standard_types": _text(merged.get("control_standard_types")),
                "mean_pchembl_mixed_endpoints": _number(merged.get("control_mean_pchembl_mixed_endpoints")),
                "min_pchembl": _number(merged.get("control_min_pchembl")),
                "max_pchembl": _number(merged.get("control_max_pchembl")),
                "assay_ids": _text(merged.get("control_assay_ids")),
                "document_ids": _text(merged.get("control_doc_ids")),
                "external_reference": _text(merged.get("control_external_reference")),
                "local_label": _text(merged.get("control_local_label")),
                "note": _text(merged.get("reference_note")) or "不同端点的 pChEMBL 不能表述为 SPR Kd",
                "source": "ChEMBL 37 original assay/document references · SPR design control audit",
                "source_path": CONTROL_PATH,
            }
        if drug_id in result["drugs"]:
            result["drugs"][drug_id]["experiments"].append(item)
        if target_id in result["targets"]:
            result["targets"][target_id]["experiments"].append(item)

    result["counts"] = {
        "spr_design_pairs": len(rows), "spr_candidate_pairs": sum(value for role, value in role_counts.items() if role != "POSITIVE_CONTROL"),
        "spr_control_pairs": role_counts["POSITIVE_CONTROL"],
        "spr_design_targets": len({_text(row.get("target_chembl_id")) for row in rows}),
        "spr_catalog_drugs": sum(bool(x["experiments"]) for x in result["drugs"].values()),
        "spr_catalog_targets": sum(bool(x["experiments"]) for x in result["targets"].values()),
        "spr_off_catalog_compounds": len(outside_drugs), "spr_off_catalog_targets": len(outside_targets),
        "spr_not_released_pairs": sum(count for status, count in release_counts.items() if "NOT_RELEASED" in status),
    }
    if (root / SUMMARY_PATH).is_file():
        with (root / SUMMARY_PATH).open(encoding="utf-8") as handle:
            summary = json.load(handle)
        result["summary"] = {key: summary[key] for key in ["status", "utc", "remaining", "interpretation",
            "high_candidates_are_not_certified_binders", "control_evidence", "control_local_conflict_exceptions"] if key in summary}
    return result
