#!/usr/bin/env python3
"""Validate the single current BioMaster project contract against audit JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "configs/biomaster_current_contract_v1.json"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def metric_by_model(summary: dict[str, Any], section: str, model: str) -> dict[str, Any]:
    metrics = summary[section]["metrics"]
    matches = [row for row in metrics if row["model"] == model]
    require(len(matches) == 1, f"Expected one {model} row in {section}, found {len(matches)}")
    return matches[0]


def validate_contract_only(contract: dict[str, Any]) -> list[str]:
    checks: list[str] = []
    universes = contract["universes"]
    require(
        universes["current_scored_core_pairs"]
        == universes["old_drugs"] * universes["current_scored_core_targets"],
        "720 x 384 pair count is inconsistent",
    )
    checks.append("current_scored_core_arithmetic")
    require(
        universes["next_primary_pairs"]
        == universes["old_drugs"] * universes["next_primary_targets"],
        "720 x 450 pair count is inconsistent",
    )
    checks.append("next_primary_arithmetic")
    require(
        universes["current_scored_core_targets"] + universes["pending_primary_targets"]
        == universes["next_primary_targets"],
        "384 + 66 must equal 450",
    )
    checks.append("primary_target_extension_arithmetic")
    require(
        universes["next_primary_targets"]
        + universes["special_direct_small_molecule_targets"]
        + universes["exploratory_assayable_no_direct_small_molecule_targets"]
        == universes["active_routed_design_targets"],
        "Active routed target count is inconsistent",
    )
    checks.append("active_route_arithmetic")
    require(
        universes["active_routed_design_targets"] + universes["registry_only_targets"]
        == universes["complete_non_gpcr_registry_targets"],
        "Complete registry count is inconsistent",
    )
    checks.append("registry_arithmetic")
    require(contract["task"]["primary_direction"] == "old_drug_to_target", "Primary direction drifted")
    require(contract["task"]["unknown_pair_policy"] == "unknown_not_negative", "Unknown policy drifted")
    require(contract["data"]["kirhub"]["affinity_measurement"] is False, "KIRHub cannot be affinity")
    require(
        contract["evaluation"]["kirhub_strict_retrospective"]["not_rank_over_384"] is True,
        "Restricted KIRHub Recall@K cannot be described as rank/384",
    )
    for role, path_text in contract["current_entrypoints"].items():
        require(resolve(path_text).is_file(), f"Missing current_entrypoints {role}: {path_text}")
    checks.extend(
        [
            "primary_direction",
            "unknown_policy",
            "kirhub_endpoint_boundary",
            "current_entrypoints",
        ]
    )
    return checks


def validate_artifacts(contract: dict[str, Any]) -> list[str]:
    checks: list[str] = []
    training = contract["data"]["comprehensive_training"]
    training_manifest = read_json(resolve(training["manifest"]))
    require(training_manifest["status"] == "PASS", "Comprehensive training manifest is not PASS")
    require(training_manifest["protocol"] == training["protocol"], "Training protocol mismatch")
    checks.append("training_manifest")

    full_fit = read_json(resolve(training["full_fit_audit"]))
    split = full_fit["split_audit"]
    require(split["full_fit_rows"] == training["feature_resolved_relations"], "Training relation count mismatch")
    require(split["full_fit_unique_drugs"] == training["unique_drugs"], "Training drug count mismatch")
    require(split["full_fit_unique_targets"] == training["unique_targets"], "Training target count mismatch")
    checks.append("full_fit_training_counts")

    model = contract["model_roles"]["directional_full_fit_neural_head"]
    scoring = read_json(resolve(model["score_summary"]))
    universes = contract["universes"]
    require(scoring["rows"] == universes["current_scored_core_pairs"], "Scoring row count mismatch")
    require(scoring["drug_queries"] == universes["old_drugs"], "Scoring drug count mismatch")
    require(scoring["candidate_targets"] == universes["current_scored_core_targets"], "Scoring target count mismatch")
    require(len(scoring["checkpoints"]) == 3, "Expected three FULL_FIT checkpoints")
    checks.append("full_fit_scoring_contract")

    target_scope_path = ROOT / "outputs/target_discovery_scope_ch37_v3/TARGET_DISCOVERY_SCOPE_SUMMARY_V3.json"
    target_scope = read_json(target_scope_path)
    target_counts = target_scope["counts"]
    expected_target_counts = {
        "non_gpcr_registry": universes["complete_non_gpcr_registry_targets"],
        "primary_direct_sm_assayable": universes["next_primary_targets"],
        "special_direct_sm": universes["special_direct_small_molecule_targets"],
        "exploratory_assayable_no_direct_sm": universes[
            "exploratory_assayable_no_direct_small_molecule_targets"
        ],
        "active_routed_design_space": universes["active_routed_design_targets"],
        "registry_only_special_no_direct_sm": universes["registry_only_targets"],
        "historical_scored_core": universes["current_scored_core_targets"],
        "new_primary_targets_beyond_historical_384": universes["pending_primary_targets"],
    }
    for key, expected in expected_target_counts.items():
        require(target_counts[key] == expected, f"Target scope mismatch for {key}")
    checks.append("target_routing_counts")

    rank_summary_path = contract["model_roles"]["general_primary_system"]["summary"]
    rank_summary = read_json(resolve(rank_summary_path))
    s5_contract = contract["evaluation"]["primary_internal"]
    s5_model = metric_by_model(rank_summary, "s5_test", "INDEPENDENT_VALIDATION_RANK")
    s5_dtiam = metric_by_model(rank_summary, "s5_test", "DTIAM_RAW")
    require(s5_model["rows"] == s5_contract["rows"], "S5 row count mismatch")
    require(s5_model["drug_groups_with_both_classes"] == s5_contract["two_class_drug_queries"], "S5 query count mismatch")
    require(s5_model["drug_macro_auprc"] == s5_contract["independent_head_drug_macro_auprc"], "S5 BioMaster metric mismatch")
    require(s5_dtiam["drug_macro_auprc"] == s5_contract["dtiam_drug_macro_auprc"], "S5 DTIAM metric mismatch")
    checks.append("s5_metrics")

    kirhub_contract = contract["evaluation"]["kirhub_strict_retrospective"]
    kirhub = rank_summary["strict_kirhub_post_audit"]
    require(kirhub["rows"] == kirhub_contract["measured_pairs"], "KIRHub row count mismatch")
    require(kirhub["positives"] == kirhub_contract["strong_inhibition_pairs"], "KIRHub positive count mismatch")
    require(kirhub["drugs"] == kirhub_contract["drugs"], "KIRHub drug count mismatch")
    require(kirhub["two_class_drugs"] == kirhub_contract["two_class_drug_queries"], "KIRHub query count mismatch")
    kirhub_model = metric_by_model(rank_summary, "strict_kirhub_post_audit", "BIOMASTER_INDEPENDENT_BORDA")
    kirhub_dtiam = metric_by_model(rank_summary, "strict_kirhub_post_audit", "DTIAM_RAW")
    require(
        kirhub_model["drug_macro_auprc"] == kirhub_contract["independent_borda_drug_macro_auprc"],
        "KIRHub BioMaster metric mismatch",
    )
    require(kirhub_dtiam["drug_macro_auprc"] == kirhub_contract["dtiam_drug_macro_auprc"], "KIRHub DTIAM metric mismatch")
    checks.append("kirhub_retrospective_metrics")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--contract-only", action="store_true", help="Skip checks against local output artifacts")
    args = parser.parse_args()

    contract_path = args.contract if args.contract.is_absolute() else ROOT / args.contract
    contract = read_json(contract_path)
    checks = validate_contract_only(contract)
    if not args.contract_only:
        checks.extend(validate_artifacts(contract))
    print(json.dumps({"status": "PASS", "contract": str(contract_path), "checks": checks}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
