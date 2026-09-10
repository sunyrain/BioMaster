#!/usr/bin/env python3
"""Build a routed non-GPCR target registry and production-screen manifests.

The ChEMBL 37 audit contains 745 non-GPCR human single-protein targets with
sequence. That number is the lossless registry size, not a claim that all 745
targets are comparable under one drug-to-target production rank.

The production screen is restricted to targets with direct small-molecule
evidence in five currently modelled assay lanes. Special systems and targets
without direct small-molecule evidence remain visible in separate routed
branches; missing relations are never converted to negatives here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/target_discovery_scope_ch37_v3.yaml"

BIOCHEMICAL_LANES = frozenset(
    {
        "ENZYME_BIOCHEMICAL",
        "KINASE_BIOCHEMICAL",
        "NUCLEAR_EPIGENETIC_DOMAIN",
    }
)
FUNCTIONAL_LANES = frozenset(
    {
        "ION_CHANNEL_FUNCTIONAL",
        "TRANSPORTER_MEMBRANE_FUNCTIONAL",
    }
)
ASSAYABLE_LANES = BIOCHEMICAL_LANES | FUNCTIONAL_LANES
SPECIAL_LANES = frozenset(
    {
        "EXTRACELLULAR_SPECIAL",
        "NONCANONICAL_REVIEW",
        "NON_GPCR_MEMBRANE_SPECIAL",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna("").astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y"}
    )


def classify_discovery_scope(
    master: pd.DataFrame,
    current_scored_target_ids: set[str],
    supervised_target_ids: set[str],
    protbert_exact_sequences: set[str],
) -> pd.DataFrame:
    """Return one explicitly routed row for every non-GPCR registry target."""

    required = {
        "target_chembl_id",
        "gene_symbol",
        "uniprot_accession",
        "sequence",
        "sequence_sha256",
        "set_non_gpcr_all",
        "set_small_molecule_moa_non_gpcr",
        "ot_project_standard_direct_sm",
        "structure_ready_permissive",
        "structure_ready_strict",
        "af_exact_sequence_model",
        "calibration_8x8",
        "assay_lane",
        "evidence_class",
    }
    missing = sorted(required - set(master.columns))
    if missing:
        raise ValueError(f"target universe lacks required columns: {missing}")

    output = master.loc[as_bool(master["set_non_gpcr_all"])].copy()
    output = output.sort_values(
        ["target_chembl_id", "gene_symbol"], kind="mergesort"
    ).reset_index(drop=True)
    output.insert(0, "target_registry_index", np.arange(len(output), dtype=np.int16))

    chembl_core = as_bool(output["set_small_molecule_moa_non_gpcr"])
    ot_extension = ~chembl_core & as_bool(output["ot_project_standard_direct_sm"])
    direct_sm = chembl_core | ot_extension
    no_direct_sm = ~direct_sm
    biochemical = output["assay_lane"].isin(BIOCHEMICAL_LANES)
    functional = output["assay_lane"].isin(FUNCTIONAL_LANES)
    assayable = biochemical | functional
    special = output["assay_lane"].isin(SPECIAL_LANES)

    primary_biochemical = direct_sm & biochemical
    primary_functional = direct_sm & functional
    primary_screen = primary_biochemical | primary_functional
    special_direct = direct_sm & special
    exploratory_assayable = no_direct_sm & assayable
    registry_only_special = no_direct_sm & special
    active_routed = primary_screen | special_direct | exploratory_assayable

    all_routed = (
        primary_screen
        | special_direct
        | exploratory_assayable
        | registry_only_special
    )
    if not all_routed.all():
        unknown_lanes = sorted(
            output.loc[~all_routed, "assay_lane"].astype(str).unique()
        )
        raise ValueError(f"unrouted non-GPCR assay lanes: {unknown_lanes}")

    output["discovery_evidence_tier"] = np.select(
        [chembl_core, ot_extension],
        [
            "A_CHEMBL_SMALL_MOLECULE_MOA",
            "B_OPENTARGETS_DIRECT_SMALL_MOLECULE_EXTENSION",
        ],
        default="C_NO_DIRECT_SMALL_MOLECULE_EVIDENCE",
    )
    output["production_route"] = np.select(
        [
            primary_biochemical,
            primary_functional,
            special_direct,
            exploratory_assayable,
        ],
        [
            "PRIMARY_BIOCHEMICAL_DIRECT_SM",
            "PRIMARY_FUNCTIONAL_DIRECT_SM",
            "SPECIAL_SYSTEM_DIRECT_SM",
            "EXPLORATORY_ASSAYABLE_NO_DIRECT_SM",
        ],
        default="REGISTRY_ONLY_SPECIAL_NO_DIRECT_SM",
    )

    output["include_in_non_gpcr_registry_745"] = True
    output["include_in_primary_direct_sm_screen_450"] = primary_screen
    output["include_in_primary_biochemical_screen_367"] = primary_biochemical
    output["include_in_primary_functional_screen_83"] = primary_functional
    output["include_in_special_direct_sm_branch_42"] = special_direct
    output["include_in_exploratory_assayable_frontier_8"] = exploratory_assayable
    output["include_in_active_routed_design_space_500"] = active_routed
    output["registry_only_special_no_direct_sm_245"] = registry_only_special

    # Preserve provenance so the two different 450-target sets cannot be
    # accidentally treated as identical downstream.
    output["is_legacy_chembl_moa_set_450"] = chembl_core
    output["is_opentargets_direct_sm_extension_42"] = ot_extension
    output["is_no_direct_sm_evidence_253"] = no_direct_sm

    target_ids = output["target_chembl_id"].astype(str)
    sequences = output["sequence"].astype(str)
    output["in_historical_scored_384"] = target_ids.isin(current_scored_target_ids)
    output["in_supervised_training_target_vocabulary"] = target_ids.isin(
        supervised_target_ids
    )
    output["target_model_warmth"] = np.where(
        output["in_supervised_training_target_vocabulary"],
        "TARGET_WARM_IN_SUPERVISED_RELATIONS",
        "TARGET_COLD_SEQUENCE_EXTRAPOLATION",
    )
    output["protbert_exact_cache_available"] = sequences.isin(protbert_exact_sequences)

    strict = as_bool(output["structure_ready_strict"])
    permissive = as_bool(output["structure_ready_permissive"])
    exact_af = as_bool(output["af_exact_sequence_model"])
    output["structure_evidence_route"] = np.select(
        [strict, permissive, exact_af],
        [
            "OPTIONAL_STRICT_POCKET_CONTEXT",
            "OPTIONAL_PERMISSIVE_POCKET_CONTEXT",
            "SEQUENCE_PRIMARY_WEAK_OR_NO_POCKET",
        ],
        default="SEQUENCE_ONLY_EXACT_STRUCTURE_PENDING",
    )
    output["structure_is_optional_within_route"] = True
    output["missing_relation_semantics"] = "UNKNOWN_NOT_NEGATIVE"
    output["assay_route"] = output["assay_lane"].astype(str)
    output["historical_calibration_route"] = np.where(
        as_bool(output["calibration_8x8"]),
        "LOCAL_CALIBRATION_AVAILABLE_8X8",
        "SPARSE_OR_UNBALANCED_CALIBRATION_REPORT_UNCERTAINTY",
    )
    output["current_score_status"] = np.select(
        [
            output["in_historical_scored_384"],
            primary_screen,
            special_direct,
            exploratory_assayable,
        ],
        [
            "SCORED_IN_FROZEN_384_CORE",
            "UNSCORED_PRIMARY_EXPANSION_REQUIRES_FEATURES_AND_CALIBRATION",
            "UNSCORED_REQUIRES_SPECIAL_SYSTEM_MODEL_AND_ASSAY_ROUTE",
            "UNSCORED_HIGH_NOVELTY_REPORT_SEPARATELY_WITH_UNCERTAINTY",
        ],
        default="REGISTRY_ONLY_NOT_IN_CURRENT_PRODUCTION_SCORING",
    )
    output["rank_contract"] = np.select(
        [primary_screen, special_direct, exploratory_assayable],
        [
            "ELIGIBLE_FOR_ROUTED_GLOBAL_RANK_450_AFTER_CALIBRATION",
            "SEPARATE_SPECIAL_SYSTEM_RANK_42",
            "SEPARATE_EXPLORATORY_LIST_8_NOT_CALIBRATED_AS_RANK_450",
        ],
        default="NO_CURRENT_PRODUCTION_RANK",
    )
    return output


def target_export_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "target_registry_index",
        "target_chembl_id",
        "gene_symbol",
        "target_name",
        "uniprot_accession",
        "sequence_key",
        "sequence",
        "sequence_length",
        "sequence_sha256",
        "target_class_l1",
        "target_class_all",
        "assay_lane",
        "evidence_class",
        "discovery_evidence_tier",
        "production_route",
        "include_in_non_gpcr_registry_745",
        "include_in_primary_direct_sm_screen_450",
        "include_in_primary_biochemical_screen_367",
        "include_in_primary_functional_screen_83",
        "include_in_special_direct_sm_branch_42",
        "include_in_exploratory_assayable_frontier_8",
        "include_in_active_routed_design_space_500",
        "registry_only_special_no_direct_sm_245",
        "is_legacy_chembl_moa_set_450",
        "is_opentargets_direct_sm_extension_42",
        "is_no_direct_sm_evidence_253",
        "in_historical_scored_384",
        "in_supervised_training_target_vocabulary",
        "target_model_warmth",
        "protbert_exact_cache_available",
        "af_exact_sequence_model",
        "af_pdb_path",
        "p2rank_status",
        "p2rank_tier",
        "p2rank_top_score",
        "p2rank_top_probability",
        "p2rank_center_x",
        "p2rank_center_y",
        "p2rank_center_z",
        "p2rank_residue_ids",
        "pocket_residue_count",
        "pocket_mean_plddt",
        "pocket_residues_plddt_ge70_pct",
        "structure_ready_permissive",
        "structure_ready_strict",
        "structure_evidence_route",
        "structure_is_optional_within_route",
        "positive_compounds",
        "negative_compounds",
        "target_calibration_tier",
        "calibration_8x8",
        "historical_calibration_route",
        "assay_route",
        "missing_relation_semantics",
        "current_score_status",
        "rank_contract",
    ]
    return [column for column in preferred if column in frame.columns]


def build_pair_manifest(
    drugs: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    scope_code: str,
    denominator: int,
    score_contract: str,
) -> pd.DataFrame:
    drug_columns = [
        column
        for column in [
            "drug_feature_index",
            "ligand_inchikey",
            "ligand_smiles",
            "drug_names",
        ]
        if column in drugs.columns
    ]
    target_columns = [
        "target_registry_index",
        "target_chembl_id",
        "gene_symbol",
        "uniprot_accession",
        "assay_lane",
        "discovery_evidence_tier",
        "production_route",
        "target_model_warmth",
        "structure_evidence_route",
        "current_score_status",
        "rank_contract",
    ]
    left = drugs[drug_columns].copy()
    right = targets[target_columns].copy()
    left["__join"] = 1
    right["__join"] = 1
    pairs = left.merge(right, on="__join", how="inner", validate="many_to_many").drop(
        columns="__join"
    )
    pairs.insert(
        0,
        "candidate_pair_id",
        scope_code
        + "::"
        + pairs["ligand_inchikey"].astype(str)
        + "::"
        + pairs["target_chembl_id"].astype(str),
    )
    pairs["candidate_target_denominator"] = denominator
    pairs["pair_label_status"] = "UNOBSERVED_OR_HISTORICAL_LOOKUP_PENDING_NOT_NEGATIVE"
    pairs["model_score_status"] = score_contract
    return pairs


def build(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = {name: ROOT / value for name, value in config["inputs"].items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    master = pd.read_csv(paths["authoritative_target_universe"], low_memory=False)
    scored = pd.read_csv(
        paths["current_scored_target_index"], usecols=["target_chembl_id"]
    )
    supervised = pd.read_csv(
        paths["supervised_training_pairs"], usecols=["target_chembl_id"]
    )
    drugs = pd.read_csv(paths["old_drug_index"], low_memory=False)
    with h5py.File(paths["protbert_exact_sequence_cache"], "r") as cache:
        protbert_sequences = set(cache.keys())

    targets = classify_discovery_scope(
        master,
        set(scored["target_chembl_id"].astype(str)),
        set(supervised["target_chembl_id"].astype(str)),
        protbert_sequences,
    )
    primary = targets[targets["include_in_primary_direct_sm_screen_450"]].copy()
    biochemical = targets[
        targets["include_in_primary_biochemical_screen_367"]
    ].copy()
    functional = targets[targets["include_in_primary_functional_screen_83"]].copy()
    special = targets[targets["include_in_special_direct_sm_branch_42"]].copy()
    exploratory = targets[
        targets["include_in_exploratory_assayable_frontier_8"]
    ].copy()
    registry_only = targets[targets["registry_only_special_no_direct_sm_245"]].copy()

    pair_specs = {
        "registry": (
            targets,
            "D2T_REGISTRY_745",
            745,
            "CATALOG_ONLY_NOT_A_GLOBAL_PRODUCTION_RANK",
        ),
        "primary": (
            primary,
            "D2T_PRIMARY_450",
            450,
            "NOT_SCORED_BY_SCOPE_BUILDER_ROUTE_CALIBRATION_REQUIRED",
        ),
        "special": (
            special,
            "D2T_SPECIAL_42",
            42,
            "NOT_SCORED_SPECIAL_SYSTEM_MODEL_REQUIRED",
        ),
        "exploratory": (
            exploratory,
            "D2T_EXPLORATORY_8",
            8,
            "NOT_SCORED_REPORT_UNCERTAINTY_SEPARATELY",
        ),
    }
    pairs = {
        name: build_pair_manifest(
            drugs,
            frame,
            scope_code=scope_code,
            denominator=denominator,
            score_contract=score_contract,
        )
        for name, (frame, scope_code, denominator, score_contract) in pair_specs.items()
    }

    contracts = {name: int(value) for name, value in config["contracts"].items()}
    checks = {
        "authoritative_888": len(master) == contracts["authoritative_targets"],
        "registry_745": len(targets) == contracts["non_gpcr_registry_targets"],
        "unique_target_ids_745": targets["target_chembl_id"].nunique() == 745,
        "unique_sequences_745": targets["sequence_sha256"].nunique() == 745,
        "all_745_have_sequence": targets["sequence"].fillna("").astype(str).str.len().gt(0).all(),
        "legacy_chembl_set_450": int(targets["is_legacy_chembl_moa_set_450"].sum())
        == contracts["legacy_chembl_small_molecule_moa_set"],
        "ot_extension_42": int(
            targets["is_opentargets_direct_sm_extension_42"].sum()
        )
        == contracts["opentargets_direct_small_molecule_extension"],
        "no_direct_253": int(targets["is_no_direct_sm_evidence_253"].sum())
        == contracts["no_direct_small_molecule_evidence"],
        "primary_450": len(primary) == contracts["primary_direct_sm_assayable_targets"],
        "biochemical_367": len(biochemical)
        == contracts["primary_biochemical_targets"],
        "functional_83": len(functional) == contracts["primary_functional_targets"],
        "special_direct_42": len(special) == contracts["special_direct_sm_targets"],
        "exploratory_8": len(exploratory)
        == contracts["exploratory_assayable_no_direct_sm_targets"],
        "active_routed_500": int(
            targets["include_in_active_routed_design_space_500"].sum()
        )
        == contracts["active_routed_design_targets"],
        "registry_only_245": len(registry_only)
        == contracts["registry_only_special_no_direct_sm_targets"],
        "historical_384_subset_of_primary": int(
            primary["in_historical_scored_384"].sum()
        )
        == contracts["current_scored_targets"],
        "old_drugs_720": len(drugs) == contracts["old_drugs"],
        "registry_pairs": len(pairs["registry"]) == contracts["registry_pairs"],
        "primary_pairs": len(pairs["primary"]) == contracts["primary_pairs"],
        "special_pairs": len(pairs["special"]) == contracts["special_pairs"],
        "exploratory_pairs": len(pairs["exploratory"])
        == contracts["exploratory_pairs"],
        "all_pair_ids_unique": all(
            frame["candidate_pair_id"].is_unique for frame in pairs.values()
        ),
        "no_pair_labelled_negative": all(
            frame["pair_label_status"].eq(
                "UNOBSERVED_OR_HISTORICAL_LOOKUP_PENDING_NOT_NEGATIVE"
            ).all()
            for frame in pairs.values()
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks, ensure_ascii=False, indent=2))

    out = ROOT / config["outputs"]["directory"]
    out.mkdir(parents=True, exist_ok=True)
    columns = target_export_columns(targets)
    target_outputs = {
        "TARGET_REGISTRY_NON_GPCR_745_V3.csv.gz": targets,
        "TARGET_PRIMARY_DIRECT_SM_ASSAYABLE_450_V3.csv.gz": primary,
        "TARGET_PRIMARY_BIOCHEMICAL_367_V3.csv.gz": biochemical,
        "TARGET_PRIMARY_FUNCTIONAL_83_V3.csv.gz": functional,
        "TARGET_SPECIAL_DIRECT_SM_42_V3.csv.gz": special,
        "TARGET_EXPLORATORY_ASSAYABLE_NO_DIRECT_SM_8_V3.csv.gz": exploratory,
        "TARGET_REGISTRY_ONLY_SPECIAL_NO_DIRECT_SM_245_V3.csv.gz": registry_only,
    }
    pair_outputs = {
        "OLD_DRUG_TARGET_REGISTRY_720X745_V3.csv.gz": pairs["registry"],
        "OLD_DRUG_TARGET_PRIMARY_720X450_V3.csv.gz": pairs["primary"],
        "OLD_DRUG_TARGET_SPECIAL_720X42_V3.csv.gz": pairs["special"],
        "OLD_DRUG_TARGET_EXPLORATORY_720X8_V3.csv.gz": pairs["exploratory"],
    }
    written_paths: list[Path] = []
    for name, frame in target_outputs.items():
        path = out / name
        frame[columns].to_csv(
            path, index=False, compression={"method": "gzip", "mtime": 0}
        )
        written_paths.append(path)
    for name, frame in pair_outputs.items():
        path = out / name
        frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
        written_paths.append(path)

    counts = {
        "official_human_single_protein_moa": int(len(master)),
        "non_gpcr_registry": int(len(targets)),
        "legacy_chembl_small_molecule_moa_set": int(
            targets["is_legacy_chembl_moa_set_450"].sum()
        ),
        "primary_direct_sm_assayable": int(len(primary)),
        "primary_biochemical": int(len(biochemical)),
        "primary_functional": int(len(functional)),
        "special_direct_sm": int(len(special)),
        "exploratory_assayable_no_direct_sm": int(len(exploratory)),
        "active_routed_design_space": int(
            targets["include_in_active_routed_design_space_500"].sum()
        ),
        "registry_only_special_no_direct_sm": int(len(registry_only)),
        "historical_scored_core": int(primary["in_historical_scored_384"].sum()),
        "new_primary_targets_beyond_historical_384": int(
            (~primary["in_historical_scored_384"]).sum()
        ),
        "old_drugs": int(len(drugs)),
        "primary_pairs": int(len(pairs["primary"])),
        "special_pairs": int(len(pairs["special"])),
        "exploratory_pairs": int(len(pairs["exploratory"])),
        "registry_pairs": int(len(pairs["registry"])),
    }
    summary = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "CHEMBL37_NON_GPCR_ROUTED_TARGET_SCOPE_V3",
        "scope_correction": (
            "745 is the lossless non-GPCR registry, not one production rank. The "
            "primary routed screen contains 450 direct-small-molecule targets in "
            "five assayable lanes; special and no-direct-evidence systems are reported "
            "in separate branches."
        ),
        "counts": counts,
        "checks": checks,
        "rank_contract": {
            "historical_metrics_denominator": 384,
            "future_primary_candidate_denominator": 450,
            "primary_biochemical_subrank": 367,
            "primary_functional_subrank": 83,
            "special_system_denominator": 42,
            "exploratory_unranked_targets": 8,
            "registry_is_not_rank_denominator": 745,
            "global_rank_450_available": False,
            "reason": (
                "66 primary targets beyond the frozen 384 still require exact features, "
                "inference and cross-route calibration."
            ),
        },
        "policy": config["policy"],
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in paths.values()},
        "outputs": {},
    }
    for path in written_paths:
        summary["outputs"][str(path.relative_to(ROOT))] = sha256(path)

    report = f"""# ChEMBL 37 非GPCR靶点空间分层说明 V3

## 结论

**745不是统一生产排名分母，而是不丢靶点的非GPCR登记目录。** 当前主生产候选集为
**450个具有直接小分子证据、且落在五类可建模实验通道中的靶点**。现有384靶点评估
全部是该450集合的子集，历史结果继续解释为rank/384。

## 正确分层

| 层级 | 数量 | 当前用途 |
|---|---:|---|
| 生化主通道：激酶/酶/核受体与表观遗传 | {counts['primary_biochemical']} | 独立校准后进入主筛 |
| 功能主通道：离子通道/转运体 | {counts['primary_functional']} | 功能实验分支校准后进入主筛 |
| 主生产候选集 | {counts['primary_direct_sm_assayable']} | 上述两路合并，未来rank/450 |
| 特殊体系且有直接小分子证据 | {counts['special_direct_sm']} | 单独模型、单独实验、rank/42 |
| 可测类别但没有直接小分子证据 | {counts['exploratory_assayable_no_direct_sm']} | 高新颖性探索清单，单列不混入校准主榜 |
| 特殊体系且没有直接小分子证据 | {counts['registry_only_special_no_direct_sm']} | 登记保留，暂不进入生产打分 |
| 非GPCR完整登记目录 | {counts['non_gpcr_registry']} | 信息不丢失，但不是rank/745 |

跨分支的活跃设计空间为 **{counts['active_routed_design_space']}个靶点**：450个主筛、
42个特殊直接证据分支和8个高新颖性探索靶点。其余245个保留在登记目录，等专门的
膜蛋白、胞外蛋白或非常规体系方法成熟后再启用。

## 为什么“两个450”不是同一批靶点

历史ChEMBL小分子MoA集合也是450个，但其中包含42个特殊体系靶点；与此同时，
Open Targets另提供42个具有直接小分子证据、且属于主实验通道的靶点。因此新的
生产450相当于：历史集合中保留408个主通道靶点，移出42个特殊体系到专项分支，
再加入42个主通道直接证据扩展靶点。数量碰巧仍为450，成员已经不同。

## 分母边界

- 当前S5、KIRHub和DTIAM比较继续使用冻结rank/384，不改写历史分母。
- 未来主榜是经生化路与功能路分别校准后合并的rank/450；尚有
  {counts['new_primary_targets_beyond_historical_384']}个主筛靶点需要补特征和评分。
- 42个特殊体系单独报告；8个无直接小分子证据靶点带不确定性单列；745只用于目录审计。
- 所有未观测pair仍是unknown，不是negative。
"""
    report_path = out / "TARGET_DISCOVERY_SCOPE_CHANGELOG_ZH_V3.md"
    report_path.write_text(report, encoding="utf-8")
    summary["outputs"][str(report_path.relative_to(ROOT))] = sha256(report_path)
    summary_path = out / "TARGET_DISCOVERY_SCOPE_SUMMARY_V3.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    build(args.config.resolve())


if __name__ == "__main__":
    main()
