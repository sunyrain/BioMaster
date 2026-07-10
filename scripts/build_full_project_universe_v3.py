#!/usr/bin/env python3
"""Build the untruncated 745-drug x 462-target production universe."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biomaster.production import (  # noqa: E402
    add_priority_score_v2,
    annotate_candidate_risk,
    assert_unique_pairs,
    bool_series,
    diverse_select,
)
from scripts.build_106k_to_1000_physics_funnel import original_target_family, standardize_smiles  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "current_pipeline_v2.yaml"
DIRECT_ACTIONS = {
    "INHIBITOR",
    "AGONIST",
    "ANTAGONIST",
    "BLOCKER",
    "MODULATOR",
    "POSITIVE ALLOSTERIC MODULATOR",
    "NEGATIVE ALLOSTERIC MODULATOR",
    "ACTIVATOR",
    "OPENER",
    "STABILISER",
    "POSITIVE MODULATOR",
    "PARTIAL AGONIST",
    "INVERSE AGONIST",
    "NEGATIVE MODULATOR",
    "DEGRADER",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_genes(value: Any) -> list[str]:
    return [token for token in re.split(r"[;,|\s]+", str(value or "").strip()) if token]


def make_pair_id(df: pd.DataFrame) -> pd.Series:
    gene = df["primary_gene"].fillna("TARGET").astype(str).str.replace(r"[^A-Za-z0-9]+", "_", regex=True)
    return df["drug_chembl_id"].astype(str) + "_" + gene + "_" + df["sequence_key"].astype(str)


def build_extra_drug_metadata(drug_library: pd.DataFrame, drug_ids: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, source in drug_library[drug_library["drug_id"].isin(drug_ids)].iterrows():
        smiles = str(source.get("canonical_smiles") or "").strip()
        standard = standardize_smiles(smiles)
        molecule = Chem.MolFromSmiles(standard["active_moiety_smiles"]) if standard["active_moiety_smiles"] else None
        mw = float(Descriptors.MolWt(molecule)) if molecule is not None else pd.to_numeric(
            pd.Series([source.get("molecular_weight")]), errors="coerce"
        ).fillna(0.0).iloc[0]
        logp = float(Crippen.MolLogP(molecule)) if molecule is not None else 0.0
        tpsa = float(Descriptors.TPSA(molecule)) if molecule is not None else 0.0
        qed = float(QED.qed(molecule)) if molecule is not None else 0.0
        ro5 = int(
            (mw > 500)
            + (logp > 5)
            + ((Lipinski.NumHDonors(molecule) if molecule is not None else 0) > 5)
            + ((Lipinski.NumHAcceptors(molecule) if molecule is not None else 0) > 10)
        )
        route = str(source.get("route") or "")
        feasibility = 0.0
        feasibility += 2.5 if 120 <= mw <= 650 else (1.0 if 650 < mw <= 900 else 0.0)
        feasibility += 2.0 if ro5 <= 1 else 0.0
        feasibility += 1.5 if qed >= 0.20 else 0.0
        feasibility += 1.0 if qed >= 0.35 else 0.0
        feasibility += 1.0 if smiles else 0.0
        feasibility += 2.0 if re.search(
            r"oral|intravenous|subcutaneous|intramuscular|inhalation|infusion|injection", route, re.I
        ) else 0.0
        row = {
            "drug_chembl_id": source["drug_id"],
            "drug_names": source.get("drug_name", ""),
            "drug_has_direct_action_label": True,
            "drug_action_types": source.get("action_type", ""),
            "drug_keep_for_target_engagement": True,
            "drug_exclusion_reasons": "v4_explicit_scope_extension",
            "fda_generic_name": source.get("drug_name", ""),
            "fda_brand_name": source.get("brand_name", ""),
            "canonical_smiles": smiles,
            "fda_therapeutic_area": source.get("therapeutic_area", ""),
            "fda_indication": source.get("indication", ""),
            "fda_moa": source.get("mechanism_of_action", ""),
            "fda_action_type": source.get("action_type", ""),
            "fda_route": route,
            "fda_target_names": source.get("target_name", ""),
            "fda_mw": mw,
            "fda_qed": qed,
            "fda_ro5_violations": ro5,
            "fda_approval_year": source.get("approval_year", ""),
            "fda_logp": logp,
            "fda_tpsa": tpsa,
            **standard,
            "drug_feasibility_score": min(10.0, feasibility),
        }
        row["fda_original_target_family"] = original_target_family(pd.Series(row))
        rows.append(row)
    missing = drug_ids - {str(row["drug_chembl_id"]) for row in rows}
    if missing:
        raise ValueError(f"Configured extra drug IDs missing from drug library: {sorted(missing)}")
    return pd.DataFrame(rows)


def build_known_controls(
    project_drugs: set[str],
    targets: pd.DataFrame,
    mechanisms: pd.DataFrame,
    legacy_known: pd.DataFrame,
) -> pd.DataFrame:
    gene_to_targets: dict[str, list[dict[str, str]]] = {}
    for _, row in targets.iterrows():
        gene = str(row.get("primary_gene") or "").strip()
        if gene:
            gene_to_targets.setdefault(gene, []).append(
                {
                    "sequence_key": str(row["sequence_key"]),
                    "gene_names": gene,
                    "target_classes": str(row.get("target_classes") or ""),
                }
            )
    base_to_project: dict[str, set[str]] = {}
    for drug in project_drugs:
        base_to_project.setdefault(drug.split("__")[0], set()).add(drug)

    mech = mechanisms[
        mechanisms["organism"].eq("Homo sapiens")
        & mechanisms["target_type"].eq("SINGLE PROTEIN")
        & mechanisms["action_type"].astype(str).str.upper().isin(DIRECT_ACTIONS)
    ].copy()
    rows: list[dict[str, Any]] = []
    for _, row in mech.iterrows():
        drugs = set(base_to_project.get(str(row.get("molecule_chembl_id")), set()))
        drugs.update(base_to_project.get(str(row.get("parent_molecule_chembl_id")), set()))
        for drug in drugs:
            for gene in split_genes(row.get("component_gene_symbols")):
                for target in gene_to_targets.get(gene, []):
                    rows.append(
                        {
                            "drug_chembl_id": drug,
                            **target,
                            "known_source": "ChEMBL37_human_single_protein_MoA",
                            "known_action_type": row.get("action_type", ""),
                            "known_mechanism": row.get("mechanism_of_action", ""),
                        }
                    )
    known = pd.DataFrame(rows)
    legacy = legacy_known[["drug_chembl_id", "sequence_key", "gene_names", "target_classes"]].copy()
    legacy["known_source"] = "FDA_workbook_target_mapping_legacy_control"
    legacy["known_action_type"] = legacy_known.get("drug_action_types", "")
    legacy["known_mechanism"] = ""
    known = pd.concat([known, legacy], ignore_index=True)
    known = known.sort_values("known_source").drop_duplicates(["drug_chembl_id", "sequence_key"], keep="first")
    known["is_known_fda_target_pair"] = True
    return known


def calibration_summary(universe: pd.DataFrame, known: pd.DataFrame) -> dict[str, Any]:
    known_pairs = universe[universe["is_known_fda_target_pair"]].copy()
    collapsed = known_pairs.sort_values(
        ["knowledge_compound_key", "sequence_key", "conplex_score"],
        ascending=[True, True, False],
        kind="mergesort",
    ).drop_duplicates(["knowledge_compound_key", "sequence_key"])

    def metrics(frame: pd.DataFrame) -> dict[str, Any]:
        drug_rank = pd.to_numeric(frame["rank_within_drug"], errors="coerce").fillna(10**9)
        full_rank = pd.to_numeric(frame["rank_within_drug_full891"], errors="coerce").fillna(10**9)
        target_rank = pd.to_numeric(frame["target_rank"], errors="coerce").fillna(10**9)
        score = pd.to_numeric(frame["conplex_score"], errors="coerce").fillna(-1)
        return {
            "recall_project_target_universe_by_drug_rank": {
                f"top{k}": float((drug_rank <= k).mean()) for k in [10, 50, 100, 300]
            },
            "recall_full891_by_drug_rank": {
                f"top{k}": float((full_rank <= k).mean()) for k in [10, 50, 100, 300]
            },
            "recall_by_target_rank": {f"top{k}": float((target_rank <= k).mean()) for k in [10, 50, 100, 300]},
            "score_threshold_recall": {
                f"ge_{cutoff}": float((score >= cutoff).mean()) for cutoff in [0.05, 0.10, 0.20, 0.30]
            },
        }

    return {
        "known_union_rows": int(len(known)),
        "known_rows_in_project_universe": int(len(known_pairs)),
        "known_unique_active_moiety_target_rows": int(len(collapsed)),
        "known_unique_drugs": int(known_pairs["drug_chembl_id"].nunique()),
        "known_unique_targets": int(known_pairs["sequence_key"].nunique()),
        **metrics(known_pairs),
        "active_moiety_collapsed_metrics": metrics(collapsed),
        "warning": "ChEMBL/label positives may overlap ConPLEx training knowledge; this is calibration, not temporal generalization.",
    }


def build(
    config_path: Path,
    *,
    output_subdir: str = "full_untruncated_universe_v3",
    version_label: str = "v3",
    use_legacy_recall_gate: bool = True,
    write_full_universe: bool = True,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = {name: ROOT / value for name, value in config["inputs"].items()}
    needed = [
        "full_conplex_predictions",
        "enhanced_discovery_pool",
        "known_controls",
        "anchor_table",
        "chembl37_mechanisms",
        "drug_library",
        "refined_top3000",
    ]
    for name in needed:
        if not paths[name].exists():
            raise FileNotFoundError(f"Missing {name}: {paths[name]}")

    full = pd.read_csv(
        paths["full_conplex_predictions"],
        sep="\t",
        header=None,
        names=["drug_chembl_id", "sequence_key", "conplex_score"],
    )
    if len(full) != 815265 or full.duplicated(["drug_chembl_id", "sequence_key"]).any():
        raise ValueError(f"Unexpected full ConPLex result shape: {len(full)}")
    old_pool = pd.read_csv(paths["enhanced_discovery_pool"], low_memory=False).fillna("")
    legacy_known = pd.read_csv(paths["known_controls"], low_memory=False).fillna("")
    anchors = pd.read_csv(paths["anchor_table"], low_memory=False).fillna("")
    mechanisms = pd.read_csv(paths["chembl37_mechanisms"], low_memory=False).fillna("")
    drug_library = pd.read_csv(paths["drug_library"], low_memory=False).fillna("")
    old3000 = pd.read_csv(paths["refined_top3000"], low_memory=False).fillna("")
    target_extension = pd.DataFrame()
    if "target_scope_extension" in paths:
        if not paths["target_scope_extension"].exists():
            raise FileNotFoundError(paths["target_scope_extension"])
        target_extension = pd.read_csv(paths["target_scope_extension"], low_memory=False).fillna("")

    extra_drug_ids = set(config.get("scope", {}).get("extra_direct_action_drug_ids", []))
    dynamic_project_drugs = set(old_pool["drug_chembl_id"].unique()) | extra_drug_ids
    dynamic_project_targets = set(old_pool["sequence_key"].unique()) | set(
        target_extension.get("sequence_key", pd.Series(dtype=str)).astype(str)
    )
    if "project_drug_manifest" in paths and "project_target_manifest" in paths:
        drug_manifest = pd.read_csv(paths["project_drug_manifest"], low_memory=False).fillna("")
        target_manifest = pd.read_csv(paths["project_target_manifest"], low_memory=False).fillna("")
        expected_drug_sha = str(config.get("scope", {}).get("project_drug_manifest_sha256", ""))
        expected_target_sha = str(config.get("scope", {}).get("project_target_manifest_sha256", ""))
        if expected_drug_sha and file_sha256(paths["project_drug_manifest"]) != expected_drug_sha:
            raise ValueError("Frozen project drug manifest SHA-256 does not match config")
        if expected_target_sha and file_sha256(paths["project_target_manifest"]) != expected_target_sha:
            raise ValueError("Frozen project target manifest SHA-256 does not match config")
        project_drugs = set(drug_manifest["drug_chembl_id"].astype(str))
        project_targets = set(target_manifest["sequence_key"].astype(str))
        if project_drugs != dynamic_project_drugs or project_targets != dynamic_project_targets:
            raise ValueError("Frozen v4 entity manifests no longer match the current source-derived scope")
    else:
        project_drugs = dynamic_project_drugs
        project_targets = dynamic_project_targets
    full["rank_within_drug_full891"] = full.groupby("drug_chembl_id")["conplex_score"].rank(
        method="average", ascending=False
    )
    project = full[full["drug_chembl_id"].isin(project_drugs) & full["sequence_key"].isin(project_targets)].copy()
    expected = len(project_drugs) * len(project_targets)
    if len(project) != expected:
        raise ValueError(f"Project Cartesian space incomplete: expected {expected}, observed {len(project)}")
    project["rank_within_drug"] = project.groupby("drug_chembl_id")["conplex_score"].rank(
        method="average", ascending=False
    )
    project["target_rank_id_weighted_v4"] = project.groupby("sequence_key")["conplex_score"].rank(
        method="average", ascending=False
    )

    drug_columns = [
        "drug_chembl_id",
        "drug_names",
        "drug_has_direct_action_label",
        "drug_action_types",
        "drug_keep_for_target_engagement",
        "drug_exclusion_reasons",
        "fda_generic_name",
        "fda_brand_name",
        "canonical_smiles",
        "fda_therapeutic_area",
        "fda_indication",
        "fda_moa",
        "fda_action_type",
        "fda_route",
        "fda_target_names",
        "fda_mw",
        "fda_qed",
        "fda_ro5_violations",
        "fda_approval_year",
        "fda_logp",
        "fda_tpsa",
        "rdkit_parse_ok",
        "active_moiety_smiles",
        "murcko_scaffold",
        "canonical_smiles_rdkit",
        "model_ligand_smiles",
        "fda_original_target_family",
        "drug_feasibility_score",
    ]
    target_columns = [
        "sequence_key",
        "representative_protein_id",
        "gene_names",
        "primary_gene",
        "protein_names",
        "target_classes",
        "druggable_modalities",
        "max_clinical_phase",
        "has_alphafold_pdb",
        "pdb_path",
        "p2rank_pocketability_tier",
        "top_pocket_probability",
        "top_pocket_score",
        "top_pocket_residue_ids",
        "puresnet_tier",
        "structure_consensus_tier",
        "strict_structure_tier",
        "p2rank_puresnet_overlap_fraction",
        "p2rank_puresnet_jaccard",
        "p2rank_top_residue_count",
        "puresnet_best_cluster_residue_count",
        "receptor_residue_count",
        "structure_bin",
        "experimental_feasibility_score",
        "target_assay_family",
        "manual_classification_note",
    ]
    target_columns.extend(column for column in old_pool.columns if column.startswith("anchor_"))
    drug_meta = old_pool[[column for column in drug_columns if column in old_pool.columns]].drop_duplicates("drug_chembl_id").copy()
    if extra_drug_ids:
        drug_meta = pd.concat(
            [drug_meta, build_extra_drug_metadata(drug_library, extra_drug_ids)],
            ignore_index=True,
        ).drop_duplicates("drug_chembl_id", keep="last")
    drug_meta["_base_drug_id"] = drug_meta["drug_chembl_id"].astype(str).str.split("__").str[0]
    base_meta = drug_meta.assign(_is_composite=drug_meta["drug_chembl_id"].astype(str).str.contains("__")).sort_values(
        ["_base_drug_id", "_is_composite"]
    ).drop_duplicates("_base_drug_id")
    for column in [col for col in drug_columns if col not in {"drug_chembl_id"} and col in drug_meta.columns]:
        fallback = drug_meta["_base_drug_id"].map(base_meta.set_index("_base_drug_id")[column])
        drug_meta[column] = drug_meta[column].where(drug_meta[column].astype(str).ne(""), fallback)
    library_smiles = drug_library.drop_duplicates("drug_id").set_index("drug_id")["canonical_smiles"].to_dict()
    model_ligands = drug_library.drop_duplicates("drug_id").set_index("drug_id").get(
        "model_ligand_smiles", pd.Series(dtype=str)
    ).to_dict()
    drug_meta["canonical_smiles"] = drug_meta["canonical_smiles"].where(
        drug_meta["canonical_smiles"].astype(str).ne(""),
        drug_meta["drug_chembl_id"].map(library_smiles),
    )
    drug_meta["model_ligand_smiles"] = drug_meta["drug_chembl_id"].map(model_ligands).fillna("")
    drug_meta["model_ligand_smiles"] = drug_meta["model_ligand_smiles"].where(
        drug_meta["model_ligand_smiles"].astype(str).ne(""), drug_meta["active_moiety_smiles"]
    )
    drug_meta = drug_meta.drop(columns=["_base_drug_id"])
    target_meta = old_pool[[column for column in target_columns if column in old_pool.columns]].drop_duplicates("sequence_key")
    if not target_extension.empty:
        anchor_extension = anchors.add_prefix("anchor_").rename(columns={"anchor_gene": "primary_gene"})
        extension = target_extension.merge(anchor_extension, on="primary_gene", how="left", suffixes=("", "_anchor"))
        extension["anchor_project_assay_family"] = extension["target_assay_family"]
        extension["anchor_project_target_engagement_class"] = True
        extension["anchor_excluded_membrane_receptor_gpcr"] = False
        extension["anchor_excluded_secreted_surface_adhesion_structural"] = False
        target_meta = pd.concat([target_meta, extension], ignore_index=True, sort=False).drop_duplicates(
            "sequence_key", keep="last"
        )
    project = project.merge(drug_meta, on="drug_chembl_id", how="left").merge(target_meta, on="sequence_key", how="left")
    if project["primary_gene"].astype(str).eq("").any() or project["canonical_smiles"].astype(str).eq("").any():
        raise ValueError("Project metadata merge produced missing target genes or SMILES")

    known = build_known_controls(project_drugs, target_meta, mechanisms, legacy_known)
    active_by_drug = drug_meta.drop_duplicates("drug_chembl_id").set_index("drug_chembl_id")[
        "active_moiety_smiles"
    ].to_dict()
    known["knowledge_compound_key"] = known["drug_chembl_id"].map(active_by_drug).fillna("")
    known["knowledge_compound_key"] = known["knowledge_compound_key"].where(
        known["knowledge_compound_key"].astype(str).ne(""), known["drug_chembl_id"]
    )
    project["knowledge_compound_key"] = project["drug_chembl_id"].map(active_by_drug).fillna("")
    project["knowledge_compound_key"] = project["knowledge_compound_key"].where(
        project["knowledge_compound_key"].astype(str).ne(""), project["drug_chembl_id"]
    )
    known_keys = set(zip(known["drug_chembl_id"], known["sequence_key"]))
    project["is_known_fda_target_pair"] = [
        key in known_keys for key in zip(project["drug_chembl_id"], project["sequence_key"])
    ]
    project["pair_id"] = make_pair_id(project)
    project["drug_pair_count_in_106k"] = project.groupby("drug_chembl_id")["sequence_key"].transform("size")
    project["target_pair_count_in_106k"] = project.groupby("sequence_key")["drug_chembl_id"].transform("size")
    scaffold = project["murcko_scaffold"].replace("", "NO_SCAFFOLD")
    project["scaffold_pair_count_in_106k"] = scaffold.map(scaffold.value_counts())
    project["drug_pair_count_in_project_space"] = project["drug_pair_count_in_106k"]
    active_key = project["model_ligand_smiles"].astype(str)
    score_spread = project.assign(_active_key=active_key).groupby(
        ["_active_key", "sequence_key"]
    )["conplex_score"].agg(lambda values: float(values.max()) - float(values.min()))
    if (score_spread > 1e-8).any():
        raise ValueError("Equivalent model-ligand structures have inconsistent ConPLEx scores")
    active_target = (
        project.assign(_active_key=active_key)
        .sort_values(["_active_key", "sequence_key", "drug_chembl_id"], kind="mergesort")
        .drop_duplicates(["_active_key", "sequence_key"])
        [["_active_key", "sequence_key", "conplex_score"]]
    )
    active_target["target_rank_active_collapsed_v4"] = active_target.groupby("sequence_key")[
        "conplex_score"
    ].rank(method="average", ascending=False)
    project = project.assign(_active_key=active_key).merge(
        active_target[["_active_key", "sequence_key", "target_rank_active_collapsed_v4"]],
        on=["_active_key", "sequence_key"],
        how="left",
        validate="many_to_one",
    )
    project["target_rank"] = project["target_rank_active_collapsed_v4"]
    project["target_pair_count_in_project_space"] = int(active_target["_active_key"].nunique())
    project["scaffold_pair_count_in_project_space"] = project["scaffold_pair_count_in_106k"]

    project = annotate_candidate_risk(project, known, anchors)
    project = add_priority_score_v2(project, project["conplex_score"])
    project["pre_boltz_priority_score_v3"] = project["priority_score_v2"] - project["pair_boltz_component_v2"]
    structure_ok = project["structure_bin"].isin(
        ["A_strict_overlapping_pocket", "B_strict_supported_overlap", "C_manual_review_structure"]
    )
    recall_gate = (
        (pd.to_numeric(project["conplex_score"], errors="coerce") >= 0.05)
        & (
            (pd.to_numeric(project["rank_within_drug"], errors="coerce") <= 300)
            | (pd.to_numeric(project["target_rank"], errors="coerce") <= 300)
        )
    )
    discovery_gate = recall_gate if use_legacy_recall_gate else pd.Series(True, index=project.index)
    eligible = project[
        structure_ok
        & bool_series(project, "anchor_project_standard_direct_sm")
        & ~bool_series(project, "exact_known_target_v2")
        & ~bool_series(project, "family_or_rediscovery_risk_v2")
        & discovery_gate
    ].copy()
    selected = diverse_select(
        eligible,
        3000,
        score_column="pre_boltz_priority_score_v3",
        drug_cap=12,
        target_cap=32,
        scaffold_cap=50,
        family_caps={
            "enzyme": 1600,
            "kinase": 800,
            "transporter": 700,
            "nuclear_epigenetic": 500,
            "ion_channel": 300,
            "other_assayable": 100,
        },
    )
    selected = selected.sort_values("selection_rank_v2").copy()
    selected["externalQueueRank"] = selected["selection_rank_v2"]
    selected["pairId"] = selected["pair_id"]
    old_completed = bool_series(old3000, "boltz_completed_refined")
    disable_boltz_reuse = bool(config.get("scope", {}).get("disable_boltz_reuse", False))
    old_keys = set(zip(old3000["drug_chembl_id"], old3000["sequence_key"]))
    reusable_keys = set()
    if not disable_boltz_reuse:
        reusable_keys = set(
            zip(old3000.loc[old_completed, "drug_chembl_id"], old3000.loc[old_completed, "sequence_key"])
        )
    selected_keys = list(zip(selected["drug_chembl_id"], selected["sequence_key"]))
    selected["has_prior_refined_row"] = [key in old_keys for key in selected_keys]
    selected["has_reusable_refined_boltz"] = [key in reusable_keys for key in selected_keys]
    # A row that exists in an older table but has no complete Boltz output is
    # still pending; prior membership alone must never suppress its rerun.
    delta = selected[~selected["has_reusable_refined_boltz"]].copy()
    delta["externalQueueRank"] = range(1, len(delta) + 1)

    calibration = calibration_summary(project, known)
    calibration["stage1_recall_gate_retained"] = int(
        (project["is_known_fda_target_pair"] & structure_ok & bool_series(project, "anchor_project_standard_direct_sm") & recall_gate).sum()
    )
    calibration["selection_scope_known_retained"] = int(
        (
            project["is_known_fda_target_pair"]
            & structure_ok
            & bool_series(project, "anchor_project_standard_direct_sm")
            & discovery_gate
        ).sum()
    )
    out_dir = ROOT / config["outputs"]["directory"] / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    output_columns = [
        "pair_id",
        "drug_chembl_id",
        "drug_names",
        "sequence_key",
        "primary_gene",
        "protein_names",
        "conplex_score",
        "rank_within_drug_full891",
        "rank_within_drug",
        "target_rank",
        "is_known_fda_target_pair",
        "target_assay_family_v2",
        "structure_bin",
        "family_or_rediscovery_risk_v2",
        "risk_notes_v2",
        "pre_boltz_priority_score_v3",
        "pair_conplex_component_v2",
        "target_pocket_prior_component_v2",
        "target_tractability_component_v2",
        "drug_feasibility_component_v2",
        "experimental_feasibility_component_v2",
        "novelty_component_v2",
        "fda_therapeutic_area",
        "fda_indication",
        "fda_target_names",
        "fda_moa",
        "fda_action_type",
        "canonical_smiles",
        "model_ligand_smiles",
        "murcko_scaffold",
    ]
    if write_full_universe:
        project[[column for column in output_columns if column in project.columns]].to_csv(
            out_dir / f"full_project_universe_{len(project)}_scored_{version_label}.csv", index=False
        )
    active_target[["conplex_score"]].to_csv(
        out_dir / f"conplex_reference_{version_label}.csv", index=False
    )
    known.to_csv(out_dir / f"known_control_union_{version_label}.csv", index=False)
    project[project["is_known_fda_target_pair"]].to_csv(
        out_dir / f"known_control_calibration_{version_label}.csv", index=False
    )
    selected.to_csv(out_dir / f"pre_boltz_top3000_{version_label}.csv", index=False)
    delta.to_csv(out_dir / f"pre_boltz_top3000_delta_to_run_{version_label}.csv", index=False)

    old106k_keys = set(zip(old_pool["drug_chembl_id"], old_pool["sequence_key"]))
    top_keys = set(zip(selected["drug_chembl_id"], selected["sequence_key"]))
    summary = {
        "created_utc": now_utc(),
        "full_conplex_rows": int(len(full)),
        "project_drug_manifest_sha256": file_sha256(paths["project_drug_manifest"])
        if "project_drug_manifest" in paths
        else "",
        "project_target_manifest_sha256": file_sha256(paths["project_target_manifest"])
        if "project_target_manifest" in paths
        else "",
        "project_drugs": int(len(project_drugs)),
        "project_unique_model_ligands": int(active_target["_active_key"].nunique()),
        "explicit_scope_extension_drugs": sorted(extra_drug_ids),
        "project_targets": int(len(project_targets)),
        "project_cartesian_rows": int(len(project)),
        "project_physical_pair_rows": int(len(active_target)),
        "project_rows_previously_outside_top300_derived_106k": int(
            sum(key not in old106k_keys for key in zip(project["drug_chembl_id"], project["sequence_key"]))
        ),
        "known_calibration": calibration,
        "eligible_rows": int(len(eligible)),
        "selected_rows": int(len(selected)),
        "selected_unique_drugs": int(selected["drug_chembl_id"].nunique()),
        "selected_unique_targets": int(selected["sequence_key"].nunique()),
        "selected_assay_families": selected["target_assay_family_v2"].value_counts().to_dict(),
        "selected_inside_old_106k": int(sum(key in old106k_keys for key in top_keys)),
        "selected_outside_old_106k": int(sum(key not in old106k_keys for key in top_keys)),
        "reusable_refined_boltz_rows": int(selected["has_reusable_refined_boltz"].sum()),
        "delta_rows_requiring_boltz": int(len(delta)),
        "boltz_reuse_disabled": disable_boltz_reuse,
        "legacy_recall_gate_applied": bool(use_legacy_recall_gate),
        "active_moiety_target_duplicates_selected": int(
            selected.assign(
                _compound=selected["active_moiety_smiles"].where(
                    selected["active_moiety_smiles"].astype(str).ne(""), selected["canonical_smiles_rdkit"]
                )
            ).duplicated(["_compound", "sequence_key"]).sum()
        ),
    }
    (out_dir / f"full_untruncated_universe_{version_label}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the untruncated production universe and Top3000 v3.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-subdir", default="full_untruncated_universe_v3")
    parser.add_argument("--version-label", default="v3")
    parser.add_argument("--no-legacy-recall-gate", action="store_true")
    parser.add_argument("--skip-full-universe-output", action="store_true")
    args = parser.parse_args()
    summary = build(
        Path(args.config).resolve(),
        output_subdir=args.output_subdir,
        version_label=args.version_label,
        use_legacy_recall_gate=not args.no_legacy_recall_gate,
        write_full_universe=not args.skip_full_universe_output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
