#!/usr/bin/env python3
"""Select the complete remote-qualified N2/N3 increment after 384-target reranking."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parents[1]
V4 = (
    ROOT
    / "outputs/evidence_routing_compute_execution_20260808_v1/final_evidence_routing_v4"
    / "PAIR_EVIDENCE_LAYER_ROUTING_ALL_720_X_384_V4.csv.gz"
)
CONTROLS = ROOT / "outputs/gnina_calibration_338_v1/CH37_PAIR_CONTROL_POOL_338_V1.csv.gz"
OUT = ROOT / "outputs/unified_pair_compute_increment_384_v1"
RDLogger.DisableLog("rdApp.*")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [
        "pairId",
        "ligand_inchikey",
        "ligand_smiles",
        "drug_names",
        "project_entity_ids",
        "target_chembl_id",
        "gene_symbol",
        "assay_lane",
        "target_route_family",
        "unified_target_route",
        "gnina_target_qualification",
        "boltz_target_qualification",
        "mainline_gnina_completed",
        "mainline_boltz_completed",
        "is_any_frozen_known_relationship",
        "dta_bidirectional_top10pct_concordant_384",
        "conplex_score",
        "conplex_rank_within_target",
        "conplex_percentile_within_target",
        "conplex_rank_within_ligand_384",
        "conplex_percentile_within_ligand_384",
        "drugclip_cosine_mean",
        "drugclip_rank_within_target",
        "drugclip_percentile_within_target",
        "drugclip_rank_within_ligand_382",
        "drugclip_percentile_within_ligand_382",
        "pair_novelty_class_384",
        "pocket_evidence_tier",
        "pair_pocket_evidence_source",
        "receptor_protocol_status",
        "redock_status",
    ]
    pairs = pd.read_csv(V4, usecols=required, low_memory=False)
    remote_route = pairs["unified_target_route"].isin(
        {"E1_DUAL_REMOTE_QUALIFIED", "E2_SINGLE_MODEL_REMOTE_QUALIFIED"}
    )
    base = pairs[
        pairs["target_route_family"].eq("EXPERIMENTAL_POCKET_MAINLINE")
        & remote_route
        & pairs["dta_bidirectional_top10pct_concordant_384"].fillna(False)
        & ~pairs["is_any_frozen_known_relationship"].fillna(False)
    ].copy()
    base["gnina_remote_model_required"] = base["gnina_target_qualification"].eq(
        "REMOTE_STRONG"
    )
    base["boltz_remote_model_required"] = base["boltz_target_qualification"].eq(
        "BOLTZ_REMOTE_QUALIFIED"
    )
    base["gnina_increment_required"] = (
        base["gnina_remote_model_required"] & ~base["mainline_gnina_completed"]
    )
    base["boltz_increment_required"] = (
        base["boltz_remote_model_required"] & ~base["mainline_boltz_completed"]
    )
    missing_model = base[
        base["gnina_increment_required"] | base["boltz_increment_required"]
    ].copy()
    if missing_model.empty:
        raise RuntimeError("No remote-qualified incremental pairs were found")

    target_ids = set(missing_model["target_chembl_id"].astype(str))
    control_parts = []
    control_columns = [
        "target_chembl_id",
        "control_class",
        "docking_domain_pass",
        "canonical_control_smiles",
    ]
    for chunk in pd.read_csv(
        CONTROLS, usecols=control_columns, chunksize=50_000, low_memory=False
    ):
        selected = chunk[
            chunk["target_chembl_id"].astype(str).isin(target_ids)
            & chunk["control_class"].eq("positive")
            & chunk["docking_domain_pass"].fillna(False)
        ].dropna(subset=["canonical_control_smiles"])
        if len(selected):
            control_parts.append(selected)
    controls = pd.concat(control_parts, ignore_index=True).drop_duplicates(
        ["target_chembl_id", "canonical_control_smiles"]
    )

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    similarities: dict[str, float] = {}
    scaffolds: dict[str, str] = {}
    reference_counts: dict[str, int] = {}
    for target, candidate_group in missing_model.groupby("target_chembl_id", sort=True):
        reference_group = controls[controls["target_chembl_id"].astype(str).eq(str(target))]
        reference_fingerprints = []
        for smiles in reference_group["canonical_control_smiles"]:
            molecule = Chem.MolFromSmiles(str(smiles))
            if molecule is not None:
                reference_fingerprints.append(generator.GetFingerprint(molecule))
        reference_counts[str(target)] = len(reference_fingerprints)
        for row in candidate_group.itertuples(index=False):
            molecule = Chem.MolFromSmiles(str(row.ligand_smiles))
            if molecule is None or not reference_fingerprints:
                similarities[str(row.pairId)] = float("nan")
                scaffolds[str(row.pairId)] = ""
                continue
            fingerprint = generator.GetFingerprint(molecule)
            similarities[str(row.pairId)] = max(
                DataStructs.BulkTanimotoSimilarity(fingerprint, reference_fingerprints)
            )
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=molecule, includeChirality=True
            )
            scaffolds[str(row.pairId)] = scaffold or Chem.MolToSmiles(
                molecule, isomericSmiles=True
            )
    missing_model["max_tanimoto_to_target_measured_positive"] = missing_model[
        "pairId"
    ].map(similarities)
    missing_model["murcko_scaffold"] = missing_model["pairId"].map(scaffolds)
    missing_model["measured_positive_reference_count"] = missing_model[
        "target_chembl_id"
    ].astype(str).map(reference_counts).fillna(0).astype(int)
    score = missing_model["max_tanimoto_to_target_measured_positive"]
    missing_model["novelty_lane"] = "N3_REMOTE"
    missing_model.loc[score.gt(0.40), "novelty_lane"] = "N2_SCAFFOLD_HOP"
    missing_model.loc[score.gt(0.60), "novelty_lane"] = "N1_LOCAL_ANALOG"
    missing_model.loc[score.isna(), "novelty_lane"] = "NOVELTY_NOT_EVALUABLE"

    missing_model["dta_priority_score_384"] = (
        0.25
        * missing_model[
            ["conplex_percentile_within_target", "drugclip_percentile_within_target"]
        ].max(axis=1)
        + 0.25
        * missing_model[
            [
                "conplex_percentile_within_ligand_384",
                "drugclip_percentile_within_ligand_382",
            ]
        ].max(axis=1)
        + 0.25 * missing_model["conplex_percentile_within_ligand_384"]
        + 0.25 * missing_model["drugclip_percentile_within_ligand_382"]
        + 0.10
    )
    eligible = missing_model[
        missing_model["novelty_lane"].isin({"N2_SCAFFOLD_HOP", "N3_REMOTE"})
    ].copy()
    eligible = eligible.sort_values(
        ["dta_priority_score_384", "target_chembl_id", "pairId"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    eligible["incremental_queue_rank"] = range(1, len(eligible) + 1)
    eligible["incremental_target_rank"] = (
        eligible.groupby("target_chembl_id").cumcount() + 1
    )
    eligible["selection_reason"] = (
        "384-target rerank bidirectional top10%; E1/E2 remote-qualified target; "
        "N2/N3 versus measured positives; missing at least one remote-qualified physical model"
    )
    eligible["hard_gate_policy"] = (
        "active 384 only; GPCR and prior unsupported-class/no-small-molecule-MoA exclusions remain excluded"
    )
    eligible["planned_model_runs"] = (
        eligible["gnina_increment_required"].astype(int)
        + eligible["boltz_increment_required"].astype(int)
    )

    if eligible["pairId"].duplicated().any():
        raise RuntimeError("Incremental queue contains duplicate pairs")
    if not eligible["novelty_lane"].isin({"N2_SCAFFOLD_HOP", "N3_REMOTE"}).all():
        raise RuntimeError("N1 or unevaluable novelty leaked into the incremental queue")
    if not eligible["receptor_protocol_status"].eq("FROZEN_PASS").all():
        raise RuntimeError("A non-frozen receptor entered the incremental queue")
    if not (eligible["gnina_increment_required"] | eligible["boltz_increment_required"]).all():
        raise RuntimeError("A pair without a missing qualified model entered the queue")

    audit_path = OUT / "UNIFIED_RERANK_REMOTE_INCREMENT_CANDIDATE_AUDIT_V1.csv"
    queue_path = OUT / "UNIFIED_RERANK_REMOTE_N2_N3_INCREMENTAL_PAIR_QUEUE_V1.csv"
    gnina_path = OUT / "TASK_GNINA_REMOTE_INCREMENT_V1.csv"
    boltz_path = OUT / "TASK_BOLTZ_REMOTE_INCREMENT_V1.csv"
    missing_model.to_csv(audit_path, index=False)
    eligible.to_csv(queue_path, index=False)
    eligible[eligible["gnina_increment_required"]].to_csv(gnina_path, index=False)
    eligible[eligible["boltz_increment_required"]].to_csv(boltz_path, index=False)

    excluded = missing_model[~missing_model["pairId"].isin(set(eligible["pairId"]))]
    summary = {
        "package_name": "UNIFIED_RERANK_REMOTE_N2_N3_INCREMENT_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "selection": {
            "bidirectional_unrecorded_pairs_in_e1_e2": int(len(base)),
            "pairs_missing_remote_qualified_model_before_novelty": int(len(missing_model)),
            "eligible_n2_n3_pairs": int(len(eligible)),
            "eligible_targets": int(eligible["target_chembl_id"].nunique()),
            "excluded_n1_local_analog": int(excluded["novelty_lane"].eq("N1_LOCAL_ANALOG").sum()),
            "excluded_novelty_not_evaluable": int(
                excluded["novelty_lane"].eq("NOVELTY_NOT_EVALUABLE").sum()
            ),
        },
        "compute": {
            "gnina_pairs": int(eligible["gnina_increment_required"].sum()),
            "boltz_pairs": int(eligible["boltz_increment_required"].sum()),
            "total_model_runs": int(eligible["planned_model_runs"].sum()),
            "novelty_counts": eligible["novelty_lane"].value_counts().to_dict(),
            "route_counts": eligible["unified_target_route"].value_counts().to_dict(),
        },
        "recovered_46_compute_gap": 0,
        "scope_boundary": (
            "This queue closes only the newly created 384-rerank remote N2/N3 gap. "
            "It does not reopen GPCR, unsupported-class, no-small-molecule-MoA, E3-local, "
            "or failed target-gate branches."
        ),
        "outputs": {
            "candidate_audit": str(audit_path),
            "pair_queue": str(queue_path),
            "gnina_task": str(gnina_path),
            "boltz_task": str(boltz_path),
        },
        "source_hashes": {str(V4.relative_to(ROOT)): sha256(V4), str(CONTROLS.relative_to(ROOT)): sha256(CONTROLS)},
    }
    summary_path = OUT / "UNIFIED_RERANK_REMOTE_N2_N3_INCREMENT_SUMMARY_V1.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
