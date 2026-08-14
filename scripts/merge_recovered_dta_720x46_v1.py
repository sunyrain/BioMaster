#!/usr/bin/env python3
"""Merge ConPLEx and predicted-pocket DrugCLIP evidence for the recovered 46."""

from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
RECOVERED_DIR = ROOT / "outputs/recovered_no_experimental_pocket_targets_ch37_v1"
RECOVERED = RECOVERED_DIR / "RECOVERED_NO_EXPERIMENTAL_POCKET_TARGETS_46_V1.csv"
LIGAND_DIR = ROOT / "outputs/strict_affinity_main_queue_2005_2026_v2"
LIGANDS = LIGAND_DIR / "STRICT_UNIQUE_STANDARD_MODEL_LIGAND_STRUCTURES.csv"
OUT = ROOT / "outputs/recovered_dta_720x46_v1"
INPUTS = OUT / "inputs"
EXPECTED_LIGANDS = 720
EXPECTED_TARGETS = 46
EXPECTED_POCKETS = 44
EXPECTED_ROWS = EXPECTED_LIGANDS * EXPECTED_TARGETS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def rank_desc(matrix: np.ndarray, axis: int) -> np.ndarray:
    return np.asarray(rankdata(-matrix, axis=axis, method="average", nan_policy="omit"), dtype=np.float32)


def percentile(rank: np.ndarray, total: int) -> np.ndarray:
    return 1.0 - (rank.astype(np.float32) - 1.0) / max(1, total - 1)


def main() -> None:
    ligands = pd.read_csv(LIGANDS, dtype=str).fillna("").sort_values("ligand_inchikey", kind="mergesort").reset_index(drop=True)
    targets = pd.read_csv(RECOVERED, dtype=str).fillna("").sort_values("target_chembl_id", kind="mergesort").reset_index(drop=True)
    if len(ligands) != EXPECTED_LIGANDS or len(targets) != EXPECTED_TARGETS:
        raise ValueError("Frozen recovered DTA universe size changed")

    conplex = pd.read_csv(
        OUT / "CONPLEX_720_X_46_PREDICTIONS.tsv", sep="\t", header=None,
        names=["ligand_inchikey", "target_chembl_id", "conplex_score"],
    )
    if len(conplex) != EXPECTED_ROWS or conplex[["ligand_inchikey", "target_chembl_id"]].duplicated().any():
        raise ValueError(f"Invalid ConPLEx result: {len(conplex)} rows")
    conplex_matrix = conplex.pivot(index="ligand_inchikey", columns="target_chembl_id", values="conplex_score")
    conplex_matrix = conplex_matrix.reindex(index=ligands["ligand_inchikey"], columns=targets["target_chembl_id"])
    if conplex_matrix.isna().any().any():
        raise ValueError("ConPLEx matrix contains missing values")
    conplex_values = conplex_matrix.to_numpy(dtype=np.float32)

    ligand_manifest = pd.read_csv(INPUTS / "STRICT720_DRUGCLIP_LIGAND_MANIFEST.csv", dtype=str).fillna("")
    ligand_manifest = ligand_manifest[ligand_manifest["status"].eq("OK")].copy()
    ligand_manifest["_key"] = pd.to_numeric(ligand_manifest["lmdb_index"]).astype(int).astype(str)
    ligand_manifest = ligand_manifest.sort_values("_key", kind="mergesort").reset_index(drop=True)
    pocket_manifest_all = pd.read_csv(INPUTS / "RECOVERED44_DRUGCLIP_POCKET_MANIFEST.csv", dtype=str).fillna("")
    pocket_manifest = pocket_manifest_all[pocket_manifest_all["status"].eq("OK")].copy()
    pocket_manifest["_key"] = pd.to_numeric(pocket_manifest["lmdb_index"]).astype(int).astype(str)
    pocket_manifest = pocket_manifest.sort_values("_key", kind="mergesort").reset_index(drop=True)

    molecule_files = sorted((OUT / "drugclip/mol_embeddings").glob("mol_reps*.npy"))
    if len(molecule_files) != 1:
        raise ValueError(f"Expected one molecule embedding file, found {molecule_files}")
    molecule_embeddings = np.asarray(np.load(molecule_files[0]), dtype=np.float32)
    with (INPUTS / "pocket_reps.pkl").open("rb") as handle:
        pocket_names, pocket_embeddings = pickle.load(handle)
    pocket_embeddings = np.asarray(pocket_embeddings, dtype=np.float32)
    if molecule_embeddings.shape != (EXPECTED_LIGANDS, 6, 128) or pocket_embeddings.shape != (EXPECTED_POCKETS, 6, 128):
        raise ValueError(f"Unexpected DrugCLIP shapes: {molecule_embeddings.shape}, {pocket_embeddings.shape}")
    if list(map(str, pocket_names)) != pocket_manifest["target_chembl_id"].tolist():
        raise ValueError("DrugCLIP pocket order differs from the recovered manifest")

    fold_scores_lexical = np.einsum("lfd,tfd->ltf", molecule_embeddings, pocket_embeddings, optimize=True)
    ligand_order = pd.Index(ligand_manifest["ligand_inchikey"]).get_indexer(ligands["ligand_inchikey"])
    if (ligand_order < 0).any():
        raise ValueError("DrugCLIP ligand manifest does not cover 720 ligands")
    eligible_target_ids = pocket_manifest_all[pocket_manifest_all["status"].eq("OK")]["target_chembl_id"].tolist()
    target_embedding_index = {target: index for index, target in enumerate(pocket_manifest["target_chembl_id"])}
    drugclip_values = np.full((EXPECTED_LIGANDS, EXPECTED_TARGETS), np.nan, dtype=np.float32)
    drugclip_std = np.full_like(drugclip_values, np.nan)
    lexical_ordered = fold_scores_lexical[ligand_order]
    for target_column, target_id in enumerate(targets["target_chembl_id"]):
        if target_id not in target_embedding_index:
            continue
        values = lexical_ordered[:, target_embedding_index[target_id], :]
        drugclip_values[:, target_column] = values.mean(axis=1)
        drugclip_std[:, target_column] = values.std(axis=1)

    conplex_target_rank = rank_desc(conplex_values, axis=0)
    conplex_ligand_rank = rank_desc(conplex_values, axis=1)
    conplex_target_pct = percentile(conplex_target_rank, EXPECTED_LIGANDS)
    conplex_ligand_pct = percentile(conplex_ligand_rank, EXPECTED_TARGETS)
    drugclip_target_rank = rank_desc(drugclip_values, axis=0)
    drugclip_ligand_rank = rank_desc(drugclip_values, axis=1)
    drugclip_target_pct = percentile(drugclip_target_rank, EXPECTED_LIGANDS)
    drugclip_ligand_pct = percentile(drugclip_ligand_rank, EXPECTED_POCKETS)

    ligand_index = np.repeat(np.arange(EXPECTED_LIGANDS), EXPECTED_TARGETS)
    target_index = np.tile(np.arange(EXPECTED_TARGETS), EXPECTED_LIGANDS)
    target_columns = [
        "target_chembl_id", "gene_symbol", "uniprot_accession", "target_name", "target_class_l1",
        "assay_lane", "calibration_status", "positive_compounds", "negative_compounds",
        "computed_pocket_evidence", "pocket_consensus_class", "best_method_match_p2rank_rank",
        "best_method_match_fpocket_rank", "evidence_compute_route",
        "target_compute_class_zh", "class_specific_compute_bundle_zh", "wetlab_priority",
    ]
    matrix = pd.DataFrame({
        "ligand_inchikey": ligands.loc[ligand_index, "ligand_inchikey"].to_numpy(),
        "ligand_smiles": ligands.loc[ligand_index, "ligand_smiles"].to_numpy(),
        "drug_names": ligands.loc[ligand_index, "drug_names"].to_numpy(),
        "project_entity_ids": ligands.loc[ligand_index, "project_entity_ids"].to_numpy(),
        **{column: targets.loc[target_index, column].to_numpy() for column in target_columns},
        "conplex_score": conplex_values.reshape(-1),
        "conplex_rank_within_target": conplex_target_rank.reshape(-1),
        "conplex_rank_within_ligand_46": conplex_ligand_rank.reshape(-1),
        "conplex_percentile_within_target": conplex_target_pct.reshape(-1),
        "conplex_percentile_within_ligand_46": conplex_ligand_pct.reshape(-1),
        "drugclip_predicted_pocket_cosine_mean": drugclip_values.reshape(-1),
        "drugclip_predicted_pocket_sixfold_std": drugclip_std.reshape(-1),
        "drugclip_rank_within_target": drugclip_target_rank.reshape(-1),
        "drugclip_rank_within_ligand_44": drugclip_ligand_rank.reshape(-1),
        "drugclip_percentile_within_target": drugclip_target_pct.reshape(-1),
        "drugclip_percentile_within_ligand_44": drugclip_ligand_pct.reshape(-1),
    })
    matrix["drugclip_formal_eligible"] = matrix["target_chembl_id"].isin(eligible_target_ids)
    matrix["dta_target_percentile_disagreement"] = (
        matrix["conplex_percentile_within_target"] - matrix["drugclip_percentile_within_target"]
    ).abs()
    matrix["dta_ligand_percentile_disagreement"] = (
        matrix["conplex_percentile_within_ligand_46"] - matrix["drugclip_percentile_within_ligand_44"]
    ).abs()
    matrix["dta_target_top10pct_concordant"] = (
        matrix["drugclip_formal_eligible"]
        & matrix["conplex_percentile_within_target"].ge(0.9)
        & matrix["drugclip_percentile_within_target"].ge(0.9)
    )
    matrix["dta_bidirectional_top10pct_concordant"] = (
        matrix["dta_target_top10pct_concordant"]
        & matrix["conplex_percentile_within_ligand_46"].ge(0.9)
        & matrix["drugclip_percentile_within_ligand_44"].ge(0.9)
    )
    matrix["sequence_only_fragment_exploratory_top5pct"] = (
        ~matrix["drugclip_formal_eligible"] & matrix["conplex_percentile_within_target"].ge(0.95)
    )

    output = OUT / "RECOVERED_DTA_720_X_46_EVIDENCE_MATRIX_V1.csv.gz"
    matrix.to_csv(output, index=False, compression={"method": "gzip", "compresslevel": 5})
    shortlist = matrix[
        matrix["dta_target_top10pct_concordant"] | matrix["sequence_only_fragment_exploratory_top5pct"]
    ].copy()
    shortlist["shortlist_route"] = np.where(
        shortlist["dta_target_top10pct_concordant"],
        "TWO_MODEL_TARGET_TOP10PCT_PREDICTED_POCKET",
        "CONPLEX_TARGET_TOP5PCT_FRAGMENT_EXPLORATORY",
    )
    shortlist = shortlist.sort_values(
        ["target_chembl_id", "shortlist_route", "dta_target_percentile_disagreement", "conplex_rank_within_target"],
        kind="mergesort", na_position="last",
    )
    shortlist_output = OUT / "RECOVERED_DTA_PHYSICAL_REVIEW_SHORTLIST_V1.csv"
    shortlist.to_csv(shortlist_output, index=False)

    target_summary = targets[[
        "target_chembl_id", "gene_symbol", "target_class_l1", "assay_lane", "calibration_status",
        "positive_compounds", "negative_compounds", "computed_pocket_evidence", "evidence_compute_route",
        "next_compute_action_zh", "wetlab_priority",
    ]].copy()
    counts = shortlist.groupby(["target_chembl_id", "shortlist_route"]).size().unstack(fill_value=0).reset_index()
    target_summary = target_summary.merge(counts, on="target_chembl_id", how="left").fillna(0)
    target_summary.to_csv(OUT / "RECOVERED_DTA_TARGET_SUMMARY_46_V1.csv", index=False)

    exact_shortlist = shortlist[shortlist["drugclip_formal_eligible"]]
    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "PASS",
        "rows": len(matrix),
        "ligands": matrix["ligand_inchikey"].nunique(),
        "targets": matrix["target_chembl_id"].nunique(),
        "conplex_targets": int(matrix.groupby("target_chembl_id")["conplex_score"].count().eq(EXPECTED_LIGANDS).sum()),
        "drugclip_predicted_pocket_targets": int(matrix.groupby("target_chembl_id")["drugclip_predicted_pocket_cosine_mean"].count().eq(EXPECTED_LIGANDS).sum()),
        "two_model_target_top10pct_pairs": int(matrix["dta_target_top10pct_concordant"].sum()),
        "two_model_bidirectional_top10pct_pairs": int(matrix["dta_bidirectional_top10pct_concordant"].sum()),
        "sequence_only_fragment_top5pct_pairs": int(matrix["sequence_only_fragment_exploratory_top5pct"].sum()),
        "physical_review_shortlist_pairs": len(shortlist),
        "targets_with_two_model_shortlist": int(exact_shortlist["target_chembl_id"].nunique()),
        "score_policy": "No cross-target total score. Candidate routing uses within-target threshold concordance; raw scores, ranks, six-fold uncertainty, and disagreement are retained.",
        "scope_policy": "Only 46 targets removed solely for lack of an experimental pocket. DrugCLIP scores are predicted-pocket retrieval evidence for 44 exact AlphaFold structures, never experimental-pocket evidence.",
        "limitations": [
            "ConPLEx and DrugCLIP scores are ranking signals, not Kd or calibrated binder probabilities.",
            "Predicted-pocket DrugCLIP evidence is structurally weaker than holo-pocket retrieval and requires control docking or experimental validation.",
            "DIO1 and RYR1 have sequence-only exploratory rankings because their fragment models are low confidence.",
            "Historical positive/negative counts describe available ChEMBL calibration evidence; they are not necessarily members of the 720-drug candidate set.",
        ],
        "outputs": {
            "matrix": str(output.relative_to(ROOT)),
            "shortlist": str(shortlist_output.relative_to(ROOT)),
        },
        "sha256": {"matrix": sha256(output), "shortlist": sha256(shortlist_output)},
    }
    (OUT / "RECOVERED_DTA_720_X_46_SUMMARY_V1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
