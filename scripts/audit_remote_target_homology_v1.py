#!/usr/bin/env python3
"""Exclude exact and homologous ChEMBL-MoA targets from the remote DTA lane."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "affinity_first_remote_discovery_v1.yaml"
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


def clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "na", "n/a"} else text


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def split_tokens(value: Any) -> list[str]:
    return [token for token in re.split(r"[;,|\s]+", clean(value)) if token]


def write_fasta(frame: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="ascii") as handle:
        for _, row in frame.iterrows():
            sequence = re.sub(r"[^A-Z]", "", clean(row["sequence"]).upper())
            if not sequence:
                continue
            handle.write(f">{row['sequence_key']}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def build_homology_table(project_targets: set[str], sequences: pd.DataFrame, out_dir: Path) -> Path:
    matrix_path = out_dir / "PROJECT463_VS_CHEMBL891_MMSEQS_HOMOLOGY_V1.tsv.gz"
    if matrix_path.exists():
        return matrix_path
    query = sequences[sequences["sequence_key"].isin(project_targets)].copy()
    if len(query) != len(project_targets):
        raise ValueError(f"Missing query sequences: {len(query)} != {len(project_targets)}")
    query_fasta = out_dir / "project463_queries.fasta"
    reference_fasta = out_dir / "chembl891_reference.fasta"
    raw_path = out_dir / "project463_vs_chembl891_mmseqs.tsv"
    tmp_dir = out_dir / "mmseqs_tmp"
    write_fasta(query, query_fasta)
    write_fasta(sequences, reference_fasta)
    command = [
        "mmseqs",
        "easy-search",
        str(query_fasta),
        str(reference_fasta),
        str(raw_path),
        str(tmp_dir),
        "--format-output",
        "query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits",
        "--min-seq-id",
        "0.20",
        "-c",
        "0.25",
        "--cov-mode",
        "0",
        "-s",
        "7.5",
        "--threads",
        "32",
        "--remove-tmp-files",
        "1",
    ]
    subprocess.run(command, check=True)
    columns = ["query", "target", "fident", "alnlen", "qlen", "tlen", "qcov", "tcov", "evalue", "bits"]
    matrix = pd.read_csv(raw_path, sep="\t", header=None, names=columns)
    for column in ["fident", "alnlen", "qlen", "tlen", "qcov", "tcov", "evalue", "bits"]:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
    matrix["shorter_sequence_coverage"] = matrix["alnlen"] / matrix[["qlen", "tlen"]].min(axis=1)
    matrix.to_csv(matrix_path, sep="\t", index=False, compression={"method": "gzip", "compresslevel": 5})
    raw_path.unlink(missing_ok=True)
    query_fasta.unlink(missing_ok=True)
    reference_fasta.unlink(missing_ok=True)
    return matrix_path


def build_known_targets(
    physical: pd.DataFrame,
    mechanisms: pd.DataFrame,
    known_controls: pd.DataFrame,
    sequences: pd.DataFrame,
) -> tuple[dict[str, set[str]], dict[str, str]]:
    sequence_by_gene: dict[str, set[str]] = {}
    gene_by_sequence: dict[str, str] = {}
    for _, row in sequences.iterrows():
        sequence_key = clean(row["sequence_key"])
        genes = split_tokens(row.get("gene_names", ""))
        gene_by_sequence[sequence_key] = ";".join(genes)
        for gene in genes:
            sequence_by_gene.setdefault(gene, set()).add(sequence_key)

    direct = mechanisms[
        mechanisms["organism"].eq("Homo sapiens")
        & mechanisms["target_type"].eq("SINGLE PROTEIN")
        & mechanisms["action_type"].astype(str).str.upper().isin(DIRECT_ACTIONS)
    ].copy()
    genes_by_chembl: dict[str, set[str]] = {}
    for _, row in direct.iterrows():
        genes = set(split_tokens(row.get("component_gene_symbols", "")))
        for column in ["molecule_chembl_id", "parent_molecule_chembl_id"]:
            compound = clean(row.get(column))
            if compound:
                genes_by_chembl.setdefault(compound, set()).update(genes)

    known_sequence_by_ligand: dict[str, set[str]] = {}
    source_by_ligand = physical[["model_ligand_smiles", "source_drug_ids"]].drop_duplicates()
    for _, row in source_by_ligand.iterrows():
        known_sequences: set[str] = set()
        for drug in split_tokens(row["source_drug_ids"]):
            base = drug.split("__")[0]
            for gene in genes_by_chembl.get(base, set()):
                known_sequences.update(sequence_by_gene.get(gene, set()))
        known_sequence_by_ligand[clean(row["model_ligand_smiles"])] = known_sequences

    # The signed project control table also contains FDA workbook mappings that
    # may not be represented by a ChEMBL mechanism row.
    drug_to_ligand: dict[str, str] = {}
    for _, row in source_by_ligand.iterrows():
        ligand = clean(row["model_ligand_smiles"])
        for drug in split_tokens(row["source_drug_ids"]):
            drug_to_ligand[drug] = ligand
    for _, row in known_controls.iterrows():
        drug = clean(row.get("drug_chembl_id"))
        ligand = drug_to_ligand.get(drug, "")
        sequence_key = clean(row.get("sequence_key"))
        if ligand and sequence_key:
            known_sequence_by_ligand.setdefault(ligand, set()).add(sequence_key)
    return known_sequence_by_ligand, gene_by_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = {name: ROOT / value for name, value in config["inputs"].items()}
    out_dir = ROOT / config["outputs"]["directory"]
    out_dir.mkdir(parents=True, exist_ok=True)

    master_path = out_dir / "PHYSICAL_PAIR_UNIVERSE_334749_V1.csv.gz"
    physical = pd.read_csv(master_path, low_memory=False).fillna("")
    sequences = pd.read_csv(paths["protein_sequences"], low_memory=False).fillna("")
    mechanisms = pd.read_csv(paths["chembl37_mechanisms"], low_memory=False).fillna("")
    controls = pd.read_csv(paths["known_controls"], low_memory=False).fillna("")
    project_targets = set(physical["sequence_key"].astype(str))
    matrix_path = build_homology_table(project_targets, sequences, out_dir)
    matrix = pd.read_csv(matrix_path, sep="\t", low_memory=False)
    matrix_lookup = {
        (clean(row.query), clean(row.target)): (
            float(row.fident),
            float(row.shorter_sequence_coverage),
            int(row.alnlen),
        )
        for row in matrix.itertuples(index=False)
    }
    known_by_ligand, gene_by_sequence = build_known_targets(physical, mechanisms, controls, sequences)

    max_identity: list[float] = []
    max_coverage: list[float] = []
    max_aligned: list[int] = []
    nearest_sequence: list[str] = []
    expanded_exact: list[bool] = []
    homology_risk: list[bool] = []
    for row in physical.itertuples(index=False):
        ligand = clean(row.model_ligand_smiles)
        candidate = clean(row.sequence_key)
        known_targets = known_by_ligand.get(ligand, set())
        best_merit = -1.0
        best = (0.0, 0.0, 0, "")
        for known_target in known_targets:
            identity, coverage, aligned = matrix_lookup.get((candidate, known_target), (0.0, 0.0, 0))
            merit = identity * min(1.0, coverage / 0.60)
            if merit > best_merit:
                best_merit = merit
                best = (identity, coverage, aligned, known_target)
        identity, coverage, aligned, nearest = best
        exact = candidate in known_targets
        risk = not exact and (
            (identity >= 0.40 and coverage >= 0.60)
            or (identity >= 0.55 and coverage >= 0.30)
        )
        max_identity.append(identity)
        max_coverage.append(coverage)
        max_aligned.append(aligned)
        nearest_sequence.append(nearest)
        expanded_exact.append(exact)
        homology_risk.append(risk)

    physical["expanded_known_target_count_v1"] = physical["model_ligand_smiles"].map(
        lambda ligand: len(known_by_ligand.get(clean(ligand), set()))
    )
    physical["expanded_exact_known_target_v1"] = expanded_exact
    physical["max_known_target_mmseqs_identity_v1"] = max_identity
    physical["max_known_target_shorter_coverage_v1"] = max_coverage
    physical["max_known_target_aligned_residues_v1"] = max_aligned
    physical["nearest_known_target_sequence_key_v1"] = nearest_sequence
    physical["nearest_known_target_gene_v1"] = [gene_by_sequence.get(key, "") for key in nearest_sequence]
    physical["expanded_target_homology_risk_v1"] = homology_risk

    exact = as_bool(physical["expanded_exact_known_target_v1"])
    homology = as_bool(physical["expanded_target_homology_risk_v1"])
    base_remote = as_bool(physical["remote_pair_eligible_v1"])
    strict = as_bool(physical["strict_structure_ready_v1"])
    review = as_bool(physical["review_structure_ready_v1"])
    physical["remote_pair_homology_audited_v1"] = base_remote & ~exact & ~homology
    physical["dta_stage1_strict_homology_audited_v1"] = (
        physical["remote_pair_homology_audited_v1"] & strict
    )
    physical["dta_stage1_review_homology_audited_v1"] = (
        physical["remote_pair_homology_audited_v1"] & review
    )
    physical["dta_queue_homology_audited_v1"] = physical["dta_queue_v1"]
    physical.loc[homology, "dta_queue_homology_audited_v1"] = "homologous_target_extension_control"
    physical.loc[exact, "dta_queue_homology_audited_v1"] = "expanded_known_positive_control"
    physical.loc[
        physical["dta_stage1_strict_homology_audited_v1"], "dta_queue_homology_audited_v1"
    ] = "remote_structure_strict_homology_audited"

    output = out_dir / "PHYSICAL_PAIR_UNIVERSE_334749_HOMOLOGY_AUDITED_V1.csv.gz"
    strict_output = out_dir / "DTA_STAGE1_REMOTE_STRICT_HOMOLOGY_AUDITED_V1.csv.gz"
    physical.to_csv(output, index=False, compression={"method": "gzip", "compresslevel": 5})
    physical.loc[physical["dta_stage1_strict_homology_audited_v1"]].to_csv(
        strict_output, index=False, compression={"method": "gzip", "compresslevel": 5}
    )
    summary = {
        "status": "passed",
        "created_utc": now_utc(),
        "rows": int(len(physical)),
        "ligands_with_expanded_known_target_context": int(
            physical.loc[physical["expanded_known_target_count_v1"] > 0, "model_ligand_smiles"].nunique()
        ),
        "expanded_exact_known_rows": int(exact.sum()),
        "expanded_homology_risk_rows": int(homology.sum()),
        "remote_homology_audited_rows": int(physical["remote_pair_homology_audited_v1"].sum()),
        "strict_structure_remote_homology_audited_rows": int(
            physical["dta_stage1_strict_homology_audited_v1"].sum()
        ),
        "review_structure_remote_homology_audited_rows": int(
            physical["dta_stage1_review_homology_audited_v1"].sum()
        ),
        "homology_method": (
            "MMseqs2 project463-vs-ChEMBL891; extension risk if identity>=0.40 and "
            "shorter-sequence coverage>=0.60, or identity>=0.55 and coverage>=0.30"
        ),
        "outputs": {
            "master": str(output.relative_to(ROOT)),
            "strict_manifest": str(strict_output.relative_to(ROOT)),
            "homology_matrix": str(matrix_path.relative_to(ROOT)),
        },
        "sha256": {
            "master": hashlib.sha256(output.read_bytes()).hexdigest(),
            "strict_manifest": hashlib.sha256(strict_output.read_bytes()).hexdigest(),
            "homology_matrix": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
        },
    }
    summary_path = out_dir / "REMOTE_TARGET_HOMOLOGY_AUDIT_V1_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
