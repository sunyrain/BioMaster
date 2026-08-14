#!/usr/bin/env python3
"""Remove Platinum structures homologous to the locked PBCNet2 mutation test."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import pandas as pd
from Bio.Data.PDBData import protein_letters_3to1_extended


class UnionFind:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pdb_chain_sequences(path: Path) -> dict[str, str]:
    chains: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, int, str]] = set()
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("ATOM") or len(line) < 27:
            continue
        altloc = line[16:17]
        if altloc not in {" ", "A", "1"}:
            continue
        try:
            position = int(line[22:26])
        except ValueError:
            continue
        chain = line[21:22].strip() or "_"
        insertion = line[26:27].strip()
        key = (chain, position, insertion)
        if key in seen:
            continue
        seen.add(key)
        resname = line[17:20].strip().upper()
        one = protein_letters_3to1_extended.get(resname, "X").upper()
        chains[chain].append(one)
    return {chain: "".join(sequence) for chain, sequence in chains.items()}


def write_fasta(records: dict[str, str], path: Path) -> None:
    with path.open("w") as handle:
        for name, sequence in sorted(records.items()):
            handle.write(f">{name}\n{sequence}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=(
            "outputs/old_drug_target_sota_v1/platinum_official_audit_v1/"
            "PLATINUM_NORMALIZED_MUTATION_MANIFEST_V1.csv"
        ),
    )
    parser.add_argument(
        "--pbcnet2-root", default=".external/pbcnet2/data/Mutation"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/old_drug_target_sota_v1/platinum_pbcnet2_homology_exclusion_v1",
    )
    parser.add_argument("--minimum-identity", type=float, default=0.30)
    parser.add_argument("--minimum-shorter-coverage", type=float, default=0.80)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    pbcnet2_root = Path(args.pbcnet2_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path, low_memory=False)

    sequences: dict[str, str] = {}
    row_sequence_id: dict[int, str] = {}
    for _, row in manifest.iterrows():
        if not bool(row["phase_a_atomic_wt_eligible_before_homology_exclusion"]):
            continue
        pdb_path = Path(str(row["wt_pdb_path"]))
        chain = str(row["chain"]) if pd.notna(row["chain"]) else "_"
        chain = chain or "_"
        sequence = pdb_chain_sequences(pdb_path).get(chain, "")
        if len(sequence) < 20:
            continue
        sequence_id = f"PLAT__{row['wt_pdb_id']}__{chain}"
        if sequence_id in sequences and sequences[sequence_id] != sequence:
            raise RuntimeError(f"inconsistent sequence for {sequence_id}")
        sequences[sequence_id] = sequence
        row_sequence_id[int(row["platinum_row_index"])] = sequence_id

    pbc_sequence_ids: set[str] = set()
    for target_dir in sorted(path for path in pbcnet2_root.iterdir() if path.is_dir()):
        wt_paths = sorted(target_dir.glob("WT_*_protein.pdb"))
        if len(wt_paths) != 1:
            raise RuntimeError(f"expected one WT protein PDB in {target_dir}, found {wt_paths}")
        for chain, sequence in pdb_chain_sequences(wt_paths[0]).items():
            if len(sequence) < 20:
                continue
            sequence_id = f"PBC__{target_dir.name}__{chain}"
            sequences[sequence_id] = sequence
            pbc_sequence_ids.add(sequence_id)

    fasta_path = output_dir / "PLATINUM_AND_PBCNET2_STRUCTURE_CHAINS_V1.fasta"
    write_fasta(sequences, fasta_path)
    search_path = output_dir / "MMSEQS_ALL_VS_ALL_V1.tsv"
    temporary_dir = output_dir / "mmseqs_tmp"
    subprocess.run(
        [
            "mmseqs",
            "easy-search",
            str(fasta_path),
            str(fasta_path),
            str(search_path),
            str(temporary_dir),
            "--min-seq-id",
            str(args.minimum_identity),
            "-c",
            "0.0",
            "--format-output",
            "query,target,fident,alnlen,qlen,tlen",
            "--threads",
            "8",
        ],
        check=True,
    )

    hits = pd.read_csv(
        search_path,
        sep="\t",
        names=["query", "target", "fident", "alnlen", "qlen", "tlen"],
    )
    identity = hits["fident"].astype(float)
    identity = identity.where(identity <= 1.0, identity / 100.0)
    shorter_coverage = hits["alnlen"] / hits[["qlen", "tlen"]].min(axis=1)
    accepted = hits[
        (identity >= args.minimum_identity)
        & (shorter_coverage >= args.minimum_shorter_coverage)
    ].copy()
    accepted["identity_fraction"] = identity[
        (identity >= args.minimum_identity)
        & (shorter_coverage >= args.minimum_shorter_coverage)
    ]
    accepted["shorter_coverage"] = shorter_coverage[
        (identity >= args.minimum_identity)
        & (shorter_coverage >= args.minimum_shorter_coverage)
    ]
    accepted_path = output_dir / "MMSEQS_ACCEPTED_HOMOLOGY_EDGES_V1.csv"
    accepted.to_csv(accepted_path, index=False)

    union_find = UnionFind(list(sequences))
    for _, hit in accepted.iterrows():
        union_find.union(str(hit["query"]), str(hit["target"]))
    components: dict[str, list[str]] = defaultdict(list)
    for sequence_id in sequences:
        components[union_find.find(sequence_id)].append(sequence_id)
    ordered_components = sorted(
        (sorted(members) for members in components.values()), key=lambda members: members[0]
    )
    sequence_cluster = {
        member: f"HC30_{index:04d}"
        for index, members in enumerate(ordered_components)
        for member in members
    }
    pbc_clusters = {sequence_cluster[sequence_id] for sequence_id in pbc_sequence_ids}

    manifest["structure_chain_sequence_id"] = manifest["platinum_row_index"].map(
        row_sequence_id
    )
    manifest["structure_homology_cluster_30"] = manifest[
        "structure_chain_sequence_id"
    ].map(sequence_cluster)
    manifest["pbcnet2_structure_homology_overlap"] = manifest[
        "structure_homology_cluster_30"
    ].isin(pbc_clusters)
    manifest["phase_a_atomic_wt_eligible_after_homology_exclusion"] = (
        manifest["phase_a_atomic_wt_eligible_before_homology_exclusion"]
        & manifest["structure_chain_sequence_id"].notna()
        & ~manifest["pbcnet2_structure_homology_overlap"]
    )

    annotated_path = output_dir / "PLATINUM_HOMOLOGY_ANNOTATED_MANIFEST_V1.csv"
    manifest.to_csv(annotated_path, index=False)
    eligible_path = output_dir / "PLATINUM_PHASE_A_HOMOLOGY_SAFE_ELIGIBLE_V1.csv"
    manifest.loc[
        manifest["phase_a_atomic_wt_eligible_after_homology_exclusion"]
    ].to_csv(eligible_path, index=False)

    component_rows = []
    for index, members in enumerate(ordered_components):
        cluster = f"HC30_{index:04d}"
        component_rows.append(
            {
                "homology_cluster": cluster,
                "members": ";".join(members),
                "member_count": len(members),
                "contains_pbcnet2": cluster in pbc_clusters,
            }
        )
    component_path = output_dir / "STRUCTURE_HOMOLOGY_COMPONENTS_V1.csv"
    pd.DataFrame(component_rows).to_csv(component_path, index=False)

    before = manifest["phase_a_atomic_wt_eligible_before_homology_exclusion"]
    after = manifest["phase_a_atomic_wt_eligible_after_homology_exclusion"]
    summary = {
        "schema_version": "PLATINUM_PBCNET2_HOMOLOGY_EXCLUSION_V1",
        "status": "PASS",
        "thresholds": {
            "minimum_identity_fraction": args.minimum_identity,
            "minimum_alignment_coverage_of_shorter_chain": args.minimum_shorter_coverage,
            "component_rule": "connected_components_of_accepted_pairwise_edges",
        },
        "counts": {
            "all_sequences": len(sequences),
            "platinum_structure_chains": sum(name.startswith("PLAT__") for name in sequences),
            "pbcnet2_structure_chains": len(pbc_sequence_ids),
            "homology_components": len(ordered_components),
            "pbcnet2_contaminated_components": len(pbc_clusters),
            "eligible_rows_before_homology_exclusion": int(before.sum()),
            "eligible_rows_removed_by_structure_homology": int(
                (before & manifest["pbcnet2_structure_homology_overlap"]).sum()
            ),
            "eligible_rows_after_homology_exclusion": int(after.sum()),
            "eligible_uniprot_ids_after_homology_exclusion": int(
                manifest.loc[after, "uniprot_id"].nunique()
            ),
            "eligible_homology_clusters_after_exclusion": int(
                manifest.loc[after, "structure_homology_cluster_30"].nunique()
            ),
        },
        "audit": {
            "no_pbcnet2_homology_cluster_in_training": bool(
                not manifest.loc[after, "structure_homology_cluster_30"].isin(pbc_clusters).any()
            ),
            "all_training_rows_have_structure_sequence": bool(
                manifest.loc[after, "structure_chain_sequence_id"].notna().all()
            ),
        },
        "input_manifest_sha256": sha256(manifest_path),
        "fasta_sha256": sha256(fasta_path),
        "files": {
            "annotated_manifest_csv": str(annotated_path),
            "homology_safe_eligible_csv": str(eligible_path),
            "accepted_homology_edges_csv": str(accepted_path),
            "homology_components_csv": str(component_path),
        },
    }
    summary_path = output_dir / "PLATINUM_PBCNET2_HOMOLOGY_EXCLUSION_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
