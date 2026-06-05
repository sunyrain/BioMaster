from __future__ import annotations

import argparse
import csv
import json
import re
import tarfile
import time
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def tar_accessions(tar_path: Path) -> set[str]:
    accessions: set[str] = set()
    pattern = re.compile(r"AF-([A-Z0-9]+)-F1-model_v\d+\.pdb\.gz$")
    with tarfile.open(tar_path, "r") as archive:
        for member in archive:
            match = pattern.match(Path(member.name).name)
            if match:
                accessions.add(match.group(1))
    return accessions


def local_pdb_exists(receptor_dir: Path, protein_id: str) -> bool:
    path = receptor_dir / f"AF-{protein_id}-F1-model_v6.pdb"
    return path.exists() and path.stat().st_size > 0


def choose_representative(
    candidate_ids: list[str],
    original_id: str,
    local_receptor_dir: Path,
    tar_ids: set[str],
) -> tuple[str, str]:
    if original_id in tar_ids:
        return original_id, "original_in_alphafold_tar"
    if local_pdb_exists(local_receptor_dir, original_id):
        return original_id, "original_local_pdb"

    for protein_id in candidate_ids:
        if protein_id.startswith(("P", "Q", "O")) and protein_id in tar_ids:
            return protein_id, "same_sequence_reviewed_like_tar"
    for protein_id in candidate_ids:
        if protein_id in tar_ids:
            return protein_id, "same_sequence_tar"
    for protein_id in candidate_ids:
        if local_pdb_exists(local_receptor_dir, protein_id):
            return protein_id, "same_sequence_local_pdb"
    return original_id, "no_replacement_found"


def main() -> int:
    parser = argparse.ArgumentParser(description="Select AlphaFold-available representatives for druggable proteome DiffDock seed groups.")
    parser.add_argument("--input", default="outputs/druggable_proteome/top10000_druggable_diffdock_seed_manifest_unique_sequences.csv")
    parser.add_argument("--proteins", default="outputs/druggable_proteome/protein_library_druggable_chembl.csv")
    parser.add_argument("--alphafold-tar", default="UP000005640_9606_HUMAN_v6.tar")
    parser.add_argument("--local-receptor-dir", default="data/processed/alphafold_receptors_v6")
    parser.add_argument("--output", default="outputs/druggable_proteome/top10000_druggable_diffdock_seed_manifest_structure_representatives.csv")
    parser.add_argument("--metadata", default="outputs/druggable_proteome/top10000_druggable_diffdock_seed_manifest_structure_representatives.metadata.json")
    args = parser.parse_args()

    rows = read_csv(Path(args.input))
    proteins_by_id = {row["protein_id"]: row for row in read_csv(Path(args.proteins))}
    local_receptor_dir = Path(args.local_receptor_dir)
    tar_ids = tar_accessions(Path(args.alphafold_tar))

    output_rows: list[dict[str, Any]] = []
    replacement_counts: dict[str, int] = {}
    missing = 0
    for row in rows:
        represented_ids = [value for value in row.get("represented_protein_ids", "").split(";") if value]
        if not represented_ids:
            represented_ids = [row["protein_id"]]
        chosen_id, reason = choose_representative(represented_ids, row["protein_id"], local_receptor_dir, tar_ids)
        replacement_counts[reason] = replacement_counts.get(reason, 0) + 1
        protein = proteins_by_id.get(chosen_id, {})
        if chosen_id not in tar_ids and not local_pdb_exists(local_receptor_dir, chosen_id):
            missing += 1

        enriched = dict(row)
        old_pair_id = row["pair_id"]
        enriched["source_pair_id"] = old_pair_id
        enriched["protein_id"] = chosen_id
        enriched["pair_id"] = f"{row['drug_id']}__{chosen_id}"
        enriched["gene_name"] = protein.get("gene_name", row.get("gene_name", ""))
        enriched["protein_name"] = protein.get("protein_name", row.get("protein_name", ""))
        enriched["protein_sequence"] = protein.get("sequence", row.get("protein_sequence", ""))
        enriched["receptor_pdb_url"] = f"https://alphafold.ebi.ac.uk/files/AF-{chosen_id}-F1-model_v6.pdb"
        enriched["receptor_cif_url"] = f"https://alphafold.ebi.ac.uk/files/AF-{chosen_id}-F1-model_v6.cif"
        enriched["diffdock_output_dir"] = f"outputs/druggable_proteome/diffdock_runs/{enriched['pair_id']}"
        enriched["representative_selection_reason"] = reason
        output_rows.append(enriched)

    fieldnames = list(rows[0].keys()) if rows else []
    for field in ["source_pair_id", "representative_selection_reason"]:
        if field not in fieldnames:
            fieldnames.append(field)
    write_csv(Path(args.output), fieldnames, output_rows)

    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": args.input,
        "output": args.output,
        "rows": len(output_rows),
        "unique_drug_sequence_groups": len({(row["drug_id"], row["sequence_key"]) for row in output_rows}),
        "unique_representative_proteins": len({row["protein_id"] for row in output_rows}),
        "alphafold_unavailable_rows": missing,
        "selection_reason_counts": replacement_counts,
    }
    Path(args.metadata).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
