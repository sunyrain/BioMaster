from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def residue_counts_by_chain(pdb_path: Path) -> dict[str, int]:
    residues_by_chain: dict[str, set[tuple[str, str]]] = {}
    with pdb_path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            chain_id = line[21].strip() or "_"
            residue_key = (line[22:26].strip(), line[26].strip())
            residues_by_chain.setdefault(chain_id, set()).add(residue_key)
    return {chain_id: len(residues) for chain_id, residues in residues_by_chain.items()}


def load_curated_receptors(path: Path, root: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    curated: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        diffdock_pdb = row.get("diffdock_pdb", "")
        if diffdock_pdb:
            row = dict(row)
            row["diffdock_pdb"] = str(resolve(root, diffdock_pdb))
            curated[row["protein_id"]] = row
    return curated


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    manifest = resolve(root, args.manifest)
    curated = load_curated_receptors(resolve(root, args.curated_receptors), root)
    rows = read_csv(manifest)

    protein_to_receptor: dict[str, str] = {}
    for row in rows:
        protein_to_receptor.setdefault(row["protein_id"], row.get("receptor_pdb_path") or row.get("receptor_pdb_url") or "")

    receptor_rows: list[dict[str, Any]] = []
    receptor_info: dict[str, dict[str, Any]] = {}
    for protein_id, receptor_value in sorted(protein_to_receptor.items()):
        original_path = resolve(root, receptor_value) if receptor_value else Path("")
        curated_row = curated.get(protein_id)
        chain_counts = residue_counts_by_chain(original_path) if receptor_value and original_path.exists() else {}
        max_chain_residues = max(chain_counts.values()) if chain_counts else 0

        if curated_row:
            diffdock_path = Path(curated_row["diffdock_pdb"])
            status = "curated_" + (curated_row.get("label") or "receptor").replace(" ", "_")
            residue_count = curated_row.get("residue_count", "")
            ready = diffdock_path.exists()
        elif receptor_value and original_path.exists() and max_chain_residues <= args.max_chain_residues:
            diffdock_path = original_path
            status = "full_length_ok"
            residue_count = max_chain_residues
            ready = True
        elif receptor_value and original_path.exists():
            diffdock_path = Path("")
            status = "needs_curated_crop"
            residue_count = max_chain_residues
            ready = False
        else:
            diffdock_path = Path("")
            status = "missing_receptor_pdb"
            residue_count = ""
            ready = False

        info = {
            "protein_id": protein_id,
            "original_receptor_pdb_path": str(original_path) if receptor_value else "",
            "diffdock_receptor_pdb_path": str(diffdock_path) if ready else "",
            "diffdock_receptor_status": status,
            "diffdock_receptor_ready": "true" if ready else "false",
            "max_chain_residues": max_chain_residues,
            "diffdock_residue_count": residue_count,
        }
        receptor_info[protein_id] = info
        receptor_rows.append(info)

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        info = receptor_info[row["protein_id"]]
        enriched.update(info)
        ligand_ready = bool(row.get("ligand_sdf_path"))
        receptor_ready = info["diffdock_receptor_ready"] == "true"
        enriched["diffdock_ready"] = "true" if ligand_ready and receptor_ready else "false"
        output_rows.append(enriched)

    out_manifest = resolve(root, args.out_manifest)
    fieldnames = list(output_rows[0].keys()) if output_rows else []
    write_csv(out_manifest, fieldnames, output_rows)
    receptor_audit = resolve(root, args.receptor_audit)
    write_csv(receptor_audit, list(receptor_rows[0].keys()), receptor_rows)

    status_counts = Counter(row["diffdock_receptor_status"] for row in receptor_rows)
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest": str(manifest),
        "out_manifest": str(out_manifest),
        "receptor_audit": str(receptor_audit),
        "max_chain_residues": args.max_chain_residues,
        "proteins_total": len(receptor_rows),
        "proteins_ready": sum(1 for row in receptor_rows if row["diffdock_receptor_ready"] == "true"),
        "pairs_total": len(output_rows),
        "pairs_diffdock_ready": sum(1 for row in output_rows if row["diffdock_ready"] == "true"),
        "receptor_status_counts": dict(status_counts),
    }
    metadata_path = out_manifest.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a DiffDock-ready manifest with ESM-length-safe receptors.")
    parser.add_argument("--manifest", default="outputs/report_scale/manifest_915k_structure_ready.csv")
    parser.add_argument("--curated-receptors", default="outputs/full_test_rerun/receptors/diffdock_ready/receptor_paths.csv")
    parser.add_argument("--out-manifest", default="outputs/report_scale/manifest_915k_diffdock_ready.csv")
    parser.add_argument("--receptor-audit", default="data/processed/diffdock_ready_receptor_paths.csv")
    parser.add_argument("--max-chain-residues", type=int, default=1022)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
