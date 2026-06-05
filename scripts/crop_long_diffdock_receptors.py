from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any


CURATED_FIELDS = [
    "protein_id",
    "label",
    "source_pdb",
    "diffdock_pdb",
    "residue_start",
    "residue_end",
    "residue_count",
    "atom_count",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ca_residues(path: Path) -> list[dict[str, Any]]:
    residues: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            chain_id = line[21].strip() or "_"
            residue_number = line[22:26].strip()
            insertion_code = line[26].strip()
            key = (chain_id, residue_number, insertion_code)
            if key in seen:
                continue
            seen.add(key)
            try:
                plddt = float(line[60:66])
            except ValueError:
                plddt = 0.0
            residues.append(
                {
                    "key": key,
                    "residue_number": residue_number,
                    "insertion_code": insertion_code,
                    "plddt": plddt,
                }
            )
    return residues


def best_window(residues: list[dict[str, Any]], max_residues: int) -> tuple[int, int]:
    if len(residues) <= max_residues:
        return 0, len(residues)
    scores = [float(residue["plddt"]) for residue in residues]
    current = sum(scores[:max_residues])
    best_sum = current
    best_start = 0
    for start in range(1, len(scores) - max_residues + 1):
        current += scores[start + max_residues - 1] - scores[start - 1]
        if current > best_sum:
            best_sum = current
            best_start = start
    return best_start, best_start + max_residues


def crop_pdb(source: Path, target: Path, selected_keys: set[tuple[str, str, str]]) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    atom_count = 0
    with source.open(encoding="utf-8", errors="ignore") as src, target.open("w", encoding="utf-8") as out:
        for line in src:
            if line.startswith(("ATOM", "HETATM", "ANISOU")):
                chain_id = line[21].strip() or "_"
                residue_number = line[22:26].strip()
                insertion_code = line[26].strip()
                if (chain_id, residue_number, insertion_code) not in selected_keys:
                    continue
                if line.startswith("ATOM"):
                    atom_count += 1
                out.write(line)
            elif line.startswith(("TER", "END")):
                continue
        out.write("END\n")
    return atom_count


def main() -> int:
    parser = argparse.ArgumentParser(description="Create DiffDock-safe receptor crops for proteins longer than the ESM residue limit.")
    parser.add_argument("--audit", default="outputs/druggable_proteome/top10000_druggable_diffdock_receptor_audit.csv")
    parser.add_argument("--base-curated", default="outputs/full_test_rerun/receptors/diffdock_ready/receptor_paths.csv")
    parser.add_argument("--out-dir", default="outputs/druggable_proteome/receptors/diffdock_ready")
    parser.add_argument("--out-curated", default="outputs/druggable_proteome/receptors/diffdock_ready/receptor_paths_with_auto_crops.csv")
    parser.add_argument("--metadata", default="outputs/druggable_proteome/receptors/diffdock_ready/auto_crop.metadata.json")
    parser.add_argument("--max-residues", type=int, default=1022)
    args = parser.parse_args()

    audit_rows = read_csv(Path(args.audit))
    curated_rows = [dict(row) for row in read_csv(Path(args.base_curated))]
    curated_ids = {row["protein_id"] for row in curated_rows if row.get("protein_id")}
    output_rows = curated_rows[:]
    crop_rows: list[dict[str, Any]] = []

    for row in audit_rows:
        if row.get("diffdock_receptor_status") != "needs_curated_crop":
            continue
        protein_id = row["protein_id"]
        if protein_id in curated_ids:
            continue
        source = Path(row["original_receptor_pdb_path"])
        residues = ca_residues(source)
        start_idx, end_idx = best_window(residues, args.max_residues)
        selected = residues[start_idx:end_idx]
        if not selected:
            continue
        selected_keys = {item["key"] for item in selected}
        start_label = selected[0]["residue_number"]
        end_label = selected[-1]["residue_number"]
        target = Path(args.out_dir) / f"AF-{protein_id}-F1-model_v6_auto_{start_label}_{end_label}.pdb"
        atom_count = crop_pdb(source, target, selected_keys)
        crop_row = {
            "protein_id": protein_id,
            "label": "auto_high_confidence_crop",
            "source_pdb": str(source),
            "diffdock_pdb": str(target.resolve()),
            "residue_start": start_label,
            "residue_end": end_label,
            "residue_count": len(selected),
            "atom_count": atom_count,
        }
        output_rows.append(crop_row)
        crop_rows.append(crop_row)
        curated_ids.add(protein_id)

    write_csv(Path(args.out_curated), CURATED_FIELDS, output_rows)
    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "audit": args.audit,
        "base_curated": args.base_curated,
        "out_curated": args.out_curated,
        "max_residues": args.max_residues,
        "base_curated_rows": len(curated_rows),
        "auto_crop_rows": len(crop_rows),
        "output_rows": len(output_rows),
        "crop_rows": crop_rows,
    }
    Path(args.metadata).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metadata).write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
