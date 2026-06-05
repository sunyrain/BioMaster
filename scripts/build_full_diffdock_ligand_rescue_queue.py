from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import Descriptors


INPUT_FIELDNAMES = ["complex_name", "protein_path", "protein_sequence", "ligand_description"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def read_first_mol(path: Path) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    mol = next((item for item in supplier if item is not None), None)
    if mol is None:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        mol = next((item for item in supplier if item is not None), None)
        if mol is None:
            raise ValueError(f"Could not read molecule from {path}")
        Chem.SanitizeMol(mol)
    return mol


def fragment_key(mol: Chem.Mol) -> tuple[int, int, float]:
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    return carbon_count, mol.GetNumHeavyAtoms(), float(Descriptors.MolWt(mol))


def write_parent_ligand(original_sdf: Path, parent_sdf: Path, ligand_id: str) -> dict[str, Any]:
    mol = read_first_mol(original_sdf)
    fragments = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    if not fragments:
        fragments = [mol]
    selected = max(fragments, key=fragment_key)
    selected.SetProp("_Name", f"{ligand_id}_parent")
    parent_sdf.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(parent_sdf))
    writer.write(selected)
    writer.close()
    return {
        "originalMolAtoms": int(mol.GetNumAtoms()),
        "originalMolHeavyAtoms": int(mol.GetNumHeavyAtoms()),
        "fragmentCount": int(len(fragments)),
        "fragmentSummaries": [
            {
                "index": index,
                "atoms": int(fragment.GetNumAtoms()),
                "heavyAtoms": int(fragment.GetNumHeavyAtoms()),
                "carbonAtoms": int(sum(1 for atom in fragment.GetAtoms() if atom.GetAtomicNum() == 6)),
                "molWt": round(float(Descriptors.MolWt(fragment)), 4),
                "smiles": Chem.MolToSmiles(fragment),
            }
            for index, fragment in enumerate(fragments)
        ],
        "selectedParentAtoms": int(selected.GetNumAtoms()),
        "selectedParentHeavyAtoms": int(selected.GetNumHeavyAtoms()),
        "selectedParentMolWt": round(float(Descriptors.MolWt(selected)), 4),
        "selectedParentSmiles": Chem.MolToSmiles(selected),
        "parentSdf": str(parent_sdf),
    }


def job_id_from_chunk_path(path: str) -> int | None:
    match = re.search(r"chunk_(\d+)", path)
    return int(match.group(1)) if match else None


def load_input_rows(run_dir: Path, ligand_id: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    prefix = f"{ligand_id}__"
    for path in sorted((run_dir / "inputs").glob("diffdock_full_chunk_*.csv")):
        job_id = job_id_from_chunk_path(path.name)
        for row in read_csv(path):
            pair_id = row.get("complex_name", "")
            if not pair_id.startswith(prefix):
                continue
            rows[pair_id] = {
                **row,
                "source_input_csv": str(path),
                "source_job_id": "" if job_id is None else str(job_id),
            }
    return rows


def collect_missing_rows(run_dir: Path, ligand_id: str, max_source_job_id: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    prefix = f"{ligand_id}__"
    for score_csv in sorted((run_dir / "scores").glob("diffdock_full_chunk_*.scores.csv")):
        job_id = job_id_from_chunk_path(score_csv.name)
        if max_source_job_id is not None and job_id is not None and job_id > max_source_job_id:
            continue
        for row in read_csv(score_csv):
            pair_id = row.get("pair_id") or row.get("complex_name", "")
            if not pair_id.startswith(prefix):
                continue
            if row.get("status") == "completed":
                continue
            rows.append(
                {
                    "pair_id": pair_id,
                    "source_score_csv": str(score_csv),
                    "source_job_id": "" if job_id is None else str(job_id),
                    "source_status": row.get("status", ""),
                    "source_error": row.get("error", ""),
                    "source_chunk": row.get("source_chunk", ""),
                }
            )
    rows.sort(key=lambda item: (int(item["source_job_id"] or 0), item["pair_id"]))
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if row["pair_id"] in seen:
            continue
        deduped.append(row)
        seen.add(row["pair_id"])
    return deduped


def build_queue(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    run_dir = resolve(root, args.run_dir)
    out_dir = resolve(root, args.out_dir)
    original_sdf = resolve(root, args.original_sdf)
    parent_sdf = resolve(root, args.parent_sdf)
    ligand_info = write_parent_ligand(original_sdf, parent_sdf, args.ligand_id)

    input_by_pair = load_input_rows(run_dir, args.ligand_id)
    missing_rows = collect_missing_rows(run_dir, args.ligand_id, args.max_source_job_id)

    rescue_inputs: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for missing in missing_rows:
        source = input_by_pair.get(missing["pair_id"])
        if not source:
            skipped_rows.append({**missing, "skip_reason": "source_input_missing"})
            continue
        protein_path = source.get("protein_path", "")
        if not protein_path or not Path(protein_path).exists():
            skipped_rows.append({**missing, "skip_reason": "protein_path_missing", **source})
            continue
        input_row = {
            "complex_name": missing["pair_id"],
            "protein_path": protein_path,
            "protein_sequence": source.get("protein_sequence", ""),
            "ligand_description": str(parent_sdf),
        }
        rescue_inputs.append(input_row)
        manifest_rows.append(
            {
                **missing,
                "source_input_csv": source.get("source_input_csv", ""),
                "protein_path": protein_path,
                "original_ligand_description": source.get("ligand_description", ""),
                "rescue_ligand_description": str(parent_sdf),
                "rescue_reason": "largest_organic_parent_for_multifragment_ligand",
            }
        )

    input_dir = out_dir / "inputs"
    output_dir = out_dir / "outputs"
    score_dir = out_dir / "scores"
    log_dir = out_dir / "logs"

    job_rows: list[dict[str, Any]] = []
    for job_id, start in enumerate(range(0, len(rescue_inputs), args.chunk_size)):
        chunk_rows = rescue_inputs[start : start + args.chunk_size]
        chunk_csv = input_dir / f"diffdock_ligand_rescue_chunk_{job_id:05d}.csv"
        write_csv(chunk_csv, INPUT_FIELDNAMES, chunk_rows)
        job_rows.append(
            {
                "job_id": job_id,
                "chunk_csv": rel(root, chunk_csv),
                "out_dir": rel(root, output_dir / f"chunk_{job_id:05d}"),
                "score_csv": rel(root, score_dir / f"diffdock_ligand_rescue_chunk_{job_id:05d}.scores.csv"),
                "log_file": rel(root, log_dir / f"diffdock_ligand_rescue_chunk_{job_id:05d}.log"),
                "row_count": len(chunk_rows),
                "gpu_slot": job_id % 4,
                "status": "pending",
            }
        )

    job_index = out_dir / "diffdock_ligand_rescue_job_index.csv"
    write_csv(
        job_index,
        ["job_id", "chunk_csv", "out_dir", "score_csv", "log_file", "row_count", "gpu_slot", "status"],
        job_rows,
    )
    write_csv(out_dir / "diffdock_ligand_rescue_manifest.csv", sorted(manifest_rows[0]) if manifest_rows else [], manifest_rows)
    if skipped_rows:
        write_csv(out_dir / "diffdock_ligand_rescue_skipped.csv", sorted(skipped_rows[0]), skipped_rows)

    summary = {
        "createdUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "Full DiffDock ligand-specific technical rescue queue.",
        "ligandId": args.ligand_id,
        "runDir": str(run_dir),
        "outDir": str(out_dir),
        "originalSdf": str(original_sdf),
        "parentSdf": str(parent_sdf),
        "inputRowsForLigand": len(input_by_pair),
        "missingRowsSelected": len(missing_rows),
        "queuedRows": len(rescue_inputs),
        "skippedRows": len(skipped_rows),
        "jobs": len(job_rows),
        "chunkSize": args.chunk_size,
        "maxSourceJobId": args.max_source_job_id,
        "jobIndex": str(job_index),
        "manifest": str(out_dir / "diffdock_ligand_rescue_manifest.csv"),
        "ligandPreparation": ligand_info,
        "executionNote": (
            "Do not run this queue while the full DiffDock queue occupies all GPUs. "
            "It is prepared for later execution with scripts/run_diffdock_dynamic_queue.py."
        ),
    }
    write_json(out_dir / "diffdock_ligand_rescue_summary.json", summary)
    (out_dir / "DIFFDOCK_LIGAND_RESCUE_QUEUE.md").write_text(
        "\n".join(
            [
                "# Full DiffDock Ligand Rescue Queue",
                "",
                f"- Generated: {summary['createdUtc']}",
                f"- Ligand: `{args.ligand_id}`",
                f"- Original SDF: `{original_sdf}`",
                f"- Parent SDF: `{parent_sdf}`",
                f"- Missing rows selected: {len(missing_rows)}",
                f"- Queued rows: {len(rescue_inputs)}",
                f"- Jobs: {len(job_rows)}",
                f"- Job index: `{job_index}`",
                "",
                "This queue is for technical rescue of DiffDock rows that failed with the original ligand representation. It is not a biological negative/positive call.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a ligand-specific rescue queue for full DiffDock technical missing outputs.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-dir", default="outputs/report_scale/diffdock_full_run")
    parser.add_argument("--ligand-id", default="CHEMBL3039504")
    parser.add_argument("--original-sdf", default="data/processed/ligands_sdf_chembl/CHEMBL3039504.sdf")
    parser.add_argument("--parent-sdf", default="data/processed/ligands_sdf_chembl_parent/CHEMBL3039504_parent.sdf")
    parser.add_argument("--out-dir", default="outputs/report_scale/diffdock_full_ligand_rescue/CHEMBL3039504_parent")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-source-job-id", type=int, default=None)
    args = parser.parse_args()
    summary = build_queue(Path(args.root).resolve(), args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
