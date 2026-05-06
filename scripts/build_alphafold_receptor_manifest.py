from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import tarfile
import time
from pathlib import Path
from typing import Any

import requests


PROTEIN_EXTRA_FIELDS = [
    "alphafold_pdb_path",
    "alphafold_cif_path",
    "alphafold_path_status",
    "alphafold_source_tar",
]

MANIFEST_EXTRA_FIELDS = [
    "receptor_pdb_path",
    "receptor_cif_path",
    "receptor_path_status",
    "structure_ready",
]


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def accession_from_member(name: str) -> str | None:
    match = re.match(r"AF-([A-Z0-9]+)-F1-model_v\d+\.(pdb|cif)\.gz$", Path(name).name)
    return match.group(1) if match else None


def build_tar_index(tar_path: Path, required: set[str]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {accession: {} for accession in required}
    with tarfile.open(tar_path, "r") as archive:
        for member in archive:
            accession = accession_from_member(member.name)
            if accession not in required:
                continue
            if member.name.endswith(".pdb.gz"):
                index[accession]["pdb_member"] = member.name
            elif member.name.endswith(".cif.gz"):
                index[accession]["cif_member"] = member.name
    return index


def extract_member(archive: tarfile.TarFile, member_name: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    extracted = archive.extractfile(member_name)
    if extracted is None:
        raise FileNotFoundError(member_name)
    output_path.write_bytes(gzip.decompress(extracted.read()))


def extract_required_structures(
    tar_path: Path,
    index: dict[str, dict[str, str]],
    out_dir: Path,
) -> dict[str, dict[str, str]]:
    extracted: dict[str, dict[str, str]] = {}
    with tarfile.open(tar_path, "r") as archive:
        for accession, members in sorted(index.items()):
            record: dict[str, str] = {}
            pdb_member = members.get("pdb_member")
            cif_member = members.get("cif_member")
            if pdb_member:
                pdb_path = out_dir / f"AF-{accession}-F1-model_v6.pdb"
                extract_member(archive, pdb_member, pdb_path)
                record["pdb_path"] = str(pdb_path)
            if cif_member:
                cif_path = out_dir / f"AF-{accession}-F1-model_v6.cif"
                extract_member(archive, cif_member, cif_path)
                record["cif_path"] = str(cif_path)
            extracted[accession] = record
    return extracted


def fetch_api_fallback(
    accessions: set[str],
    extracted: dict[str, dict[str, str]],
    out_dir: Path,
    timeout: int = 30,
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    session = requests.Session()
    session.headers.update({"User-Agent": "BioMaster/0.1 AlphaFold receptor path rebuild"})
    for accession in sorted(accessions):
        current = extracted.get(accession, {})
        if current.get("pdb_path") and current.get("cif_path"):
            statuses[accession] = "tar"
            continue
        try:
            response = session.get(f"https://alphafold.ebi.ac.uk/api/prediction/{accession}", timeout=timeout)
            if response.status_code == 404:
                statuses[accession] = "api_not_found"
                continue
            response.raise_for_status()
            candidates = response.json()
        except Exception:
            statuses[accession] = "api_failed"
            continue

        # Only accept exact accession matches. Isoform-only records are not used for canonical sequences.
        exact = [item for item in candidates if item.get("uniprotAccession") == accession]
        if not exact:
            statuses[accession] = "api_no_exact_accession_match"
            continue
        item = exact[0]
        try:
            pdb_path = out_dir / f"AF-{accession}-F1-model_v{item.get('latestVersion', 6)}.pdb"
            cif_path = out_dir / f"AF-{accession}-F1-model_v{item.get('latestVersion', 6)}.cif"
            pdb_path.parent.mkdir(parents=True, exist_ok=True)
            if not pdb_path.exists() or pdb_path.stat().st_size == 0:
                pdb_path.write_bytes(session.get(item["pdbUrl"], timeout=timeout).content)
            if not cif_path.exists() or cif_path.stat().st_size == 0:
                cif_path.write_bytes(session.get(item["cifUrl"], timeout=timeout).content)
            extracted.setdefault(accession, {})["pdb_path"] = str(pdb_path)
            extracted.setdefault(accession, {})["cif_path"] = str(cif_path)
            statuses[accession] = "api_exact_accession"
        except Exception:
            statuses[accession] = "api_download_failed"
    return statuses


def rebuild(
    proteins_csv: Path,
    drugs_csv: Path,
    manifest_csv: Path,
    alphafold_tar: Path,
    receptor_dir: Path,
    protein_out: Path,
    manifest_out: Path,
    ready_manifest_out: Path,
    metadata_out: Path,
) -> dict[str, Any]:
    protein_fields, proteins = read_rows(proteins_csv)
    drug_fields, drugs = read_rows(drugs_csv)
    manifest_fields, manifest_rows = read_rows(manifest_csv)

    required = {row["protein_id"] for row in proteins if row.get("protein_id")}
    index = build_tar_index(alphafold_tar, required)
    extracted = extract_required_structures(alphafold_tar, index, receptor_dir)
    api_status = fetch_api_fallback(required, extracted, receptor_dir)

    protein_status: dict[str, str] = {}
    protein_paths: dict[str, dict[str, str]] = {}
    for accession in required:
        paths = extracted.get(accession, {})
        if paths.get("pdb_path") and paths.get("cif_path"):
            status = "ok" if api_status.get(accession) == "tar" else "ok_api"
        elif paths.get("pdb_path"):
            status = "pdb_only"
        elif paths.get("cif_path"):
            status = "cif_only"
        else:
            status = api_status.get(accession, "missing_in_alphafold_tar")
        protein_status[accession] = status
        protein_paths[accession] = paths

    enriched_proteins: list[dict[str, Any]] = []
    for row in proteins:
        accession = row.get("protein_id", "")
        paths = protein_paths.get(accession, {})
        enriched = dict(row)
        enriched["alphafold_pdb_path"] = paths.get("pdb_path", "")
        enriched["alphafold_cif_path"] = paths.get("cif_path", "")
        enriched["alphafold_path_status"] = protein_status.get(accession, "missing_protein_id")
        enriched["alphafold_source_tar"] = str(alphafold_tar)
        enriched_proteins.append(enriched)

    drug_by_id = {row.get("drug_id", ""): row for row in drugs}
    enriched_manifest: list[dict[str, Any]] = []
    ready_manifest: list[dict[str, Any]] = []
    for row in manifest_rows:
        drug = drug_by_id.get(row.get("drug_id", ""), {})
        accession = row.get("protein_id", "")
        paths = protein_paths.get(accession, {})
        ligand_sdf = drug.get("sdf_path") or row.get("ligand_sdf_path", "")
        receptor_pdb = paths.get("pdb_path", "")
        receptor_cif = paths.get("cif_path", "")
        structure_ready = bool(ligand_sdf and receptor_pdb)
        enriched = dict(row)
        enriched["ligand_sdf_path"] = ligand_sdf
        if drug.get("isomeric_smiles") or drug.get("canonical_smiles"):
            enriched["ligand_smiles"] = drug.get("isomeric_smiles") or drug.get("canonical_smiles")
        enriched["receptor_pdb_url"] = receptor_pdb
        enriched["receptor_cif_url"] = receptor_cif
        enriched["receptor_pdb_path"] = receptor_pdb
        enriched["receptor_cif_path"] = receptor_cif
        enriched["receptor_path_status"] = protein_status.get(accession, "missing_protein_id")
        enriched["structure_ready"] = "true" if structure_ready else "false"
        enriched_manifest.append(enriched)
        if structure_ready:
            ready_manifest.append(enriched)

    protein_out_fields = protein_fields + [field for field in PROTEIN_EXTRA_FIELDS if field not in protein_fields]
    manifest_out_fields = manifest_fields + [field for field in MANIFEST_EXTRA_FIELDS if field not in manifest_fields]
    write_rows(protein_out, protein_out_fields, enriched_proteins)
    write_rows(manifest_out, manifest_out_fields, enriched_manifest)
    write_rows(ready_manifest_out, manifest_out_fields, ready_manifest)

    status_counts: dict[str, int] = {}
    for status in protein_status.values():
        status_counts[status] = status_counts.get(status, 0) + 1

    metadata = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "alphafold_tar": str(alphafold_tar),
        "receptor_dir": str(receptor_dir),
        "input_protein_csv": str(proteins_csv),
        "input_drug_csv": str(drugs_csv),
        "input_manifest_csv": str(manifest_csv),
        "protein_output_csv": str(protein_out),
        "manifest_output_csv": str(manifest_out),
        "structure_ready_manifest_csv": str(ready_manifest_out),
        "protein_rows": len(proteins),
        "drug_rows": len(drugs),
        "manifest_rows": len(manifest_rows),
        "protein_status_counts": status_counts,
        "api_fallback_status_counts": {status: sum(1 for value in api_status.values() if value == status) for status in sorted(set(api_status.values()))},
        "proteins_with_pdb_path": sum(1 for item in protein_paths.values() if item.get("pdb_path")),
        "proteins_with_cif_path": sum(1 for item in protein_paths.values() if item.get("cif_path")),
        "manifest_rows_with_ligand_sdf": sum(1 for row in enriched_manifest if row.get("ligand_sdf_path")),
        "manifest_rows_with_receptor_pdb": sum(1 for row in enriched_manifest if row.get("receptor_pdb_path")),
        "structure_ready_rows": len(ready_manifest),
        "missing_proteins": sorted(accession for accession, paths in protein_paths.items() if not (paths.get("pdb_path") and paths.get("cif_path"))),
        "notes": "Structure-ready means ligand_sdf_path and receptor_pdb_path are both nonempty. The original manifest is not overwritten.",
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract AlphaFold receptors and rebuild structure-ready manifests.")
    parser.add_argument("--proteins", default="outputs/report_scale/protein_library_1000.csv")
    parser.add_argument("--drugs", default="data/processed/drug_library_pubchem_chembl_mapped.csv")
    parser.add_argument("--manifest", default="outputs/report_scale/manifest_915k.csv")
    parser.add_argument("--alphafold-tar", default="UP000005640_9606_HUMAN_v6.tar")
    parser.add_argument("--receptor-dir", default="data/processed/alphafold_receptors_v6")
    parser.add_argument("--protein-out", default="data/processed/protein_library_1000_alphafold_paths.csv")
    parser.add_argument("--manifest-out", default="outputs/report_scale/manifest_915k_with_structures.csv")
    parser.add_argument("--ready-manifest-out", default="outputs/report_scale/manifest_915k_structure_ready.csv")
    parser.add_argument("--metadata-out", default="data/processed/alphafold_receptor_manifest.metadata.json")
    args = parser.parse_args()

    metadata = rebuild(
        proteins_csv=Path(args.proteins),
        drugs_csv=Path(args.drugs),
        manifest_csv=Path(args.manifest),
        alphafold_tar=Path(args.alphafold_tar),
        receptor_dir=Path(args.receptor_dir),
        protein_out=Path(args.protein_out),
        manifest_out=Path(args.manifest_out),
        ready_manifest_out=Path(args.ready_manifest_out),
        metadata_out=Path(args.metadata_out),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
