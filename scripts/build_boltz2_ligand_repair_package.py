from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import Descriptors


def json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def completed_stems(run_dir: Path) -> set[str]:
    confidence = {
        p.name.replace("confidence_", "").replace("_model_0.json", "")
        for p in run_dir.rglob("confidence_*_model_0.json")
    }
    affinity = {p.name.replace("affinity_", "").replace(".json", "") for p in run_dir.rglob("affinity_*.json")}
    return confidence & affinity


def fragment_score(mol: Chem.Mol) -> tuple[int, int, float]:
    heavy = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    carbon = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    exact = float(Descriptors.ExactMolWt(mol))
    return heavy, carbon, exact


def parent_smiles(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"ok": False, "parentSmiles": "", "repairReason": "rdkit_parse_failed", "fragments": []}
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    frag_records: list[dict[str, Any]] = []
    for frag in fragments:
        try:
            Chem.SanitizeMol(frag)
            smi = Chem.MolToSmiles(frag, canonical=True)
            heavy, carbon, exact = fragment_score(frag)
            frag_records.append({"smiles": smi, "heavyAtoms": heavy, "carbonAtoms": carbon, "exactMolWt": exact})
        except Exception as exc:  # noqa: BLE001
            frag_records.append({"smiles": "", "heavyAtoms": 0, "carbonAtoms": 0, "exactMolWt": 0.0, "error": str(exc)})
    usable = [item for item in frag_records if item.get("smiles") and item.get("heavyAtoms", 0) > 1]
    if not usable:
        return {"ok": False, "parentSmiles": "", "repairReason": "no_usable_fragment", "fragments": frag_records}
    usable.sort(key=lambda item: (item["carbonAtoms"], item["heavyAtoms"], item["exactMolWt"]), reverse=True)
    parent = str(usable[0]["smiles"])
    if parent == Chem.MolToSmiles(mol, canonical=True):
        reason = "unchanged_single_component"
    elif "." in smiles:
        reason = "largest_organic_fragment_from_salt_or_solvent"
    else:
        reason = "canonicalized_parent"
    return {"ok": True, "parentSmiles": parent, "repairReason": reason, "fragments": frag_records}


def repair_payload(original_yaml: Path, parent: str) -> dict[str, Any]:
    payload = yaml.safe_load(original_yaml.read_text(encoding="utf-8"))
    sequences = payload.get("sequences", [])
    for entry in sequences:
        ligand = entry.get("ligand") if isinstance(entry, dict) else None
        if ligand and ligand.get("id") == "B":
            ligand["smiles"] = parent
    return payload


def build(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    out_dir = root / args.out_dir
    input_dir = out_dir / "inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = root / args.source_manifest
    previous_run_dir = root / args.previous_run_dir
    source_df = pd.read_csv(source_manifest).fillna("")
    done = completed_stems(previous_run_dir)

    records: list[dict[str, Any]] = []
    for _, row in source_df.iterrows():
        original_yaml = root / str(row["yamlPath"])
        stem = Path(str(row["yamlFile"])).stem
        if stem in done:
            continue
        repair = parent_smiles(str(row["canonicalSmiles"]))
        repaired_name = f"{stem}.yaml"
        repaired_yaml = input_dir / repaired_name
        if repair["ok"]:
            payload = repair_payload(original_yaml, str(repair["parentSmiles"]))
            repaired_yaml.write_text(yaml.safe_dump(payload, sort_keys=False, width=120), encoding="utf-8")
        records.append(
            {
                **row.to_dict(),
                "originalYamlFile": row["yamlFile"],
                "yamlFile": repaired_name,
                "yamlPath": str(repaired_yaml) if repair["ok"] else "",
                "originalSmiles": row["canonicalSmiles"],
                "repairedSmiles": repair["parentSmiles"],
                "repairOk": repair["ok"],
                "repairReason": repair["repairReason"],
                "fragmentCount": len(repair["fragments"]),
                "fragmentsJson": json.dumps(json_safe(repair["fragments"]), ensure_ascii=False),
                "boltzAlreadyCompleted": False,
            }
        )

    record_df = pd.DataFrame(records).fillna("")
    manifest_path = out_dir / "boltz2_ligand_repair_manifest.csv"
    record_df.to_csv(manifest_path, index=False)

    ready_df = record_df[record_df["repairOk"].astype(bool)].copy() if not record_df.empty else record_df
    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "Boltz-2 failed-input ligand parent/fragment repair package.",
        "sourceManifest": str(source_manifest),
        "previousRunDir": str(previous_run_dir),
        "previousCompletedRows": len(done),
        "failedRowsDetected": int(len(record_df)),
        "repairReadyRows": int(len(ready_df)),
        "repairReasonCounts": record_df["repairReason"].value_counts(dropna=False).to_dict() if not record_df.empty else {},
        "artifacts": {
            "manifestCsv": str(manifest_path),
            "inputDir": str(input_dir),
        },
    }
    write_json(out_dir / "boltz2_ligand_repair_summary.json", summary)

    lines = [
        "# Boltz-2 Ligand Repair Package",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "## Summary",
        "",
        f"- Failed rows detected from the initial Top50 run: {summary['failedRowsDetected']}",
        f"- Repair-ready rows: {summary['repairReadyRows']}",
        f"- Repair reasons: {summary['repairReasonCounts']}",
        "",
        "## Method",
        "",
        "Failed ligands were converted from salt, solvate, or counter-ion forms to the largest organic parent fragment before rerunning Boltz-2. The original SMILES strings remain recorded in the manifest.",
        "",
    ]
    (out_dir / "BOLTZ2_LIGAND_REPAIR_PACKAGE.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument(
        "--source-manifest",
        default="outputs/sota_validation/boltz2_complex_validation/boltz2_input_manifest.csv",
    )
    parser.add_argument(
        "--previous-run-dir",
        default="outputs/sota_validation/boltz2_complex_validation/runs_top50",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/sota_validation/boltz2_complex_validation/ligand_repair_failed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build(Path(args.root).resolve(), args)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
