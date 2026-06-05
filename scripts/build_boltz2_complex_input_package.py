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


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}


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


def safe_token(value: Any) -> str:
    text = str(value or "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned[:120] or "item"


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y"}


def abs_path(root: Path, value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else root / path


def parse_pdb_sequence(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "sequence": "", "chains": [], "residueCount": 0, "error": "pdb_missing"}
    residues: dict[str, list[tuple[str, str, str]]] = {}
    seen: set[tuple[str, str, str, str]] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            resn = line[17:20].strip().upper()
            if resn not in AA3_TO_1:
                continue
            chain = line[21].strip() or "A"
            resi = line[22:26].strip()
            icode = line[26].strip()
            key = (chain, resi, icode, resn)
            if key in seen:
                continue
            seen.add(key)
            residues.setdefault(chain, []).append((resi, icode, resn))
    if not residues:
        return {"ok": False, "sequence": "", "chains": [], "residueCount": 0, "error": "no_standard_residues"}
    chain_items = sorted(residues.items(), key=lambda item: len(item[1]), reverse=True)
    chain, chain_residues = chain_items[0]
    sequence = "".join(AA3_TO_1[resn] for _, _, resn in chain_residues)
    return {
        "ok": True,
        "sequence": sequence,
        "chain": chain,
        "chains": [{"chain": item_chain, "residueCount": len(item_residues)} for item_chain, item_residues in chain_items],
        "residueCount": len(sequence),
        "error": "",
    }


def candidate_yaml_payload(row: dict[str, Any], protein_sequence: str, receptor_pdb: Path | None, use_template: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": protein_sequence,
                    "msa": "empty",
                }
            },
            {
                "ligand": {
                    "id": "B",
                    "smiles": str(row.get("canonicalSmiles") or row.get("canonical_smiles") or "").strip(),
                }
            },
        ],
        "properties": [{"affinity": {"binder": "B"}}],
    }
    if use_template and receptor_pdb is not None and receptor_pdb.exists():
        payload["templates"] = [{"pdb": str(receptor_pdb), "chain_id": "A"}]
    return payload


def row_to_record(root: Path, row: pd.Series, input_dir: Path, use_template: bool) -> dict[str, Any]:
    data = row.to_dict()
    pair_id = str(data.get("pairId") or "")
    rank = int(float(data.get("externalQueueRank") or 0))
    receptor_pdb = abs_path(root, data.get("receptorPdbPath"))
    pdb_seq = parse_pdb_sequence(receptor_pdb) if receptor_pdb else {"ok": False, "sequence": "", "residueCount": 0, "error": "receptor_pdb_missing"}
    full_seq = str(data.get("proteinSequence") or "").strip()
    if pdb_seq.get("ok") and pdb_seq.get("sequence"):
        protein_sequence = str(pdb_seq["sequence"])
        sequence_source = "receptor_pdb_atom_sequence"
    else:
        protein_sequence = full_seq
        sequence_source = "protein_library_sequence"

    yaml_name = f"{rank:03d}_{safe_token(pair_id)}.yaml"
    yaml_path = input_dir / yaml_name
    payload = candidate_yaml_payload(data, protein_sequence, receptor_pdb, use_template)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False, width=120), encoding="utf-8")

    return {
        "externalQueueRank": rank,
        "direction": data.get("direction", ""),
        "pairId": pair_id,
        "drugId": data.get("drugId", ""),
        "drug": data.get("drug", ""),
        "target": data.get("target", ""),
        "protein": data.get("protein", ""),
        "proteinName": data.get("proteinName", ""),
        "knownDrugTargetPair": data.get("knownDrugTargetPair", ""),
        "noveltyClass": data.get("noveltyClass", ""),
        "canonicalSmiles": data.get("canonicalSmiles") or data.get("canonical_smiles") or "",
        "sourceProteinSequenceLength": len(full_seq),
        "boltzProteinSequenceLength": len(protein_sequence),
        "boltzSequenceSource": sequence_source,
        "receptorPdbPath": str(receptor_pdb) if receptor_pdb else "",
        "receptorPdbSequenceOk": bool(pdb_seq.get("ok")),
        "receptorPdbResidueCount": pdb_seq.get("residueCount", 0),
        "receptorPdbChain": pdb_seq.get("chain", ""),
        "yamlPath": str(yaml_path),
        "yamlFile": yaml_name,
        "boltzInputReady": bool(protein_sequence and (data.get("canonicalSmiles") or data.get("canonical_smiles"))),
    }


def select_smoke_rows(records: pd.DataFrame, count: int) -> pd.DataFrame:
    if count <= 0:
        return records.head(0).copy()
    scored = records.copy()
    scored["_knownSort"] = scored["knownDrugTargetPair"].apply(lambda value: 0 if truthy(value) else 1)
    scored["_lenSort"] = pd.to_numeric(scored["boltzProteinSequenceLength"], errors="coerce").fillna(999999)
    scored["_rankSort"] = pd.to_numeric(scored["externalQueueRank"], errors="coerce").fillna(999999)
    return scored.sort_values(["_knownSort", "_lenSort", "_rankSort"]).drop(columns=["_knownSort", "_lenSort", "_rankSort"]).head(count).copy()


def markdown(summary: dict[str, Any], smoke_df: pd.DataFrame) -> str:
    lines = [
        "# Boltz-2 Complex Input Package",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "## Scope",
        "",
        summary["scope"],
        "",
        "## Headline",
        "",
        f"- Source rows: {summary['sourceRows']}",
        f"- YAML inputs generated: {summary['yamlRows']}",
        f"- Input-ready rows: {summary['inputReadyRows']}/{summary['yamlRows']}",
        f"- Smoke-test rows: {summary['smokeRows']}",
        f"- Sequence source counts: {summary['sequenceSourceCounts']}",
        "",
        "## Smoke-Test Queue",
        "",
        "| Rank | Pair | Drug | Target | Known | Boltz seq len | YAML |",
        "| ---: | --- | --- | --- | ---: | ---: | --- |",
    ]
    for _, row in smoke_df.iterrows():
        lines.append(
            f"| {row['externalQueueRank']} | {row['pairId']} | {row['drug']} | {row['target']} | "
            f"{row['knownDrugTargetPair']} | {row['boltzProteinSequenceLength']} | `{row['yamlFile']}` |"
        )
    lines.extend(
        [
            "",
            "## Method Note",
            "",
            "Protein sequences are taken from the receptor PDB ATOM records when available, so cropped DiffDock-ready receptors remain computationally tractable. MSA is set to `empty` for the first local smoke test; MSA-server runs can be enabled later for stronger final inference.",
        ]
    )
    return "\n".join(lines) + "\n"


def build(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    source = root / args.source
    out_dir = root / args.out_dir
    input_dir = out_dir / "inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source, low_memory=False).fillna("")
    df["_rankSort"] = pd.to_numeric(df["externalQueueRank"], errors="coerce").fillna(999999)
    selected = df.sort_values("_rankSort").head(args.top_n).copy() if args.top_n > 0 else df.sort_values("_rankSort").copy()

    records = [row_to_record(root, row, input_dir, args.use_template) for _, row in selected.iterrows()]
    record_df = pd.DataFrame(records).fillna("")
    smoke_df = select_smoke_rows(record_df, args.smoke_n)

    record_path = out_dir / "boltz2_input_manifest.csv"
    smoke_path = out_dir / "boltz2_smoke_queue.csv"
    summary_path = out_dir / "boltz2_input_summary.json"
    md_path = out_dir / "BOLTZ2_COMPLEX_INPUT_PACKAGE.md"
    record_df.to_csv(record_path, index=False)
    smoke_df.to_csv(smoke_path, index=False)

    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": f"Boltz-2 YAML input package for Top{len(selected)} protein-ligand complex validation candidates.",
        "source": args.source,
        "sourceRows": int(len(df)),
        "yamlRows": int(len(record_df)),
        "inputReadyRows": int(sum(record_df["boltzInputReady"].apply(truthy))),
        "smokeRows": int(len(smoke_df)),
        "sequenceSourceCounts": record_df["boltzSequenceSource"].value_counts().to_dict(),
        "medianBoltzProteinSequenceLength": float(pd.to_numeric(record_df["boltzProteinSequenceLength"], errors="coerce").median()),
        "maxBoltzProteinSequenceLength": int(pd.to_numeric(record_df["boltzProteinSequenceLength"], errors="coerce").max()),
        "artifacts": {
            "inputManifestCsv": str(record_path),
            "smokeQueueCsv": str(smoke_path),
            "inputDir": str(input_dir),
            "markdown": str(md_path),
        },
    }
    write_json(summary_path, summary)
    md_path.write_text(markdown(summary, smoke_df), encoding="utf-8")
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Boltz-2 YAML inputs for external SOTA complex validation.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--source", default="outputs/sota_validation/external_sota_model_inputs/boltz_chai_top50_complex_queue.csv")
    parser.add_argument("--out-dir", default="outputs/sota_validation/boltz2_complex_validation")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--smoke-n", type=int, default=3)
    parser.add_argument("--use-template", action="store_true")
    args = parser.parse_args()

    build(Path(args.root).resolve(), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
