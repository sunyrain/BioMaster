#!/usr/bin/env python3
"""Prepare strict ChEMBL positive/negative pairs for ConPLEx calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_106k_to_1000_physics_funnel import standardize_smiles  # noqa: E402


DEFAULT_PAIRS = (
    ROOT
    / "outputs/current_production_package_v2/chembl37_target_calibration_api_v5"
    / "PROJECT_TARGET_CHEMBL37_API_NUMERIC_PAIRS_V5.csv.gz"
)
SEQUENCES = ROOT / "outputs/full_conplex_active_moiety_v4/protein_sequence_representatives.csv"
MODEL = ROOT / "third_party/ConPLex/models/BindingDB_ExperimentalValidModel.pt"
DEFAULT_OUT = ROOT / "outputs/current_production_package_v2/conplex_target_calibration_v5"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collapse_target_compound_duplicates(pairs: pd.DataFrame) -> pd.DataFrame:
    """Collapse alternate ChEMBL target records without hiding label conflicts."""
    keys = ["sequence_key", "parent_molecule_chembl_id"]
    if not pairs.duplicated(keys).any():
        return pairs.copy()
    rows = []
    for _, group in pairs.groupby(keys, sort=False, dropna=False):
        row = group.iloc[0].copy()
        labels = set(group["calibration_label"].dropna().astype(str))
        if "conflicting_exclude" in labels or {
            "positive",
            "negative_or_inactive",
        }.issubset(labels):
            row["calibration_label"] = "conflicting_exclude"
        elif "positive" in labels:
            row["calibration_label"] = "positive"
        elif "negative_or_inactive" in labels:
            row["calibration_label"] = "negative_or_inactive"
        else:
            row["calibration_label"] = "grey_or_unresolved"
        for column in ["activity_rows", "assay_count", "document_count", "numeric_rows"]:
            if column in group:
                row[column] = pd.to_numeric(group[column], errors="coerce").fillna(0).sum()
        for column in ["min_pchembl", "min_document_year"]:
            if column in group:
                row[column] = pd.to_numeric(group[column], errors="coerce").min()
        for column in ["max_pchembl", "max_document_year"]:
            if column in group:
                row[column] = pd.to_numeric(group[column], errors="coerce").max()
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def scaffold_balanced_cap(group: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if maximum <= 0 or len(group) <= maximum:
        return group.copy()
    work = group.copy()
    work["_sample_hash"] = [
        hashlib.sha256(f"{sequence}|{compound}".encode("utf-8")).hexdigest()
        for sequence, compound in zip(work["sequence_key"], work["parent_molecule_chembl_id"])
    ]
    work["_scaffold_key"] = work["murcko_scaffold"].fillna("").replace("", "NO_SCAFFOLD")
    representatives = (
        work.sort_values("_sample_hash", kind="mergesort")
        .drop_duplicates("_scaffold_key", keep="first")
        .sort_values("_sample_hash", kind="mergesort")
    )
    selected = list(representatives.head(maximum).index)
    if len(selected) < maximum:
        remainder = work.loc[~work.index.isin(selected)].sort_values("_sample_hash", kind="mergesort")
        selected.extend(remainder.head(maximum - len(selected)).index)
    return work.loc[selected].drop(columns=["_sample_hash", "_scaffold_key"])


def deterministic_pre_cap(group: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if maximum <= 0 or len(group) <= maximum:
        return group.copy()
    work = group.copy()
    work["_pre_hash"] = [
        hashlib.sha256(f"{sequence}|{compound}".encode("utf-8")).hexdigest()
        for sequence, compound in zip(work["sequence_key"], work["parent_molecule_chembl_id"])
    ]
    return work.sort_values("_pre_hash", kind="mergesort").head(maximum).drop(columns="_pre_hash")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-per-label-target", type=int, default=150)
    args = parser.parse_args()
    if not args.pairs.is_absolute():
        args.pairs = (ROOT / args.pairs).resolve()
    if not args.output_dir.is_absolute():
        args.output_dir = (ROOT / args.output_dir).resolve()
    for path in [args.pairs, SEQUENCES, MODEL]:
        if not path.is_file():
            raise FileNotFoundError(path)

    pairs = collapse_target_compound_duplicates(pd.read_csv(args.pairs, low_memory=False))
    pairs = pairs[pairs["calibration_label"].isin(["positive", "negative_or_inactive"])].copy()
    pairs["binary_label"] = pairs["calibration_label"].eq("positive").astype(int)
    pre_cap = max(args.max_per_label_target * 3, args.max_per_label_target)
    pairs = pd.concat(
        [
            deterministic_pre_cap(group, pre_cap)
            for _, group in pairs.groupby(["sequence_key", "calibration_label"], sort=True)
        ],
        ignore_index=True,
    )
    source_smiles = pairs["parent_canonical_smiles"].fillna("").astype(str)
    standardized_lookup = {
        value: standardize_smiles(value) for value in source_smiles.drop_duplicates()
    }
    standardized = source_smiles.map(standardized_lookup)
    pairs["model_ligand_smiles"] = standardized.map(lambda item: item["active_moiety_smiles"])
    pairs["murcko_scaffold"] = standardized.map(lambda item: item["murcko_scaffold"])
    pairs["rdkit_parse_ok"] = pairs["model_ligand_smiles"].ne("")
    pairs = pairs[pairs["rdkit_parse_ok"]].copy()
    pairs = pd.concat(
        [
            scaffold_balanced_cap(group, args.max_per_label_target)
            for _, group in pairs.groupby(["sequence_key", "calibration_label"], sort=True)
        ],
        ignore_index=True,
    )

    sequences = pd.read_csv(SEQUENCES, usecols=["sequence_key", "sequence"], dtype=str)
    pairs = pairs.merge(sequences, on="sequence_key", how="left", validate="many_to_one")
    if pairs["sequence"].isna().any():
        missing = sorted(pairs.loc[pairs["sequence"].isna(), "sequence_key"].unique())
        raise ValueError(f"Missing sequences: {missing[:20]}")
    pairs["calibration_pair_id"] = (
        "CAL37_"
        + pairs["sequence_key"].astype(str)
        + "_"
        + pairs["parent_molecule_chembl_id"].astype(str)
    )
    if pairs["calibration_pair_id"].duplicated().any():
        raise ValueError("Calibration pairs are not unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = args.output_dir / "CHEMBL37_CONPLEX_CALIBRATION_LABELS_V5.csv.gz"
    input_tsv = args.output_dir / "CHEMBL37_CONPLEX_CALIBRATION_INPUT_V5.tsv"
    predictions = args.output_dir / "CHEMBL37_CONPLEX_CALIBRATION_PREDICTIONS_V5.tsv"
    cache = args.output_dir / "conplex_cache"
    cache.mkdir(exist_ok=True)
    run_script = args.output_dir / "run_conplex_calibration_v5.sh"
    summary_path = args.output_dir / "CONPLEX_CALIBRATION_PREPARATION_V5.json"

    pairs.drop(columns=["sequence"]).to_csv(labels, index=False, compression="gzip")
    pairs[["sequence_key", "calibration_pair_id", "sequence", "model_ligand_smiles"]].to_csv(
        input_tsv, sep="\t", header=False, index=False
    )
    relative = lambda path: path.relative_to(ROOT).as_posix()
    conplex_python = ROOT / ".venvs/conplex/bin/python"
    python_command = relative(conplex_python) if conplex_python.is_file() else "python"
    run_script.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH=third_party/ConPLex:.local_deps/conplex:${{PYTHONPATH:-}}
{python_command} -m conplex_dti predict \\
  --data-file {relative(input_tsv)} \\
  --model-path {relative(MODEL)} \\
  --outfile {relative(predictions)} \\
  --data-cache-dir {relative(cache)} \\
  --device {args.device} \\
  --batch-size {args.batch_size}
""",
        encoding="utf-8",
    )
    run_script.chmod(0o755)
    summary = {
        "status": "prepared",
        "created_utc": now(),
        "source_pair_rows": int(len(pd.read_csv(args.pairs, usecols=["calibration_label"]))),
        "calibration_rows": int(len(pairs)),
        "max_per_label_target": int(args.max_per_label_target),
        "deterministic_pre_cap_per_label_target": int(pre_cap),
        "positive_rows": int(pairs["binary_label"].sum()),
        "negative_rows": int((pairs["binary_label"] == 0).sum()),
        "targets": int(pairs["sequence_key"].nunique()),
        "compounds": int(pairs["parent_molecule_chembl_id"].nunique()),
        "source_sha256": sha256(args.pairs),
        "sequence_sha256": sha256(SEQUENCES),
        "model_sha256": sha256(MODEL),
        "labels": relative(labels),
        "input_tsv": relative(input_tsv),
        "predictions": relative(predictions),
        "run_script": relative(run_script),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
