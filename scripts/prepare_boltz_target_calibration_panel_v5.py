#!/usr/bin/env python3
"""Select a balanced ChEMBL positive/negative panel for Boltz target calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors


ROOT = Path(__file__).resolve().parents[1]
LABELS = (
    ROOT
    / "outputs/current_production_package_v2/conplex_target_calibration_v5"
    / "CHEMBL37_CONPLEX_CALIBRATION_LABELS_V5.csv.gz"
)
TOP3000 = ROOT / "outputs/current_production_package_v2/formal_full_universe_v4/refined_top3000_v4_complete.csv"
FINAL1000 = ROOT / "outputs/current_production_package_v2/final_delivery_v4/FINAL1000_RESERVE_FULL_V4.csv"
SEQUENCES = ROOT / "outputs/full_conplex_active_moiety_v4/protein_sequence_representatives.csv"
DEFAULT_OUT = ROOT / "outputs/current_production_package_v2/boltz_target_calibration_v5"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def descriptors(smiles: str) -> tuple[float, float, float]:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return np.nan, np.nan, np.nan
    return float(Descriptors.MolWt(molecule)), float(Crippen.MolLogP(molecule)), float(Descriptors.TPSA(molecule))


def pick_diverse(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    work = frame.copy()
    work["_hash"] = [
        hashlib.sha256(f"{seq}|{compound}".encode()).hexdigest()
        for seq, compound in zip(work["sequence_key"], work["parent_molecule_chembl_id"])
    ]
    work["_scaffold"] = work["murcko_scaffold"].fillna("").replace("", "NO_SCAFFOLD")
    unique = work.sort_values("_hash", kind="mergesort").drop_duplicates("_scaffold")
    selected = list(unique.head(count).index)
    if len(selected) < count:
        remainder = work.loc[~work.index.isin(selected)].sort_values("_hash", kind="mergesort")
        selected.extend(remainder.head(count - len(selected)).index)
    return work.loc[selected].drop(columns=["_hash", "_scaffold"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--targets", type=int, default=100)
    parser.add_argument("--per-label-target", type=int, default=5)
    args = parser.parse_args()
    if not args.labels.is_absolute():
        args.labels = (ROOT / args.labels).resolve()
    if not args.output_dir.is_absolute():
        args.output_dir = (ROOT / args.output_dir).resolve()
    for path in [args.labels, TOP3000, FINAL1000, SEQUENCES]:
        if not path.is_file():
            raise FileNotFoundError(path)

    labels = pd.read_csv(args.labels, low_memory=False)
    labels = labels[labels["calibration_label"].isin(["positive", "negative_or_inactive"])].copy()
    top = pd.read_csv(TOP3000, low_memory=False)
    final = pd.read_csv(FINAL1000, usecols=["sequence_key"], low_memory=False)
    context_columns = [
        "sequence_key",
        "primary_gene",
        "target_assay_family_v2",
        "representative_protein_id",
        "pdb_path",
        "top_pocket_residue_ids",
        "protein_names",
    ]
    context = top[context_columns].drop_duplicates("sequence_key")
    top_counts = top["sequence_key"].value_counts().rename("top3000_candidate_n")
    final_counts = final["sequence_key"].value_counts().rename("final1000_candidate_n")
    coverage = labels.groupby(["sequence_key", "calibration_label"]).size().unstack(fill_value=0)
    coverage = coverage.rename(
        columns={"positive": "positive_n", "negative_or_inactive": "negative_n"}
    )
    for column in ["positive_n", "negative_n"]:
        if column not in coverage:
            coverage[column] = 0
    target_priority = (
        context[["sequence_key", "primary_gene", "target_assay_family_v2"]]
        .merge(coverage.reset_index(), on="sequence_key", how="left")
        .merge(top_counts, left_on="sequence_key", right_index=True, how="left")
        .merge(final_counts, left_on="sequence_key", right_index=True, how="left")
        .fillna(0)
    )
    eligible = target_priority[
        target_priority["positive_n"].ge(args.per_label_target)
        & target_priority["negative_n"].ge(args.per_label_target)
    ].copy()
    eligible["balanced_calibration_n"] = eligible[["positive_n", "negative_n"]].min(axis=1)
    eligible = eligible.sort_values(
        ["final1000_candidate_n", "top3000_candidate_n", "balanced_calibration_n", "primary_gene"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).head(args.targets)

    selected_rows = []
    selected_targets = set(eligible["sequence_key"])
    for (sequence_key, label), group in labels[labels["sequence_key"].isin(selected_targets)].groupby(
        ["sequence_key", "calibration_label"], sort=True
    ):
        selected_rows.append(pick_diverse(group, args.per_label_target))
    panel = pd.concat(selected_rows, ignore_index=True)
    panel = panel.merge(context, on=["sequence_key", "primary_gene"], how="left", validate="many_to_one")
    sequences = pd.read_csv(SEQUENCES, usecols=["sequence_key", "sequence"], dtype=str)
    panel = panel.merge(sequences, on="sequence_key", how="left", validate="many_to_one")
    if panel["sequence"].isna().any() or panel["pdb_path"].isna().any():
        raise ValueError("Missing sequence or receptor path in calibration panel")

    properties = panel["model_ligand_smiles"].map(descriptors)
    panel[["calibration_mw", "calibration_logp", "calibration_tpsa"]] = pd.DataFrame(
        properties.tolist(), index=panel.index
    )
    panel["pairId"] = (
        "CAL37_"
        + panel["calibration_label"].map(
            {"positive": "POS", "negative_or_inactive": "NEG"}
        )
        + "_"
        + panel["sequence_key"].astype(str)
        + "_"
        + panel["parent_molecule_chembl_id"].astype(str)
    )
    panel = panel.sort_values(
        ["sequence_key", "calibration_label", "parent_molecule_chembl_id"], kind="mergesort"
    ).reset_index(drop=True)
    panel.insert(0, "externalQueueRank", range(1, len(panel) + 1))
    panel["drugId"] = panel["parent_molecule_chembl_id"]
    panel["drug"] = panel["parent_molecule_name"].fillna("").replace("", np.nan).fillna(panel["drugId"])
    panel["target"] = panel["primary_gene"]
    panel["protein"] = panel["representative_protein_id"]
    panel["proteinName"] = panel["protein_names"]
    panel["knownDrugTargetPair"] = panel["calibration_label"].eq("positive")
    panel["noveltyClass"] = "chembl37_" + panel["calibration_label"]
    panel["canonicalSmiles"] = panel["model_ligand_smiles"]
    panel["proteinSequence"] = panel["sequence"]
    panel["receptorPdbPath"] = panel["pdb_path"]
    panel["topPocketResidueIds"] = panel["top_pocket_residue_ids"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.output_dir / "BOLTZ_TARGET_CALIBRATION_SOURCE_QUEUE_V5.csv"
    target_path = args.output_dir / "BOLTZ_TARGET_CALIBRATION_TARGET_SELECTION_V5.csv"
    panel.to_csv(source, index=False)
    eligible.to_csv(target_path, index=False)
    relative_source = source.relative_to(ROOT).as_posix()
    package_dir = (args.output_dir / "input_package").relative_to(ROOT).as_posix()
    run_path = args.output_dir / "prepare_and_run_boltz_target_calibration_v5.sh"
    run_path.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
python scripts/build_boltz2_complex_input_package.py \\
  --source {relative_source} \\
  --out-dir {package_dir} \\
  --top-n {len(panel)} --smoke-n 10 --use-template --use-pocket-constraint
python scripts/run_boltz2_batched_queue.py \\
  --input-manifest {package_dir}/boltz2_input_manifest.csv \\
  --input-dir {package_dir}/inputs \\
  --out-dir {args.output_dir.relative_to(ROOT).as_posix()}/run \\
  --top-n {len(panel)} --batch-size 10 --gpus 0,1 \\
  --recycling-steps 3 --sampling-steps 200 --diffusion-samples 2 \\
  --sampling-steps-affinity 200 --diffusion-samples-affinity 5 \\
  --num-workers 2 --preprocessing-threads 4 --no-kernels --stop-on-failure
""",
        encoding="utf-8",
    )
    run_path.chmod(0o755)
    summary: dict[str, Any] = {
        "status": "prepared_source_queue",
        "created_utc": now(),
        "selected_targets": int(panel["sequence_key"].nunique()),
        "panel_rows": int(len(panel)),
        "positive_rows": int(panel["calibration_label"].eq("positive").sum()),
        "negative_rows": int(panel["calibration_label"].eq("negative_or_inactive").sum()),
        "per_label_target": int(args.per_label_target),
        "selection_basis": "targets prioritized by Final1000 count, Top3000 count, then balanced ChEMBL coverage; compounds scaffold-diverse deterministic sample",
        "inputs": {str(path): sha256(path) for path in [args.labels, TOP3000, FINAL1000, SEQUENCES]},
        "run_script": run_path.relative_to(ROOT).as_posix(),
    }
    (args.output_dir / "BOLTZ_TARGET_CALIBRATION_PREPARATION_V5.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
