#!/usr/bin/env python3
"""Select diverse ChEMBL positive/negative controls for target-wise docking calibration."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINDING = (
    ROOT
    / "outputs/current_production_package_v2/chembl37_target_calibration_v5/"
    "PROJECT463_CHEMBL37_STRICT_BINDING_PAIR_CALIBRATION_V5.csv.gz"
)
DEFAULT_TARGETS = ROOT / "outputs/affinity_first_remote_discovery_v1/TARGET_MODEL_READINESS_463_V1.csv"
DEFAULT_HOLO = (
    ROOT
    / "outputs/affinity_first_remote_discovery_v1/experimental_structure_atlas_v1/"
    "TARGET_EXPERIMENTAL_STRUCTURE_ATLAS_463_V1.csv"
)
DEFAULT_OUT = ROOT / "outputs/affinity_first_remote_discovery_v1/target_docking_calibration_v1"


def clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "na", "n/a"} else text


def canonical_smiles(value: Any) -> str:
    molecule = Chem.MolFromSmiles(clean(value))
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True) if molecule is not None else ""


def scaffold(value: Any) -> str:
    molecule = Chem.MolFromSmiles(clean(value))
    if molecule is None:
        return ""
    try:
        core = MurckoScaffold.GetScaffoldForMol(molecule)
        return Chem.MolToSmiles(core, canonical=True) if core.GetNumAtoms() else "ACYCLIC"
    except Exception:
        return ""


def docking_domain_properties(value: Any) -> dict[str, Any]:
    smiles = canonical_smiles(value)
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if molecule is None:
        return {
            "canonical_control_smiles": "", "control_mw": None, "control_heavy_atoms": None,
            "control_rotatable_bonds": None, "control_formal_charge": None, "control_unusual_atoms": None,
            "control_fragment_count": None,
        }
    allowed = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}
    return {
        "canonical_control_smiles": smiles,
        "control_mw": float(Descriptors.MolWt(molecule)),
        "control_heavy_atoms": int(molecule.GetNumHeavyAtoms()),
        "control_rotatable_bonds": int(Lipinski.NumRotatableBonds(molecule)),
        "control_formal_charge": int(Chem.GetFormalCharge(molecule)),
        "control_unusual_atoms": int(sum(atom.GetAtomicNum() not in allowed for atom in molecule.GetAtoms())),
        "control_fragment_count": int(len(Chem.GetMolFrags(molecule))),
    }


def diverse_take(group: pd.DataFrame, limit: int, positive: bool) -> pd.DataFrame:
    work = group.copy()
    work["numeric_score"] = pd.to_numeric(work["max_pchembl"], errors="coerce")
    work["evidence_rows"] = pd.to_numeric(work["activity_rows"], errors="coerce").fillna(0)
    work["documents"] = pd.to_numeric(work["document_count"], errors="coerce").fillna(0)
    if positive:
        work = work.sort_values(
            ["numeric_score", "documents", "evidence_rows"], ascending=[False, False, False], kind="mergesort"
        )
    else:
        work["explicit_inactive"] = pd.to_numeric(work["any_explicit_inactive"], errors="coerce").fillna(0)
        work = work.sort_values(
            ["explicit_inactive", "numeric_score", "documents", "evidence_rows"],
            ascending=[False, True, False, False],
            kind="mergesort",
        )
    selected_indices: list[int] = []
    selected_smiles: dict[int, str] = {}
    selected_scaffolds: dict[int, str] = {}
    seen_scaffolds: set[str] = set()
    fallback: list[tuple[int, str, str]] = []
    for index, row in work.iterrows():
        smiles = canonical_smiles(row.get("parent_canonical_smiles"))
        core = scaffold(smiles)
        if not smiles or not core:
            continue
        fallback.append((index, smiles, core))
        if core in seen_scaffolds:
            continue
        selected_indices.append(index)
        selected_smiles[index] = smiles
        selected_scaffolds[index] = core
        seen_scaffolds.add(core)
        if len(selected_indices) == limit:
            break
    if len(selected_indices) < limit:
        for index, smiles, core in fallback:
            if index in selected_smiles:
                continue
            selected_indices.append(index)
            selected_smiles[index] = smiles
            selected_scaffolds[index] = core
            if len(selected_indices) == limit:
                break
    selected = work.loc[selected_indices].copy()
    selected["canonical_control_smiles"] = [selected_smiles[index] for index in selected.index]
    selected["murcko_scaffold"] = [selected_scaffolds[index] for index in selected.index]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", default=str(DEFAULT_BINDING))
    parser.add_argument("--targets", default=str(DEFAULT_TARGETS))
    parser.add_argument("--holo", default=str(DEFAULT_HOLO))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--include-targets",
        default="",
        help="Optional CSV/CSV.GZ containing sequence_key values to retain.",
    )
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--output-version", default="V1")
    parser.add_argument("--docking-domain-filter", action="store_true")
    parser.add_argument("--min-mw", type=float, default=100.0)
    parser.add_argument("--max-mw", type=float, default=900.0)
    parser.add_argument("--min-heavy-atoms", type=int, default=7)
    parser.add_argument("--max-heavy-atoms", type=int, default=65)
    parser.add_argument("--max-rotatable-bonds", type=int, default=20)
    parser.add_argument("--max-absolute-charge", type=int, default=2)
    args = parser.parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    binding = pd.read_csv(args.binding, low_memory=False).fillna("")
    targets = pd.read_csv(args.targets, low_memory=False).fillna("")
    holo = pd.read_csv(args.holo, low_memory=False).fillna("")
    if args.include_targets:
        include = pd.read_csv(args.include_targets, usecols=["sequence_key"])
        include_keys = set(include["sequence_key"].astype(str))
        binding = binding[binding["sequence_key"].astype(str).isin(include_keys)].copy()
        targets = targets[targets["sequence_key"].astype(str).isin(include_keys)].copy()
        holo = holo[holo["sequence_key"].astype(str).isin(include_keys)].copy()
    binding = binding[binding["parent_canonical_smiles"].astype(str).str.strip().ne("")].copy()
    binding["control_class"] = binding["calibration_label"].map(
        {"positive": "positive", "negative_or_inactive": "negative"}
    ).fillna("")
    binding = binding[binding["control_class"].ne("")].copy()
    input_labeled_rows = int(len(binding))
    excluded = pd.DataFrame()
    if args.docking_domain_filter:
        unique_smiles = pd.DataFrame(
            {"parent_canonical_smiles": binding["parent_canonical_smiles"].astype(str).unique()}
        )
        properties = pd.DataFrame(
            [docking_domain_properties(value) for value in unique_smiles["parent_canonical_smiles"]]
        )
        properties["parent_canonical_smiles"] = unique_smiles["parent_canonical_smiles"].values
        binding = binding.merge(properties, on="parent_canonical_smiles", how="left", validate="many_to_one")
        keep = (
            binding["control_mw"].between(args.min_mw, args.max_mw)
            & binding["control_heavy_atoms"].between(args.min_heavy_atoms, args.max_heavy_atoms)
            & binding["control_rotatable_bonds"].le(args.max_rotatable_bonds)
            & binding["control_formal_charge"].abs().le(args.max_absolute_charge)
            & binding["control_unusual_atoms"].eq(0)
            & binding["control_fragment_count"].eq(1)
        )
        excluded = binding[~keep].copy()
        binding = binding[keep].copy()

    selected_rows = []
    stats_rows = []
    for sequence_key, group in binding.groupby("sequence_key", sort=True):
        positive_pool = group[group["control_class"].eq("positive")]
        negative_pool = group[group["control_class"].eq("negative")]
        positive = diverse_take(positive_pool, args.per_class, True) if len(positive_pool) else positive_pool
        negative = diverse_take(negative_pool, args.per_class, False) if len(negative_pool) else negative_pool
        selected = pd.concat([positive, negative], ignore_index=False).copy()
        selected["target_control_rank"] = selected.groupby("control_class").cumcount() + 1
        selected_rows.append(selected)
        stats_rows.append(
            {
                "sequence_key": sequence_key,
                "positive_pool_rows": int(len(positive_pool)),
                "negative_pool_rows": int(len(negative_pool)),
                "selected_positive_rows": int(len(positive)),
                "selected_negative_rows": int(len(negative)),
                "calibration_ready_8x8": bool(len(positive) >= 8 and len(negative) >= 8),
                "calibration_ready_12x12": bool(len(positive) >= 12 and len(negative) >= 12),
            }
        )
    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    target_stats = pd.DataFrame(stats_rows)
    target_info = targets[
        [
            "sequence_key",
            "primary_gene",
            "target_assay_family",
            "structure_bin",
            "pdb_path",
            "top_pocket_center_x",
            "top_pocket_center_y",
            "top_pocket_center_z",
        ]
    ].drop_duplicates("sequence_key")
    target_stats = target_info.merge(target_stats, on="sequence_key", how="left").merge(
        holo[
            [
                "sequence_key",
                "has_candidate_experimental_holo",
                "best_holo_pdb_id",
                "best_holo_resolution",
                "best_holo_coverage",
                "holo_validation_status",
            ]
        ],
        on="sequence_key",
        how="left",
        validate="one_to_one",
    )
    count_columns = [
        "positive_pool_rows",
        "negative_pool_rows",
        "selected_positive_rows",
        "selected_negative_rows",
    ]
    target_stats[count_columns] = target_stats[count_columns].fillna(0).astype(int)
    target_stats["calibration_ready_8x8"] = target_stats["calibration_ready_8x8"].fillna(False).astype(bool)
    target_stats["calibration_ready_12x12"] = target_stats["calibration_ready_12x12"].fillna(False).astype(bool)
    if not selected.empty:
        selected = selected.merge(target_info, on=["sequence_key", "primary_gene", "target_assay_family"], how="left")
        selected = selected.merge(
            holo[["sequence_key", "has_candidate_experimental_holo", "best_holo_pdb_id"]],
            on="sequence_key",
            how="left",
            validate="many_to_one",
        )
        selected["control_pair_id"] = (
            selected["sequence_key"].astype(str)
            + "_"
            + selected["parent_molecule_chembl_id"].astype(str)
            + "_"
            + selected["control_class"].astype(str)
        )

    version = clean(args.output_version).upper() or "V1"
    manifest_path = out_dir / f"GNINA_TARGET_CALIBRATION_CONTROLS_{version}.csv.gz"
    target_stats_path = out_dir / f"TARGET_DOCKING_CALIBRATION_READINESS_463_{version}.csv"
    exclusions_path = out_dir / f"GNINA_CONTROL_DOMAIN_EXCLUSIONS_{version}.csv.gz"
    selected.to_csv(manifest_path, index=False, compression={"method": "gzip", "compresslevel": 5})
    target_stats.to_csv(target_stats_path, index=False)
    if args.docking_domain_filter:
        excluded.to_csv(exclusions_path, index=False, compression={"method": "gzip", "compresslevel": 5})
    summary = {
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "project_targets": int(len(target_stats)),
        "control_rows": int(len(selected)),
        "input_labeled_binding_rows": input_labeled_rows,
        "domain_eligible_binding_rows": int(len(binding)),
        "domain_excluded_binding_rows": int(len(excluded)),
        "positive_control_rows": int(selected["control_class"].eq("positive").sum()),
        "negative_control_rows": int(selected["control_class"].eq("negative").sum()),
        "targets_with_any_positive": int(target_stats["positive_pool_rows"].gt(0).sum()),
        "targets_with_any_negative": int(target_stats["negative_pool_rows"].gt(0).sum()),
        "targets_ready_8x8": int(target_stats["calibration_ready_8x8"].sum()),
        "targets_ready_12x12": int(target_stats["calibration_ready_12x12"].sum()),
        "targets_ready_8x8_and_candidate_holo": int(
            (target_stats["calibration_ready_8x8"] & target_stats["has_candidate_experimental_holo"].astype(bool)).sum()
        ),
        "policy": (
            "Controls calibrate docking per target. They are not nomination evidence and are selected with "
            "Murcko-scaffold diversity from strict ChEMBL binding labels. When enabled, the docking-domain "
            "filter is matched to the 99th-percentile envelope of project FDA structures and excludes salts, "
            "metals, peptides and very large/flexible molecules that GNINA cannot calibrate fairly."
        ),
        "docking_domain_filter": bool(args.docking_domain_filter),
        "docking_domain_thresholds": {
            "mw": [args.min_mw, args.max_mw],
            "heavy_atoms": [args.min_heavy_atoms, args.max_heavy_atoms],
            "max_rotatable_bonds": args.max_rotatable_bonds,
            "max_absolute_charge": args.max_absolute_charge,
            "unusual_atoms": 0,
            "fragment_count": 1,
        },
        "outputs": {
            "control_manifest": str(manifest_path.relative_to(ROOT)),
            "target_readiness": str(target_stats_path.relative_to(ROOT)),
            "domain_exclusions": str(exclusions_path.relative_to(ROOT)) if args.docking_domain_filter else "",
        },
    }
    (out_dir / f"TARGET_DOCKING_CALIBRATION_{version}_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
