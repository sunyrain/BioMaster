#!/usr/bin/env python3
"""Build the label-free BioMaster pocket-context feature store V2.

V1 contains 19 useful but mostly global quality/availability fields.  V2 keeps
the same pair alignment and exact-fallback contract, while replacing raw
coordinate-frame-dependent values with four semantic groups:

* ``quality_*``: structure, sequence, pocket and local-confidence evidence;
* ``chem_*``: amino-acid chemistry of the selected pocket;
* ``geom_*``: invariant pocket size/shape/compactness descriptors;
* ``consensus_*``: top-vs-alternative pocket and experimental-structure support.

No activity label, affinity value, GNINA score, positive/negative control or
split assignment is read by this builder.  Missing or unparseable pockets are
emitted with ``structure_mask=0`` so the model falls back exactly to its
sequence/chemistry base.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRS = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
DEFAULT_TARGET_STRUCTURE = ROOT / "outputs/affinity_first_remote_discovery_v1/DTA_STAGE1_REMOTE_STRICT_STRUCTURE_V1.csv.gz"
DEFAULT_TARGET_ATLAS = ROOT / "outputs/affinity_first_remote_discovery_v1/experimental_structure_atlas_v1/TARGET_EXPERIMENTAL_STRUCTURE_ATLAS_463_V1.csv"
DEFAULT_P2RANK_DIR = ROOT / "outputs/p2rank26_pocket_audit_afcomplete_v1/p2rank_raw"
DEFAULT_OUT = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/ODTI_STRUCTURE_CONTEXT_V2.csv.gz"
DEFAULT_MANIFEST = ROOT / "outputs/old_drug_target_sota_v1/feature_store_v1/ODTI_STRUCTURE_CONTEXT_V2_MANIFEST.json"


QUALITY_COLUMNS = [
    "quality_sequence_exact_match",
    "quality_strict_tier",
    "quality_experimental_holo",
    "quality_pocket_probability",
    "quality_puresnet_overlap",
    "quality_pocket_plddt_mean",
    "quality_pocket_plddt_std",
    "quality_pocket_plddt_q10",
    "quality_best_holo_coverage",
    "quality_inverse_holo_resolution",
]

CHEMISTRY_COLUMNS = [
    "chem_hydrophobic_fraction",
    "chem_aromatic_fraction",
    "chem_polar_fraction",
    "chem_positive_fraction",
    "chem_negative_fraction",
    "chem_charged_fraction",
    "chem_hbond_donor_fraction",
    "chem_hbond_acceptor_fraction",
    "chem_sulfur_fraction",
    "chem_gly_pro_fraction",
    "chem_cysteine_fraction",
    "chem_histidine_fraction",
    "chem_mean_hydropathy_scaled",
    "chem_residue_entropy_normalized",
]

GEOMETRY_COLUMNS = [
    "geom_log1p_volume",
    "geom_log1p_residue_count",
    "geom_log1p_volume_per_residue",
    "geom_relative_residue_count",
    "geom_sphericity",
    "geom_radius_of_gyration",
    "geom_axis_ratio_21",
    "geom_axis_ratio_31",
    "geom_pocket_ca_radius_of_gyration",
    "geom_pocket_ca_contact_density_8a",
    "geom_centroid_offset_over_protein_rg",
    "geom_sequence_span_fraction",
]

CONSENSUS_COLUMNS = [
    "consensus_log1p_top_score",
    "consensus_top_second_score_gap",
    "consensus_top_second_probability_gap",
    "consensus_top5_probability_entropy",
    "consensus_top1_top2_center_distance_over_protein_rg",
    "consensus_top1_top2_residue_jaccard",
    "consensus_pockets_probability_ge_0_5",
    "consensus_structure_strong",
    "consensus_log1p_experimental_entries",
    "consensus_log1p_candidate_holo_entries",
    "consensus_candidate_holo_fraction",
]

FEATURE_GROUPS = {
    "quality": QUALITY_COLUMNS,
    "chemistry": CHEMISTRY_COLUMNS,
    "geometry": GEOMETRY_COLUMNS,
    "consensus": CONSENSUS_COLUMNS,
}
FEATURE_COLUMNS = [column for values in FEATURE_GROUPS.values() for column in values]


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}
HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
AA_GROUPS = {
    "hydrophobic": set("AVILMFWY"),
    "aromatic": set("FWYH"),
    "polar": set("STNQCY"),
    "positive": set("KRH"),
    "negative": set("DE"),
    "donor": set("KRHSTNQWY"),
    "acceptor": set("DENQHSTY"),
    "sulfur": set("CM"),
    "gly_pro": set("GP"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_complete_by_target(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return frame[["sequence_key"]].drop_duplicates().copy()
    work = frame[["sequence_key", *available]].copy()
    work["__coverage"] = work[available].notna().sum(axis=1)
    work = work.sort_values(["sequence_key", "__coverage"], ascending=[True, False])
    return work.drop_duplicates("sequence_key", keep="first").drop(columns="__coverage")


def _number(value: Any, default: float = 0.0) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    return output if math.isfinite(output) else default


def _truth(value: Any) -> float:
    return float(str(value).strip().lower() in {"1", "true", "yes", "y"})


def parse_pocket_ids(value: Any) -> list[tuple[str, int]]:
    """Parse P2Rank identifiers such as ``A_248`` deterministically."""

    parsed: list[tuple[str, int]] = []
    for token in str(value or "").split():
        if "_" not in token:
            continue
        chain, residue = token.split("_", 1)
        match = re.match(r"(-?\d+)", residue)
        if match:
            parsed.append((chain.strip() or " ", int(match.group(1))))
    return list(dict.fromkeys(parsed))


def parse_pdb_ca(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Read one CA record per residue from the first PDB model."""

    residues: dict[tuple[str, int], dict[str, Any]] = {}
    in_first_model = True
    saw_model = False
    with path.open("rt", errors="replace") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "MODEL":
                if saw_model:
                    in_first_model = False
                saw_model = True
                continue
            if record == "ENDMDL" and saw_model:
                break
            if not in_first_model or record != "ATOM" or line[12:16].strip() != "CA":
                continue
            altloc = line[16:17]
            if altloc not in {" ", "A"}:
                continue
            try:
                chain = line[21:22].strip() or " "
                residue_id = int(line[22:26])
                coord = np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=np.float64,
                )
                bfactor = float(line[60:66])
            except ValueError:
                continue
            residues.setdefault(
                (chain, residue_id),
                {
                    "aa": THREE_TO_ONE.get(line[17:20].strip().upper(), "X"),
                    "coord": coord,
                    "bfactor": bfactor,
                },
            )
    return residues


def _radius_of_gyration(coords: np.ndarray) -> float:
    if len(coords) == 0:
        return 0.0
    centered = coords - coords.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.square(centered).sum(axis=1).mean()))


def _pair_geometry(coords: np.ndarray) -> tuple[float, float]:
    if len(coords) < 2:
        return 0.0, 0.0
    delta = coords[:, None, :] - coords[None, :, :]
    distance = np.sqrt(np.square(delta).sum(axis=2))
    upper = distance[np.triu_indices(len(coords), k=1)]
    return float(upper.mean()), float((upper <= 8.0).mean())


def _residue_chemistry(amino_acids: list[str]) -> dict[str, float]:
    valid = [aa for aa in amino_acids if aa in HYDROPATHY]
    if not valid:
        return {column: 0.0 for column in CHEMISTRY_COLUMNS}
    count = float(len(valid))
    frequencies = Counter(valid)

    def fraction(group: set[str]) -> float:
        return sum(frequencies[aa] for aa in group) / count

    probabilities = np.array(list(frequencies.values()), dtype=np.float64) / count
    entropy = float(-(probabilities * np.log(probabilities)).sum() / math.log(20.0))
    positive = fraction(AA_GROUPS["positive"])
    negative = fraction(AA_GROUPS["negative"])
    return {
        "chem_hydrophobic_fraction": fraction(AA_GROUPS["hydrophobic"]),
        "chem_aromatic_fraction": fraction(AA_GROUPS["aromatic"]),
        "chem_polar_fraction": fraction(AA_GROUPS["polar"]),
        "chem_positive_fraction": positive,
        "chem_negative_fraction": negative,
        "chem_charged_fraction": positive + negative,
        "chem_hbond_donor_fraction": fraction(AA_GROUPS["donor"]),
        "chem_hbond_acceptor_fraction": fraction(AA_GROUPS["acceptor"]),
        "chem_sulfur_fraction": fraction(AA_GROUPS["sulfur"]),
        "chem_gly_pro_fraction": fraction(AA_GROUPS["gly_pro"]),
        "chem_cysteine_fraction": frequencies["C"] / count,
        "chem_histidine_fraction": frequencies["H"] / count,
        "chem_mean_hydropathy_scaled": float(
            np.mean([HYDROPATHY[aa] for aa in valid]) / 4.5
        ),
        "chem_residue_entropy_normalized": entropy,
    }


def load_p2rank_context(
    pdb_path: Path,
    directory: Path | None,
) -> tuple[pd.DataFrame | None, list[set[tuple[str, int]]], list[dict[str, str]]]:
    if directory is None:
        return None, [], []
    descriptor_path = directory / f"{pdb_path.name}_pocket_descriptors.csv.gz"
    prediction_path = directory / f"{pdb_path.name}_predictions.csv"
    provenance: list[dict[str, str]] = []
    descriptors = None
    residue_sets: list[set[tuple[str, int]]] = []
    if descriptor_path.is_file():
        descriptors = pd.read_csv(descriptor_path, low_memory=False)
        descriptors.columns = [str(column).strip() for column in descriptors.columns]
        descriptors = descriptors.sort_values("rank").reset_index(drop=True)
        provenance.append({"path": str(descriptor_path), "sha256": sha256(descriptor_path)})
    if prediction_path.is_file():
        predictions = pd.read_csv(prediction_path, low_memory=False)
        predictions.columns = [str(column).strip() for column in predictions.columns]
        predictions = predictions.sort_values("rank").reset_index(drop=True)
        residue_sets = [set(parse_pocket_ids(value)) for value in predictions["residue_ids"]]
        provenance.append({"path": str(prediction_path), "sha256": sha256(prediction_path)})
    return descriptors, residue_sets, provenance


def _probability_summary(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values >= 0)]
    if values.size == 0 or values.sum() <= 0:
        return 0.0, 0.0
    probabilities = values / values.sum()
    entropy = 0.0 if len(probabilities) == 1 else float(
        -(probabilities * np.log(probabilities.clip(1e-12))).sum() / math.log(len(probabilities))
    )
    return entropy, float(probabilities[0])


def target_feature_row(
    row: pd.Series,
    p2rank_dir: Path | None,
) -> tuple[dict[str, float], dict[str, Any]]:
    features = {column: 0.0 for column in FEATURE_COLUMNS}
    audit: dict[str, Any] = {
        "sequence_key": str(row.get("sequence_key", "")),
        "pdb_parsed": False,
        "descriptor_available": False,
        "pocket_residue_count_requested": 0,
        "pocket_residue_count_mapped": 0,
        "source_files": [],
        "error": None,
    }

    pdb_value = row.get("pdb_path")
    pdb_path = Path(str(pdb_value)) if pd.notna(pdb_value) and str(pdb_value) else None
    pocket_ids = parse_pocket_ids(row.get("top_pocket_residue_ids"))
    audit["pocket_residue_count_requested"] = len(pocket_ids)
    features.update({
        "quality_structure_available": float(pdb_path is not None and pdb_path.is_file()),
        "quality_pocket_definition_present": float(bool(pocket_ids)),
        "quality_sequence_exact_match": float(str(row.get("sequence_match_status", "")) == "exact_match"),
        "quality_strict_tier": float(str(row.get("strict_structure_tier", "")).startswith("A_")),
        "quality_experimental_holo": _truth(row.get("has_candidate_experimental_holo")),
        "quality_pocket_probability": _number(row.get("top_pocket_probability")),
        "quality_puresnet_overlap": _number(row.get("p2rank_puresnet_overlap_fraction")),
        "quality_puresnet_jaccard": _number(row.get("p2rank_puresnet_jaccard")),
        "quality_best_holo_coverage": _number(row.get("best_holo_coverage")),
        "quality_inverse_holo_resolution": (
            1.0 / _number(row.get("best_holo_resolution"))
            if _number(row.get("best_holo_resolution")) > 0 else 0.0
        ),
        "consensus_log1p_top_score": math.log1p(max(_number(row.get("top_pocket_score")), 0.0)),
        "consensus_puresnet_supported": float(
            str(row.get("puresnet_tier", "")).startswith(("A_", "B_"))
        ),
        "consensus_structure_strong": float(
            str(row.get("structure_consensus_tier", "")).startswith("A_")
        ),
        "consensus_log1p_experimental_entries": math.log1p(
            max(_number(row.get("experimental_entry_count_top20")), 0.0)
        ),
        "consensus_log1p_candidate_holo_entries": math.log1p(
            max(_number(row.get("candidate_holo_entry_count_top20")), 0.0)
        ),
    })
    experimental_count = max(_number(row.get("experimental_entry_count_top20")), 0.0)
    holo_count = max(_number(row.get("candidate_holo_entry_count_top20")), 0.0)
    features["consensus_candidate_holo_fraction"] = (
        min(holo_count / experimental_count, 1.0) if experimental_count > 0 else 0.0
    )

    if pdb_path is None or not pdb_path.is_file() or not pocket_ids:
        return features, audit
    try:
        residues = parse_pdb_ca(pdb_path)
        audit["pdb_parsed"] = bool(residues)
        mapped = [residues[key] for key in pocket_ids if key in residues]
        audit["pocket_residue_count_mapped"] = len(mapped)
        mapping_fraction = len(mapped) / max(len(pocket_ids), 1)
        features["quality_pocket_residue_mapping_fraction"] = mapping_fraction
        if not mapped:
            return features, audit

        pocket_coords = np.stack([value["coord"] for value in mapped])
        protein_coords = np.stack([value["coord"] for value in residues.values()])
        pocket_plddt = np.array([value["bfactor"] for value in mapped], dtype=np.float64)
        # AlphaFold stores pLDDT in the B-factor field.  If a future source is
        # experimental, the source/quality indicator remains explicit and the
        # values are still finite local confidence-like summaries rather than
        # activity-derived evidence.
        features["quality_pocket_plddt_mean"] = float(pocket_plddt.mean() / 100.0)
        features["quality_pocket_plddt_std"] = float(pocket_plddt.std() / 100.0)
        features["quality_pocket_plddt_q10"] = float(np.quantile(pocket_plddt, 0.10) / 100.0)
        features.update(_residue_chemistry([value["aa"] for value in mapped]))

        pocket_rg = _radius_of_gyration(pocket_coords)
        protein_rg = _radius_of_gyration(protein_coords)
        mean_distance, contact_density = _pair_geometry(pocket_coords)
        centroid_offset = float(np.linalg.norm(pocket_coords.mean(axis=0) - protein_coords.mean(axis=0)))
        residue_numbers = np.array([key[1] for key in pocket_ids if key in residues], dtype=np.float64)
        receptor_count = max(_number(row.get("receptor_residue_count"), len(residues)), 1.0)
        features.update({
            "geom_pocket_ca_radius_of_gyration": pocket_rg,
            "geom_pocket_ca_mean_pair_distance": mean_distance,
            "geom_pocket_ca_contact_density_8a": contact_density,
            "geom_centroid_offset_over_protein_rg": centroid_offset / max(protein_rg, 1e-6),
            "geom_sequence_span_fraction": (
                float(residue_numbers.max() - residue_numbers.min() + 1) / receptor_count
                if len(residue_numbers) else 0.0
            ),
        })

        descriptors, residue_sets, provenance = load_p2rank_context(pdb_path, p2rank_dir)
        audit["source_files"] = provenance
        if descriptors is not None and len(descriptors):
            audit["descriptor_available"] = True
            top = descriptors.iloc[0]
            residue_count = max(_number(top.get("num_residues"), len(mapped)), 0.0)
            volume = max(_number(top.get("volume"), row.get("p2rank26_top_pocket_volume")), 0.0)
            lambda1 = max(_number(top.get("principal_moments.lambda1")), 0.0)
            lambda2 = max(_number(top.get("principal_moments.lambda2")), 0.0)
            lambda3 = max(_number(top.get("principal_moments.lambda3")), 0.0)
            features.update({
                "geom_log1p_volume": math.log1p(volume),
                "geom_log1p_residue_count": math.log1p(residue_count),
                "geom_log1p_volume_per_residue": math.log1p(volume / max(residue_count, 1.0)),
                "geom_relative_residue_count": residue_count / receptor_count,
                "geom_sphericity": _number(top.get("sphericity")),
                "geom_radius_of_gyration": _number(top.get("radius_of_gyration")),
                "geom_axis_ratio_21": lambda2 / max(lambda1, 1e-8),
                "geom_axis_ratio_31": lambda3 / max(lambda1, 1e-8),
                "geom_anisotropy": (lambda1 - lambda3) / max(lambda1, 1e-8),
                "geom_descriptor_available": 1.0,
            })
            top5 = descriptors.head(5)
            scores = pd.to_numeric(top5["score"], errors="coerce").to_numpy(dtype=np.float64)
            probabilities = pd.to_numeric(top5["probability"], errors="coerce").to_numpy(dtype=np.float64)
            entropy, dominance = _probability_summary(probabilities)
            features["consensus_top5_probability_entropy"] = entropy
            features["consensus_top5_probability_dominance"] = dominance
            features["consensus_pockets_probability_ge_0_5"] = float((probabilities >= 0.5).sum())
            if len(top5) > 1:
                second = top5.iloc[1]
                features["consensus_top_second_score_gap"] = max(scores[0] - scores[1], 0.0)
                features["consensus_top_second_probability_gap"] = max(
                    probabilities[0] - probabilities[1], 0.0
                )
                center1 = np.array([_number(top.get(key)) for key in ["center_x", "center_y", "center_z"]])
                center2 = np.array([_number(second.get(key)) for key in ["center_x", "center_y", "center_z"]])
                features["consensus_top1_top2_center_distance_over_protein_rg"] = (
                    float(np.linalg.norm(center1 - center2)) / max(protein_rg, 1e-6)
                )
            if len(residue_sets) > 1:
                union = residue_sets[0] | residue_sets[1]
                features["consensus_top1_top2_residue_jaccard"] = (
                    len(residue_sets[0] & residue_sets[1]) / len(union) if union else 0.0
                )
        else:
            # Preserve stable invariant geometry even if the descriptor sidecar
            # is absent.
            residue_count = float(len(mapped))
            volume = max(_number(row.get("p2rank26_top_pocket_volume")), 0.0)
            features.update({
                "geom_log1p_volume": math.log1p(volume),
                "geom_log1p_residue_count": math.log1p(residue_count),
                "geom_log1p_volume_per_residue": math.log1p(volume / max(residue_count, 1.0)),
                "geom_relative_residue_count": residue_count / receptor_count,
            })
    except Exception as error:  # fail closed per target; audit records the reason
        audit["error"] = f"{type(error).__name__}: {error}"
    return features, audit


def _correlation_audit(target_features: pd.DataFrame) -> list[dict[str, Any]]:
    matrix = target_features[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    std = matrix.std(axis=0)
    variable = np.flatnonzero(std > 1e-12)
    if len(variable) < 2:
        return []
    correlation = np.corrcoef(matrix[:, variable], rowvar=False)
    findings: list[dict[str, Any]] = []
    for i in range(len(variable)):
        for j in range(i):
            value = float(correlation[i, j])
            if math.isfinite(value) and abs(value) >= 0.95:
                findings.append({
                    "feature_a": FEATURE_COLUMNS[int(variable[j])],
                    "feature_b": FEATURE_COLUMNS[int(variable[i])],
                    "pearson": value,
                })
    return findings


def build_structure_context_v2(
    pairs_path: Path,
    target_structure_path: Path,
    atlas_path: Path | None,
    p2rank_dir: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pairs = pd.read_csv(pairs_path, low_memory=False)
    required = {"calibration_pair_id", "sequence_key"}
    missing = sorted(required - set(pairs.columns))
    if missing:
        raise ValueError(f"pair table missing required columns: {missing}")
    if pairs["calibration_pair_id"].duplicated().any():
        raise ValueError("pair table has duplicate calibration_pair_id")

    dta = pd.read_csv(target_structure_path, low_memory=False)
    safe_dta = [
        "pdb_path", "top_pocket_score", "top_pocket_probability",
        "top_pocket_residue_ids", "p2rank26_top_pocket_volume",
        "p2rank26_top_pocket_num_residues", "p2rank_puresnet_overlap_fraction",
        "p2rank_puresnet_jaccard", "receptor_residue_count", "structure_bin",
        "strict_structure_tier", "sequence_match_status", "pocket_definition_status",
        "puresnet_tier", "structure_consensus_tier", "receptor_pdb_sha256",
    ]
    target = first_complete_by_target(dta, safe_dta)
    atlas_summary: dict[str, Any] = {"provided": False}
    if atlas_path is not None:
        atlas = pd.read_csv(atlas_path, low_memory=False)
        safe_atlas = [
            "experimental_entry_count_top20", "candidate_holo_entry_count_top20",
            "has_candidate_experimental_holo", "best_holo_resolution",
            "best_holo_coverage", "holo_validation_status",
        ]
        target = target.merge(
            first_complete_by_target(atlas, safe_atlas),
            on="sequence_key", how="outer", validate="one_to_one",
        )
        atlas_summary = {
            "provided": True,
            "rows": int(len(atlas)),
            "targets": int(atlas["sequence_key"].nunique()),
        }

    target_rows: list[dict[str, Any]] = []
    target_audits: list[dict[str, Any]] = []
    for _, row in target.iterrows():
        feature_values, audit = target_feature_row(row, p2rank_dir)
        target_rows.append({"sequence_key": str(row["sequence_key"]), **feature_values})
        target_audits.append(audit)
    target_features = pd.DataFrame(target_rows)
    target_features[FEATURE_COLUMNS] = (
        target_features[FEATURE_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )
    audit_by_target = {str(item["sequence_key"]): item for item in target_audits}
    mapped_targets = {
        key for key, item in audit_by_target.items()
        if item["pdb_parsed"] and item["pocket_residue_count_mapped"] > 0
    }

    output = pairs[["calibration_pair_id", "sequence_key"]].merge(
        target_features, on="sequence_key", how="left", validate="many_to_one"
    )
    output[FEATURE_COLUMNS] = output[FEATURE_COLUMNS].fillna(0.0).astype(np.float32)
    output.insert(
        1,
        "structure_mask",
        output["sequence_key"].astype(str).isin(mapped_targets).astype(np.float32),
    )
    output = output[["calibration_pair_id", "structure_mask", *FEATURE_COLUMNS]]
    if len(output) != len(pairs) or output["calibration_pair_id"].duplicated().any():
        raise RuntimeError("V2 structure store is not a one-to-one pair alignment")
    values = output[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("V2 structure store contains non-finite values")

    parse_errors = [item for item in target_audits if item["error"]]
    present_target_features = target_features[
        target_features["sequence_key"].astype(str).isin(mapped_targets)
    ]
    constants = [
        column for column in FEATURE_COLUMNS
        if present_target_features[column].nunique(dropna=False) <= 1
    ]
    provenance_digest = hashlib.sha256()
    provenance_rows = []
    for item in target_audits:
        for source in item["source_files"]:
            provenance_rows.append(source)
            provenance_digest.update(source["path"].encode())
            provenance_digest.update(source["sha256"].encode())

    manifest: dict[str, Any] = {
        "status": "PASS" if not constants else "PASS_WITH_CONSTANT_COLUMNS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "LABEL_FREE_POCKET_CONTEXT_V2",
        "label_dependency": "NONE_IN_FEATURE_COLUMNS",
        "pair_rows": int(len(pairs)),
        "pair_targets": int(pairs["sequence_key"].nunique()),
        "targets_with_structure_context": int(len(mapped_targets & set(pairs["sequence_key"].astype(str)))),
        "pairs_with_structure_context": int(output["structure_mask"].sum()),
        "feature_columns": FEATURE_COLUMNS,
        "feature_dim": len(FEATURE_COLUMNS),
        "feature_groups": {
            name: {"columns": columns, "dim": len(columns)}
            for name, columns in FEATURE_GROUPS.items()
        },
        "coordinate_contract": {
            "raw_absolute_xyz_in_model_features": False,
            "invariant_geometry_only": True,
            "shared_ligand_pocket_frame_assumed": False,
        },
        "target_parse_audit": {
            "target_rows": len(target_rows),
            "pdb_parsed": sum(bool(item["pdb_parsed"]) for item in target_audits),
            "descriptor_available": sum(bool(item["descriptor_available"]) for item in target_audits),
            "parse_error_count": len(parse_errors),
            "parse_errors": parse_errors[:25],
            "constant_columns": constants,
            "correlation_scope": "targets_with_structure_context_only",
            "high_correlation_pairs_abs_ge_0_95": _correlation_audit(
                present_target_features
            ),
        },
        "source_tables": {
            "pairs": {"path": str(pairs_path), "sha256": sha256(pairs_path)},
            "target_structure": {"path": str(target_structure_path), "sha256": sha256(target_structure_path)},
            "atlas": ({"path": str(atlas_path), "sha256": sha256(atlas_path)} if atlas_path else None),
            "p2rank_directory": {
                "path": str(p2rank_dir) if p2rank_dir else None,
                "used_file_count": len(provenance_rows),
                "used_file_inventory_sha256": provenance_digest.hexdigest(),
            },
        },
        "atlas_summary": atlas_summary,
        "excluded_label_derived_sources": [
            "GNINA scores", "positive/negative control selection",
            "activity labels", "affinity labels", "split assignments",
        ],
        "formal_use": (
            "eligible as a grouped target-pocket residual; pair-specific distances "
            "still require a jointly posed ligand-pocket coordinate frame"
        ),
    }
    return output, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    parser.add_argument("--target-structure", default=str(DEFAULT_TARGET_STRUCTURE))
    parser.add_argument("--atlas", default=str(DEFAULT_TARGET_ATLAS))
    parser.add_argument("--p2rank-dir", default=str(DEFAULT_P2RANK_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()

    pairs_path = Path(args.pairs)
    target_structure_path = Path(args.target_structure)
    atlas_path = Path(args.atlas) if args.atlas else None
    p2rank_dir = Path(args.p2rank_dir) if args.p2rank_dir else None
    for path in [pairs_path, target_structure_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if atlas_path is not None and not atlas_path.is_file():
        raise FileNotFoundError(atlas_path)
    if p2rank_dir is not None and not p2rank_dir.is_dir():
        raise FileNotFoundError(p2rank_dir)

    features, manifest = build_structure_context_v2(
        pairs_path, target_structure_path, atlas_path, p2rank_dir
    )
    out = Path(args.out)
    manifest_path = Path(args.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out, index=False, compression="gzip")
    manifest["output"] = {"path": str(out), "sha256": sha256(out)}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
