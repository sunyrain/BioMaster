#!/usr/bin/env python3
"""Extract frozen label-blind local WT/mutant ESM2 delta representations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/biomaster_mutation_local_esm2_delta_feature_freeze_20260814.json"
DEFAULT_OUTPUT = ROOT / "outputs/evidence_routing_compute_execution_20260808_v1/mutation_local_esm2_delta_v11"
SUBSTITUTION_PATTERN = re.compile(
    r"(?:^|_)([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])(?=_|$)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def parse_substitutions(construct: str) -> list[tuple[str, int, str]]:
    inside = (
        construct[construct.find("(") + 1 : construct.rfind(")")]
        if "(" in construct and ")" in construct
        else construct
    )
    return [
        (source, int(position), destination)
        for source, position, destination in SUBSTITUTION_PATTERN.findall(inside)
    ]


def request_key(sequence: str, center_offset_zero_based: int) -> str:
    payload = f"{center_offset_zero_based}|{sequence}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def signed_max_abs(values: np.ndarray) -> np.ndarray:
    if values.shape[0] == 1:
        return values[0]
    index = np.abs(values).argmax(axis=0)
    return values[index, np.arange(values.shape[1])]


def build_manifests(
    constructs: pd.DataFrame,
    radius: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[str, int]]]:
    construct_rows: list[dict[str, Any]] = []
    site_rows: list[dict[str, Any]] = []
    requests: dict[str, tuple[str, int]] = {}
    for construct_index, row in enumerate(constructs.itertuples(index=False)):
        construct = str(row.variant_construct)
        sequence = str(row.sequence)
        substitutions = parse_substitutions(construct)
        mismatch = []
        for source, position, destination in substitutions:
            if position < 1 or position > len(sequence):
                mismatch.append(f"{source}{position}{destination}:OUT_OF_RANGE")
            elif sequence[position - 1] != source:
                mismatch.append(
                    f"{source}{position}{destination}:REFERENCE_IS_{sequence[position - 1]}"
                )
        if not substitutions:
            status = "NO_PARSABLE_SUBSTITUTION"
            available = False
        elif mismatch:
            status = "REFERENCE_RESIDUE_MISMATCH_OR_ISOFORM_NUMBERING"
            available = False
        elif len(substitutions) > 4:
            status = "MORE_THAN_FOUR_SUBSTITUTIONS"
            available = False
        else:
            status = "AVAILABLE_EXACT_REFERENCE_SUBSTITUTION"
            available = True
        construct_rows.append(
            {
                "construct_feature_index": construct_index,
                "target_chembl_id": str(row.target_chembl_id),
                "gene_symbol": str(row.gene_symbol),
                "variant_construct": construct,
                "reference_sequence_length": len(sequence),
                "parsed_substitution_count": len(substitutions),
                "substitutions_json": json.dumps(substitutions, separators=(",", ":")),
                "reconstruction_available": available,
                "reconstruction_status": status,
                "mismatch_details": ";".join(mismatch),
            }
        )
        if not available:
            continue
        mutant_sequence = list(sequence)
        for _, position, destination in substitutions:
            mutant_sequence[position - 1] = destination
        mutant_sequence = "".join(mutant_sequence)
        for site_number, (source, position, destination) in enumerate(substitutions, start=1):
            position_zero = position - 1
            start = max(0, position_zero - radius)
            stop = min(len(sequence), position_zero + radius + 1)
            center = position_zero - start
            wt_window = sequence[start:stop]
            mutant_window = mutant_sequence[start:stop]
            wt_key = request_key(wt_window, center)
            mutant_key = request_key(mutant_window, center)
            requests.setdefault(wt_key, (wt_window, center))
            requests.setdefault(mutant_key, (mutant_window, center))
            site_rows.append(
                {
                    "construct_feature_index": construct_index,
                    "target_chembl_id": str(row.target_chembl_id),
                    "gene_symbol": str(row.gene_symbol),
                    "variant_construct": construct,
                    "site_number": site_number,
                    "source_amino_acid": source,
                    "position_one_based": position,
                    "destination_amino_acid": destination,
                    "window_start_one_based": start + 1,
                    "window_stop_one_based_inclusive": stop,
                    "window_length": len(wt_window),
                    "center_offset_zero_based": center,
                    "wt_center_amino_acid": wt_window[center],
                    "mutant_center_amino_acid": mutant_window[center],
                    "wt_request_key": wt_key,
                    "mutant_request_key": mutant_key,
                    "wt_window_sequence": wt_window,
                    "mutant_window_sequence": mutant_window,
                }
            )
    return pd.DataFrame(construct_rows), pd.DataFrame(site_rows), requests


def extract_requests(
    requests: dict[str, tuple[str, int]],
    checkpoint: Path,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen ESM2-650M extraction")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["TORCH_HOME"] = str(checkpoint.parents[2])
    import esm  # noqa: PLC0415

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    device = torch.device("cuda")
    model.to(device).eval()
    converter = alphabet.get_batch_converter()
    layer = model.num_layers
    ordered = sorted(requests.items(), key=lambda item: (len(item[1][0]), item[0]))
    extracted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    batches = 0
    with torch.inference_mode():
        for start in range(0, len(ordered), batch_size):
            selected = ordered[start : start + batch_size]
            labels_and_sequences = [(key, payload[0]) for key, payload in selected]
            _, _, tokens = converter(labels_and_sequences)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                result = model(
                    tokens.to(device, non_blocking=True),
                    repr_layers=[layer],
                    return_contacts=False,
                )
            representation = result["representations"][layer]
            for batch_position, (key, (sequence, center)) in enumerate(selected):
                center_embedding = representation[batch_position, center + 1].float().cpu().numpy()
                window_mean = (
                    representation[batch_position, 1 : len(sequence) + 1]
                    .float()
                    .mean(dim=0)
                    .cpu()
                    .numpy()
                )
                extracted[key] = (
                    center_embedding.astype(np.float32, copy=False),
                    window_mean.astype(np.float32, copy=False),
                )
            batches += 1
            if batches % 20 == 0 or start + len(selected) == len(ordered):
                print(
                    json.dumps(
                        {
                            "esm2_request_windows_completed": start + len(selected),
                            "total": len(ordered),
                            "batches": batches,
                            "gpu_memory_allocated_mib": round(
                                torch.cuda.memory_allocated() / (1024**2), 1
                            ),
                        }
                    ),
                    flush=True,
                )
    del model
    torch.cuda.empty_cache()
    return extracted, {
        "requests": len(ordered),
        "batches": batches,
        "batch_size": batch_size,
        "layer": layer,
        "precision": "CUDA_AUTOCAST_FLOAT16_SAVE_FLOAT32",
    }


def aggregate_constructs(
    construct_manifest: pd.DataFrame,
    site_manifest: pd.DataFrame,
    extracted: dict[str, tuple[np.ndarray, np.ndarray]],
    dimension: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    count = len(construct_manifest)
    matrices = {
        "wt_center_mean": np.zeros((count, dimension), dtype=np.float32),
        "mutant_center_mean": np.zeros((count, dimension), dtype=np.float32),
        "center_delta_mean": np.zeros((count, dimension), dtype=np.float32),
        "center_delta_signed_maxabs": np.zeros((count, dimension), dtype=np.float32),
        "window_mean_delta_mean": np.zeros((count, dimension), dtype=np.float32),
    }
    norm_rows: list[dict[str, Any]] = []
    for construct_index, sites in site_manifest.groupby(
        "construct_feature_index", sort=False, observed=True
    ):
        wt_center = np.stack(
            [extracted[key][0] for key in sites["wt_request_key"]]
        ).astype(np.float32)
        mutant_center = np.stack(
            [extracted[key][0] for key in sites["mutant_request_key"]]
        ).astype(np.float32)
        wt_window = np.stack(
            [extracted[key][1] for key in sites["wt_request_key"]]
        ).astype(np.float32)
        mutant_window = np.stack(
            [extracted[key][1] for key in sites["mutant_request_key"]]
        ).astype(np.float32)
        center_delta = mutant_center - wt_center
        window_delta = mutant_window - wt_window
        index = int(construct_index)
        matrices["wt_center_mean"][index] = wt_center.mean(axis=0)
        matrices["mutant_center_mean"][index] = mutant_center.mean(axis=0)
        matrices["center_delta_mean"][index] = center_delta.mean(axis=0)
        matrices["center_delta_signed_maxabs"][index] = signed_max_abs(center_delta)
        matrices["window_mean_delta_mean"][index] = window_delta.mean(axis=0)
        norm_rows.append(
            {
                "construct_feature_index": index,
                "wt_center_mean_l2": float(
                    np.linalg.norm(matrices["wt_center_mean"][index])
                ),
                "mutant_center_mean_l2": float(
                    np.linalg.norm(matrices["mutant_center_mean"][index])
                ),
                "center_delta_mean_l2": float(
                    np.linalg.norm(matrices["center_delta_mean"][index])
                ),
                "center_delta_signed_maxabs_l2": float(
                    np.linalg.norm(matrices["center_delta_signed_maxabs"][index])
                ),
                "window_mean_delta_mean_l2": float(
                    np.linalg.norm(matrices["window_mean_delta_mean"][index])
                ),
            }
        )
    norms = pd.DataFrame(norm_rows)
    manifest = construct_manifest.merge(
        norms, on="construct_feature_index", how="left", validate="one_to_one"
    )
    norm_columns = [column for column in manifest if column.endswith("_l2")]
    manifest[norm_columns] = manifest[norm_columns].fillna(0.0)
    return matrices, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config["status"] != "FROZEN_BEFORE_LABEL_BLIND_FEATURE_EXTRACTION":
        raise RuntimeError("Unexpected feature freeze status")
    input_specs = config["inputs"]
    paths = {
        name: (ROOT / spec["path"]).resolve() if not Path(spec["path"]).is_absolute() else Path(spec["path"])
        for name, spec in input_specs.items()
    }
    for name, path in paths.items():
        observed = sha256(path)
        if observed != input_specs[name]["sha256"]:
            raise RuntimeError(f"Input hash mismatch for {name}: {observed}")
    panel_columns = input_specs["mutation_panel"]["columns_allowed"]
    sequence_columns = input_specs["reference_sequences"]["columns_allowed"]
    panel = pd.read_csv(paths["mutation_panel"], usecols=panel_columns, low_memory=False)
    constructs = panel.drop_duplicates(panel_columns).copy()
    sequences = pd.read_csv(
        paths["reference_sequences"], usecols=sequence_columns, low_memory=False
    ).drop_duplicates("target_chembl_id")
    constructs = constructs.merge(
        sequences, on="target_chembl_id", how="left", validate="many_to_one"
    ).sort_values(
        ["target_chembl_id", "variant_construct"], kind="stable"
    ).reset_index(drop=True)
    if constructs["sequence"].isna().any():
        raise RuntimeError("Reference sequences are incomplete")
    radius = int(config["representation"]["window_radius_residues"])
    construct_manifest, site_manifest, requests = build_manifests(constructs, radius)
    covered = panel.merge(
        construct_manifest[
            ["target_chembl_id", "variant_construct", "reconstruction_available"]
        ],
        on=["target_chembl_id", "variant_construct"],
        how="left",
        validate="many_to_one",
    )
    observed_counts = {
        "unique_constructs": int(len(construct_manifest)),
        "targets": int(construct_manifest["target_chembl_id"].nunique()),
        "constructs_with_one_to_four_parsable_substitutions": int(
            construct_manifest["parsed_substitution_count"].between(1, 4).sum()
        ),
        "constructs_with_all_reference_residues_matching": int(
            construct_manifest["reconstruction_available"].sum()
        ),
        "constructs_with_numbering_or_isoform_mismatch": int(
            construct_manifest["reconstruction_status"]
            .eq("REFERENCE_RESIDUE_MISMATCH_OR_ISOFORM_NUMBERING")
            .sum()
        ),
        "constructs_without_parsable_substitutions": int(
            construct_manifest["reconstruction_status"].eq("NO_PARSABLE_SUBSTITUTION").sum()
        ),
        "measurement_rows_covered_by_reconstructable_substitutions": int(
            covered["reconstruction_available"].sum()
        ),
        "total_measurement_rows": int(len(covered)),
    }
    expected_counts = config["frozen_counts_before_extraction"]
    if observed_counts != expected_counts:
        raise RuntimeError(
            f"Frozen reconstruction counts changed: {observed_counts} != {expected_counts}"
        )
    validation = {
        "config_sha256": sha256(config_path),
        "input_hashes": {name: sha256(path) for name, path in paths.items()},
        "counts": observed_counts,
        "site_rows": int(len(site_manifest)),
        "unique_embedding_requests": int(len(requests)),
        "columns_read": {
            "mutation_panel": panel_columns,
            "reference_sequences": sequence_columns,
        },
        "forbidden_label_columns_read": [],
    }
    if args.validate_only:
        print(json.dumps({"status": "VALID", **validation}, ensure_ascii=False, indent=2))
        return
    extracted, extraction_metadata = extract_requests(
        requests,
        paths["esm2_checkpoint"],
        int(config["representation"]["batch_size"]),
        int(config["representation"]["deterministic_seed"]),
    )
    if set(extracted) != set(requests):
        raise RuntimeError("ESM2 request extraction incomplete")
    dimension = int(config["representation"]["embedding_dimension"])
    matrices, construct_manifest = aggregate_constructs(
        construct_manifest, site_manifest, extracted, dimension
    )
    available = construct_manifest["reconstruction_available"].to_numpy(dtype=bool)
    unavailable = ~available
    checks = {
        "all_source_hashes_match": True,
        "only_allowed_columns_read": True,
        "forbidden_label_columns_read_zero": True,
        "frozen_counts_exact": observed_counts == expected_counts,
        "site_center_reference_checks_exact": bool(
            site_manifest["wt_center_amino_acid"].eq(site_manifest["source_amino_acid"]).all()
            and site_manifest["mutant_center_amino_acid"].eq(
                site_manifest["destination_amino_acid"]
            ).all()
        ),
        "all_requests_extracted": set(extracted) == set(requests),
        "all_matrices_exact_299x1280_float32": all(
            value.shape == (299, 1280) and value.dtype == np.float32
            for value in matrices.values()
        ),
        "all_embeddings_finite": all(np.isfinite(value).all() for value in matrices.values()),
        "all_available_center_deltas_nonzero": bool(
            np.linalg.norm(matrices["center_delta_mean"][available], axis=1).min() > 0
        ),
        "all_unavailable_vectors_exact_zero": all(
            np.count_nonzero(value[unavailable]) == 0 for value in matrices.values()
        ),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    construct_path = output_dir / "MUTATION_LOCAL_ESM2_CONSTRUCT_MANIFEST_299_V1.csv"
    site_path = output_dir / "MUTATION_LOCAL_ESM2_SITE_MANIFEST_V1.csv.gz"
    construct_manifest.to_csv(construct_path, index=False)
    site_manifest.to_csv(
        site_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 5, "mtime": 0},
    )
    matrix_paths = {}
    for name, value in matrices.items():
        path = output_dir / f"MUTATION_LOCAL_ESM2_{name.upper()}_299X1280_FLOAT32_V1.npy"
        np.save(path, value, allow_pickle=False)
        matrix_paths[name] = path
    output_paths = [construct_path, site_path, *matrix_paths.values()]
    summary = {
        "schema_version": "MUTATION_LOCAL_ESM2_DELTA_FEATURE_SUMMARY_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "config": {"path": relative(config_path), "sha256": sha256(config_path)},
        "script": {
            "path": relative(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "validation": validation,
        "extraction": extraction_metadata,
        "checks": checks,
        "availability": {
            "constructs_available": int(available.sum()),
            "constructs_unavailable": int(unavailable.sum()),
            "measurement_rows_covered": observed_counts[
                "measurement_rows_covered_by_reconstructable_substitutions"
            ],
            "measurement_rows_total": observed_counts["total_measurement_rows"],
        },
        "delta_norms": {
            "center_delta_mean_available_min": float(
                construct_manifest.loc[available, "center_delta_mean_l2"].min()
            ),
            "center_delta_mean_available_median": float(
                construct_manifest.loc[available, "center_delta_mean_l2"].median()
            ),
            "center_delta_mean_available_max": float(
                construct_manifest.loc[available, "center_delta_mean_l2"].max()
            ),
            "window_mean_delta_available_median": float(
                construct_manifest.loc[available, "window_mean_delta_mean_l2"].median()
            ),
        },
        "claim_boundary": config["claim_boundary"],
        "labels_read": [],
        "model_training_authorized": False,
        "outputs": {
            relative(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in output_paths
        },
    }
    summary_path = output_dir / "MUTATION_LOCAL_ESM2_DELTA_FEATURE_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
