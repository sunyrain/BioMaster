#!/usr/bin/env python3
"""Build a label-blind, residue-level ESM2 target feature store.

The existing BioMaster feature store contains a pooled 1280-dimensional ESM2
vector per target.  This builder preserves the same frozen target index while
also exporting residue-level representations for a later pocket/cross-
attention branch.  Long proteins are encoded in overlapping windows because
ESM2-t33-650M is evaluated with a bounded context length; overlapping windows
are averaged at residues that occur in more than one window.

Outputs are a concatenated ``.npy`` matrix and an index table with offsets,
which avoids padding all 428 targets to the longest protein during training.
No activity, assay, label, docking score, or benchmark split field is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/old_drug_target_sota_v1"
TARGET_INDEX = BASE / "feature_store_v1/TARGET_FEATURE_INDEX_V1.csv.gz"
DEFAULT_OUT = BASE / "public_retrained_v1/dtiam_official_feature_store_v1"
DEFAULT_TARGET_STRUCTURE = ROOT / "outputs/affinity_first_remote_discovery_v1/DTA_STAGE1_REMOTE_STRICT_STRUCTURE_V1.csv.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def window_bounds(length: int, max_length: int, overlap: int) -> list[tuple[int, int]]:
    if length <= 0:
        return []
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if overlap < 0 or overlap >= max_length:
        raise ValueError("overlap must satisfy 0 <= overlap < max_length")
    if length <= max_length:
        return [(0, length)]
    step = max_length - overlap
    starts = list(range(0, length - max_length + 1, step))
    final_start = length - max_length
    if starts[-1] != final_start:
        starts.append(final_start)
    return [(start, min(length, start + max_length)) for start in starts]


def load_esm2(device: torch.device):
    try:
        import esm  # type: ignore
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("fair-esm is required to build token features") from error
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device).eval()
    return model, alphabet


def build_pocket_masks(
    target_index: pd.DataFrame,
    target_structure_path: Path | None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Project audited PDB pocket residue IDs to sequence positions.

    The source table stores identifiers such as ``A_248``.  We only use the
    numeric component as a 1-based sequence position when the source marks an
    exact sequence match; unresolved or non-exact targets receive an all-zero
    mask and remain valid for the full-token branch.
    """

    total = int(target_index["protein_sequence"].astype(str).str.len().sum())
    mask = np.zeros(total, dtype=np.uint8)
    if target_structure_path is None:
        return mask, {"enabled": False, "targets_with_pocket_mask": 0}
    if not target_structure_path.is_file():
        raise FileNotFoundError(target_structure_path)
    table = pd.read_csv(target_structure_path, low_memory=False)
    required = {"sequence_key", "top_pocket_residue_ids"}
    if not required.issubset(table.columns):
        raise ValueError(f"target structure table is missing {sorted(required - set(table.columns))}")
    if "sequence_match_status" in table.columns:
        exact = table[table["sequence_match_status"].astype(str).eq("exact_match")].copy()
    else:
        exact = table.copy()
    exact["__ids"] = exact["top_pocket_residue_ids"].fillna("").astype(str)
    exact["__n"] = exact["__ids"].str.len()
    exact = exact.sort_values(["sequence_key", "__n"], ascending=[True, False])
    selected = exact.drop_duplicates("sequence_key", keep="first").set_index("sequence_key")
    offsets = 0
    count = 0
    target_masks: list[int] = []
    for row in target_index.itertuples(index=False):
        sequence = str(row.protein_sequence)
        residue_ids = ""
        if str(row.sequence_key) in selected.index:
            residue_ids = str(selected.loc[str(row.sequence_key), "__ids"])
        local = np.zeros(len(sequence), dtype=np.uint8)
        if residue_ids and residue_ids.lower() not in {"nan", "none"}:
            for token in residue_ids.split():
                match = re.search(r"_(\d+)", token)
                if not match:
                    continue
                position = int(match.group(1)) - 1
                if 0 <= position < len(sequence):
                    local[position] = 1
        if local.any():
            count += 1
        mask[offsets : offsets + len(sequence)] = local
        offsets += len(sequence)
        target_masks.append(int(local.sum()))
    return mask, {
        "enabled": True,
        "targets_with_pocket_mask": int(count),
        "targets_without_pocket_mask": int(len(target_index) - count),
        "projection": "exact_sequence_match_PDB_residue_number_to_1_based_sequence_position",
        "source": str(target_structure_path),
        "source_sha256": sha256(target_structure_path),
        "pocket_residue_count_total": int(sum(target_masks)),
    }


def build(args: argparse.Namespace) -> dict[str, object]:
    target_index_path = Path(args.target_index)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    index_path = output_dir / args.index_name
    manifest_path = output_dir / args.manifest_name
    pocket_mask_path = output_dir / args.pocket_mask_name
    if not target_index_path.is_file():
        raise FileNotFoundError(target_index_path)
    if output_path.exists() or index_path.exists() or manifest_path.exists():
        if not args.force:
            raise FileExistsError(
                "token feature artifacts already exist; use --force or a new --output-dir"
            )
        for path in (output_path, index_path, manifest_path, pocket_mask_path):
            path.unlink(missing_ok=True)

    index = pd.read_csv(target_index_path, low_memory=False).sort_values(
        "target_feature_index"
    ).reset_index(drop=True)
    required = {"target_feature_index", "sequence_key", "protein_sequence"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"target index is missing columns: {sorted(missing)}")
    expected = np.arange(len(index), dtype=np.int64)
    actual = index["target_feature_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual, expected):
        raise ValueError("target_feature_index must be dense and sorted")
    sequences = index["protein_sequence"].astype(str).tolist()
    if any(not sequence for sequence in sequences):
        raise ValueError("target index contains an empty protein sequence")

    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=np.int64)
    if "sequence_length" in index.columns and not np.array_equal(
        lengths, index["sequence_length"].to_numpy(dtype=np.int64)
    ):
        raise ValueError("protein_sequence and sequence_length disagree")
    total_residues = int(lengths.sum())
    features = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=(total_residues, int(args.feature_dim)),
    )
    pocket_mask, pocket_manifest = build_pocket_masks(
        index,
        None if args.no_pocket_mask else Path(args.target_structure),
    )
    pocket_output = np.lib.format.open_memmap(
        pocket_mask_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total_residues,),
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    model, alphabet = load_esm2(device)
    batch_converter = alphabet.get_batch_converter()
    representation_layer = int(args.representation_layer)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    offset = 0
    for target_row, sequence in enumerate(sequences):
        bounds = window_bounds(len(sequence), args.max_window_length, args.window_overlap)
        residue_sum = np.zeros((len(sequence), args.feature_dim), dtype=np.float32)
        residue_count = np.zeros(len(sequence), dtype=np.float32)
        for start in range(0, len(bounds), args.batch_size):
            batch_bounds = bounds[start : start + args.batch_size]
            batch = [
                (f"target_{target_row}_{left}", sequence[left:right])
                for left, right in batch_bounds
            ]
            _, _, tokens = batch_converter(batch)
            tokens = tokens.to(device, non_blocking=True)
            with torch.inference_mode():
                result = model(tokens, repr_layers=[representation_layer], return_contacts=False)
            representations = result["representations"][representation_layer].float().cpu().numpy()
            for local, (left, right) in enumerate(batch_bounds):
                token_count = right - left
                values = representations[local, 1 : token_count + 1]
                if values.shape != (token_count, args.feature_dim):
                    raise RuntimeError(
                        f"unexpected ESM2 token shape for target {target_row}: {values.shape}"
                    )
                residue_sum[left:right] += values
                residue_count[left:right] += 1.0
        if not np.all(residue_count > 0):
            failures.append(
                {
                    "target_feature_index": target_row,
                    "sequence_key": str(index.iloc[target_row]["sequence_key"]),
                    "reason": "one or more residues were not covered by any window",
                }
            )
            continue
        stitched = residue_sum / residue_count[:, None]
        if not np.isfinite(stitched).all():
            failures.append(
                {
                    "target_feature_index": target_row,
                    "sequence_key": str(index.iloc[target_row]["sequence_key"]),
                    "reason": "non-finite stitched representation",
                }
            )
            continue
        features[offset : offset + len(sequence)] = stitched.astype(np.float16)
        pocket_output[offset : offset + len(sequence)] = pocket_mask[offset : offset + len(sequence)]
        rows.append(
            {
                "target_feature_index": target_row,
                "sequence_key": str(index.iloc[target_row]["sequence_key"]),
                "token_offset": offset,
                "token_length": len(sequence),
                "sequence_length": len(sequence),
            }
        )
        offset += len(sequence)
        if args.progress_every > 0 and (target_row + 1) % args.progress_every == 0:
            print(f"processed {target_row + 1}/{len(sequences)} targets", flush=True)
    features.flush()
    pocket_output.flush()
    if failures:
        raise RuntimeError(json.dumps({"failures": failures[:10]}, ensure_ascii=False))
    if offset != total_residues or len(rows) != len(index):
        raise RuntimeError("token feature output is incomplete")
    token_index = pd.DataFrame(rows).sort_values("target_feature_index")
    token_index.to_csv(index_path, index=False, compression="gzip")
    manifest = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "route": "ESM2_T33_650M_RESIDUE_TOKEN_STITCHED_OVERLAPPING_WINDOWS",
        "label_dependency": "NONE",
        "target_count": len(index),
        "total_residues": total_residues,
        "feature_dim": int(args.feature_dim),
        "dtype": "float16",
        "max_window_length": int(args.max_window_length),
        "window_overlap": int(args.window_overlap),
        "representation_layer": int(args.representation_layer),
        "target_index": str(target_index_path),
        "target_index_sha256": sha256(target_index_path),
        "features": str(output_path),
        "features_sha256": sha256(output_path),
        "index": str(index_path),
        "index_sha256": sha256(index_path),
        "pocket_mask": str(pocket_mask_path),
        "pocket_mask_sha256": sha256(pocket_mask_path),
        "pocket_mask_manifest": pocket_manifest,
        "device": str(device),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-index", default=str(TARGET_INDEX))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--output-name", default="DTIAM_ESM2_T33_650M_RESIDUE_FLOAT16_V1.npy")
    parser.add_argument("--index-name", default="DTIAM_ESM2_T33_650M_RESIDUE_INDEX_V1.csv.gz")
    parser.add_argument("--manifest-name", default="DTIAM_ESM2_T33_650M_RESIDUE_MANIFEST_V1.json")
    parser.add_argument("--pocket-mask-name", default="DTIAM_ESM2_T33_650M_POCKET_MASK_UINT8_V1.npy")
    parser.add_argument("--target-structure", default=str(DEFAULT_TARGET_STRUCTURE))
    parser.add_argument("--no-pocket-mask", action="store_true")
    parser.add_argument("--feature-dim", type=int, default=1280)
    parser.add_argument("--representation-layer", type=int, default=33)
    parser.add_argument("--max-window-length", type=int, default=1022)
    parser.add_argument("--window-overlap", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
