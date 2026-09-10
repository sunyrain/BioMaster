#!/usr/bin/env python3
"""Score indexed pairs using an existing BioMaster V3 checkpoint and support fold.

Only canonical drug indices represented in the checkpoint's prepared data are
accepted. Query labels are never model inputs. This predicts activity-label
logits; it does not estimate affinity or calibrated probability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts.prepare_biomaster_selectivity_v3 import prepare_training_data
from scripts.train_biomaster_selectivity_v3 import Runtime, now, write_json

CHECKPOINT_FORMAT = "BIOMASTER_SELECTIVITY_V3_2"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True, help="Trusted locally produced V3 .pt checkpoint inside this project")
    p.add_argument("--pairs", type=Path, required=True, help="CSV/.csv.gz with canonical drug_feature_index,target_feature_index")
    p.add_argument("--out", type=Path, required=True, help="Output .csv or .csv.gz; metadata is written alongside")
    p.add_argument("--device", default=None, help="Default: cuda when available, otherwise cpu")
    p.add_argument("--batch-size", type=positive_integer, default=256)
    return p


def load_project_checkpoint(path: str | Path) -> dict:
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT.resolve()) or path.suffix != ".pt":
        raise ValueError("Only trusted local .pt checkpoints produced inside this BioMaster project are supported")
    if not path.is_file():
        raise FileNotFoundError(path)
    # Runtime.checkpoint contains local argparse.Path/RNG objects in addition to
    # tensors. This entry point is restricted to trusted project checkpoints.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported BioMaster V3 checkpoint format")
    required = {"arguments", "data_audit", "model_state", "base_config", "support_config",
                "local_state", "local_config", "family_count", "feature_identity"}
    if not required.issubset(checkpoint):
        raise ValueError(f"Checkpoint lacks required fields: {sorted(required - set(checkpoint))}")
    return checkpoint


def restored_arguments(checkpoint: dict, *, device: str | None, batch_size: int) -> argparse.Namespace:
    if not isinstance(checkpoint.get("arguments"), dict):
        raise ValueError("Checkpoint arguments must be a dictionary")
    arguments = dict(checkpoint["arguments"])
    required = {"relations", "features", "target_features", "target_aux", "cache_dir", "max_entities",
                "support_k", "width", "pair_hidden", "dropout", "support", "local", "precision"}
    if not required.issubset(arguments):
        raise ValueError(f"Checkpoint arguments lack required fields: {sorted(required - set(arguments))}")
    if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    arguments["device"] = device or ("cuda" if torch.cuda.is_available() else "cpu")
    arguments["eval_batch_size"] = int(batch_size)
    return argparse.Namespace(**arguments)


def verify_existing_preparation(args: argparse.Namespace) -> dict:
    """Require complete matching artifacts before calling the reusable preparer.

    The preparer can construct missing data when used by training; scoring must
    never silently create a training split or repair a missing support cache.
    """
    directory = Path(args.cache_dir).resolve()
    manifest_path = directory / "DATA_MANIFEST_V3.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Scoring requires an existing DATA_MANIFEST_V3.json: {manifest_path}; prepare the exact training fold first")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "COMPLETE":
        raise ValueError("Scoring requires a COMPLETE prepared-data manifest")
    identity = manifest.get("identity", {})
    expected = {"relations_path": str(Path(args.relations).resolve()), "features_path": str(Path(args.features).resolve()),
                "max_entities": args.max_entities, "support_k": args.support_k}
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint arguments do not match the existing prepared-data manifest")
    required_files = [directory / "RELATIONS_V3.csv.gz", directory / "support_store/SUPPORT_STORE_MANIFEST_V3.json",
                      directory / "support_store/support_index.npz"]
    for key in ["support_indices_path", "support_similarities_path", "support_mask_path"]:
        path = Path(manifest.get(key, ""))
        if not path.is_absolute() or not path.resolve().is_relative_to(directory / "support_normal"):
            raise ValueError(f"Prepared manifest {key} must reference this fold's existing support_normal cache")
        required_files.append(path)
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Scoring refuses to rebuild missing training artifacts: {missing}")
    indices_path = Path(manifest["support_indices_path"])
    cache_manifest_path = indices_path.parent / "manifest.json"
    if not cache_manifest_path.is_file():
        raise FileNotFoundError("Scoring requires the completed training support-cache manifest")
    cache_manifest = json.loads(cache_manifest_path.read_text())
    if cache_manifest.get("identity", {}).get("store_hash") != manifest["store_hash"]:
        raise ValueError("Training support-cache identity differs from prepared data")
    for name in ["indices", "similarities", "mask"]:
        path = Path(manifest[f"support_{name}_path"])
        if path.parent != indices_path.parent or file_sha256(path) != cache_manifest.get("array_sha256", {}).get(name):
            raise ValueError(f"Training support cache {name} is damaged; scoring will not rebuild it")
    return manifest


def _validated_indices(series: pd.Series, name: str) -> np.ndarray:
    try:
        values = pd.to_numeric(series, errors="raise").to_numpy()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain nonnegative integer feature indices") from exc
    if values.dtype.kind == "b" or not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{name} must contain nonnegative integer feature indices")
    if len(values) and (values > np.iinfo(np.int64).max).any():
        raise ValueError(f"{name} exceeds int64 index range")
    return values.astype(np.int64)


def validate_pairs(pairs: pd.DataFrame, prepared: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"drug_feature_index", "target_feature_index"}
    if not required.issubset(pairs):
        raise ValueError(f"Input pairs lack required columns: {sorted(required - set(pairs))}")
    if {"v3_logit", "support_gate"} & set(pairs):
        raise ValueError("Input contains reserved output columns v3_logit/support_gate; rename them before scoring")
    drugs = _validated_indices(pairs.drug_feature_index, "drug_feature_index")
    targets = _validated_indices(pairs.target_feature_index, "target_feature_index")
    canonical = np.asarray(prepared.drug_feature_index.unique(), dtype=np.int64)
    unknown_drugs = np.setdiff1d(drugs, canonical)
    if len(unknown_drugs):
        raise ValueError(
            f"Noncanonical or unrepresented drug_feature_index values: {unknown_drugs[:12].tolist()}. "
            "Only the prepared fold's canonical drug indices are accepted, so aliases cannot bypass entity self-exclusion. "
            "Map source relation rows through RELATIONS_V3.csv.gz (source_row -> drug_feature_index); "
            "unrepresented molecules need an explicitly validated identity mapping."
        )
    if prepared.groupby("target_feature_index").family_index.nunique().gt(1).any():
        raise ValueError("Prepared target-to-family mapping is ambiguous")
    mapping = prepared.drop_duplicates("target_feature_index").set_index("target_feature_index").family_index.to_dict()
    unknown_targets = np.setdiff1d(targets, np.asarray(list(mapping), dtype=np.int64))
    if len(unknown_targets):
        raise ValueError(f"Unknown target_feature_index values {unknown_targets[:12].tolist()}; no fixed trained family mapping exists")
    families = np.asarray([mapping[int(target)] for target in targets], dtype=np.int64)
    return drugs, targets, families


@torch.no_grad()
def score_pairs(runtime: Runtime, pairs: pd.DataFrame, cache_dir: str | Path, batch_size: int) -> pd.DataFrame:
    if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    drugs, targets, families = validate_pairs(pairs, runtime.data)
    evidence = runtime.store.cache_retrieve(cache_dir, drugs, targets, k=runtime.args.support_k)
    runtime.mode(False)
    scores, gates = np.empty(len(pairs), np.float32), np.empty(len(pairs), np.float32)
    for start in range(0, len(pairs), batch_size):
        end = min(start + batch_size, len(pairs))
        rows = np.arange(start, end, dtype=np.int64)
        with runtime.autocast():
            # Only the two validated feature indices and the training family map
            # determine inputs. Every extra input column, including labels, is inert.
            output = runtime.model(**runtime.inputs(drugs[rows], targets[rows], families[rows], evidence, rows, rng=None))
        scores[start:end] = output["final_logit"].float().cpu().numpy()
        gates[start:end] = output["support_gate"].float().cpu().numpy()
    if not np.isfinite(scores).all() or not np.isfinite(gates).all():
        raise FloatingPointError("Checkpoint produced nonfinite scores or support gates")
    result = pairs.copy()
    result["v3_logit"], result["support_gate"] = scores, gates
    return result


def run(args: argparse.Namespace) -> dict:
    checkpoint_path, pairs_path, output_path = Path(args.checkpoint).resolve(), Path(args.pairs).resolve(), Path(args.out).resolve()
    if not (output_path.name.endswith(".csv") or output_path.name.endswith(".csv.gz")):
        raise ValueError("--out must be a .csv or .csv.gz file")
    if output_path in {checkpoint_path, pairs_path}:
        raise ValueError("Output must differ from the checkpoint and input pairs")
    checkpoint = load_project_checkpoint(checkpoint_path)
    training_args = restored_arguments(checkpoint, device=args.device, batch_size=args.batch_size)
    manifest = verify_existing_preparation(training_args)
    if checkpoint["data_audit"].get("assignment_sha256") != manifest.get("assignment_sha256"):
        raise ValueError("Checkpoint data assignment differs from the prepared manifest")
    data, audit, store, support = prepare_training_data(
        training_args.relations, training_args.features, training_args.cache_dir,
        max_entities=training_args.max_entities, k=training_args.support_k)
    pairs = pd.read_csv(pairs_path, low_memory=False)
    validate_pairs(pairs, data)  # Fail before allocating GPU feature banks.
    torch.set_num_threads(int(getattr(training_args, "threads", 4)))
    runtime = Runtime(training_args, data, audit, store, support)
    if len(audit["families"]) != checkpoint["family_count"]:
        raise ValueError("Checkpoint family vocabulary differs from prepared data")
    runtime.restore(checkpoint)
    query_cache = output_path.parent / (output_path.name + ".support_cache")
    scored = score_pairs(runtime, pairs, query_cache, args.batch_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".pending")
    scored.to_csv(temporary, index=False, compression="gzip" if output_path.name.endswith(".gz") else None)
    temporary.replace(output_path)
    report = {
        "status": "completed", "finished_utc": now(), "rows": len(scored),
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"), "checkpoint_format": checkpoint["format"],
        "feature_identity": checkpoint["feature_identity"],
        "pairs_path": str(pairs_path), "pairs_sha256": file_sha256(pairs_path),
        "output_path": str(output_path), "output_sha256": file_sha256(output_path),
        "data_assignment_sha256": audit["assignment_sha256"], "support_store_hash": store.store_hash,
        "query_cache_path": str(query_cache), "device": training_args.device, "batch_size": args.batch_size,
        "support_mode": training_args.support, "support_k": training_args.support_k,
        "score_scope": "activity-label logits for prepared canonical drug indices and targets with the fixed trained family mapping",
        "score_is_calibrated_probability": False, "score_is_quantitative_affinity": False,
        "input_labels_used": False, "input_row_order_preserved": True,
        "support_exclusion": "all query entity records, independent of query batch companions",
        "local_representation": (("predicted pockets plus a full-sequence coverage region" if getattr(training_args, "local_pocket_store", None) is not None else "full-sequence coverage-preserving regions, not binding pockets") if training_args.local else None),
        "production_promoted": False,
    }
    write_json(Path(str(output_path) + ".manifest.json"), report)
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    run(parser().parse_args())
