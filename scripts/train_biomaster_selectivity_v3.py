#!/usr/bin/env python3
"""Train and evaluate V3 on a fixed drug-entity holdout with train-only support.

Checkpoint selection reads validation drug-macro AP only. Test predictions are
written once after selecting the checkpoint. This is an activity-label model,
not a quantitative affinity model or a prospective temporal benchmark.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from biomaster.odti_v2 import ODTIV2Config
from biomaster.odti_support_v3 import SupportInteractionRankerV3, SupportV3Config, drug_macro_pairwise_loss
from biomaster.selectivity_training_v3 import (
    QueryBatchStream, observed_metrics, query_microbatches, support_summary_features,
)


PACKAGE = ROOT / "outputs/retrain_20260901/comprehensive_training_v1"
AUGMENT = ROOT / "outputs/biomaster_deployment_augmentation_v1"


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


class Runtime:
    """Shared feature preparation for training, selected-checkpoint testing and scoring."""

    def __init__(self, args, data, audit, store, support):
        self.args, self.data, self.audit, self.store, self.support = args, data, audit, store, support
        self.device = torch.device(args.device)
        self.drugs = data.drug_feature_index.to_numpy(np.int64)
        self.targets = data.target_feature_index.to_numpy(np.int64)
        self.families = data.family_index.to_numpy(np.int64)
        self.labels = data.binary_label.to_numpy(np.float32)
        self.features = np.load(args.features, mmap_mode="r")
        self.target_features = np.load(args.target_features, mmap_mode="r")
        self.target_aux = np.load(args.target_aux, mmap_mode="r")
        if self.targets.max() >= min(len(self.target_features), len(self.target_aux)):
            raise ValueError("target features do not cover indexed relations")
        if self.drugs.max() >= len(self.features):
            raise ValueError("drug features do not cover indexed relations")
        # A uint8 device bank avoids repeatedly transferring large support tensors.
        self.drug_bank = torch.tensor(np.asarray(self.features), device=self.device, dtype=torch.uint8)
        self.target_bank = torch.tensor(np.asarray(self.target_features), device=self.device, dtype=torch.float32)
        self.aux_bank = torch.tensor(np.asarray(self.target_aux), device=self.device, dtype=torch.float32)
        self.base_config = ODTIV2Config(
            drug_input_dim=self.features.shape[1], target_input_dim=self.target_features.shape[1],
            target_aux_input_dim=self.target_aux.shape[1], embedding_dim=args.width,
            hidden_dim=args.pair_hidden, interaction_mode="low_rank_film", interaction_rank=48,
            structure_input_dim=0, dropout=args.dropout, directional_heads_enabled=False,
            contrastive_weight=0, observation_weight=0, affinity_weight=0,
        )
        self.support_config = SupportV3Config(hidden_dim=args.width, support_enabled=args.support != "none", dropout=args.dropout)
        self.model = SupportInteractionRankerV3(self.base_config, len(audit["families"]), self.support_config).to(self.device)
        self.local_model = None
        self.local_store = None
        if args.local:
            from biomaster.odti_local_features_v3 import LocalFeatureStore
            from biomaster.odti_local_v3 import LocalInteractionBackboneV3, LocalInteractionConfigV3
            self.local_store = LocalFeatureStore(
                drug_index=args.drug_index, max_regions=args.local_regions,
                tokens_per_region=args.local_tokens, max_atoms=args.local_max_atoms,
                cache_size=args.local_cache_size,
                **({"pocket_store": args.local_pocket_store} if getattr(args, "local_pocket_store", None) is not None else {}),
            )
            self.local_model = LocalInteractionBackboneV3(LocalInteractionConfigV3(
                width=args.width, hidden_dim=args.width, dropout=args.dropout,
                residue_input_dim=self.local_store.residue_dim,
            )).to(self.device)
        self.scaffolds = np.full(len(self.features), "", dtype=object)
        if "murcko_scaffold" in data:
            for row in data.drop_duplicates("drug_feature_index").itertuples():
                self.scaffolds[row.drug_feature_index] = "" if pd.isna(row.murcko_scaffold) else str(row.murcko_scaffold)

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" and self.args.precision == "bf16" else nullcontext()

    def parameters(self):
        return list(self.model.parameters()) + ([] if self.local_model is None else list(self.local_model.parameters()))

    def mode(self, train):
        self.model.train(train)
        if self.local_model is not None:
            self.local_model.train(train)

    def inputs(self, drug_ids, target_ids, family_ids, support, support_rows=None, rng=None):
        device = self.device
        drugs = torch.tensor(np.asarray(drug_ids), dtype=torch.long, device=device)
        targets = torch.tensor(np.asarray(target_ids), dtype=torch.long, device=device)
        batch = {"drug": self.drug_bank[drugs].float(), "target": self.target_bank[targets],
                 "target_aux": self.aux_bank[targets],
                 "family": torch.tensor(np.asarray(family_ids), dtype=torch.long, device=device)}
        if self.args.support != "none":
            indices = np.asarray(support.indices if support_rows is None else support.indices[support_rows])
            mask = np.array(support.mask if support_rows is None else support.mask[support_rows], dtype=bool, copy=True)
            similarities = np.asarray(support.similarities if support_rows is None else support.similarities[support_rows])
            if self.args.support == "positive":
                mask[:, 0] = False
            if rng is not None:
                # Make whole-support dropout consistent within each drug query.
                unique, inverse = np.unique(drug_ids, return_inverse=True)
                keep_drug = rng.random(len(unique)) >= self.args.support_dropout
                mask &= keep_drug[inverse, None, None]
                mask &= rng.random(mask.shape) >= self.args.support_item_dropout
                drop_scaffold = rng.random(len(unique)) < self.args.scaffold_dropout
                current_scaffold = self.scaffolds[np.asarray(drug_ids)]
                same = self.scaffolds[np.maximum(indices, 0)] == current_scaffold[:, None, None]
                same &= (current_scaffold != "")[:, None, None]
                mask &= ~(same & drop_scaffold[inverse, None, None])
            safe_ids = torch.as_tensor(np.maximum(indices, 0).astype(np.int64), device=device)
            batch["support_drug_features"] = self.drug_bank[safe_ids].float()
            batch["support_similarity"] = torch.as_tensor(np.array(similarities), dtype=torch.float32, device=device)
            batch["support_mask"] = torch.as_tensor(mask, device=device)
        if self.local_model is not None:
            local = self.local_model(**self.local_store.batch(drug_ids, target_ids, device=device))
            batch["local_hidden"] = local["local_hidden"]
            batch["local_available"] = local["local_available"]
            if rng is not None and self.args.local_dropout > 0:
                unique, inverse = np.unique(drug_ids, return_inverse=True)
                keep = rng.random(len(unique)) >= self.args.local_dropout
                batch["local_available"] = batch["local_available"] & torch.as_tensor(keep[inverse], device=device)
        return batch

    def forward_rows(self, rows, rng=None):
        return self.model(**self.inputs(self.drugs[rows], self.targets[rows], self.families[rows], self.support, rows, rng))

    @torch.no_grad()
    def predict(self, positions):
        self.mode(False)
        scores, gates = [], []
        for start in range(0, len(positions), self.args.eval_batch_size):
            rows = positions[start:start + self.args.eval_batch_size]
            with self.autocast():
                output = self.forward_rows(rows)
            scores.append(output["final_logit"].float().cpu().numpy())
            gates.append(output["support_gate"].float().cpu().numpy())
        return np.concatenate(scores), np.concatenate(gates)

    def _feature_identity(self):
        """Bind resumed training to the actual feature files, including targets."""
        if not hasattr(self, "_checkpoint_feature_files"):
            identity = {}
            names = ["features", "target_features", "target_aux"]
            if self.args.local:
                names.append("drug_index")
                if getattr(self.args, "local_pocket_store", None) is not None:
                    names.append("local_pocket_store")
            paths = {name: Path(getattr(self.args, name)) for name in names}
            if self.args.local:
                from biomaster.odti_local_features_v3 import (
                    DEFAULT_RESIDUE_ARRAY, DEFAULT_RESIDUE_INDEX, DEFAULT_TARGET_INDICES,
                )
                paths.update({"local_residue_array": DEFAULT_RESIDUE_ARRAY,
                              "local_residue_index": DEFAULT_RESIDUE_INDEX})
                paths.update({f"local_target_index_{index}": path
                              for index, path in enumerate(DEFAULT_TARGET_INDICES)})
            for name, source_path in paths.items():
                path = Path(source_path).resolve()
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                identity[name] = {"path": str(path), "sha256": digest.hexdigest(),
                                  "size_bytes": path.stat().st_size}
            self._checkpoint_feature_files = identity
        return self._checkpoint_feature_files

    def checkpoint(self, epoch, optimizer, best_value, *, best_epoch=None, history=None,
                   best_snapshot=None):
        """A self-contained resume bundle; tensor snapshots never alias training."""
        def cpu_copy(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {key: cpu_copy(item) for key, item in value.items()}
            if isinstance(value, list):
                return [cpu_copy(item) for item in value]
            if isinstance(value, tuple):
                return tuple(cpu_copy(item) for item in value)
            return value

        model_state = cpu_copy(self.model.state_dict())
        local_state = None if self.local_model is None else cpu_copy(self.local_model.state_dict())
        if best_snapshot is None:
            best_snapshot = {"model_state": model_state, "local_state": local_state}
        return {"format": "BIOMASTER_SELECTIVITY_V3_2", "epoch": epoch,
                "base_config": asdict(self.base_config), "support_config": asdict(self.support_config),
                "family_count": len(self.audit["families"]),
                "model_state": model_state,
                "local_config": None if self.local_model is None else asdict(self.local_model.config),
                "local_state": local_state,
                "optimizer_state": cpu_copy(optimizer.state_dict()),
                "best_validation_drug_macro_ap": best_value,
                "best_epoch": epoch if best_epoch is None else int(best_epoch),
                "history": cpu_copy([] if history is None else history),
                "best_snapshot": cpu_copy(best_snapshot),
                "arguments": cpu_copy(vars(self.args)), "data_audit": cpu_copy(self.audit),
                "feature_identity": cpu_copy(self._feature_identity()),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None}

    def restore(self, checkpoint):
        if checkpoint["data_audit"]["assignment_sha256"] != self.audit["assignment_sha256"]:
            raise ValueError("checkpoint data assignment does not match")
        if checkpoint.get("feature_identity") is not None:
            if checkpoint["feature_identity"] != self._feature_identity():
                raise ValueError("checkpoint feature files differ in path or content")
        if checkpoint["base_config"] != asdict(self.base_config):
            raise ValueError("checkpoint base configuration does not match")
        if checkpoint["support_config"] != asdict(self.support_config):
            raise ValueError("checkpoint support configuration does not match")
        local_config = None if self.local_model is None else asdict(self.local_model.config)
        if checkpoint.get("local_config") != local_config:
            raise ValueError("checkpoint local configuration does not match")
        self.model.load_state_dict(checkpoint["model_state"])
        if self.local_model is not None:
            self.local_model.load_state_dict(checkpoint["local_state"])


def train_batch(runtime, positions, optimizer, query, rng, ranking_active):
    runtime.mode(True)
    optimizer.zero_grad(set_to_none=True)
    if query:
        chunks = query_microbatches(positions, runtime.drugs, runtime.args.micro_batch_size)
        total_queries = len(np.unique(runtime.drugs[positions]))
    else:
        chunks = (positions[i:i + runtime.args.micro_batch_size] for i in range(0, len(positions), runtime.args.micro_batch_size))
        total_queries = 0
    bce_value, rank_value = 0., 0.
    for rows in chunks:
        with runtime.autocast():
            output = runtime.forward_rows(rows, rng)
            logits = output["final_logit"].float()
            labels = torch.as_tensor(runtime.labels[rows], device=runtime.device)
            bce = F.binary_cross_entropy_with_logits(logits, labels)
            row_fraction = len(rows) / len(positions)
            loss = runtime.args.observed_weight * bce * row_fraction
            rank = logits.sum() * 0.
            query_fraction = 0.
            if query and ranking_active:
                groups = torch.as_tensor(runtime.drugs[rows], device=runtime.device)
                rank = drug_macro_pairwise_loss(logits, labels, groups, torch.ones_like(labels, dtype=torch.bool), mode=runtime.args.rank_loss)
                query_fraction = len(np.unique(runtime.drugs[rows])) / total_queries
                loss = loss + runtime.args.rank_weight * rank * query_fraction
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        loss.backward()
        bce_value += float(bce.detach()) * row_fraction
        rank_value += float(rank.detach()) * query_fraction
    norm = torch.nn.utils.clip_grad_norm_(runtime.parameters(), runtime.args.grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return bce_value, rank_value, float(norm)


def baseline_predictions(runtime, train, validation, test, out):
    """Same evidence candidates, no held-out labels used to fit baseline."""
    features = support_summary_features(runtime.support.similarities, runtime.support.mask)
    estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=1.0, random_state=0))
    estimator.fit(features[train], runtime.labels[train])
    results, predictions = {}, {}
    for name, positions in [("validation", validation), ("test", test)]:
        p = {"positive_max_tanimoto": features[positions, 1],
             "positive_negative_logistic": estimator.decision_function(features[positions])}
        predictions[name] = p
        results[name] = {key: observed_metrics(runtime.labels[positions], value, runtime.drugs[positions], runtime.targets[positions]) for key, value in p.items()}
    # Baselines do not influence neural epoch selection. Keep coefficients for reproduction.
    write_json(out / "BASELINES_V3.json", {"metrics": results,
               "feature_order": ["max_negative", "max_positive", "mean_negative", "mean_positive", "has_negative", "has_positive", "max_delta", "mean_delta"],
               "scaler_mean": estimator[0].mean_.tolist(), "scaler_scale": estimator[0].scale_.tolist(),
               "coefficient": estimator[1].coef_.tolist(), "intercept": estimator[1].intercept_.tolist()})
    return results, predictions


@torch.no_grad()
def dense_test(runtime, positions, out):
    from biomaster.odti_v3_panel_metrics import positive_retrieval_metrics
    frame = runtime.data.iloc[positions]
    eligible = frame.loc[frame.binary_label.eq(1), "drug_feature_index"].unique()
    # Fixed entity-index ordering, independent of model predictions and test metrics.
    eligible = np.asarray(sorted(eligible, key=lambda d: hashlib.sha256(f"V3_DENSE:{d}".encode()).hexdigest())[:runtime.args.dense_test_drugs])
    targets = np.asarray(sorted(runtime.data.target_feature_index.unique()), dtype=np.int64)
    mapping = runtime.data.drop_duplicates("target_feature_index").set_index("target_feature_index").family_index.to_dict()
    drug_ids = np.repeat(eligible, len(targets))
    target_ids = np.tile(targets, len(eligible))
    evidence = runtime.store.cache_retrieve(Path(runtime.args.cache_dir) / f"dense_test_{len(eligible)}", drug_ids, target_ids, k=runtime.args.support_k)
    scores = []
    runtime.mode(False)
    for start in range(0, len(drug_ids), runtime.args.eval_batch_size):
        ix = np.arange(start, min(start + runtime.args.eval_batch_size, len(drug_ids)))
        with runtime.autocast():
            output = runtime.model(**runtime.inputs(drug_ids[ix], target_ids[ix], np.array([mapping[int(t)] for t in target_ids[ix]]), evidence, ix))
        scores.append(output["final_logit"].float().cpu().numpy())
    matrix = np.concatenate(scores).reshape(len(eligible), len(targets))
    positive_pairs = set(zip(frame.loc[frame.binary_label.eq(1), "drug_feature_index"], frame.loc[frame.binary_label.eq(1), "target_feature_index"]))
    positive = np.array([(int(d), int(t)) in positive_pairs for d, t in zip(drug_ids, target_ids)]).reshape(matrix.shape)
    metrics = positive_retrieval_metrics(positive, matrix, targets)
    np.savez_compressed(out / "DENSE_TEST_PREDICTIONS_V3.npz", scores=matrix, known_positive=positive, drug_indices=eligible, target_indices=targets)
    write_json(out / "DENSE_TEST_METRICS_V3.json", metrics)
    return metrics


def run(args):
    from prepare_biomaster_selectivity_v3 import prepare_training_data

    def save_atomic(path, value):
        temporary = path.with_name(path.name + ".tmp")
        torch.save(value, temporary)
        temporary.replace(path)

    def selected_checkpoint(bundle):
        # BEST is a scoring artifact. LAST alone carries optimizer/RNG/history
        # and the best snapshot needed to recreate BEST after any interruption.
        omit = {"model_state", "local_state", "optimizer_state", "torch_rng", "cuda_rng",
                "history", "best_snapshot"}
        selected = {key: value for key, value in bundle.items() if key not in omit}
        selected.update(bundle["best_snapshot"])
        selected["epoch"] = bundle["best_epoch"]
        selected["artifact_role"] = "validation_selected_scoring_model"
        selected["source_resume_bundle_epoch"] = bundle["epoch"]
        return selected

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "RUN_SUMMARY_V3.json").exists():
        raise FileExistsError("completed run already exists; choose a new output directory")
    if (out / "LAST_MODEL_V3.pt").exists() and not args.resume:
        raise FileExistsError("checkpoint exists; pass --resume or choose a new output directory")
    write_json(out / "STATUS_V3.json", {"status": "preparing", "pid": os.getpid(), "started_utc": now(), "arguments": vars(args)})
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    data, audit, store, support = prepare_training_data(args.relations, args.features, args.cache_dir, max_entities=args.max_entities, k=args.support_k)
    runtime = Runtime(args, data, audit, store, support)
    positions = {name: np.flatnonzero(data.split.eq(name)) for name in ["train", "validation", "test"]}
    train, validation, test = (positions[k] for k in ["train", "validation", "test"])
    stream = QueryBatchStream(runtime.drugs, runtime.labels, train, args.query_drugs, args.query_items)
    if not len(stream.keys):
        raise ValueError("no two-class training query available")
    optimizer = torch.optim.AdamW(runtime.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_value, best_epoch, start_epoch = -float("inf"), 0, 0
    history = []
    best_snapshot = None
    if args.resume:
        checkpoint = torch.load(out / "LAST_MODEL_V3.pt", map_location="cpu", weights_only=False)
        if checkpoint.get("format") != "BIOMASTER_SELECTIVITY_V3_2":
            raise ValueError("resume requires a self-contained V3_2 LAST bundle")
        operational = {"resume", "out", "device", "log_every", "threads", "eval_batch_size"}
        previous_arguments, requested_arguments = checkpoint["arguments"], vars(args)
        for key in sorted(set(previous_arguments) | set(requested_arguments)):
            if key in operational:
                continue
            if key not in previous_arguments or key not in requested_arguments or previous_arguments[key] != requested_arguments[key]:
                raise ValueError(f"resume argument changed: {key}")
        runtime.restore(checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if runtime.device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint["cuda_rng"]])
        start_epoch = checkpoint["epoch"]
        best_value = checkpoint["best_validation_drug_macro_ap"]
        best_epoch = checkpoint["best_epoch"]
        history = checkpoint["history"]
        best_snapshot = checkpoint["best_snapshot"]
        if not history or history[-1]["epoch"] != start_epoch or not 1 <= best_epoch <= start_epoch:
            raise ValueError("resume bundle epoch, best epoch and history are inconsistent")
        if best_snapshot is None:
            raise ValueError("resume bundle is missing its selected model snapshot")
        # Repair both derived artifacts from LAST, ignoring stale/corrupt files.
        save_atomic(out / "BEST_MODEL_V3.pt", selected_checkpoint(checkpoint))
        write_json(out / "TRAINING_HISTORY_V3.json", history)
        changed_runtime = {key: {"stored": previous_arguments.get(key), "requested": requested_arguments.get(key)}
                           for key in operational - {"resume", "out"}
                           if previous_arguments.get(key) != requested_arguments.get(key)}
        if changed_runtime:
            print(json.dumps(json_safe({"event": "resume_runtime_overrides", "changes": changed_runtime,
                                      "note": "Device, thread or evaluation-batch changes may cause numerical differences."})), flush=True)
    write_json(out / "DATA_AUDIT_V3.json", audit)
    print(json.dumps({"event": "training_start", "utc": now(), "train_rows": len(train),
                      "query_drugs": len(stream.keys), "parameters": sum(p.numel() for p in runtime.parameters()),
                      "support": args.support, "local": args.local, "seed": args.seed}), flush=True)
    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)
    overall_started = time.monotonic()
    for epoch in range(start_epoch, args.epochs):
        started = time.monotonic()
        coverage_rng = np.random.default_rng(args.seed + epoch * 104729)
        evidence_rng = np.random.default_rng(args.seed + epoch * 104729 + 11)
        coverage = coverage_rng.permutation(train)
        queries = stream.batches(args.seed + epoch * 104729 + 7)
        batch_count = math.ceil(len(coverage) / args.batch_size)
        if args.max_steps > 0:
            batch_count = min(batch_count, args.max_steps)
        lr = args.lr * (0.1 + 0.9 * (1 + math.cos(math.pi * epoch / max(args.epochs, 1))) / 2)
        for group in optimizer.param_groups:
            group["lr"] = lr
        values, exposure = [], hashlib.sha256()
        touched = []
        for step in range(batch_count):
            rows = coverage[step * args.batch_size:(step + 1) * args.batch_size]
            query_rows = next(queries)
            for part in [rows, query_rows]:
                exposure.update(part.astype("<i8").tobytes())
            touched.append(rows)
            cov = train_batch(runtime, rows, optimizer, False, evidence_rng, False)
            query = train_batch(runtime, query_rows, optimizer, True, evidence_rng,
                                epoch >= args.warmup_epochs and args.rank_weight > 0)
            values.append([cov[0], query[0], query[1], max(cov[2], query[2])])
            if (step + 1) % args.log_every == 0 or step + 1 == batch_count:
                status = {"status": "training", "pid": os.getpid(), "utc": now(), "epoch": epoch + 1,
                          "step": step + 1, "steps": batch_count, "last_losses": values[-1],
                          "elapsed_epoch_seconds": time.monotonic() - started}
                write_json(out / "STATUS_V3.json", status)
                print(json.dumps(status), flush=True)
        score, _ = runtime.predict(validation)
        metrics = observed_metrics(runtime.labels[validation], score, runtime.drugs[validation], runtime.targets[validation])
        selection = metrics["drug_macro_ap"]
        if selection is None:
            raise ValueError("validation has no two-class drug queries for checkpoint selection")
        improved = selection > best_value
        if improved:
            best_value, best_epoch = selection, epoch + 1
        row = {"epoch": epoch + 1, "seconds": time.monotonic() - started, "lr": lr,
               "mean_losses": dict(zip(["coverage_bce", "query_bce", "query_rank", "grad_norm"], np.mean(values, axis=0).tolist())),
               "coverage_unique_rows": len(np.unique(np.concatenate(touched))),
               "exposure_sha256": exposure.hexdigest(), "validation": metrics, "selected": improved}
        history.append(row)
        checkpoint = runtime.checkpoint(epoch + 1, optimizer, best_value,
                                        best_epoch=best_epoch, history=history,
                                        best_snapshot=None if improved else best_snapshot)
        best_snapshot = checkpoint["best_snapshot"]
        save_atomic(out / "BEST_MODEL_V3.pt", selected_checkpoint(checkpoint))
        save_atomic(out / "LAST_MODEL_V3.pt", checkpoint)
        write_json(out / "TRAINING_HISTORY_V3.json", history)
        print(json.dumps(json_safe({"event": "epoch_complete", **row})), flush=True)
        if args.patience > 0 and epoch + 1 - best_epoch >= args.patience:
            break
    runtime.restore(torch.load(out / "BEST_MODEL_V3.pt", map_location=runtime.device, weights_only=False))
    val_scores, val_gates = runtime.predict(validation)
    test_scores, test_gates = runtime.predict(test)
    baselines, baseline_scores = baseline_predictions(runtime, train, validation, test, out)
    test_metrics = observed_metrics(runtime.labels[test], test_scores, runtime.drugs[test], runtime.targets[test])
    for name, rows, scores, gates in [("validation", validation, val_scores, val_gates), ("test", test, test_scores, test_gates)]:
        predicted = data.iloc[rows].copy()
        predicted["v3_logit"], predicted["support_gate"] = scores, gates
        for key, values in baseline_scores[name].items():
            predicted[key] = values
        predicted.to_csv(out / f"{name.upper()}_PREDICTIONS_V3.csv.gz", index=False)
    dense = dense_test(runtime, test, out) if args.dense_test_drugs > 0 else None
    summary = {"status": "completed", "finished_utc": now(), "arguments": vars(args),
               "protocol": audit["protocol"], "data_assignment_sha256": audit["assignment_sha256"],
               "selected_epoch": best_epoch, "selection_metric": "validation measured drug_macro_ap",
               "best_validation_drug_macro_ap": best_value, "test": test_metrics, "dense_test": dense,
               "baselines": baselines, "parameters": sum(p.numel() for p in runtime.parameters()),
               "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(runtime.device) / 2**30 if runtime.device.type == "cuda" else None,
               "training_and_evaluation_seconds": time.monotonic() - overall_started,
               "training_coverage_complete_per_epoch": args.max_steps == 0,
               "smoke_subset": args.max_entities > 0 or args.max_steps > 0,
               "local_representation": (("predicted pockets plus a full-sequence coverage region" if getattr(args, "local_pocket_store", None) is not None else "full-sequence coverage-preserving regions, not binding pockets") if args.local else None),
               "production_promoted": False}
    write_json(out / "RUN_SUMMARY_V3.json", summary)
    write_json(out / "STATUS_V3.json", {"status": "completed", "pid": os.getpid(), "utc": now(), "selected_epoch": best_epoch, "test_drug_macro_ap": test_metrics["drug_macro_ap"]})
    print(json.dumps(json_safe({"event": "completed", "selected_epoch": best_epoch, "test": test_metrics})), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--relations", type=Path, default=PACKAGE / "COMPREHENSIVE_TRAINING_RELATIONS_V1.csv.gz")
    p.add_argument("--features", type=Path, default=PACKAGE / "MORGAN2048_UINT8_COMPREHENSIVE_V1.npy")
    p.add_argument("--drug-index", type=Path, default=PACKAGE / "DRUG_FEATURE_INDEX_COMPREHENSIVE_V1.csv.gz")
    p.add_argument("--target-features", type=Path, default=AUGMENT / "PROTBERT1024_FLOAT32_DEPLOYMENT_AUGMENTED_V1.npy")
    p.add_argument("--target-aux", type=Path, default=AUGMENT / "ESM2_650M_1280_FLOAT32_DEPLOYMENT_AUGMENTED_V1.npy")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "outputs/biomaster_v3_20260905/data")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--micro-batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--query-drugs", type=int, default=64)
    p.add_argument("--query-items", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--dropout", type=float, default=0.12)
    p.add_argument("--observed-weight", type=float, default=0.25)
    p.add_argument("--rank-weight", type=float, default=1.0)
    p.add_argument("--rank-loss", choices=["pairwise", "hard"], default="pairwise")
    p.add_argument("--support", choices=["none", "both", "positive"], default="both")
    p.add_argument("--support-k", type=int, default=16)
    p.add_argument("--support-dropout", type=float, default=0.2)
    p.add_argument("--support-item-dropout", type=float, default=0.1)
    p.add_argument("--scaffold-dropout", type=float, default=0.2)
    p.add_argument("--width", type=int, default=192)
    p.add_argument("--pair-hidden", type=int, default=256)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--local", action="store_true")
    p.add_argument("--local-pocket-store", type=Path, help="frozen validated predicted-pocket JSON; omitted uses full-sequence regions")
    p.add_argument("--local-dropout", type=float, default=0.2, help="train global fallback on query-consistent missing-local episodes")
    p.add_argument("--local-regions", type=int, default=4)
    p.add_argument("--local-tokens", type=int, default=32)
    p.add_argument("--local-max-atoms", type=int, default=128)
    p.add_argument("--local-cache-size", type=int, default=8192)
    p.add_argument("--max-entities", type=int, default=0, help="label-independent smoke subset; 0 uses all entities")
    p.add_argument("--max-steps", type=int, default=0, help="smoke-only cap; 0 covers all training rows")
    p.add_argument("--dense-test-drugs", type=int, default=0)
    p.add_argument("--patience", type=int, default=0, help="0 keeps equal fixed budgets across ablations")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--resume", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if min(args.epochs, args.batch_size, args.micro_batch_size, args.eval_batch_size, args.log_every) < 1:
        raise ValueError("epochs and batch/log sizes must be positive")
    try:
        run(args)
    except Exception as exc:
        write_json(Path(args.out) / "STATUS_V3.json", {"status": "failed", "pid": os.getpid(), "utc": now(), "error": repr(exc)})
        raise
