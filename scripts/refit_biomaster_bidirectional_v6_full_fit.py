#!/usr/bin/env python3
"""Head-only FULL_FIT refit for the selected bidirectional V6 architecture."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from biomaster.comprehensive_balanced import QueryFirstBatchSampler, QuerySamplingConfig  # noqa: E402
from biomaster.odti_v2 import ODTIV2Config, RoutedInteractionRankerV2, directional_retrieval_loss  # noqa: E402
from train_biomaster_bidirectional_v6 import (  # noqa: E402
    _arrays_from_checkpoint,
    _model_forward,
    _tensor_values,
    json_safe,
)
from train_biomaster_comprehensive_balanced_v2 import DRUG_FEATURE_INDEX  # noqa: E402
from train_biomaster_comprehensive_full_fit_v1 import (  # noqa: E402
    ESM2,
    MORGAN,
    PACKAGE_MANIFEST,
    PROTBERT,
    RELATIONS,
    exact_keys,
    split_positions,
    target_structure_arrays,
)
from train_biomaster_odti_v2 import set_seed  # noqa: E402


STAGE_ROOT = ROOT / "outputs/biomaster_bidirectional_v6_stage_a_dense"
V5_ROOT = ROOT / "outputs/biomaster_comprehensive_balanced_full_fit_v2"
DEFAULT_OUT = ROOT / "outputs/biomaster_bidirectional_v6_full_fit"


def reference_checkpoint(seed: int) -> Path:
    return V5_ROOT / f"FULL_FIT_2026_COMPREHENSIVE_BALANCED__seed_{seed}" / (
        "FULL_FIT_MODEL_COMPREHENSIVE_BALANCED_V2.pt"
    )


def portable_path(path: Path) -> str:
    """Prefer repository-relative paths in portable checkpoints and summaries."""

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reference-checkpoint")
    parser.add_argument("--stage-summary")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--rows-per-query", type=int, default=16)
    parser.add_argument("--d2t-steps", type=int, default=48)
    parser.add_argument("--t2d-steps", type=int, default=32)
    parser.add_argument("--scheduler-horizon", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--bce-weight", type=float, default=0.25)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--listwise-weight", type=float, default=0.25)
    parser.add_argument("--residual-weight", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    reference = Path(args.reference_checkpoint or reference_checkpoint(args.seed)).resolve()
    stage_summary_path = Path(
        args.stage_summary
        or STAGE_ROOT / f"seed_{args.seed}" / "STAGE_A_SUMMARY_V6.json"
    ).resolve()
    required = [
        reference,
        stage_summary_path,
        RELATIONS,
        MORGAN,
        PROTBERT,
        ESM2,
        PACKAGE_MANIFEST,
        DRUG_FEATURE_INDEX,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if json.loads(PACKAGE_MANIFEST.read_text()).get("status") != "PASS":
        raise RuntimeError("comprehensive package manifest is not PASS")
    stage_summary = json.loads(stage_summary_path.read_text())
    if stage_summary.get("status") != "PASS":
        raise RuntimeError("the corresponding Stage-A seed did not pass")
    selected_epochs = int(stage_summary["selection"]["best_epoch"])
    if selected_epochs < 1:
        raise RuntimeError("Stage-A selected no trained epoch")

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(reference, map_location="cpu", weights_only=False)
    contract = checkpoint.get("full_fit_contract", {})
    if not contract.get("all_feature_resolved_relations_eligible", False):
        raise RuntimeError("reference is not the comprehensive FULL_FIT checkpoint")
    data = pd.read_csv(RELATIONS, low_memory=False)
    data["exact_pair_key"] = exact_keys(data)
    drug_index = pd.read_csv(
        DRUG_FEATURE_INDEX,
        usecols=["drug_feature_index", "murcko_scaffold"],
        low_memory=False,
    ).set_index("drug_feature_index")
    data["murcko_scaffold"] = (
        drug_index.loc[
            data["drug_feature_index"].to_numpy(dtype=np.int64), "murcko_scaffold"
        ]
        .reset_index(drop=True)
        .fillna("")
        .astype(str)
        .to_numpy()
    )
    positions, split_audit = split_positions(data)
    full_fit = positions["full_fit"]
    drug_features = np.load(MORGAN, mmap_mode="r")
    target_features = np.load(PROTBERT, mmap_mode="r")
    target_aux = np.load(ESM2, mmap_mode="r")
    structure_features, structure_mask, structure_columns, _, structure_audit = (
        target_structure_arrays(data, len(target_features))
    )
    arrays = _arrays_from_checkpoint(data, checkpoint)
    config = ODTIV2Config(
        **{
            **checkpoint["config"],
            "directional_heads_enabled": True,
            "directional_hidden_dim": 64,
            "directional_dropout": 0.0,
        }
    )
    model = RoutedInteractionRankerV2(
        len(checkpoint["families"]), config, use_conplex=False
    )
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if incompatible.unexpected_keys or any(
        not key.startswith(("drug_to_target_head.", "target_to_drug_head."))
        for key in incompatible.missing_keys
    ):
        raise RuntimeError(f"FULL_FIT checkpoint mismatch: {incompatible}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for head in (model.drug_to_target_head, model.target_to_drug_head):
        for parameter in head.parameters():
            parameter.requires_grad_(True)
    model.to(device)

    drug_cache = torch.from_numpy(np.asarray(drug_features, dtype=np.float32)).to(device)
    target_cache = torch.from_numpy(
        (np.asarray(target_features, dtype=np.float32) - arrays["target_mean"])
        / arrays["target_std"]
    ).to(device)
    target_aux_cache = torch.from_numpy(
        (np.asarray(target_aux, dtype=np.float32) - arrays["target_aux_mean"])
        / arrays["target_aux_std"]
    ).to(device)
    d2t_config = QuerySamplingConfig(
        batch_size=args.batch_size,
        steps_per_epoch=args.d2t_steps,
        rows_per_query=args.rows_per_query,
        query_frequency_power=0.0,
    )
    t2d_config = QuerySamplingConfig(
        batch_size=args.batch_size,
        steps_per_epoch=args.t2d_steps,
        rows_per_query=args.rows_per_query,
        query_frequency_power=0.25,
    )
    d2t_sampler = QueryFirstBatchSampler(
        full_fit, data, "drug_feature_index", d2t_config
    )
    t2d_sampler = QueryFirstBatchSampler(
        full_fit, data, "target_feature_index", t2d_config
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.scheduler_horizon,
        eta_min=args.learning_rate / 20,
    )
    history: list[dict[str, Any]] = []
    for epoch in range(1, selected_epochs + 1):
        model.eval()
        model.drug_to_target_head.train()
        model.target_to_drug_head.train()
        schedule = [
            ("drug_to_target", batch)
            for batch in d2t_sampler.batches(args.seed + 1000 * epoch)
        ] + [
            ("target_to_drug", batch)
            for batch in t2d_sampler.batches(args.seed + 1000 * epoch + 1)
        ]
        random.Random(args.seed + epoch).shuffle(schedule)
        components: dict[str, float] = {}
        for direction, batch in schedule:
            values = _tensor_values(
                batch,
                data,
                drug_features,
                target_features,
                target_aux,
                structure_features,
                structure_mask,
                arrays,
                config,
                device,
                drug_cache,
                target_cache,
                target_aux_cache,
            )
            optimizer.zero_grad(set_to_none=True)
            output = _model_forward(model, values)
            group = (
                values["drug_group"]
                if direction == "drug_to_target"
                else values["target_group"]
            )
            losses = directional_retrieval_loss(
                output,
                values["labels"],
                group,
                direction,
                values["binary_observed"],
                bce_weight=args.bce_weight,
                rank_weight=args.rank_weight,
                listwise_weight=args.listwise_weight,
                residual_weight=args.residual_weight,
                max_pairs=config.rank_max_pairs,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.gradient_clip,
            )
            optimizer.step()
            for name, value in losses.items():
                key = f"{direction}_{name}"
                components[key] = components.get(key, 0.0) + float(value.detach())
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            **{
                f"loss_{name}": value
                / (args.d2t_steps if name.startswith("drug_to_target") else args.t2d_steps)
                for name, value in components.items()
            },
        }
        history.append(row)
        print(json.dumps(row, default=json_safe), flush=True)

    run_dir = Path(args.out_dir).resolve() / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    checkpoint_path = run_dir / "FULL_FIT_BIDIRECTIONAL_V6.pt"
    full_fit_contract = {
        "protocol": "BIOMASTER_BIDIRECTIONAL_V6_HEAD_ONLY_FULL_FIT",
        "training_mode": "FROZEN_V5_FULL_FIT_BACKBONE_DIRECTIONAL_HEAD_REFIT",
        "selected_epochs_from_stage_a": selected_epochs,
        "all_feature_resolved_relations_eligible": True,
        "full_fit_rows": int(len(full_fit)),
        "backbone_frozen": True,
        "pair_logit_unchanged_from_reference": True,
        "external_current_case_labels_used_for_selection": False,
        "statistical_status": "EXPLORATORY_PASS_WITH_UNCERTAINTY",
    }
    torch.save(
        {
            "model_state_dict": state,
            "model_class": "RoutedInteractionRankerV2",
            "config": config.__dict__,
            "families": checkpoint["families"],
            "normalization": checkpoint["normalization"],
            "temperature": checkpoint.get("temperature", 1.0),
            "base_checkpoint": portable_path(reference),
            "full_fit_contract": full_fit_contract,
        },
        checkpoint_path,
    )
    pd.DataFrame(history).to_csv(run_dir / "FULL_FIT_TRAINING_HISTORY_V6.csv", index=False)
    summary = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "device": str(device),
        "reference_checkpoint": portable_path(reference),
        "stage_summary": portable_path(stage_summary_path),
        "full_fit_contract": full_fit_contract,
        "samplers": {
            "drug_to_target": {
                **d2t_config.to_dict(),
                "eligible_two_class_queries": d2t_sampler.query_count,
            },
            "target_to_drug": {
                **t2d_config.to_dict(),
                "eligible_two_class_queries": t2d_sampler.query_count,
            },
        },
        "split_audit": split_audit,
        "structure": {**structure_audit, "columns": structure_columns},
        "checkpoint": portable_path(checkpoint_path),
        "claim_boundary": (
            "This is a production-scoring candidate refit after architecture and epoch "
            "selection. It has no unbiased post-FULL_FIT temporal performance estimate. "
            "The three-seed Stage-A ensemble remains exploratory because its test bootstrap "
            "confidence interval crosses zero."
        ),
    }
    (run_dir / "FULL_FIT_SUMMARY_V6.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_safe) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_safe), flush=True)


if __name__ == "__main__":
    main()
