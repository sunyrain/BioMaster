"""CPU integration of trainer runtime, logical batches and serialized state."""

from copy import deepcopy
import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.train_biomaster_selectivity_v3 import Runtime, parser, train_batch
import scripts.train_biomaster_selectivity_v3 as trainer


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


@pytest.fixture
def prepared(tmp_path):
    rng = np.random.default_rng(53)
    for name, array in {
        "drugs": rng.integers(0, 2, (5, 16), dtype=np.uint8),
        "targets": rng.normal(size=(6, 12)).astype(np.float32),
        "aux": rng.normal(size=(6, 8)).astype(np.float32),
    }.items():
        np.save(tmp_path / f"{name}.npy", array)
    args = parser().parse_args(["--out", str(tmp_path / "out"), "--device", "cpu"])
    args.features, args.target_features, args.target_aux = (tmp_path / f"{name}.npy" for name in ("drugs", "targets", "aux"))
    args.width, args.pair_hidden = 12, 24
    args.dropout = args.support_dropout = args.support_item_dropout = args.scaffold_dropout = 0.0
    args.grad_clip = 1e8
    args.threads = 1
    args.micro_batch_size = 4
    rows = [
        {
            "drug_feature_index": drug,
            "target_feature_index": target,
            "family_index": target % 3,
            "binary_label": target % 2,
            "murcko_scaffold": "same" if drug == 0 else f"scaffold{drug}",
        }
        for drug, count in enumerate((2, 4, 6))
        for target in range(count)
    ]
    data = pd.DataFrame(rows)
    support = SimpleNamespace(
        indices=np.broadcast_to(np.array([3, 4]), (len(data), 2, 2)).copy(),
        mask=np.ones((len(data), 2, 2), dtype=bool),
        similarities=rng.uniform(0.2, 0.9, size=(len(data), 2, 2)).astype(np.float32),
    )
    audit = {"families": ["a", "b", "c"], "assignment_sha256": "synthetic-fixed-assignment",
             "protocol": "SYNTHETIC_DRUG_ENTITY_HOLDOUT"}
    return args, data, audit, support


def make_runtime(prepared, micro_batch_size=4):
    args, data, audit, support = prepared
    args = deepcopy(args)
    args.micro_batch_size = micro_batch_size
    torch.manual_seed(121)
    return Runtime(args, data.copy(), deepcopy(audit), None, support)


@pytest.mark.parametrize("query", [False, True])
def test_accumulated_microbatches_match_whole_logical_batch_update(prepared, query):
    whole = make_runtime(prepared, micro_batch_size=12)
    split = make_runtime(prepared, micro_batch_size=4)
    positions = np.arange(len(whole.data))
    whole_optimizer = torch.optim.SGD(whole.parameters(), lr=0.01)
    split_optimizer = torch.optim.SGD(split.parameters(), lr=0.01)
    whole_losses = train_batch(whole, positions, whole_optimizer, query, np.random.default_rng(99), True)
    split_losses = train_batch(split, positions, split_optimizer, query, np.random.default_rng(99), True)
    assert split_losses == pytest.approx(whole_losses, rel=2e-5, abs=2e-6)
    for first, second in zip(whole.parameters(), split.parameters(), strict=True):
        torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-5)
        if first.grad is None:
            assert second.grad is None
        else:
            torch.testing.assert_close(first.grad, second.grad, atol=2e-6, rtol=2e-5)
    # Unequal query lengths 2/4/6 make row-weighted rank accumulation wrong;
    # the trainer instead weights each microbatch by its number of queries.
    if query:
        assert whole_losses[1] > 0
    else:
        assert whole_losses[1] == 0


def test_runtime_masks_control_class_query_and_scaffold_dropout(prepared):
    runtime = make_runtime(prepared)
    positions = np.arange(len(runtime.data))
    arguments = (runtime.drugs, runtime.targets, runtime.families, runtime.support, positions)
    runtime.args.support = "positive"
    inputs = runtime.inputs(*arguments)
    assert not inputs["support_mask"][:, 0].any()
    assert inputs["support_mask"][:, 1].all()
    runtime.args.support = "both"
    runtime.args.support_dropout = 1.0
    inputs = runtime.inputs(*arguments, rng=np.random.default_rng(7))
    assert not inputs["support_mask"].any()
    output = runtime.model(**inputs)
    assert torch.equal(output["final_logit"], output["base_logit"])
    runtime.args.support_dropout = 0.5
    inputs = runtime.inputs(*arguments, rng=np.random.default_rng(7))
    for drug in np.unique(runtime.drugs):
        masks = inputs["support_mask"][runtime.drugs == drug]
        assert torch.equal(masks, masks[0].expand_as(masks))
    runtime.args.support_dropout = 0.0
    runtime.args.scaffold_dropout = 1.0
    runtime.scaffolds[3], runtime.scaffolds[4] = "same", "unrelated"
    inputs = runtime.inputs(*arguments, rng=np.random.default_rng(7))
    drug_zero = runtime.drugs == 0
    assert not inputs["support_mask"][drug_zero, :, 0].any()
    assert inputs["support_mask"][drug_zero, :, 1].all()
    assert inputs["support_mask"][~drug_zero].all()
    # Inference does not apply stochastic evidence dropout.
    assert runtime.inputs(*arguments)["support_mask"].all()


def test_checkpoint_roundtrip_restores_selected_predictions_and_optimizer(prepared, tmp_path):
    runtime = make_runtime(prepared)
    optimizer = torch.optim.AdamW(runtime.parameters(), lr=3e-4)
    positions = np.arange(len(runtime.data))
    train_batch(runtime, positions, optimizer, True, np.random.default_rng(11), True)
    expected_scores, expected_gates = runtime.predict(positions)
    path = tmp_path / "checkpoint.pt"
    torch.save(runtime.checkpoint(1, optimizer, 0.75), path)
    restored = make_runtime(prepared)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    restored.restore(checkpoint)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.1)
    restored_optimizer.load_state_dict(checkpoint["optimizer_state"])
    scores, gates = restored.predict(positions)
    np.testing.assert_array_equal(scores, expected_scores)
    np.testing.assert_array_equal(gates, expected_gates)
    assert checkpoint["epoch"] == 1
    assert checkpoint["best_validation_drug_macro_ap"] == 0.75
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    # A second identical update verifies that moment state, not just weights,
    # survived serialization.
    train_batch(runtime, positions, optimizer, True, np.random.default_rng(12), True)
    train_batch(restored, positions, restored_optimizer, True, np.random.default_rng(12), True)
    for original_parameter, restored_parameter in zip(runtime.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(original_parameter, restored_parameter, atol=0, rtol=0)
    checkpoint["data_audit"]["assignment_sha256"] = "different-split"
    with pytest.raises(ValueError, match="assignment"):
        restored.restore(checkpoint)


def test_checkpoint_cpu_snapshots_do_not_alias_training_or_each_other(prepared):
    runtime = make_runtime(prepared)
    optimizer = torch.optim.AdamW(runtime.parameters(), lr=3e-4)
    history = [{"epoch": 1, "validation": {"drug_macro_ap": 0.75}}]
    checkpoint = runtime.checkpoint(1, optimizer, 0.75, best_epoch=1, history=history)
    name = "base.drug_encoder.net.0.weight"
    saved = checkpoint["model_state"][name].clone()
    with torch.no_grad():
        runtime.model.base.drug_encoder.net[0].weight.add_(1)
    torch.testing.assert_close(checkpoint["model_state"][name], saved, atol=0, rtol=0)
    checkpoint["model_state"][name].add_(2)
    torch.testing.assert_close(checkpoint["best_snapshot"]["model_state"][name], saved, atol=0, rtol=0)
    history[0]["validation"]["drug_macro_ap"] = 0
    assert checkpoint["history"][0]["validation"]["drug_macro_ap"] == 0.75


def test_restore_rejects_changed_target_feature_content(prepared):
    runtime = make_runtime(prepared)
    checkpoint = runtime.checkpoint(1, torch.optim.AdamW(runtime.parameters()), 0.75)
    changed = np.load(runtime.args.target_aux)
    changed[0, 0] += 1
    np.save(runtime.args.target_aux, changed)
    restored = make_runtime(prepared)
    with pytest.raises(ValueError, match="feature files"):
        restored.restore(checkpoint)


@pytest.mark.parametrize("changed_asset", ["residue_array", "residue_index", "target_0", "target_1", "target_2"])
def test_resume_identity_binds_every_local_protein_asset(prepared, tmp_path, monkeypatch, changed_asset):
    import biomaster.odti_local_features_v3 as local_features

    assets = {}
    for name in ("drug_index", "residue_array", "residue_index", "target_0", "target_1", "target_2"):
        assets[name] = tmp_path / f"{name}.asset"
        assets[name].write_bytes(f"original {name}".encode())
    monkeypatch.setattr(local_features, "DEFAULT_RESIDUE_ARRAY", assets["residue_array"])
    monkeypatch.setattr(local_features, "DEFAULT_RESIDUE_INDEX", assets["residue_index"])
    monkeypatch.setattr(local_features, "DEFAULT_TARGET_INDICES", tuple(assets[f"target_{index}"] for index in range(3)))

    # Exercise identity checks without loading a local network or parsing these
    # small hash-only fixtures as production residue/target tables.
    runtime = make_runtime(prepared)
    runtime.args.local, runtime.args.drug_index = True, assets["drug_index"]
    checkpoint = runtime.checkpoint(1, torch.optim.AdamW(runtime.parameters()), 0.75)
    expected_keys = {"drug_index", "local_residue_array", "local_residue_index",
                     "local_target_index_0", "local_target_index_1", "local_target_index_2"}
    assert expected_keys.issubset(checkpoint["feature_identity"])
    assets[changed_asset].write_bytes(f"changed {changed_asset}".encode())
    restored = make_runtime(prepared)
    restored.args.local, restored.args.drug_index = True, assets["drug_index"]
    with pytest.raises(ValueError, match="feature files"):
        restored.restore(checkpoint)


def test_resume_uses_last_bundle_after_history_write_interrupt_and_rejects_objective_change(
    prepared, tmp_path, monkeypatch
):
    args, data, audit, support = prepared
    args = deepcopy(args)
    args.out = tmp_path / "interrupted"
    args.epochs, args.batch_size, args.max_steps, args.query_drugs = 2, 2, 1, 1
    data = data.copy()
    data["split"] = data.drug_feature_index.map({0: "train", 1: "validation", 2: "test"})
    preparation = importlib.import_module("prepare_biomaster_selectivity_v3")
    monkeypatch.setattr(
        preparation, "prepare_training_data",
        lambda *unused_args, **unused_kwargs: (data.copy(), deepcopy(audit), None, support),
    )
    real_write_json = trainer.write_json

    def interrupted_write(path, value):
        if path.name == "TRAINING_HISTORY_V3.json":
            raise RuntimeError("simulated failure after committing LAST")
        return real_write_json(path, value)

    monkeypatch.setattr(trainer, "write_json", interrupted_write)
    with pytest.raises(RuntimeError, match="simulated failure"):
        trainer.run(args)
    checkpoint = torch.load(args.out / "LAST_MODEL_V3.pt", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == checkpoint["history"][-1]["epoch"] == 1
    assert checkpoint["best_epoch"] == 1
    assert checkpoint["best_snapshot"]["model_state"]
    # These two derived files may be absent, stale or corrupt after a crash.
    (args.out / "BEST_MODEL_V3.pt").write_bytes(b"corrupt selected artifact")
    (args.out / "TRAINING_HISTORY_V3.json").write_text("corrupt history")
    monkeypatch.setattr(trainer, "write_json", real_write_json)
    args.resume = True
    changed = deepcopy(args)
    changed.rank_loss = "hard"
    with pytest.raises(ValueError, match="resume argument changed: rank_loss"):
        trainer.run(changed)
    trainer.run(args)
    resumed = torch.load(args.out / "LAST_MODEL_V3.pt", map_location="cpu", weights_only=False)
    selected = torch.load(args.out / "BEST_MODEL_V3.pt", map_location="cpu", weights_only=False)
    assert resumed["epoch"] == 2
    assert selected["epoch"] == resumed["best_epoch"]
    for name, tensor in resumed["best_snapshot"]["model_state"].items():
        torch.testing.assert_close(selected["model_state"][name], tensor, atol=0, rtol=0)
    uninterrupted_args = deepcopy(args)
    uninterrupted_args.out, uninterrupted_args.resume = tmp_path / "uninterrupted", False
    trainer.run(uninterrupted_args)
    uninterrupted = torch.load(uninterrupted_args.out / "LAST_MODEL_V3.pt", map_location="cpu", weights_only=False)
    assert resumed["best_epoch"] == uninterrupted["best_epoch"]
    assert resumed["best_validation_drug_macro_ap"] == uninterrupted["best_validation_drug_macro_ap"]
    assert [row["exposure_sha256"] for row in resumed["history"]] == [row["exposure_sha256"] for row in uninterrupted["history"]]
    for name, tensor in resumed["model_state"].items():
        torch.testing.assert_close(tensor, uninterrupted["model_state"][name], atol=0, rtol=0)
