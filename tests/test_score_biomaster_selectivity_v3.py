from __future__ import annotations

from argparse import Namespace
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from biomaster.odti_support_data_v3 import SupportStore
from scripts import score_biomaster_selectivity_v3 as scorer


def prepared_rows():
    # Deliberately noncontiguous families: input ordering must never renumber them.
    return pd.DataFrame({"drug_feature_index": [0, 1, 2], "target_feature_index": [10, 20, 10],
                         "family_index": [2, 5, 2], "binary_label": [1, 0, 0]})


class MockRuntime:
    def __init__(self):
        self.data = prepared_rows()
        self.args = SimpleNamespace(support_k=2)
        self.store = SupportStore(np.eye(3, dtype=np.uint8), [0, 1, 2], [10, 20, 10], [1, 0, 0],
                                  [0, 1], backend="numpy")
        self.training = True

    def mode(self, train):
        self.training = train

    def autocast(self):
        return nullcontext()

    def inputs(self, drugs, targets, families, evidence, rows, rng=None):
        assert rng is None
        return {"drugs": torch.tensor(drugs), "targets": torch.tensor(targets),
                "families": torch.tensor(families), "mask": torch.tensor(evidence.mask[rows])}

    def model(self, drugs, targets, families, mask):
        assert not self.training
        return {"final_logit": (100 * families + targets + drugs).float(),
                "support_gate": mask.float().mean((1, 2))}


def minimal_arguments(tmp_path: Path) -> dict:
    return {"relations": tmp_path / "raw.csv", "features": tmp_path / "features.npy",
            "target_features": tmp_path / "target.npy", "target_aux": tmp_path / "aux.npy",
            "cache_dir": tmp_path / "prepared", "max_entities": 0, "support_k": 2,
            "width": 192, "pair_hidden": 256, "dropout": 0.1, "support": "both", "local": False,
            "precision": "fp32", "device": "cuda"}


def completed_preparation(tmp_path: Path):
    args = Namespace(**minimal_arguments(tmp_path))
    directory = Path(args.cache_dir)
    cache = directory / "support_normal/cache_identity"
    store = directory / "support_store"
    cache.mkdir(parents=True)
    store.mkdir()
    (directory / "RELATIONS_V3.csv.gz").write_bytes(b"prepared")
    (store / "SUPPORT_STORE_MANIFEST_V3.json").write_text("{}")
    (store / "support_index.npz").write_bytes(b"index")
    paths = {}
    for name in ["indices", "similarities", "mask"]:
        paths[name] = cache / f"{name}.npy"
        np.save(paths[name], np.zeros((3, 2, 2)))
    (cache / "manifest.json").write_text(json.dumps({"identity": {"store_hash": "store"},
                                                   "array_sha256": {name: scorer.file_sha256(path) for name, path in paths.items()}}))
    manifest = {"status": "COMPLETE", "identity": {"relations_path": str(args.relations.resolve()),
                 "features_path": str(args.features.resolve()), "support_k": 2, "max_entities": 0},
                "store_hash": "store", "assignment_sha256": "assignment",
                **{f"support_{name}_path": str(path) for name, path in paths.items()}}
    (directory / "DATA_MANIFEST_V3.json").write_text(json.dumps(manifest))
    return args, manifest


def test_cli_requires_checkpoint_pairs_output_and_positive_batch_size():
    parser = scorer.parser()
    for argv in [[], ["--checkpoint", "x.pt"], ["--checkpoint", "x.pt", "--pairs", "x.csv", "--out", "o.csv", "--batch-size", "0"]]:
        with pytest.raises(SystemExit):
            parser.parse_args(argv)
    args = parser.parse_args(["--checkpoint", "x.pt", "--pairs", "x.csv", "--out", "o.csv.gz", "--device", "cpu", "--batch-size", "7"])
    assert args.batch_size == 7 and args.device == "cpu"


def test_only_trusted_local_current_format_checkpoints_are_loaded(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(scorer, "ROOT", tmp_path / "project")
    with pytest.raises(ValueError, match="inside this BioMaster"):
        scorer.load_project_checkpoint(tmp_path / "outside.pt")
    path = tmp_path / "project" / "inside.pt"
    path.parent.mkdir()
    torch.save({"format": "OTHER"}, path)
    with pytest.raises(ValueError, match="format"):
        scorer.load_project_checkpoint(path)
    checkpoint = {"format": scorer.CHECKPOINT_FORMAT, "arguments": {}, "data_audit": {}, "model_state": {},
                  "base_config": {}, "support_config": {}, "local_state": None, "local_config": None,
                  "family_count": 1, "feature_identity": {}}
    torch.save(checkpoint, path)
    assert scorer.load_project_checkpoint(path)["format"] == "BIOMASTER_SELECTIVITY_V3_2"


def test_checkpoint_arguments_restored_without_mutating_training_configuration(tmp_path: Path):
    saved = minimal_arguments(tmp_path)
    args = scorer.restored_arguments({"arguments": saved}, device="cpu", batch_size=7)
    assert args.device == "cpu" and args.eval_batch_size == 7
    assert saved["device"] == "cuda" and "eval_batch_size" not in saved
    with pytest.raises(ValueError, match="required fields"):
        scorer.restored_arguments({"arguments": {}}, device="cpu", batch_size=7)


def test_existing_preparation_refuses_missing_changed_or_corrupt_artifacts(tmp_path: Path):
    args, manifest = completed_preparation(tmp_path)
    assert scorer.verify_existing_preparation(args)["assignment_sha256"] == "assignment"
    args.support_k = 3
    with pytest.raises(ValueError, match="do not match"):
        scorer.verify_existing_preparation(args)
    args.support_k = 2
    indices = Path(manifest["support_indices_path"])
    indices.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="damaged"):
        scorer.verify_existing_preparation(args)
    indices.unlink()
    with pytest.raises(FileNotFoundError, match="refuses to rebuild"):
        scorer.verify_existing_preparation(args)


def test_missing_data_manifest_is_rejected_before_any_preparation(tmp_path: Path, monkeypatch):
    checkpoint = {"arguments": minimal_arguments(tmp_path)}
    monkeypatch.setattr(scorer, "load_project_checkpoint", lambda path: checkpoint)
    monkeypatch.setattr(scorer, "prepare_training_data", lambda *args, **kwargs: pytest.fail("must not rebuild a fold"))
    args = Namespace(checkpoint=tmp_path / "x.pt", pairs=tmp_path / "pairs.csv", out=tmp_path / "score.csv", device="cpu", batch_size=3)
    with pytest.raises(FileNotFoundError, match="existing DATA_MANIFEST"):
        scorer.run(args)


@pytest.mark.parametrize("column,value", [("drug_feature_index", -1), ("drug_feature_index", 0.5),
                                           ("drug_feature_index", np.nan), ("target_feature_index", "broken"),
                                           ("target_feature_index", True)])
def test_invalid_feature_indices_fail_before_scoring(column, value):
    pairs = pd.DataFrame({"drug_feature_index": [0], "target_feature_index": [10]})
    pairs[column] = value
    with pytest.raises(ValueError, match="nonnegative integer"):
        scorer.validate_pairs(pairs, prepared_rows())


def test_alias_and_unknown_target_rejection_has_actionable_mapping_errors():
    with pytest.raises(ValueError, match="source_row -> drug_feature_index"):
        scorer.validate_pairs(pd.DataFrame({"drug_feature_index": [3], "target_feature_index": [10]}), prepared_rows())
    with pytest.raises(ValueError, match="fixed trained family mapping"):
        scorer.validate_pairs(pd.DataFrame({"drug_feature_index": [0], "target_feature_index": [99]}), prepared_rows())
    with pytest.raises(ValueError, match="required columns"):
        scorer.validate_pairs(pd.DataFrame({"drug_feature_index": [0]}), prepared_rows())


def test_scoring_ignores_labels_preserves_row_order_and_batch_invariance(tmp_path: Path):
    pairs = pd.DataFrame({"row_key": ["C", "A", "D", "B"], "drug_feature_index": [2, 0, 2, 1],
                          "target_feature_index": [20, 10, 10, 20], "binary_label": ["do not", "interpret", "these", "values"]})
    a = scorer.score_pairs(MockRuntime(), pairs, tmp_path / "a", batch_size=1)
    b = scorer.score_pairs(MockRuntime(), pairs, tmp_path / "b", batch_size=3)
    assert a.row_key.tolist() == pairs.row_key.tolist()
    np.testing.assert_array_equal(a.v3_logit, [522, 210, 212, 521])
    np.testing.assert_array_equal(a[["v3_logit", "support_gate"]], b[["v3_logit", "support_gate"]])
    reordered = pairs.iloc[[2, 0, 3, 1]].copy()
    reordered["binary_label"] = [0, 1, 0, 1]
    c = scorer.score_pairs(MockRuntime(), reordered, tmp_path / "c", batch_size=2)
    np.testing.assert_array_equal(a.set_index("row_key").loc[c.row_key, ["v3_logit", "support_gate"]], c[["v3_logit", "support_gate"]])


def test_empty_scoring_and_ambiguous_family_mapping(tmp_path: Path):
    empty = pd.DataFrame({"drug_feature_index": pd.Series(dtype=int), "target_feature_index": pd.Series(dtype=int)})
    assert scorer.score_pairs(MockRuntime(), empty, tmp_path / "empty", batch_size=2).shape == (0, 4)
    ambiguous = pd.concat([prepared_rows(), pd.DataFrame({"drug_feature_index": [1], "target_feature_index": [10], "family_index": [5]})])
    with pytest.raises(ValueError, match="ambiguous"):
        scorer.validate_pairs(pd.DataFrame({"drug_feature_index": [0], "target_feature_index": [10]}), ambiguous)
