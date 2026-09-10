from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import subprocess
import sys

from biomaster.odti_support_data_v3 import SupportStore


def make_store(*, backend="numpy", train=None):
    features = np.asarray([
        [1, 1, 0, 0],  # query / entity A
        [1, 1, 0, 0],  # duplicate entity A
        [1, 0, 0, 0],  # active B
        [1, 0, 0, 0],  # active C, tied similarity, same scaffold as query
        [0, 0, 1, 0],  # negative D
        [1, 1, 0, 0],  # held-out E, would be perfect match
        [1, 1, 1, 1],  # unknown F
        [0, 1, 1, 0],  # contradictory G, never valid support
    ], dtype=np.uint8)
    drugs = np.asarray([0, 1, 2, 2, 3, 4, 5, 6, 7, 7])
    labels = np.asarray([1, 1, 1, 1, 1, 0, 1, -1, 0, 1])
    train = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9]) if train is None else train
    return SupportStore(features, drugs, np.zeros(len(drugs), dtype=int), labels, train,
                        entity_keys=["A", "A", "B", "C", "D", "E", "F", "G"],
                        scaffold_keys=["S", "S", "B", "S", "D", "E", None, "G"], backend=backend)


def test_training_boundary_duplicate_entity_exclusion_and_conflicts():
    store = make_store()
    batch = store.retrieve([0], [0], k=4)
    assert batch.indices.tolist() == [[[4, -1, -1, -1], [2, 3, -1, -1]]]
    assert batch.similarities[0, 1, :2].tolist() == [0.5, 0.5]
    assert store.metadata["conflicting_target_entities_excluded"] == 1
    assert store.metadata["support_rows"] == 4  # A, B, C, D; duplicate A/B collapsed
    assert store.metadata["observed_binary_training_rows"] == 8
    assert not np.isin(batch.indices[batch.mask], [0, 1, 5, 6, 7]).any()


def test_same_entity_different_feature_row_cannot_retrieve_self():
    batch = make_store().retrieve([1], [0], k=4)
    assert not np.isin(batch.indices[batch.mask], [0, 1]).any()


def test_unknown_and_unobserved_rows_never_become_support():
    features = np.eye(5, dtype=np.uint8)
    store = SupportStore(features, np.arange(5), [0] * 5, [0, 1, np.nan, -1, 0], np.arange(5),
                         observed=[1, 1, 1, 1, 0], backend="numpy")
    batch = store.retrieve([2], [0], k=4)
    assert set(batch.indices[batch.mask]) == {0, 1}


def test_reordering_and_batch_companions_do_not_change_retrieval():
    store = make_store()
    original = store.retrieve([0, 2, 4, 5, 0], [0] * 5, k=2)
    perm = [3, 1, 4, 0, 2]
    reordered = store.retrieve(np.asarray([0, 2, 4, 5, 0])[perm], [0] * 5, k=2)
    for name in ("indices", "similarities", "mask"):
        np.testing.assert_array_equal(getattr(original, name)[perm], getattr(reordered, name))
    for index, drug in enumerate([0, 2, 4, 5, 0]):
        single = store.retrieve([drug], [0], k=2)
        np.testing.assert_array_equal(original.indices[index], single.indices[0])
    # Equal-similarity support is selected by ascending feature index.
    assert store.retrieve([0], [0], k=1).indices[0, 1, 0] == 2


def test_scaffold_exclusion_preserves_missing_scaffolds():
    store = make_store()
    batch = store.retrieve([0], [0], k=4, exclude_scaffold=True)
    assert batch.indices[0, 1].tolist() == [2, -1, -1, -1]
    assert store.retrieve([6], [0], k=4, exclude_scaffold=True).mask.sum() == 4


def test_no_support_and_empty_batch_have_explicit_masks():
    store = make_store()
    batch = store.retrieve([0], [999], k=3)
    assert not batch.mask.any()
    assert (batch.indices == -1).all()
    assert (batch.similarities == 0).all()
    assert store.retrieve([], [], k=3).indices.shape == (0, 2, 3)
    empty = make_store(train=np.asarray([], dtype=int))
    assert not empty.retrieve([0], [0]).mask.any()


def test_rdkit_and_numpy_compute_identical_exact_tanimoto():
    pytest.importorskip("rdkit")
    rng = np.random.default_rng(113)
    # Include empty vectors and bit counts not divisible by one packed byte.
    features = rng.integers(0, 2, size=(35, 71), dtype=np.uint8)
    features[:2] = 0
    args = (features, np.arange(35), np.zeros(35, dtype=int), np.arange(35) % 2, np.arange(35))
    a, b = SupportStore(*args, backend="numpy"), SupportStore(*args, backend="rdkit")
    for name in ("indices", "similarities", "mask"):
        np.testing.assert_array_equal(getattr(a.retrieve(np.arange(35), [0] * 35, k=4), name),
                                      getattr(b.retrieve(np.arange(35), [0] * 35, k=4), name))


def test_save_load_references_features_and_detects_mutation(tmp_path: Path):
    store = make_store()
    source = tmp_path / "original_features.npy"
    np.save(source, store.features)
    manifest_path = store.save(tmp_path / "snapshot", feature_path=source)
    assert sorted(path.name for path in manifest_path.parent.iterdir()) == ["SUPPORT_STORE_MANIFEST_V3.json", "support_index.npz"]
    loaded = SupportStore.load(manifest_path, backend="numpy")
    assert loaded.store_hash == store.store_hash
    np.testing.assert_array_equal(loaded.retrieve([0, 4], [0, 0]).indices, store.retrieve([0, 4], [0, 0]).indices)
    changed = store.features.copy()
    changed[0, 0] = 0
    np.save(source, changed)
    with pytest.raises(ValueError, match="Feature content hash"):
        SupportStore.load(manifest_path, backend="numpy")


def test_cache_reuses_mmap_and_invalidates_changed_snapshot(tmp_path: Path):
    store = make_store()
    cached = store.cache_retrieve(tmp_path, [0, 4], [0, 0], k=3)
    assert isinstance(cached.indices, np.memmap)
    first_cache_count = len(list(tmp_path.iterdir()))
    # If valid, a repeated call must not retrieve again.
    store.retrieve = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("cache miss"))
    same = store.cache_retrieve(tmp_path, [0, 4], [0, 0], k=3)
    np.testing.assert_array_equal(cached.indices, same.indices)
    assert len(list(tmp_path.iterdir())) == first_cache_count
    changed = make_store(train=np.arange(10))  # Newly allowed held-out E changes identity.
    refreshed = changed.cache_retrieve(tmp_path, [0, 4], [0, 0], k=3)
    assert len(list(tmp_path.iterdir())) == first_cache_count + 1
    assert refreshed.indices[0, 1, 0] == 5


def test_cache_checksum_detects_corrupt_support_indices(tmp_path: Path):
    store = make_store()
    expected = store.cache_retrieve(tmp_path, [0], [0], k=2).indices.copy()
    path = next(tmp_path.glob("*/indices.npy"))
    np.save(path, np.full_like(expected, 123))
    repaired = store.cache_retrieve(tmp_path, [0], [0], k=2)
    np.testing.assert_array_equal(repaired.indices, expected)


def test_invalid_inputs_fail_explicitly():
    store = make_store()
    with pytest.raises(ValueError):
        store.retrieve([0], [0], k=0)
    with pytest.raises(IndexError):
        store.retrieve([-1], [0])
    with pytest.raises(ValueError):
        store.retrieve([0, 1], [0])
    with pytest.raises(ValueError, match="binary"):
        SupportStore(np.asarray([[3, 1]]), [0], [0], [1], [0])
    no_scaffolds = SupportStore(np.eye(2, dtype=np.uint8), [0], [0], [1], [0], backend="numpy")
    with pytest.raises(ValueError, match="scaffold"):
        no_scaffolds.retrieve([1], [0], exclude_scaffold=True)


def test_conflict_is_resolved_across_duplicate_chemical_entity_rows():
    store = SupportStore(np.eye(3, dtype=np.uint8), [0, 1], [0, 0], [0, 1], [0, 1],
                         entity_keys=["same", "same", "query"], backend="numpy")
    assert store.metadata["conflicting_target_entities_excluded"] == 1
    assert not store.retrieve([2], [0]).mask.any()


def test_cli_builds_support_and_query_cache(tmp_path: Path):
    import pandas as pd
    source = tmp_path / "features.npy"
    np.save(source, np.eye(3, dtype=np.uint8))
    train = tmp_path / "train.npy"
    np.save(train, np.asarray([0, 1]))
    relations = tmp_path / "relations.csv"
    pd.DataFrame({"drug_feature_index": [0, 1, 2], "target_feature_index": [0, 0, 0],
                  "binary_label": [0, 1, 1], "binary_observed": [1, 1, 1]}).to_csv(relations, index=False)
    index = tmp_path / "index.csv"
    pd.DataFrame({"drug_feature_index": [0, 1, 2], "model_ligand_smiles": ["C", "CC", "CCC"],
                  "murcko_scaffold": ["C", "CC", "CCC"]}).to_csv(index, index=False)
    script = Path(__file__).resolve().parents[1] / "scripts/build_biomaster_support_store_v3.py"
    result = subprocess.run([sys.executable, str(script), "--relations", str(relations),
                             "--features", str(source), "--feature-index", str(index),
                             "--train-positions", str(train), "--out", str(tmp_path / "out"),
                             "--cache-all-relations", "--k", "2", "--backend", "numpy"],
                            capture_output=True, text=True, check=True)
    assert "support_cache_complete" in result.stdout
    store = SupportStore.load(tmp_path / "out", backend="numpy")
    assert store.metadata["training_rows"] == 2
    cache_indices = np.load(next((tmp_path / "out" / "retrieval_cache").glob("*/indices.npy")))
    assert cache_indices.shape == (3, 2, 2)
    assert 2 not in cache_indices


def test_prepare_training_data_reuses_only_identical_experiment(tmp_path: Path):
    import pandas as pd
    from biomaster.selectivity_training_v3 import entity_hash
    from scripts.prepare_biomaster_selectivity_v3 import prepare_training_data
    keys = []
    counts = {"train": 0, "validation": 0, "test": 0}
    for index in range(200):
        key = f"ENTITY_{index}"
        bucket = entity_hash(key) % 100
        split = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
        if counts[split] < 2:
            keys.append(key)
            counts[split] += 1
        if min(counts.values()) == 2:
            break
    rows = [{"drug_feature_index": index, "target_feature_index": target,
             "binary_label": target, "binary_observed": 1,
             "parent_standard_inchi_key": key, "target_assay_family": "enzyme",
             "murcko_scaffold": f"S{index}", "calibration_pair_id": f"P{index}_{target}"}
            for index, key in enumerate(keys) for target in [0, 1]]
    relation_path = tmp_path / "relations.csv"
    pd.DataFrame(rows).to_csv(relation_path, index=False)
    feature_path = tmp_path / "features.npy"
    np.save(feature_path, np.eye(len(keys), dtype=np.uint8))
    out = tmp_path / "prepared"
    data, audit, store, batch = prepare_training_data(relation_path, feature_path, out)
    assert set(data.split) == {"train", "validation", "test"}
    assert set(store.pool_drug_indices) <= set(data.loc[data.split.eq("train"), "drug_feature_index"])
    repeated = prepare_training_data(relation_path, feature_path, out)
    np.testing.assert_array_equal(batch.indices, repeated[3].indices)
    with pytest.raises(ValueError, match="differs from requested experiment"):
        prepare_training_data(relation_path, feature_path, out, max_entities=3)
