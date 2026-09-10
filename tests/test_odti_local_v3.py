import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from biomaster.odti_local_features_v3 import LocalFeatureStore, molecular_graph, sequence_regions, pocket_sequence_regions
from biomaster.odti_local_v3 import LocalInteractionBackboneV3, LocalInteractionConfigV3
from biomaster.odti_pockets_v3 import POCKET_CACHE_VERSION, select_diverse_pockets, validate_candidate, load_pocket_store


def _store(tmp_path=None):
    sequences = ["ACDEFGHIKLMN", "MNPQRS"]
    hashes = [hashlib.sha256(s.encode()).hexdigest() for s in sequences]
    # The target indices and token index rows intentionally disagree in order.
    targets = pd.DataFrame({"target_feature_index": [91, 7, 99], "protein_sequence": [sequences[1], sequences[0], "VVV"], "sequence_key": ["B", "A", "C"]})
    drugs = pd.DataFrame({"drug_feature_index": [42, 5, 6], "model_ligand_smiles": ["N[C@@H](C)C(=O)O", "CCO", "C" * 20]})
    residue_index = pd.DataFrame({"target_feature_index": [1000, 1001], "sequence_key": [hashes[0], hashes[1]], "token_offset": [0, 12], "token_length": [12, 6], "sequence_length": [12, 6]})
    array = np.arange(18 * 8, dtype=np.float32).reshape(18, 8) / 100
    return LocalFeatureStore(drugs, targets, array, residue_index, max_regions=2, tokens_per_region=3, max_atoms=12, cache_dir=tmp_path), array


def _model():
    torch.manual_seed(19)
    return LocalInteractionBackboneV3(LocalInteractionConfigV3(residue_input_dim=8, width=24, hidden_dim=16, pair_dim=8, blocks=2, dropout=0)).eval()


def test_stereochemistry_is_explicit_and_large_graph_is_not_truncated():
    r = molecular_graph("N[C@H](C)C(=O)O")
    s = molecular_graph("N[C@@H](C)C(=O)O")
    assert not np.array_equal(r["atom_features"], s["atom_features"])
    assert r["atom_features"][:, 30:34].sum() == 1
    assert s["atom_features"][:, 30:34].sum() == 1
    e = molecular_graph("F/C=C/F")
    z = molecular_graph("F/C=C\\F")
    assert not np.array_equal(e["bond_stereo"], z["bond_stereo"])
    oversized = molecular_graph("C" * 15, max_atoms=10)
    assert not oversized["available"]
    assert len(oversized["atom_features"]) == 0
    assert oversized["reason"] == "atom_count_exceeds_limit"


def test_complete_sequence_coverage_including_long_protein_tail():
    residues = np.zeros((2057, 5), np.float32)
    residues[1900:, 0] = 7
    item = sequence_regions(residues, max_regions=4, tokens_per_region=8)
    spans = item["coverage_spans"][item["residue_mask"]]
    assert spans[0, 0] == 0 and spans[-1, 1] == 2057
    np.testing.assert_array_equal(spans[:-1, 1], spans[1:, 0])
    weights = spans[:, 1] - spans[:, 0]
    represented = item["residue_features"][item["residue_mask"]].astype(np.float32)
    np.testing.assert_allclose(np.average(represented, weights=weights, axis=0), residues.mean(0), atol=1e-4)
    assert item["residue_features"][-1, -1, 0] == 7
    assert item["region_kind"] == "SEQUENCE_COVERAGE_WINDOWS"


def test_hash_alignment_missing_targets_and_lazy_disk_cache(tmp_path):
    store, array = _store(tmp_path)
    item = store.target(91)
    np.testing.assert_allclose(item["residue_features"][0, 0], array[12], atol=1e-3)
    assert store.target(7)["sequence_length"] == 12
    assert not store.target(99)["available"]
    assert len(store.drug_cache) == 0
    original = store.drug(42)["atom_features"].copy()
    store.drug_cache.clear()
    np.testing.assert_array_equal(store.drug(42)["atom_features"], original)
    assert len(list(tmp_path.glob("drug_*.npz"))) == 1


def test_truncated_cache_and_conflicting_hash_rejected():
    sequence = "ACDEFGH"
    targets = pd.DataFrame({"target_feature_index": [1], "protein_sequence": [sequence], "sequence_key": ["A"]})
    drugs = pd.DataFrame({"drug_feature_index": [1], "model_ligand_smiles": ["CC"]})
    tokens = pd.DataFrame({"sequence_key": ["A"], "token_offset": [0], "token_length": [5], "sequence_length": [7]})
    with pytest.raises(ValueError, match="truncated"):
        LocalFeatureStore(drugs, targets, np.zeros((5, 8)), tokens)
    targets["sequence_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="does not match"):
        LocalFeatureStore(drugs, targets, np.zeros((5, 8)), tokens)


def test_all_missing_and_mixed_missing_outputs_are_exact_zero():
    store, _ = _store()
    model = _model()
    batch = store.batch([42, 6, 5], [7, 7, 99])
    result = model(**batch)
    assert result["local_hidden"].shape == (3, 16)
    assert result["local_available"].tolist() == [True, False, False]
    assert torch.count_nonzero(result["local_hidden"][1:]) == 0
    torch.testing.assert_close(result["region_weights"].sum(-1), torch.tensor([1., 0., 0.]))
    batch["atom_mask"][:] = False
    batch["atom_features"][:] = float("nan")
    empty = model(**batch)
    assert torch.isfinite(empty["local_hidden"]).all()
    assert torch.count_nonzero(empty["local_hidden"]) == 0


def test_padding_and_atom_residue_region_permutation_invariance():
    store, _ = _store()
    model = _model()
    batch = store.batch([42], [7])
    expected = model(**batch)["local_hidden"]
    atom_perm = torch.tensor([3, 1, 5, 0, 4, 2])
    residue_perm = torch.tensor([2, 0, 1])
    permuted = {key: value.clone() for key, value in batch.items()}
    for key in ("atom_features", "atom_mask"):
        permuted[key] = permuted[key][:, atom_perm]
    for key in ("bond_type", "bond_stereo"):
        permuted[key] = permuted[key][:, atom_perm][:, :, atom_perm]
    for key in ("residue_features", "residue_positions", "residue_mask"):
        permuted[key] = permuted[key].flip(1)[:, :, residue_perm]
    permuted["region_mask"] = permuted["region_mask"].flip(1)
    torch.testing.assert_close(model(**permuted)["local_hidden"], expected, atol=2e-6, rtol=2e-6)
    padded = {key: value.clone() for key, value in batch.items()}
    padded["atom_features"] = torch.nn.functional.pad(padded["atom_features"], (0, 0, 0, 3), value=999)
    padded["atom_mask"] = torch.nn.functional.pad(padded["atom_mask"], (0, 3), value=False)
    for key in ("bond_type", "bond_stereo"):
        padded[key] = torch.nn.functional.pad(padded[key], (0, 3, 0, 3), value=0)
    for key in ("residue_features", "residue_positions"):
        padded[key] = torch.nn.functional.pad(padded[key], (0, 0, 0, 2), value=999)
    padded["residue_mask"] = torch.nn.functional.pad(padded["residue_mask"], (0, 2), value=False)
    torch.testing.assert_close(model(**padded)["local_hidden"], expected, atol=2e-6, rtol=2e-6)


def test_pair_states_are_updated_and_gradients_reach_both_entities():
    store, _ = _store()
    model = _model()
    batch = store.batch([42, 5], [7, 91])
    batch["atom_features"].requires_grad_()
    batch["residue_features"].requires_grad_()
    before = []
    handle = model.blocks[0].register_forward_hook(lambda module, inputs, output: before.append(output[2].detach().clone()))
    result = model(**batch, return_pair_states=True)
    handle.remove()
    assert not torch.allclose(before[0], result["pair_states"].reshape_as(before[0]))
    result["local_hidden"].square().sum().backward()
    assert batch["atom_features"].grad.abs().sum() > 0
    assert batch["residue_features"].grad.abs().sum() > 0
    assert model.blocks[0].pair_update[0].weight.grad.abs().sum() > 0
    changed = {key: value.detach().clone() for key, value in batch.items()}
    changed["residue_features"] = changed["residue_features"].flip(0)
    assert not torch.allclose(result["local_hidden"], model(**changed)["local_hidden"])


def test_predicted_pocket_probability_AA_and_bounds_filters():
    sequence = "ACDEFGHIKLMN"
    ca = {("A", i + 1, ""): aa for i, aa in enumerate(sequence)}
    candidate = {"name": "pocket1", "probability": .8, "score": 20, "rank": 1, "residue_ids": "A_1 A_2 A_3 A_4 A_5"}
    valid, reason = validate_candidate(candidate, sequence, ca)
    assert reason == "" and valid["residue_indices"] == [0, 1, 2, 3, 4]
    assert validate_candidate({**candidate, "probability": .19}, sequence, ca)[0] is None
    assert validate_candidate({**candidate, "residue_ids": "A_1 A_2 A_3 A_4 A_50"}, sequence, ca)[1] == "residue_outside_canonical_sequence"
    assert validate_candidate(candidate, sequence, {**ca, ("A", 3, ""): "W"})[1] == "residue_CA_missing_or_AA_mismatch"
    assert validate_candidate({**candidate, "residue_ids": "B_1 B_2 B_3 B_4 B_5"}, sequence, ca)[0] is None


def test_predicted_pocket_sort_and_jaccard_deduplication():
    first = {"pocket_id": "best", "probability": .9, "residue_indices": [0, 1, 2, 3, 4]}
    duplicate = {"pocket_id": "overlap", "probability": .85, "residue_indices": [0, 1, 2, 3, 5]}
    second = {"pocket_id": "second", "probability": .7, "residue_indices": [6, 7, 8, 9, 10]}
    selected = select_diverse_pockets([second, duplicate, first], max_pockets=3, jaccard_threshold=.5)
    assert [x["pocket_id"] for x in selected] == ["best", "second"]


def test_pocket_region_has_explicit_membership_and_retains_complete_sequence():
    residues = np.arange(60, dtype=np.float32).reshape(20, 3)
    pocket = {"residue_indices": [0, 2, 4, 8, 10], "probability": .8}
    item = pocket_sequence_regions(residues, [pocket], max_regions=4, tokens_per_region=3)
    assert item["region_kind"] == "P2RANK_POCKETS_AND_FULL_SEQUENCE"
    assert item["full_sequence_region_index"] == 3
    assert item["region_mask"].tolist() == [True, False, False, True]
    assert sum(item["pocket_residue_index_bins"][0], []) == [0, 2, 4, 8, 10]
    assert np.all(item["coverage_spans"][0] == -1)
    spans = item["coverage_spans"][3]
    assert spans[0, 0] == 0 and spans[-1, 1] == 20
    np.testing.assert_array_equal(spans[:-1, 1], spans[1:, 0])
    np.testing.assert_allclose(item["residue_features"][0, 0], residues[0])
    fallback = pocket_sequence_regions(residues, [], max_regions=4, tokens_per_region=3)
    expected = sequence_regions(residues, 4, 3)
    np.testing.assert_array_equal(fallback["residue_features"], expected["residue_features"])


def test_store_pocket_hash_alignment_and_sequence_only_fallback(tmp_path):
    store, array = _store()
    digest = store.target_hashes[7]
    payload = {"format": POCKET_CACHE_VERSION, "label_dependency": "NONE", "targets": {
        digest: {"sequence_sha256": digest, "sequence_length": 12, "pockets": [
            {"source_kind": "P2RANK_PREDICTED_POCKET", "probability": .8, "residue_indices": [0, 2, 4, 6, 8]},
        ]},
    }}
    path = tmp_path / "pockets.json"
    path.write_text(json.dumps(payload))
    baseline = store.target(91)["residue_features"].copy()
    # Reuse the fixture's explicit IDs and residue index; new optional path must
    # not change the sequence-only targets or mask behavior.
    loaded = LocalFeatureStore(
        pd.DataFrame({"drug_feature_index": [42], "model_ligand_smiles": ["CCO"]}),
        pd.DataFrame({"target_feature_index": [91, 7], "protein_sequence": ["MNPQRS", "ACDEFGHIKLMN"]}),
        array,
        pd.DataFrame({"sequence_key": [store.target_hashes[7], store.target_hashes[91]], "token_offset": [0, 12], "token_length": [12, 6], "sequence_length": [12, 6]}),
        max_regions=2, tokens_per_region=3, pocket_store=path,
    )
    assert loaded.target(7)["region_kind"] == "P2RANK_POCKETS_AND_FULL_SEQUENCE"
    np.testing.assert_array_equal(loaded.target(91)["residue_features"], baseline)
    output = _model()(**loaded.batch([42, 42], [7, 91]))
    assert output["local_available"].tolist() == [True, True]
    payload["targets"][digest]["pockets"][0]["residue_indices"] = [0, 12]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="canonical residue indices"):
        load_pocket_store(path)
