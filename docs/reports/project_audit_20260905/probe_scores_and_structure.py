"""Read-only checks of frozen scores and one existing FULL_FIT checkpoint."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PACKAGE = ROOT / "outputs/retrain_20260901/comprehensive_training_v1"
os.environ["BIOMASTER_COMPREHENSIVE_PACKAGE"] = str(PACKAGE)
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from biomaster.odti_v2 import ODTIV2Config, RoutedInteractionRankerV2  # noqa: E402
from train_biomaster_comprehensive_full_fit_v1 import target_structure_arrays  # noqa: E402


def main():
    torch.set_num_threads(2)
    base = ROOT / "outputs/old_drug_target_sota_v1"
    s5 = pd.read_csv(base / "drug_centric_ranker_v1/BIOMASTER_DRUG_TO_TARGET_S5_TEST_V1.csv.gz")
    baseline = pd.read_csv(base / "baseline_results_v1/S5_OLD_DRUG_ENTITY_COLD/fixed_split/BASELINE_TEST_PREDICTIONS_V1.csv.gz")
    column = "TRAIN_POSITIVE_MAX_MORGAN_TANIMOTO"
    merged = s5.merge(baseline[["calibration_pair_id", column]], on="calibration_pair_id", validate="one_to_one")
    assert len(merged) == 2556
    ap = []
    for _, g in merged.groupby("parent_standard_inchi_key"):
        if g.binary_label.nunique() == 2:
            ap.append([average_precision_score(g.binary_label, g[c]) for c in [column, "biomaster_logit", "independent_validation_rank_score"]])
    ap = np.array(ap)
    rng = np.random.default_rng(20260905)
    difference = ap[:, 2] - ap[:, 0]
    bootstrap = difference[rng.integers(len(ap), size=(5000, len(ap)))].mean(axis=1)
    result = {"s5_existing_scores": {
        "queries": len(ap), "tanimoto_drug_macro_ap": float(ap[:, 0].mean()),
        "raw_neural_drug_macro_ap": float(ap[:, 1].mean()),
        "independent_rank_drug_macro_ap": float(ap[:, 2].mean()),
        "rank_minus_tanimoto_ap": float(difference.mean()),
        "rank_minus_tanimoto_ap_ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
        "note": "Retrospective diagnostic on existing S5 predictions; no model fitted or selected.",
    }}
    run = ROOT / "outputs/retrain_20260901/comprehensive_balanced_full_fit_v2/FULL_FIT_2026_COMPREHENSIVE_BALANCED__seed_20260816"
    history = pd.read_csv(run / "FULL_FIT_TRAINING_HISTORY_V2.csv")
    replay = json.loads((OUT / "AUDIT_DIAGNOSTICS.json").read_text())["sampling_exposure_seed20260816"]
    actual_counts = history["balanced_unique_rows_emitted"].astype(int).tolist()
    replay_counts = [r["unique_rows_this_epoch"] for r in replay["epochs"]]
    assert actual_counts == replay_counts
    result["sampler_replay_matches_all_existing_epoch_unique_counts"] = True
    checkpoint = torch.load(run / "FULL_FIT_MODEL_COMPREHENSIVE_BALANCED_V2.pt", map_location="cpu", weights_only=False)
    cfg = ODTIV2Config(**checkpoint["config"])
    model = RoutedInteractionRankerV2(len(checkpoint["families"]), cfg, use_conplex=False).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    data = pd.read_csv(PACKAGE / "COMPREHENSIVE_TRAINING_RELATIONS_V1.csv.gz", low_memory=False)
    aug = ROOT / "outputs/biomaster_deployment_augmentation_v1"
    targets = np.load(aug / "PROTBERT1024_FLOAT32_DEPLOYMENT_AUGMENTED_V1.npy", mmap_mode="r")
    aux = np.load(aug / "ESM2_650M_1280_FLOAT32_DEPLOYMENT_AUGMENTED_V1.npy", mmap_mode="r")
    drugs = np.load(PACKAGE / "MORGAN2048_UINT8_COMPREHENSIVE_V1.npy", mmap_mode="r")
    structure, mask, columns, _, _ = target_structure_arrays(data, len(targets))
    sample = data.loc[mask > 0].drop_duplicates("target_feature_index").head(128)
    indices = sample.index.to_numpy()
    drug_indices = sample.drug_feature_index.to_numpy(dtype=int)
    target_indices = sample.target_feature_index.to_numpy(dtype=int)
    norm = checkpoint["normalization"]
    def tensor(x):
        return torch.tensor(np.asarray(x, dtype=np.float32))
    family_map = {name: i for i, name in enumerate(checkpoint["families"])}
    families = torch.tensor(sample.target_assay_family.map(family_map).to_numpy(dtype=int))
    arguments = [tensor(drugs[drug_indices]), tensor((targets[target_indices] - norm["target_mean"]) / norm["target_std"]), families]
    target_aux = tensor((aux[target_indices] - norm["target_aux_mean"]) / norm["target_aux_std"])
    raw_structure = structure[indices].copy()
    shifted = raw_structure.copy()
    shifted[:, columns.index("pocket_center_x")] += 100.0
    def predict(raw):
        with torch.inference_mode():
            return model(*arguments, structure=tensor((raw - norm["structure_mean"]) / norm["structure_std"]), structure_mask=torch.ones(len(indices)), target_aux=target_aux)["final_logit"].numpy()
    delta = predict(shifted) - predict(raw_structure)
    result["absolute_coordinate_probe"] = {
        "checkpoint": str((run / "FULL_FIT_MODEL_COMPREHENSIVE_BALANCED_V2.pt").relative_to(ROOT)),
        "pairs": len(sample), "perturbation": "Translate each receptor 100 angstrom along x; all other chemistry and geometry descriptors held fixed.",
        "absolute_logit_change_mean": float(np.abs(delta).mean()),
        "absolute_logit_change_max": float(np.abs(delta).max()),
        "pairs_logit_change_above_1e6_inverse": int((np.abs(delta) > 1e-6).sum()),
        "note": "Tests coordinate invariance of existing scores; does not estimate performance loss.",
    }
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    left = generator.GetFingerprintAsNumPy(Chem.MolFromSmiles("C[C@H](O)C(=O)O"))
    right = generator.GetFingerprintAsNumPy(Chem.MolFromSmiles("C[C@@H](O)C(=O)O"))
    result["default_morgan_chirality_example"] = {"enantiomer_fingerprints_equal": bool(np.array_equal(left, right)), "note": "Configuration matches comprehensive builder; this is a representation limitation, not evidence of a particular candidate failure."}
    (OUT / "SCORE_AND_STRUCTURE_PROBES.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
