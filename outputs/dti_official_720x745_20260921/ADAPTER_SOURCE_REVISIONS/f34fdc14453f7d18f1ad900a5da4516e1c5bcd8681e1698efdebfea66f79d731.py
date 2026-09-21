#!/usr/bin/env python3
"""Official ConPLex checkpoint, exact cached sequence features, both directions."""
import sys
import time
import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from dti_official_runtime_20260921 import ROOT, inputs, sha, contract, status, save_scores, historical, dump, directory


def main():
    start = time.time()
    model_name = 'ConPLex'
    status(model_name, 'RUNNING')
    sys.path.insert(0, str(ROOT / 'third_party/ConPLex'))
    from conplex_dti.model.architectures import SimpleCoembeddingNoSigmoid
    from conplex_dti.featurizer.molecule import MorganFeaturizer
    torch.set_num_threads(4)
    RDLogger.DisableLog('rdApp.warning')
    drugs, targets = inputs()
    feature_dir = ROOT / 'outputs/retrain_20260901/target_registry_745_feature_store_v1'
    index = pd.read_csv(feature_dir / 'TARGET_REGISTRY_745_FEATURE_INDEX_V1.csv.gz')
    match = targets.merge(index[['target_chembl_id', 'sequence', 'target_feature_index']],
                          left_on='target_id', right_on='target_chembl_id', validate='one_to_one', suffixes=('', '_cache'))
    assert len(match) == 745 and match.sequence.eq(match.sequence_cache).all()
    target_path = feature_dir / 'TARGET_REGISTRY_745_PROTBERT1024_FLOAT32_V1.npy'
    protein = torch.tensor(np.load(target_path)[match.target_feature_index.to_numpy()], dtype=torch.float32)
    feat = MorganFeaturizer()
    drug = torch.stack([feat._transform(s) for s in drugs.smiles])
    assert drug.shape == (720, 2048) and drug.sum(1).gt(0).all()
    checkpoint = ROOT / 'third_party/ConPLex/models/BindingDB_ExperimentalValidModel.pt'
    model = SimpleCoembeddingNoSigmoid(2048, 1024, 1024).eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    with torch.inference_mode():
        # Same official forward, factorized only to avoid recomputing entity projections.
        dp = model.drug_projector(drug)
        tp = model.target_projector(protein)
        scores = torch.stack([model.activator(dp, row.expand_as(dp)) for row in tp], dim=1).numpy()
        di = np.arange(128) * 5 % 720
        ti = np.arange(128) * 13 % 745
        direct = model(drug[di], protein[ti]).numpy()
    delta = float(np.max(np.abs(direct - scores[di, ti])))
    assert delta < 1e-6, delta
    frame = pd.DataFrame({'drug_id': np.repeat(drugs.drug_id, 745),
                          'target_id': np.tile(match.target_id, 720), 'score': scores.reshape(-1)})
    replay = frame.merge(historical(model_name), on=['drug_id', 'target_id'], suffixes=('', '_old'))
    historical_delta = float((replay.score - replay.score_old).abs().max())
    # The historical catalogue used a different device/batching path. All current
    # scores are recomputed uniformly; keep this cross-runtime audit separate from
    # the stricter same-runtime native forward equivalence check above.
    assert len(replay) == 276480 and historical_delta < 1e-5, historical_delta
    contract(model_name, checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha(checkpoint),
             source='third_party/ConPLex/conplex_dti/cli/predict.py', adapter_sha256=sha(__file__),
             head='official cosine of ReLU coembeddings', precision='float32_cpu',
             preprocessing='Native Morgan radius2 2048 + exact sequence cached ProtBert residue mean, first1022 residues',
             target_features_sha256=sha(target_path), scores_are_binding_probability=False)
    dump(directory(model_name) / 'REPLAY_CHECK.json', dict(native_forward_pairs=128, native_forward_max_abs_delta=delta,
         historical_pairs=len(replay), historical_max_abs_delta=historical_delta, status='PASS'))
    save_scores(model_name, frame)
    status(model_name, 'COMPLETE', scored_pairs=len(frame), elapsed_seconds=time.time() - start)
    print('ConPLex COMPLETE', len(frame), 'replay max delta', historical_delta, flush=True)


if __name__ == '__main__':
    main()
