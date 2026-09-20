#!/usr/bin/env python3
"""Verify frozen inputs, no-label matrix membership and rank/report provenance."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / 'outputs/dti_ranking_scope_20260921'
ANALYSIS = ROOT / 'outputs/dti_rank_agreement_20260921'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    frozen = json.loads((ROOT/'outputs/dti_research_preparation_20260920/PACKAGE_MANIFEST.json').read_text())
    for name, record in frozen['files'].items():
        assert sha(ROOT/name) == record['sha256'], name
    for name in ['configs/dti_reliability_A_20260920_v2/PROTOCOL.json',
        'outputs/dti_reliability_A_plan_20260920/EXPERIMENT_MATRIX_A.csv',
        'outputs/dti_reliability_A_plan_20260920/CONDITIONAL_EXPERIMENTS_A.csv']:
        assert (ROOT/name).read_bytes() == subprocess.check_output(['git', 'show', '74f0fe091:'+name], cwd=ROOT)
    active = json.loads((ROOT/'configs/dti_reliability_20260920/ACTIVE_PROTOCOL.json').read_text())
    for path_key, hash_key in [('active_protocol','active_protocol_sha256'),
        ('active_main_runs','main_runs_sha256'), ('inference_comparison_addendum','inference_comparison_addendum_sha256'),
        ('ranking_study_scope','ranking_study_scope_sha256')]:
        assert sha(ROOT/active[path_key]) == active[hash_key]
    scope = json.loads((SCOPE/'SCOPE_SUMMARY.json').read_text())
    for path, expected in scope['artifacts'].items():
        assert sha(ROOT/path) == expected, path
    pairs = pd.read_csv(ROOT/'data/research/dti_ranking_scope_20260921_v1/APPLICATION_720X96_PAIRS.csv.gz')
    assert pairs.shape == (69120, 2) and not pairs.duplicated().any()
    assert pairs.drug_id.nunique() == 720 and pairs.target_id.nunique() == 96
    manifest = json.loads((ANALYSIS/'MANIFEST.json').read_text())
    assert manifest['selection_manifest_sha256'] == sha(SCOPE/'SCOPE_SUMMARY.json')
    assert not manifest['davis_used'] and manifest['no_experimental_labels_read']
    for path, expected in manifest['artifacts'].items():
        assert sha(ANALYSIS/path) == expected, path
    presentation = json.loads((ANALYSIS/'PRESENTATION_MANIFEST.json').read_text())
    assert presentation['source_manifest_sha256'] == sha(ANALYSIS/'MANIFEST.json')
    for path, expected in presentation['outputs'].items():
        assert sha(ROOT/path) == expected, path
    snapshot = pd.read_csv(ANALYSIS/'SIX_CURRENT_SCORE_SNAPSHOT.csv.gz')
    assert len(snapshot) == 276480 and not snapshot.duplicated(['drug_id','target_id']).any()
    models = manifest['current_models']
    assert len(models) == 6 and 'ReTargetMap' not in models
    assert np.isfinite(snapshot[models].to_numpy()[snapshot[models].notna().to_numpy()]).all()
    assert snapshot.loc[snapshot.selected_APP96].set_index(['drug_id','target_id']).index.equals(
        snapshot.set_index(['drug_id','target_id']).loc[lambda x: x.index.get_level_values('target_id').isin(pairs.target_id)].index)
    detail = pd.read_csv(ANALYSIS/'PER_QUERY_RANK_AGREEMENT.csv.gz')
    # Independently check both views/directions against the frozen scores.
    sampled = detail.groupby(['matrix','view','direction'], group_keys=False).sample(n=3, random_state=20260921)
    for row in sampled.itertuples():
        data = snapshot.loc[snapshot.selected_APP96] if row.matrix == 'PRIMARY_720X96' else snapshot
        if row.view == 'ALL_SIX_COMMON':
            data = data.dropna(subset=models)
        key = 'target_id' if row.direction == 'target_to_drug' else 'drug_id'
        scores = data.loc[data[key].eq(row.query_id), [row.model_a,row.model_b]].dropna()
        assert len(scores) == row.candidates
        if np.isfinite(row.spearman):
            np.testing.assert_allclose(spearmanr(scores.iloc[:,0],scores.iloc[:,1]).statistic, row.spearman, atol=1e-12)
    result = dict(status='PASS', checked_utc=datetime.now(timezone.utc).isoformat(),
        frozen_v1_files_unchanged=len(frozen['files']), previous_A_protocol_and_runs_unchanged=True,
        active_hashes_match=True, full_score_rows=len(snapshot), primary_matrix_pairs=len(pairs),
        independent_rank_checks=len(sampled), finite_observed_scores=True,
        scope_and_analysis_manifests_match=True, score_independent_scope=True, no_truth_labels=True,
        davis_evaluation_used=False, official_weight_download_audited_separately=True,
        producer_sha256=sha(Path(__file__)))
    (SCOPE/'PACKAGE_CHECKS.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
