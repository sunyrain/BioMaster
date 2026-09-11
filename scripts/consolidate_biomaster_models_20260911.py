#!/usr/bin/env python3
"""Consolidate frozen research fits and recommend existing weights by validation.

No fitting, ensemble construction, production replacement or SPR reselection.
The existing test panels are retrospective diagnostics, not fresh confirmation.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biomaster.endpoint_multitask import AuxiliaryInteraction

OUT = ROOT / 'outputs/biomaster_model_consolidation_20260911'
RUNS = {
    'fixed_6000': ROOT / 'outputs/biomaster_endpoint_ablation_20260911',
    'multitask': ROOT / 'outputs/biomaster_endpoint_multitask_20260911',
    'assay_aware': ROOT / 'outputs/biomaster_assay_aware_20260911',
}
OLD = ROOT / 'outputs/biomaster_old_production_comparison_20260911'
KD = 'AFFINITY_KD_KI'
KEYS = ['panel', 'pair_id']
ROLES = {'binding': 'validation_binding_ap', 'ranking': 'validation_balanced_query_ap',
         'activity': 'validation_activity_ap'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def query_summary(frame, scores):
    y = frame.binary_label.to_numpy(int)
    result = {}
    for direction, column in [('target', 'target_id'), ('drug', 'molecule_id')]:
        ap, p5, p20 = [], [], []
        for indices in frame.groupby(column, sort=True).indices.values():
            truth = y[indices]
            if len(indices) < 10 or not 0 < truth.sum() < len(truth):
                continue
            s = scores[indices]
            ap.append(average_precision_score(truth, s))
            order = np.argsort(-s, kind='stable')
            p5.append(truth[order[:5]].mean())
            if len(indices) >= 20:
                p20.append(truth[order[:20]].mean())
        result.update({f'{direction}_queries': len(ap),
                       f'{direction}_macro_ap': float(np.mean(ap)) if ap else np.nan,
                       f'{direction}_p5': float(np.mean(p5)) if p5 else np.nan,
                       f'{direction}_p20': float(np.mean(p20)) if p20 else np.nan})
    result['balanced_query_ap'] = (result['target_macro_ap'] + result['drug_macro_ap']) / 2
    return result


def pair_summary(frame, scores):
    y = frame.binary_label.to_numpy(int)
    both = 0 < y.sum() < len(y)
    return dict(pairs=len(y), positive=int(y.sum()), negative=int(len(y)-y.sum()),
                ap=average_precision_score(y, scores) if both else np.nan,
                auroc=roc_auc_score(y, scores) if both else np.nan)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protected = json.loads((RUNS['assay_aware']/'DATA_MANIFEST.json').read_text())['frozen_inputs']
    for path, expected in protected.items():
        assert digest(ROOT/path) == expected, path
    original = pd.read_parquet(OLD/'TEST_PREDICTIONS.parquet')
    assert not original.duplicated(KEYS).any()
    frames = {}
    input_hashes = {str((OLD/'TEST_PREDICTIONS.parquet').relative_to(ROOT)): digest(OLD/'TEST_PREDICTIONS.parquet')}
    for stage, folder in RUNS.items():
        frame = pd.read_parquet(folder/'TEST_PREDICTIONS.parquet')
        assert frame[KEYS + ['binary_label']].equals(original[KEYS + ['binary_label']]), stage
        frames[stage] = frame
        input_hashes[str((folder/'TEST_PREDICTIONS.parquet').relative_to(ROOT))] = digest(folder/'TEST_PREDICTIONS.parquet')

    ledger, scores, panels = [], {}, []
    kd_mask = original.panel.eq(KD).to_numpy()
    kd = original.loc[kd_mask].reset_index(drop=True)
    for stage, folder in RUNS.items():
        for path in sorted(folder.glob('*seed_*/RESULT.json')):
            r = json.loads(path.read_text())
            input_hashes[str(path.relative_to(ROOT))] = digest(path)
            variant = r.get('variant', 'binary')
            ident = f"{stage}__{r['arm']}__{variant}__{r['seed']}"
            model_path = path.parent/'model.pt'
            assert digest(model_path) == r['checkpoint_sha256'], ident
            prefix = (f"{r['arm']}_{r['seed']}" if stage == 'fixed_6000'
                      else f"{r['arm']}__{variant}__{r['seed']}")
            score = frames[stage][prefix+'_score'].to_numpy(float)
            assert np.isfinite(score).all()
            scores[ident] = score
            row = dict(model_id=ident, stage=stage, arm=r['arm'], variant=variant, seed=r['seed'],
                       status=r['status'], checkpoint=str(model_path.relative_to(ROOT)),
                       checkpoint_sha256=r['checkpoint_sha256'], optimizer_steps=r['optimizer_steps'],
                       best_step=r.get('best_step'), train_pairs=337570 if r['arm']=='kdki_inactive' else 1116270,
                       validation_binding_ap=np.nan, validation_balanced_query_ap=np.nan, validation_activity_ap=np.nan,
                       **pair_summary(kd, score[kd_mask]), **query_summary(kd, score[kd_mask]))
            if 'best_validation' in r:
                v = r['best_validation']
                row.update(validation_binding_ap=v['panels'][KD]['ap'],
                           validation_balanced_query_ap=v['selection_score'],
                           validation_activity_ap=(v['panels']['ACTIVITY_IC50']['ap']+v['panels']['ACTIVITY_EC50']['ap'])/2)
                assert r['status'] == 'CONVERGED_VALIDATION_PLATEAU'
            ledger.append(row)
            for panel in [KD, 'ACTIVITY_IC50', 'ACTIVITY_EC50', 'ALL_ENDPOINT_UNION']:
                use = original.panel.eq(panel).to_numpy()
                panels.append(dict(model_id=ident, stage=stage, arm=r['arm'], variant=variant, seed=r['seed'],
                                   panel=panel, **pair_summary(original.loc[use], score[use])))
        print(f'Consolidated {stage}', flush=True)
    assert len(ledger) == 60
    legacy_id = 'legacy_production'
    scores[legacy_id] = original.old_production_score.to_numpy(float)
    legacy = dict(model_id=legacy_id, stage='legacy', arm='legacy', variant='mean_logits', seed=0,
                  status='EXISTING_FULL_FIT', train_pairs=383638,
                  checkpoint='outputs/biomaster_best_model_20260906/retargetmap_selected_v1/model.pt',
                  **pair_summary(kd, scores[legacy_id][kd_mask]),
                  **query_summary(kd, scores[legacy_id][kd_mask]))
    legacy['checkpoint_sha256'] = digest(ROOT/legacy['checkpoint'])
    ledger.append(legacy)
    for panel in [KD, 'ACTIVITY_IC50', 'ACTIVITY_EC50', 'ALL_ENDPOINT_UNION']:
        use = original.panel.eq(panel).to_numpy()
        panels.append(dict(model_id=legacy_id, stage='legacy', arm='legacy', variant='mean_logits', seed=0,
                           panel=panel, **pair_summary(original.loc[use], scores[legacy_id][use])))
    per_fit = pd.DataFrame(ledger)
    per_fit.to_csv(OUT/'MODEL_LEDGER.csv', index=False)
    metrics = ['ap', 'auroc', 'target_macro_ap', 'drug_macro_ap', 'balanced_query_ap',
               'target_p5', 'drug_p5', 'target_p20', 'drug_p20', *ROLES.values()]
    groups = ['stage', 'arm', 'variant']
    means = per_fit.groupby(groups)[metrics].agg(['mean', 'std'])
    means.columns = ['_'.join(c) for c in means.columns]
    means['models'] = per_fit.groupby(groups).size()
    means.reset_index().to_csv(OUT/'MODEL_FAMILY_COMPARISON.csv', index=False)
    endpoint_metrics = pd.DataFrame(panels)
    endpoint_metrics.to_csv(OUT/'ENDPOINT_METRICS_PER_MODEL.csv', index=False)
    endpoint_metrics.groupby(groups+['panel'])[['pairs', 'positive', 'negative', 'ap', 'auroc']].mean().reset_index().to_csv(OUT/'ENDPOINT_FAMILY_COMPARISON.csv', index=False)

    # Use validation only for role recommendations, first family mean then member.
    # These role policies are a present synthesis, not a newly preregistered trial.
    converged = per_fit[per_fit.status.eq('CONVERGED_VALIDATION_PLATEAU')]
    assert len(converged) == 54
    recommendations = {}
    for role, metric in ROLES.items():
        family_mean = converged.groupby(groups)[metric].mean()
        winner = family_mean.idxmax()
        members = converged.loc[(converged[groups] == pd.Series(winner, index=groups)).all(axis=1)]
        best = members.sort_values([metric, 'seed'], ascending=[False, True]).iloc[0]
        recommendations[role] = dict(model_id=best.model_id, stage=best.stage, arm=best.arm, variant=best.variant,
            seed=int(best.seed), checkpoint=best.checkpoint, checkpoint_sha256=best.checkpoint_sha256,
            calibration=str((Path(best.checkpoint).parent/'CALIBRATION.json')),
            validation_metric=metric, family_validation_mean=float(family_mean.loc[winner]),
            representative_validation_value=float(best[metric]),
            test_metric_panel=KD,
            representative_test={c:float(best[c]) for c in metrics if not c.startswith('validation_')},
            family_test_mean={c:float(members[c].mean()) for c in metrics if not c.startswith('validation_')},
            family_member_ids=members.model_id.tolist(), score='mean of the two binary logits',
            auxiliary_regression_trained=best.variant in ['regression','joint'], deployed=False)
        cal = json.loads((ROOT/recommendations[role]['calibration']).read_text())
        recommendations[role]['calibration_fit_panel'] = cal['fit_panel']
        recommendations[role]['representative_endpoint_test'] = {
            row.panel:dict(pairs=int(row.pairs),positive=int(row.positive),negative=int(row.negative),
                           ap=float(row.ap),auroc=float(row.auroc))
            for row in endpoint_metrics[endpoint_metrics.model_id.eq(best.model_id)].itertuples()}

    write_json('RECOMMENDED_MODELS.json', dict(status='RESEARCH_RECOMMENDATIONS_EXISTING_WEIGHTS',
        primary_role_for_current_SPR='binding', roles=recommendations,
        selection='Among 18 validation-converged families: highest three-seed mean validation metric for the use case, then highest validation member within that family; no ensemble.',
        policy_disclosure='Role-specific synthesis performed after earlier test reports were viewed; no claim of a fresh independent selection experiment. Existing checkpoints were originally selected by validation balanced query AP.',
        legacy_scope='Legacy and fixed-budget models are retained in comparisons, not included in the 54-converged-fit family selection.',
        production_replaced=False, wetlab_reselected=False))

    # Apply identical cohort membership to the three recommended families + legacy.
    flags = pd.read_parquet(RUNS['assay_aware']/'STRICT_SOURCE_ASSAY_FLAGS.parquet')
    strict = flags[flags.panel.eq(KD) & flags.document_disjoint &
                   flags.kdki_inactive_source_assay_disjoint & flags.all_inactive_source_assay_disjoint]
    target_table = pd.read_csv(ROOT/'outputs/biomaster_model_decision_audit_20260911/WETLAB112_TARGET_BENCHMARK.csv')
    wetlab_targets = set(target_table.sequence_target_id.dropna())
    masks = dict(all=np.ones(len(kd), bool), old_training_pair_unseen=~kd.old_train_pair_seen.to_numpy(bool),
                 document_disjoint=kd.document_disjoint.to_numpy(bool),
                 raw_assay_and_document_disjoint=kd.pair_id.isin(set(strict.pair_id)).to_numpy(),
                 neither_arm_target_supervised=kd.target_training_coverage.eq('neither').to_numpy(),
                 wetlab112_old_pair_unseen=(kd.target_id.isin(wetlab_targets)&~kd.old_train_pair_seen).to_numpy())
    assert masks['raw_assay_and_document_disjoint'].sum() == 488
    selected_ids = sorted({m for r in recommendations.values() for m in r['family_member_ids']} | {legacy_id})
    strata = []
    for ident in selected_ids:
        row = per_fit[per_fit.model_id.eq(ident)].iloc[0]
        s = scores[ident][kd_mask]
        for scope, mask in masks.items():
            f = kd.loc[mask].reset_index(drop=True)
            strata.append(dict(model_id=ident, stage=row.stage, arm=row.arm, variant=row.variant, seed=row.seed,
                               scope=scope, **pair_summary(f,s[mask]), **query_summary(f,s[mask])))
    strata = pd.DataFrame(strata)
    strata.to_csv(OUT/'FINALIST_COHORTS_PER_MODEL.csv', index=False)
    strata.groupby(groups+['scope'])[['pairs','positive','negative','target_queries','drug_queries',
                                     'ap','auroc','target_macro_ap','drug_macro_ap','balanced_query_ap','target_p5','drug_p5']].mean().reset_index().to_csv(OUT/'FINALIST_COHORTS_FAMILY.csv', index=False)
    # Keep legacy deployed directional scores separate from the harmonized mean.
    directional = []
    for scope, mask in masks.items():
        q = query_summary(kd.loc[mask].reset_index(drop=True), kd.old_production_head0.to_numpy(float)[mask])
        t = query_summary(kd.loc[mask].reset_index(drop=True), kd.old_production_head1.to_numpy(float)[mask])
        directional.append(dict(scope=scope, drug_macro_ap=q['drug_macro_ap'], drug_p5=q['drug_p5'],
                                target_macro_ap=t['target_macro_ap'], target_p5=t['target_p5'],
                                balanced_query_ap=(q['drug_macro_ap']+t['target_macro_ap'])/2))
    pd.DataFrame(directional).to_csv(OUT/'LEGACY_DIRECTIONAL_DIAGNOSTICS.csv', index=False)

    # Paired query uncertainty for the two A families, conditional on their seeds.
    intervals = []
    y = kd.binary_label.to_numpy(int)
    rng = np.random.default_rng(20260911)
    for direction, column in [('target', 'target_id'), ('drug', 'molecule_id')]:
        groups_indices = [ids for ids in kd.groupby(column,sort=True).indices.values()
                          if len(ids)>=10 and 0<y[ids].sum()<len(ids)]
        values = {}
        for role in ['binding', 'ranking']:
            seed_values = []
            for ident in recommendations[role]['family_member_ids']:
                s = scores[ident][kd_mask]
                seed_values.append([[average_precision_score(y[ids],s[ids]),
                    float(y[ids][np.argsort(-s[ids],kind='stable')[:5]].mean())] for ids in groups_indices])
            values[role] = np.mean(seed_values, axis=0)
        delta = values['ranking']-values['binding']
        bootstrap = delta[rng.integers(len(delta),size=(2000,len(delta)))].mean(axis=1)
        for j, metric in enumerate(['macro_ap','p5']):
            intervals.append(dict(direction=direction,metric=metric,queries=len(delta),
                binding_family_value=float(values['binding'][:,j].mean()),
                ranking_family_value=float(values['ranking'][:,j].mean()),
                ranking_minus_binding=float(delta[:,j].mean()),
                ci95_low=float(np.quantile(bootstrap[:,j],.025)),ci95_high=float(np.quantile(bootstrap[:,j],.975)),
                interpretation='Exploratory paired query bootstrap conditional on three seeds; no multiplicity correction or independent confirmation.'))
    pd.DataFrame(intervals).to_csv(OUT/'PAIRED_FINALIST_QUERY_INTERVALS.csv',index=False)

    # Reload selected weights and reproduce one complete BF16 evaluation batch.
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    test = pd.read_parquet(RUNS['fixed_6000']/'COMMON_TEST.parquet').iloc[:4096]
    assert test[KEYS].reset_index(drop=True).equals(original.iloc[:4096][KEYS])
    feature_dir = RUNS['fixed_6000']/'features'
    d, t = test.drug_feature_index.to_numpy(int), test.target_feature_index.to_numpy(int)
    def feature(name, indices):
        return torch.tensor(np.load(feature_dir/name, mmap_mode='r')[indices], device='cuda').float()
    batch = dict(drug_global=torch.cat([feature('DRUG_CLIP.npy',d),feature('MORGAN.npy',d)],-1),
                 drug_graph_mean=feature('GRAPH.npy',d), pretrained_available=feature('AVAILABLE.npy',d),
                 target_global=feature('TARGET.npy',t))
    inference_checks = []
    for role, r in recommendations.items():
        state = torch.load(ROOT/r['checkpoint'], map_location='cpu', weights_only=True)
        model = AuxiliaryInteraction(state['config'], state['endpoints']).cuda().eval()
        model.load_state_dict(state['model'], strict=True)
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
            logits, _ = model(batch)
        actual = logits.float().mean(1).cpu().numpy()
        error = float(np.max(np.abs(actual-scores[r['model_id']][:4096])))
        assert error == 0., (role,error)
        inference_checks.append(dict(role=role,model_id=r['model_id'],rows=4096,max_abs_error=error,
                                     checkpoint_sha256=r['checkpoint_sha256']))
        del model
    for path, expected in protected.items():
        assert digest(ROOT/path) == expected, path
    write_json('VERIFICATION.json',dict(status='PASS',completed_utc=datetime.now(timezone.utc).isoformat(),
        research_fits=60,validation_converged_fits=54,legacy_checkpoints=1,
        identical_prediction_panel_keys_and_labels=True,all_61_checkpoint_hashes_checked=True,
        selected_weight_inference=inference_checks,protected_files_unchanged=True,
        input_artifacts_sha256=input_hashes,producer_sha256=digest(Path(__file__)),
        no_training=True,no_new_ensemble=True,production_replaced=False,wetlab_reselected=False))
    print(json.dumps({role:{k:r[k] for k in ['model_id','family_validation_mean','representative_validation_value']}
                      for role,r in recommendations.items()},indent=2),flush=True)


if __name__ == '__main__':
    main()
