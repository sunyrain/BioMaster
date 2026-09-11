#!/usr/bin/env python3
"""Audit actual SPR scorer provenance, component sensitivity, and convergence evidence."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/biomaster_model_lineage_convergence_20260911'
BASE = ROOT / 'outputs/old_drug_target_sota_v1'
AB = ROOT / 'outputs/biomaster_endpoint_ablation_20260911'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    paths = [BASE/'biomaster_odti_deployment_v1/BIOMASTER_ODTI_720X384_SCORES_V1.csv.gz',
             BASE/'drug_centric_ranker_v1/BIOMASTER_DRUG_TO_TARGET_720X384_V1.csv.gz',
             ROOT/'outputs/spr384_final_experiment_table_20260910/SPR384_FINAL_DETAILED.csv',
             ROOT/'outputs/biomaster_ab_ranking_selection_20260911/SELECTION_LINEAGE.json',
             ROOT/'outputs/biomaster_spr384_ab_rank_explanation_20260911/SPR384_AB_RANKING_EXPLANATION.csv',
             BASE/'public_retrained_v1/dtiam_same_data_compatible_v1/S5_OLD_DRUG_ENTITY_COLD__fold_-1__OFFICIAL_DEFAULT_COMPAT_V1/RUN_SUMMARY_V1.json',
             BASE/'biomaster_odti_deployment_v1/BIOMASTER_ODTI_DEPLOYMENT_SUMMARY_V1.json',
             ROOT/'outputs/biomaster_best_model_20260906/retargetmap_selected_v1/training_data.json',
             ROOT/'scripts/run_endpoint_ablation_20260911.py', AB/'PROTOCOL.json',
             ROOT/'scripts/run_strict_dta_720x338_v1.sh', ROOT/'scripts/run_recovered_dta_720x46_v1.sh',
             ROOT/'third_party/ConPLex/conplex_dti/cli/download.py']
    for seed in range(20260813, 20260818):
        paths.append(BASE/f'biomaster_odti_routed_ranker_v1/S5_OLD_DRUG_ENTITY_COLD__fold_-1__seed_{seed}__CORE/RUN_SUMMARY_V1.json')
    for arm in ['kdki_inactive', 'all_inactive']:
        for seed in [20260921, 20260922, 20260923]:
            paths += [AB/f'{arm}_seed_{seed}/{name}' for name in ['HISTORY.json', 'RESULT.json', 'VALIDATION_METRICS.csv']]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    raw = pd.read_csv(paths[0], low_memory=False)
    legacy = pd.read_csv(paths[1])
    final = pd.read_csv(paths[2])
    assert raw.pairId.equals(legacy.pairId) and raw.pairId.nunique() == 276480
    assert len(final) == final.pair_id.nunique() == 384
    cfg = json.loads(paths[3].read_text())['actual_rank_features']
    z = {name: (raw['biomaster_ensemble_logit' if name == 'biomaster_logit' else name] - cfg['standardization_mean'][name]) / cfg['standardization_scale'][name] for name in cfg['features']}
    n, c, p = z['biomaster_logit'], z['train_positive_max_tanimoto'], z['target_train_prior']
    scores = {'historical_neural_only': n, 'chemical_neighbor_only': c,
              'historical_neural_plus_chemical': n+2.5*c, 'historical_neural_plus_prior': n+.25*p,
              'original_complete_combination': n+2.5*c+.25*p}
    assert np.allclose(scores['original_complete_combination'], legacy.independent_validation_rank_score, rtol=0, atol=1e-10)
    detail = final[['排序', '小分子药物名称', '新靶点名称', '当前实验建议', 'pair_id']].copy()
    stats = []
    for name, score in scores.items():
        ranks = score.groupby(raw.ligand_inchikey).rank(method='first', ascending=False)
        lookup = pd.Series(ranks.to_numpy(), index=raw.pairId)
        detail[name+'_rank384'] = detail.pair_id.map(lookup).astype(int)
        r = detail[name+'_rank384']
        stats.append(dict(scorer=name, original384_top20=int(r.le(20).sum()), median_rank=float(r.median()), mean_rank=float(r.mean())))
    detail.to_csv(OUT/'SPR384_HISTORICAL_COMPONENT_RANKS.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(stats).to_csv(OUT/'HISTORICAL_COMPONENT_SUMMARY.csv', index=False)

    history_rows, lineage_rows = [], []
    for seed in range(20260813, 20260818):
        r = json.loads((BASE/f'biomaster_odti_routed_ranker_v1/S5_OLD_DRUG_ENTITY_COLD__fold_-1__seed_{seed}__CORE/RUN_SUMMARY_V1.json').read_text())
        assert r['split_counts']['train'] == 65276
        lineage_rows.append(dict(model='historical_S5_neural', seed=seed, train_pairs=r['split_counts']['train'],
                                 validation_pairs=r['split_counts']['valid'], test_pairs=r['split_counts']['test'],
                                 best_epoch=r['training']['best_epoch'], epochs_completed=r['training']['epochs_completed'],
                                 objectives=json.dumps(r['objectives'], ensure_ascii=False)))
    pd.DataFrame(lineage_rows).to_csv(OUT/'HISTORICAL_S5_TRAINING.csv', index=False)
    for arm in ['kdki_inactive', 'all_inactive']:
        for seed in [20260921, 20260922, 20260923]:
            run = AB/f'{arm}_seed_{seed}'
            hist = pd.DataFrame(json.loads((run/'HISTORY.json').read_text())).set_index('step')
            r = json.loads((run/'RESULT.json').read_text())
            v = pd.read_csv(run/'VALIDATION_METRICS.csv')
            val = v[v.panel.eq('AFFINITY_KD_KI') & v.scope.eq('all')].iloc[0]
            history_rows.append(dict(model='A' if arm=='kdki_inactive' else 'B', seed=seed, train_pairs=r['pairs'],
                                     steps=r['optimizer_steps'], sampled_rows=r['sampled_rows'], equivalent_passes=r['sampled_rows']/r['pairs'],
                                     training_loss_steps4501_5000=float(hist.loc[5000, 'loss']),
                                     training_loss_steps5501_6000=float(hist.loc[6000, 'loss']),
                                     final_validation_ap=float(val.ap), final_validation_auroc=float(val.auroc),
                                     validation_checkpoints_during_training=0, convergence_certified=False,
                                     stopping_policy='fixed_6000_steps_final_EMA_validation_after_fit'))
    pd.DataFrame(history_rows).to_csv(OUT/'AB_CONVERGENCE_EVIDENCE.csv', index=False)

    # Scheduling review aid, not a new binding assessment or final laboratory release.
    ab = pd.read_csv(paths[4])
    schedule = final[['排序','小分子药物名称','新靶点名称','当前实验建议','需先确认的事项','评价理由','pair_id']].merge(
        ab[['pair_id','A药内排名_384','B药内排名_384','共识Top20组合']], on='pair_id', validate='one_to_one')
    schedule['至少一个AB药内Top20'] = schedule['A药内排名_384'].le(20) | schedule['B药内排名_384'].le(20)
    schedule['安排复核建议_未更改实验表'] = np.select([
        schedule['当前实验建议'].eq('有证据支持降低投入'),
        schedule['当前实验建议'].eq('先解决具体问题'), schedule['至少一个AB药内Top20']],
        ['先核降低投入的原始依据，再决定后置或替换', '分清操作前提与证据不足；逐项确认后安排',
         '优先核对靶点实验就绪；模型支持不替代质控'], default='保留探索；按靶点就绪及代表性分批')
    schedule.to_csv(OUT/'SPR384_SCHEDULING_REVIEW_AID.csv', index=False, encoding='utf-8-sig')
    schedule.groupby(['当前实验建议','共识Top20组合']).size().rename('pairs').reset_index().to_csv(OUT/'SCHEDULING_STRATA.csv', index=False)
    dtiam = json.loads(paths[5].read_text())
    assert dtiam['counts']['train_rows'] == 65276
    summary = dict(status='COMPLETE_AUDIT_NO_RETRAIN_NO_LAB_CHANGE', inputs_unchanged=True, input_sha256=hashes,
                   original_scorer='historical 5-seed S5 neural + training positive neighbor + target prior',
                   original_neural_train_pairs_per_seed=65276,
                   later_neural_train_pairs=json.loads(paths[7].read_text())['rows'],
                   dtiam_train_counts=dtiam['counts'], dtiam_time_limit=dtiam['fit_policy']['this_call']['time_limit_seconds'],
                   historical_component_results=stats,
                   conplex_source='local scripts use official BindingDB_ExperimentalValidModel.pt; no new-data refit on this scoring route',
                   drugclip_source='existing public six-fold pocket-ligand retrieval; A/B reuse frozen molecule features, not full pocket scorer',
                   scheduling_counts=schedule['当前实验建议'].value_counts().to_dict(),
                   exploratory_with_any_AB_top20=int((schedule['当前实验建议'].eq('仍可探索') & schedule['至少一个AB药内Top20']).sum()),
                   caveats=['Component removal explains this frozen score mathematically; it is not new binding validation or a causal allocation of total disagreement.',
                            'No plateau/early-stopping validation curve was collected during A/B training; convergence is unestablished, not proven absent.',
                            'Scheduling aid reuses prior reviews without a new per-pair literature audit; conditional items are not all impossible mechanisms.'],
                   artifacts={p.name:digest(p) for p in OUT.glob('*.csv')}, source_sha256=digest(Path(__file__)))
    assert all(digest(ROOT/p)==sha for p,sha in hashes.items())
    (OUT/'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(pd.DataFrame(stats).to_string(index=False))
    print('exploratory with A/B support', summary['exploratory_with_any_AB_top20'])


if __name__ == '__main__':
    main()
