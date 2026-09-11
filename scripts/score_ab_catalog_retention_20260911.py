#!/usr/bin/env python3
"""Frozen-model catalog scores and counterfactual SPR384 selections, never release edits."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biomaster.model_registry import build_model
from biomaster.portable_ranker_v2 import CatalogRanker, digest

OUT = ROOT / 'outputs/biomaster_ab_ranking_selection_20260911'
AB = ROOT / 'outputs/biomaster_endpoint_ablation_20260911'
MATRIX = ROOT / 'outputs/biomaster_matrix_720x890_20260910'
BUNDLE = ROOT / 'outputs/biomaster_best_model_20260906/retargetmap_selected_v1'
REVIEW = ROOT / 'outputs/joint384_comprehensive_20260909'
DATA = ROOT / 'data/processed/biomaster_training_full_20260910_v1'
FINAL = ROOT / 'outputs/spr384_final_experiment_table_20260910/SPR384_FINAL_DETAILED.csv'
SEEDS = [20260921, 20260922, 20260923]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def ranks(score, axis):
    order = np.argsort(-score, axis=axis, kind='stable')
    return np.argsort(order, axis=axis, kind='stable') + 1


@torch.inference_mode()
def score_matrix(model, features):
    nd, nt = len(features['drug_global']), len(features['target_global'])
    result = np.empty((nd * nt, 2), np.float32)
    for start in range(0, len(result), 4096):
        index = torch.arange(start, min(start + 4096, len(result)), device='cuda')
        d, t = index // nt, index % nt
        batch = {key: arr[t if key == 'target_global' else d] for key, arr in features.items()}
        result[start:start + len(index)] = model(batch).float().cpu().numpy()
    assert np.isfinite(result).all()
    return result.reshape(nd, nt, 2)


def select_fixed_slots(pool, values, slots):
    """Maximize within-target ranks under frozen target slots and original diversity caps."""
    p = pool[pool.target_chembl_id.isin(slots)].sort_values('pair_id').reset_index(drop=True)
    p['selection_score'] = p.pair_id.map(values)
    assert p.selection_score.notna().all()
    drugs = sorted(p.ligand_inchikey.unique())
    n = len(p)
    di = {d: n + i for i, d in enumerate(drugs)}
    rr, cc, vv, low, high = [], [], [], [], []
    def add(indices, weights, lo, hi):
        row = len(low)
        low.append(lo); high.append(hi)
        rr.extend([row] * len(indices)); cc.extend(indices); vv.extend(weights)
    for target, quota in slots.items():
        ix = p.index[p.target_chembl_id.eq(target)].tolist()
        add(ix, [1] * len(ix), quota, quota)
    for drug, g in p.groupby('ligand_inchikey'):
        ix = g.index.tolist()
        add(ix + [di[drug]], [1] * len(ix) + [-4], -np.inf, 0)
        add(ix + [di[drug]], [1] * len(ix) + [-1], 0, np.inf)
    for _, g in p.groupby('connectivity_key'):
        add(g.index.tolist(), [1] * len(g), 0, 4)
    add(list(di.values()), [1] * len(di), 128, np.inf)
    # Ranking percentiles are comparable within a target; target quotas are fixed.
    utility = p.groupby('target_chembl_id').selection_score.rank(method='first', pct=True)
    c = np.r_[-utility.to_numpy(), np.zeros(len(drugs))]
    a = coo_matrix((vv, (rr, cc)), shape=(len(low), n + len(drugs))).tocsc()
    res = milp(c, integrality=np.ones(len(c)), bounds=Bounds(np.zeros(len(c)), np.ones(len(c))),
               constraints=LinearConstraint(a, low, high), options={'time_limit':60, 'mip_rel_gap':0.00001})
    meta = dict(status=int(res.status), message=res.message,
                mip_gap=None if getattr(res, 'mip_gap', None) is None else float(res.mip_gap))
    if res.x is None:
        return None, meta
    x = np.rint(res.x)
    actual = a @ x
    assert np.all(actual >= np.asarray(low) - 1e-6) and np.all(actual <= np.asarray(high) + 1e-6)
    chosen = p.iloc[np.flatnonzero(x[:n])].copy()
    assert len(chosen) == 384 and chosen.pair_id.is_unique
    assert chosen.groupby('target_chembl_id').size().to_dict() == slots
    return chosen, meta


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths = [FINAL, REVIEW / 'RECOMMENDED_CANDIDATES_384.csv', REVIEW / 'FULL574_COMPREHENSIVE_AUDIT.csv',
             MATRIX / 'DRUG_INDEX.csv', MATRIX / 'TARGET_INDEX.csv.gz', MATRIX / 'TARGET_GLOBAL.npy',
             BUNDLE / 'model.pt', AB / 'PROTOCOL.json',
             ROOT / 'outputs/joint_screen_720x384_20260909/FULL_276480_PAIR_AUDIT.parquet']
    paths += [AB / f'{arm}_seed_{s}/model.pt' for arm in ['kdki_inactive', 'all_inactive'] for s in SEEDS]
    paths += [AB / f'{arm}_seed_{s}/CALIBRATION.json' for arm in ['kdki_inactive', 'all_inactive'] for s in SEEDS]
    frozen = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    write_json(OUT / 'SELECTION_PROTOCOL.json', dict(
        input_hashes=frozen, inference_scope='720 x 890; selection restricted to historical 720 x 384',
        precision='FP32 for all catalog models, TF32 disabled',
        primary='Fixed 112 targets and exact per-target slots, drug/connectivity cap 4, at least 128 drugs; maximize target-wise rank percentile',
        non_model_gates='Frozen identity, chemistry, known relation, interpretable TxGNN, same disease TxGNN top50 / OT >=0.3, biochemical assay lanes; apply available reviewed holds',
        binding_rank_gate='Do not impose old model top20; its rank is the quantity being replaced. Report new top20 retention separately.',
        evidence_policy='Available exclusion/hold findings retained; no new LLM review and no inference that unreviewed pairs passed review',
        latest_evidence_sensitivity='Additionally exclude all exact molecule+sequence pairs in final September evidence labels, including grey/conflicting records',
        consensus='Mean validation-calibrated probabilities of three seeds, exploratory ensemble; report all individual seeds',
        diagnostic='Global top384 by mean-logit/consensus probability in same eligible core pool, without quota constraints',
        baseline='Frozen 384 discovery pairs, reference controls extra; no wet-lab or website modification'))
    final = pd.read_csv(FINAL)
    assert len(final) == 384 and final.pair_id.is_unique
    finalids = set(final.pair_id)
    reviewed = pd.read_csv(REVIEW / 'FULL574_COMPREHENSIVE_AUDIT.csv')
    pool = pd.read_parquet(paths[8])
    drugs = pd.read_csv(MATRIX / 'DRUG_INDEX.csv')
    targets = pd.read_csv(MATRIX / 'TARGET_INDEX.csv.gz')
    ranker = CatalogRanker(BUNDLE, device='cuda', verify=True)
    assert drugs.drug_id.tolist() == ranker.drugs.drug_id.tolist()
    features = ranker.features
    features['target_global'] = torch.from_numpy(np.load(MATRIX / 'TARGET_GLOBAL.npy')).cuda()
    di = {d: i for i, d in enumerate(drugs.drug_id)}
    ti = {t: i for i, t in enumerate(targets.uniprot_accession)}
    pool['di'] = pool.ligand_inchikey.map(di)
    pool['ti'] = pool.uniprot_accession.map(ti)
    assert pool[['di', 'ti']].notna().all().all()
    corecols = np.sort(pool.ti.unique())
    assert len(corecols) == 384
    coreposition = {i: j for j, i in enumerate(corecols)}
    pool['core_ti'] = pool.ti.map(coreposition)
    matrices = {}
    for arm in ['old_production', 'kdki_inactive', 'all_inactive']:
        for seed in ([0] if arm == 'old_production' else SEEDS):
            name = arm if seed == 0 else f'{arm}_{seed}'
            statepath = BUNDLE / 'model.pt' if seed == 0 else AB / f'{arm}_seed_{seed}/model.pt'
            path = OUT / f'{name}_DIRECTIONAL_LOGITS.npy'
            state = torch.load(statepath, map_location='cpu', weights_only=True)
            model = build_model(state['architecture'], state['config']).cuda().eval()
            model.load_state_dict(state['model'], strict=True)
            matrix = score_matrix(model, features)
            np.save(path, matrix)
            matrices[name] = matrix
            del model
            print('Scored', name, matrix.shape, flush=True)
    # Compare the unchanged production model to its existing complete catalog matrix.
    olddiff = float(np.max(np.abs(matrices['old_production'] - np.load(MATRIX / 'DIRECTIONAL_LOGITS.npy'))))
    assert olddiff < 0.001, olddiff
    catalog = pd.DataFrame(dict(ligand_inchikey=np.repeat(drugs.drug_id, len(targets)),
                                uniprot_accession=np.tile(targets.uniprot_accession, len(drugs))))
    scoremaps, targetmaps = {}, {}
    for name, matrix in matrices.items():
        score = matrix.mean(axis=2)
        targetscore = matrix[:, :, 1] if name == 'old_production' else score
        drugscore = matrix[:, :, 0] if name == 'old_production' else score
        scoremaps[name], targetmaps[name] = score, targetscore
        catalog[name + '_score'] = score.ravel()
        pool[name + '_score'] = score[pool.di, pool.ti]
        pool[name + '_target_score'] = targetscore[pool.di, pool.ti]
        pool[name + '_drug_rank384'] = ranks(drugscore[:, corecols], 1)[pool.di, pool.core_ti]
        pool[name + '_drug_rank890'] = ranks(drugscore, 1)[pool.di, pool.ti]
        pool[name + '_target_rank720'] = ranks(targetscore, 0)[pool.di, pool.ti]
    for arm in ['kdki_inactive', 'all_inactive']:
        prob = []
        for seed in SEEDS:
            cal = json.loads((AB / f'{arm}_seed_{seed}/CALIBRATION.json').read_text())
            prob.append(expit(cal['slope'] * scoremaps[f'{arm}_{seed}'] + cal['intercept']))
        s = np.mean(prob, axis=0)
        name = arm + '_consensus'
        scoremaps[name] = targetmaps[name] = s
        catalog[name + '_score'] = s.ravel()
        pool[name + '_score'] = pool[name + '_target_score'] = s[pool.di, pool.ti]
        pool[name + '_drug_rank384'] = ranks(s[:, corecols], 1)[pool.di, pool.core_ti]
        pool[name + '_drug_rank890'] = ranks(s, 1)[pool.di, pool.ti]
        pool[name + '_target_rank720'] = ranks(s, 0)[pool.di, pool.ti]
    catalog.to_parquet(OUT / 'CATALOG_720x890_SCORES.parquet', index=False, compression='zstd')
    # Preserve known review exclusions, including molecular-identity findings across targets.
    allowed = reviewed.set_index('pair_id').eligible_after_comprehensive_review
    pool['reviewed_in_original574'] = pool.pair_id.isin(allowed.index)
    pool['known_review_hold'] = pool.pair_id.map(allowed).eq(False)
    identity_hold = set(reviewed.loc[reviewed.new_identity_hold, 'ligand_inchikey'])
    pool['known_identity_hold'] = pool.ligand_inchikey.isin(identity_hold)
    lanes = ['ENZYME_BIOCHEMICAL', 'KINASE_BIOCHEMICAL', 'NUCLEAR_EPIGENETIC_DOMAIN']
    pool['counterfactual_eligible'] = (pool.base_eligible & pool.joint_r50_ot03.eq(True) &
                                      pool.assay_lane.isin(lanes) & ~pool.known_review_hold & ~pool.known_identity_hold)
    assert set(pool.loc[pool.counterfactual_eligible, 'pair_id']) >= finalids
    pool['sequence_target_id'] = 'SEQ:' + pool.uniprot_accession.map(targets.set_index('uniprot_accession').sequence_sha256)
    pool['training_pair_id'] = pool.ligand_inchikey + '__' + pool.sequence_target_id
    labels = pd.read_parquet(DATA / 'ALL_PAIR_TASK_LABELS.parquet', columns=['pair_id', 'connectivity_pair'])
    pool['latest_exact_evidence'] = pool.training_pair_id.isin(set(labels.pair_id))
    pool['latest_connectivity_evidence'] = (pool.connectivity_key + '__' + pool.sequence_target_id).isin(set(labels.connectivity_pair))
    for arm in ['kdki_inactive', 'all_inactive']:
        train = pd.read_parquet(AB / f'{arm}_TRAIN.parquet', columns=['pair_id'])
        pool[arm + '_training_seen'] = pool.training_pair_id.isin(set(train.pair_id))
    assert not pool.loc[pool.pair_id.isin(finalids), ['kdki_inactive_training_seen', 'all_inactive_training_seen']].any().any()
    pool['in_frozen384'] = pool.pair_id.isin(finalids)
    pool.to_parquet(OUT / 'CORE276480_SELECTION_AUDIT.parquet', index=False, compression='zstd')
    slots = final.groupby('新靶点ChEMBL编号').size().to_dict()
    eligible = pool[pool.counterfactual_eligible].copy()
    results, choices = [], []
    finalout = final[['排序', '原候选编号', '小分子药物名称', '新靶点名称', '新靶点基因', '当前实验建议', 'pair_id']].merge(
        pool.drop(columns=['drug_names', 'gene_symbol']), on='pair_id', validate='one_to_one')
    for name in scoremaps:
        modelvalues = pool.set_index('pair_id')[name + '_target_score']
        for scenario, candidates in [('fixed112_frozen_evidence', eligible),
                                      ('fixed112_latest_exact_excluded', eligible[~eligible.latest_exact_evidence])]:
            # The evidence sensitivity is run for the old model and the two consensus models.
            if scenario.endswith('excluded') and name not in ['old_production', 'kdki_inactive_consensus', 'all_inactive_consensus']:
                continue
            selected, solver = select_fixed_slots(candidates, modelvalues, slots)
            if selected is None:
                results.append(dict(model=name, scenario=scenario, candidate_pool=len(candidates), selected=0, **solver))
                continue
            selected['model'] = name; selected['scenario'] = scenario
            ids = set(selected.pair_id)
            finalout[name + '__' + scenario + '__retained'] = finalout.pair_id.isin(ids)
            results.append(dict(model=name, scenario=scenario, candidate_pool=len(candidates), selected=len(selected),
                                retained=len(ids & finalids), retained_percent=100 * len(ids & finalids) / 384,
                                drugs=selected.ligand_inchikey.nunique(), targets=selected.target_chembl_id.nunique(),
                                original_reviewed=int(selected.reviewed_in_original574.sum()),
                                latest_exact_evidence=int(selected.latest_exact_evidence.sum()),
                                a_train_seen=int(selected.kdki_inactive_training_seen.sum()),
                                b_train_seen=int(selected.all_inactive_training_seen.sum()), **solver))
            choices.append(selected)
            print(name, scenario, 'retained', len(ids & finalids), 'solver', solver, flush=True)
        selected = eligible.sort_values([name + '_score', 'pair_id'], ascending=[False, True]).head(384).copy()
        ids = set(selected.pair_id)
        scenario = 'global_top384_frozen_evidence'
        selected['model'] = name; selected['scenario'] = scenario
        selected['selection_score'] = selected[name + '_score']
        choices.append(selected)
        finalout[name + '__' + scenario + '__retained'] = finalout.pair_id.isin(ids)
        results.append(dict(model=name, scenario=scenario, candidate_pool=len(eligible), selected=384,
                            retained=len(ids & finalids), retained_percent=100 * len(ids & finalids) / 384,
                            drugs=selected.ligand_inchikey.nunique(), targets=selected.target_chembl_id.nunique(),
                            original_reviewed=int(selected.reviewed_in_original574.sum()),
                            latest_exact_evidence=int(selected.latest_exact_evidence.sum()),
                            a_train_seen=int(selected.kdki_inactive_training_seen.sum()),
                            b_train_seen=int(selected.all_inactive_training_seen.sum())))
    pd.DataFrame(results).to_csv(OUT / 'RETENTION_SUMMARY.csv', index=False)
    choicecols = ['model', 'scenario', 'pair_id', 'ligand_inchikey', 'drug_names', 'gene_symbol', 'target_chembl_id',
                  'selection_score', 'in_frozen384', 'reviewed_in_original574', 'latest_exact_evidence',
                  'kdki_inactive_training_seen', 'all_inactive_training_seen']
    pd.concat(choices)[choicecols].to_csv(OUT / 'COUNTERFACTUAL_SELECTIONS.csv', index=False, encoding='utf-8-sig')
    display = ['排序', '原候选编号', '小分子药物名称', '新靶点名称', '新靶点基因', '当前实验建议', 'pair_id',
               'binding_rank_384', 'latest_exact_evidence', 'latest_connectivity_evidence']
    display += [c for c in finalout if any(c.startswith(n + '_') for n in scoremaps)]
    finalout[list(dict.fromkeys(display))].to_csv(OUT / 'SPR384_MODEL_RANK_AND_RETENTION.csv', index=False, encoding='utf-8-sig')
    rankcounts = []
    for name in scoremaps:
        rankcounts.append(dict(model=name, drug_top10_core384=int(finalout[name + '_drug_rank384'].le(10).sum()),
                               drug_top20_core384=int(finalout[name + '_drug_rank384'].le(20).sum()),
                               drug_top20_full890=int(finalout[name + '_drug_rank890'].le(20).sum()),
                               target_top20_drugs720=int(finalout[name + '_target_rank720'].le(20).sum())))
    pd.DataFrame(rankcounts).to_csv(OUT / 'SPR384_RANK_BAND_COUNTS.csv', index=False)
    assert all(digest(ROOT / p) == h for p, h in frozen.items())
    write_json(OUT / 'SELECTION_VERIFICATION.json', dict(all_pass=True, full_scoring_shape=[720,890,2], models=7,
               consensus_models=2, frozen_inputs_unchanged=True, old_matrix_max_abs_difference=olddiff,
               same_frozen384_present_and_not_in_ab_training=True, exact_target_slots_and_diversity_asserted=True,
               eligible_core_pairs=len(eligible), eligible_fixed_target_pairs=int(eligible.target_chembl_id.isin(slots).sum()),
               original384_latest_exact_evidence=int(finalout.latest_exact_evidence.sum()),
               original384_latest_connectivity_evidence=int(finalout.latest_connectivity_evidence.sum()),
               original384_old_production_rank_matches=int((finalout.binding_rank_384 == finalout.old_production_drug_rank384).sum()),
               source_sha256=digest(Path(__file__))))
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
