#!/usr/bin/env python3
"""Agreement of frozen catalog ranks; missing scores are never counted as disagreement."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from biomaster.portable_ranker_v2 import digest
from score_ab_catalog_retention_20260911 import select_fixed_slots

OUT = ROOT / 'outputs/biomaster_ab_external_agreement_20260911'
RANK = ROOT / 'outputs/biomaster_ab_ranking_selection_20260911'
COMPARE = ROOT / 'outputs/biomaster_old_drug_bidirectional_20260906'
EXTRA = ROOT / 'outputs/biomaster_target_queries_20260908/lyve1_slc8a1_hgf/comparators'
FINAL = ROOT / 'outputs/spr384_final_experiment_table_20260910/SPR384_FINAL_DETAILED.csv'


def rank_matrix(scores, axis):
    return np.argsort(np.argsort(-scores, axis=axis, kind='stable'), axis=axis, kind='stable') + 1


def agreement(a, b):
    """Rows are queries, columns are identical candidates in identical order."""
    assert a.shape == b.shape and a.shape[1] >= 20
    assert np.isfinite(a).all() and np.isfinite(b).all()
    ra, rb = rankdata(a, axis=1), rankdata(b, axis=1)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    denom = np.sqrt((ra*ra).sum(axis=1) * (rb*rb).sum(axis=1))
    rho = np.divide((ra*rb).sum(axis=1), denom, out=np.full(len(a),np.nan), where=denom>0)
    hits = ((rank_matrix(a,1)<=20) & (rank_matrix(b,1)<=20)).sum(axis=1)
    return dict(queries=len(a), candidates_per_query=a.shape[1], valid_spearman_queries=int(np.isfinite(rho).sum()),
                mean_spearman=float(np.nanmean(rho)), median_spearman=float(np.nanmedian(rho)),
                mean_top20_intersection=float(hits.mean()), mean_top20_overlap_fraction=float(hits.mean()/20),
                random_independent_top20_overlap_fraction=20/a.shape[1]), rho, hits


def main():
    OUT.mkdir(exist_ok=True)
    paths = [FINAL, RANK/'CORE276480_SELECTION_AUDIT.parquet', COMPARE/'OLD_DRUGS_720.csv',
             COMPARE/'TARGETS_384.csv.gz', COMPARE/'scores/dtiam.npz', COMPARE/'scores/drugclip.npz',
             COMPARE/'DRUGCLIP_COMMON_SCOPE.npz', COMPARE/'scores/drugclip.json', COMPARE/'scores/dtiam.json']
    paths += [EXTRA/f'{gene}_THREE_MODEL_COMPARISON_720.csv' for gene in ['LYVE1','SLC8A1']]
    paths += [RANK/'CATALOG_720x890_SCORES.parquet', ROOT/'outputs/biomaster_matrix_720x890_20260910/TARGET_INDEX.csv.gz']
    paths += [RANK/'old_production_DIRECTIONAL_LOGITS.npy', ROOT/'outputs/biomaster_matrix_720x890_20260910/DRUG_INDEX.csv',
              ROOT/'scripts/score_ab_catalog_retention_20260911.py']
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in paths}
    d = pd.read_csv(COMPARE/'OLD_DRUGS_720.csv')
    t = pd.read_csv(COMPARE/'TARGETS_384.csv.gz')
    final = pd.read_csv(FINAL)
    assert d.ligand_inchikey.is_unique and t.target_chembl_id.is_unique
    # Use score-cache identity axes and require the same sequence identities in the new catalog.
    newtargets = pd.read_csv(ROOT/'outputs/biomaster_matrix_720x890_20260910/TARGET_INDEX.csv.gz')
    joint = t.merge(newtargets[['target_chembl_id','sequence_sha256']], on='target_chembl_id',validate='one_to_one',suffixes=['_old','_new'])
    assert len(joint)==384 and joint.sequence_sha256_old.eq(joint.sequence_sha256_new).all()
    core = pd.read_parquet(RANK/'CORE276480_SELECTION_AUDIT.parquet')
    keyframe = pd.MultiIndex.from_product([d.ligand_inchikey,t.target_chembl_id], names=['ligand_inchikey','target_chembl_id'])
    core = core.set_index(['ligand_inchikey','target_chembl_id']).reindex(keyframe)
    assert len(core)==720*384 and core.pair_id.notna().all()
    def matrix(column):
        return core[column].to_numpy().reshape(720,384)
    scores = {
        'legacy_spr_ranker': {'drug_to_target':matrix('independent_validation_rank_score'),'target_to_drug':matrix('independent_validation_rank_score')},
        'old_neural': {'drug_to_target':2*matrix('old_production_score')-matrix('old_production_target_score'),'target_to_drug':matrix('old_production_target_score')},
        'A_consensus': {direction:matrix('kdki_inactive_consensus_score') for direction in ['drug_to_target','target_to_drug']},
        'B_consensus': {direction:matrix('all_inactive_consensus_score') for direction in ['drug_to_target','target_to_drug']},
    }
    # Reconstructing head0 from the stored average can introduce rounding; use the exact frozen tensor.
    catalogdrugs = pd.read_csv(ROOT/'outputs/biomaster_matrix_720x890_20260910/DRUG_INDEX.csv')
    ix = pd.Index(catalogdrugs.drug_id).get_indexer(d.ligand_inchikey)
    it = pd.Index(newtargets.uniprot_accession).get_indexer(t.uniprot_accession)
    assert (ix>=0).all() and (it>=0).all()
    old = np.load(RANK/'old_production_DIRECTIONAL_LOGITS.npy')[ix][:,it,:]
    scores['old_neural'] = {'drug_to_target':old[:,:,0],'target_to_drug':old[:,:,1]}
    for name in ['dtiam','drugclip']:
        a = np.load(COMPARE/f'scores/{name}.npz')
        scores[name] = {direction:a[direction] for direction in ['drug_to_target','target_to_drug']}
        assert all(v.shape==(720,384) for v in scores[name].values())
    common = np.load(COMPARE/'DRUGCLIP_COMMON_SCOPE.npz')
    assert len(common['drug_indices'])==548 and len(common['target_indices'])==269
    rows, queryrows, supportrows = [], [], []
    details = final[['排序','原候选编号','小分子药物名称','新靶点名称','当前实验建议','pair_id','药物完整InChIKey','新靶点ChEMBL编号']].copy()
    for scope, dids, tids, names in [
        ('full720x384',np.arange(720),np.arange(384),[n for n in scores if n!='drugclip']),
        ('drugclip_common548x269',common['drug_indices'],common['target_indices'],list(scores))]:
        subd, subt = d.iloc[dids], t.iloc[tids]
        dd = pd.Index(subd.ligand_inchikey).get_indexer(final['药物完整InChIKey'])
        tt = pd.Index(subt.target_chembl_id).get_indexer(final['新靶点ChEMBL编号'])
        covered = (dd>=0)&(tt>=0)
        details[scope+'_covered'] = covered
        for direction in ['drug_to_target','target_to_drug']:
            subs = {name:scores[name][direction][np.ix_(dids,tids)] for name in names}
            assert all(np.isfinite(s).all() for s in subs.values())
            for left,right in itertools.combinations(names,2):
                a,b = subs[left],subs[right]
                if direction=='target_to_drug': a,b=a.T,b.T
                result,rho,hits = agreement(a,b)
                rows.append(dict(scope=scope,direction=direction,left=left,right=right,**result))
                keys = subd.ligand_inchikey if direction=='drug_to_target' else subt.target_chembl_id
                queryrows.extend(dict(scope=scope,direction=direction,left=left,right=right,query=k,
                                      spearman=float(r),top20_intersection=int(h)) for k,r,h in zip(keys,rho,hits))
            for name,s in subs.items():
                rr = rank_matrix(s,1 if direction=='drug_to_target' else 0)
                v = np.full(384,np.nan)
                v[covered] = rr[dd[covered],tt[covered]]
                details[f'{scope}__{name}__{direction}_rank'] = v
                supportrows.append(dict(scope=scope,model=name,direction=direction,covered_pairs=int(covered.sum()),
                                        top10=int(np.sum(v<=10)),top20=int(np.sum(v<=20)),top50=int(np.sum(v<=50)),
                                        top20_fraction_of_covered=float(np.sum(v<=20)/covered.sum()),
                                        median_rank_covered=float(np.nanmedian(v))))
    pd.DataFrame(rows).to_csv(OUT/'RANK_AGREEMENT_SUMMARY.csv',index=False)
    pd.DataFrame(queryrows).to_csv(OUT/'QUERY_AGREEMENT.csv.gz',index=False)
    pd.DataFrame(supportrows).to_csv(OUT/'SPR384_RANK_SUPPORT_COUNTS.csv',index=False)
    # Distinguish coverage from directional support, and report shared support on identical scopes.
    supportjoint=[]
    for scope,names in [('full720x384',['legacy_spr_ranker','old_neural','A_consensus','B_consensus','dtiam']),
                        ('drugclip_common548x269',list(scores))]:
        for direction in ['drug_to_target','target_to_drug']:
            for left,right in itertools.combinations(names,2):
                a=details[f'{scope}__{left}__{direction}_rank'];b=details[f'{scope}__{right}__{direction}_rank']
                valid=a.notna()&b.notna()
                supportjoint.append(dict(scope=scope,direction=direction,left=left,right=right,covered_pairs=int(valid.sum()),
                                         both_top20=int((a.le(20)&b.le(20)&valid).sum()),
                                         either_top20=int(((a.le(20)|b.le(20))&valid).sum())))
    pd.DataFrame(supportjoint).to_csv(OUT/'SPR384_SHARED_TOP20_SUPPORT.csv',index=False)
    details.to_csv(OUT/'SPR384_EXTERNAL_MODEL_AGREEMENT.csv',index=False,encoding='utf-8-sig')
    concise=details[['排序','原候选编号','小分子药物名称','新靶点名称','当前实验建议','pair_id','drugclip_common548x269_covered']].copy()
    for name in ['A_consensus','B_consensus','dtiam']:
        concise[name+'_drug_rank384']=details[f'full720x384__{name}__drug_to_target_rank']
        concise[name+'_target_rank720']=details[f'full720x384__{name}__target_to_drug_rank']
    concise['drugclip_drug_rank269']=details['drugclip_common548x269__drugclip__drug_to_target_rank']
    concise['drugclip_target_rank548']=details['drugclip_common548x269__drugclip__target_to_drug_rank']
    concise.to_csv(OUT/'SPR384_CONSENSUS_CHECK.csv',index=False,encoding='utf-8-sig')
    # DTIAM has complete coverage and can replay the preceding exact same fixed-slot scenario.
    selectionpool=core.reset_index()
    selectionpool['dtiam_score']=scores['dtiam']['target_to_drug'].ravel()
    slots=final.groupby('新靶点ChEMBL编号').size().to_dict()
    retention=[]
    for scenario,mask in [('fixed112_frozen_evidence',selectionpool.counterfactual_eligible),
                          ('fixed112_latest_exact_excluded',selectionpool.counterfactual_eligible & ~selectionpool.latest_exact_evidence)]:
        selected,solver=select_fixed_slots(selectionpool[mask],selectionpool.set_index('pair_id').dtiam_score,slots)
        assert selected is not None
        selected[['pair_id','drug_names','gene_symbol','target_chembl_id','selection_score','in_frozen384',
                  'reviewed_in_original574','latest_exact_evidence']].to_csv(OUT/('DTIAM_'+scenario+'_384.csv'),index=False)
        retention.append(dict(scenario=scenario,retained=int(selected.in_frozen384.sum()),selected=len(selected),
                              original_reviewed=int(selected.reviewed_in_original574.sum()),**solver))
    pd.DataFrame(retention).to_csv(OUT/'DTIAM_FIXED112_RETENTION.csv',index=False)
    # The two requested extra targets have a newer, different pocket-scoring route; keep it separate.
    catalog = pd.read_parquet(RANK/'CATALOG_720x890_SCORES.parquet')
    extrarows=[]
    for gene in ['LYVE1','SLC8A1']:
        f=pd.read_csv(EXTRA/f'{gene}_THREE_MODEL_COMPARISON_720.csv').sort_values('drug_id')
        accession=newtargets.loc[newtargets.gene_symbol.eq(gene),'uniprot_accession'].item()
        extra=catalog[catalog.uniprot_accession.eq(accession)].set_index('ligand_inchikey').reindex(f.drug_id)
        assert len(f)==720 and extra.notna().all().all()
        ss={'A_consensus':extra.kdki_inactive_consensus_score.to_numpy(),
            'B_consensus':extra.all_inactive_consensus_score.to_numpy(),
            'old_neural':f.target_to_drug_logit.to_numpy(),
            'dtiam':f.dtiam_probability.to_numpy(),'drugclip':f.drugclip_score.to_numpy()}
        for left,right in itertools.combinations(ss,2):
            stats,_,_=agreement(ss[left][None,:],ss[right][None,:])
            extrarows.append(dict(gene=gene,left=left,right=right,**stats))
    pd.DataFrame(extrarows).to_csv(OUT/'LYVE1_SLC8A1_AGREEMENT.csv',index=False)
    assert all(digest(ROOT/p)==h for p,h in hashes.items())
    summary=dict(status='COMPLETE',input_sha256=hashes,source_sha256=digest(Path(__file__)),
                 dtiam_version=json.loads((COMPARE/'scores/dtiam.json').read_text()),
                 drugclip_version=json.loads((COMPARE/'scores/drugclip.json').read_text()),
                 drugclip_covered_final_pairs=int(details.drugclip_common548x269_covered.sum()),
                 missing_not_negative=True,exact_axis_and_sequence_mapping_verified=True,
                 unchanged_inputs=True,no_training=True,no_new_external_model_inference=True,
                 interpretation='Rank agreement, not assay validation; public encoders/data are shared, models are not independent witnesses.',
                 output_sha256={p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='SUMMARY.json'})
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(pd.DataFrame(supportrows).to_string(index=False))
    print(pd.DataFrame(rows)[lambda f: f.right.isin(['dtiam','drugclip'])].to_string(index=False))


if __name__=='__main__':
    main()
