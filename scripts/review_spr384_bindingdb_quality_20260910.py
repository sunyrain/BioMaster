"""Read-only comparison of frozen SPR384 with September BindingDB evidence."""
from pathlib import Path
import hashlib
import json
import pandas as pd
from evaluate_affinity_refresh_predictor_20260910 import metrics, clustered_ci

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/spr384_bindingdb_quality_review_20260910'
AUDIT=ROOT/'outputs/affinity_predictor_audit_20260910'


def main():
    OUT.mkdir(exist_ok=True)
    baseline=ROOT/'outputs/joint384_comprehensive_20260909/RECOMMENDED_CANDIDATES_384.csv'
    final=ROOT/'outputs/spr384_final_experiment_table_20260910/SPR384_FINAL_DETAILED.csv'
    b=pd.read_csv(baseline);d=pd.read_csv(final)
    a=pd.read_csv(AUDIT/'PAIR_LABELS_AND_FROZEN_SCORES.csv.gz')
    obs=pd.read_csv(AUDIT/'LABELED_OBSERVATIONS.csv.gz')
    binding=obs[obs.source.eq('BindingDB_202609')&obs.endpoint.isin(['Kd','Ki'])]
    z=a[a.endpoint_group.eq('Kd_Ki')&a.refresh_endpoint_covered&~a.local_training_connectivity_pair&a.scored&a.binary_label.notna()].copy()
    bdb=z[z.pair_id.isin(binding.pair_id)].copy()
    # Preserve the previous audit's labels, including cross-source conflict checks.
    bdb.to_csv(OUT/'BINDINGDB_COVERED_EVALUATION_PAIRS.csv',index=False,encoding='utf-8-sig')
    cohorts={'MERGED_REFRESH':z,'BINDINGDB_COVERED_MERGED_QC':bdb,
        'BINDINGDB_IDENTITY_CHEMISTRY_PASS':bdb[bdb.identity_scope_pass.eq(True)&bdb.chemistry_policy_pass.eq(True)]}
    results=[];bands=[]
    for name,g in cohorts.items():
        for model in ['SPR_current','neural_full_fit']:
            results.append(dict(cohort=name,**metrics(g,model)))
        for lo,hi in [(1,10),(11,20),(21,384)]:
            v=g[g.rank_SPR_current.between(lo,hi)]
            bands.append(dict(cohort=name,rank_band=f'{lo}-{hi}',observed_pairs=len(v),
                positive=int(v.binary_label.sum()),weak_negative=int((v.binary_label==0).sum()),
                observed_positive_fraction=float(v.binary_label.mean()),drugs=v.ligand_inchikey.nunique()))
    pd.DataFrame(results).to_csv(OUT/'COHORT_METRICS.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(bands).to_csv(OUT/'RANK_BAND_OBSERVED_OUTCOMES.csv',index=False,encoding='utf-8-sig')
    ci=clustered_ci(bdb,'SPR_current')
    raw=pd.read_csv(ROOT/'outputs/affinity_evidence_refresh_20260910/PROJECT_EXACT_NUMERIC_EVIDENCE.csv.gz',low_memory=False).fillna('')
    records=raw[raw.id.isin(binding[binding.pair_id.isin(bdb.pair_id)].id)]
    dates=[]
    for r in records.itertuples():
        meta=json.loads(r.metadata_json)
        dates.append(dict(pair_id=r.pair_id,record_id=r.record_id,endpoint=r.endpoint,
            publication_date=meta.get('Date of publication',''),bindingdb_date=meta.get('Date in BindingDB',''),doi=r.doi))
    dates=pd.DataFrame(dates)
    for col in ['publication_date','bindingdb_date']:
        dates[col]=pd.to_datetime(dates[col],format='mixed',errors='coerce')
    dates.to_csv(OUT/'BINDINGDB_EVIDENCE_DATES.csv',index=False)
    date_stats={}
    for col in ['publication_date','bindingdb_date']:
        valid=dates[dates[col].notna()]
        fresh=valid[valid[col].ge('2026-09-01')]
        date_stats[col]=dict(dated_rows=len(valid),earliest=str(valid[col].min().date()),
            latest=str(valid[col].max().date()),september_or_later_rows=len(fresh),
            pairs_with_september_or_later_record=fresh.pair_id.nunique())
    novel=bdb[bdb.known_relation_excluded.eq(False)&~bdb.existing_ChEMBL37_numeric_pair]
    composition=d[['排序','原候选编号','小分子药物名称','新靶点基因','结合排名_每药384靶点','当前实验建议','FDA身份状态','pair_id']].merge(
        b[['pair_id','ensemble_drug_to_target_logit_rank_within_drug_384','usable_chemical_support_ge04','recommended_disease_is_cross_area']],on='pair_id',validate='one_to_one')
    composition.to_csv(OUT/'FROZEN384_QUALITY_CONTEXT.csv',index=False,encoding='utf-8-sig')
    summary=dict(candidate_pairs=len(d),drugs=b.ligand_inchikey.nunique(),targets=b.target_chembl_id.nunique(),
        current_rank_top10=int(b.binding_rank_384.le(10).sum()),current_rank_11_20=int(b.binding_rank_384.between(11,20).sum()),
        action_counts=d['当前实验建议'].value_counts().to_dict(),cross_area=int(b.recommended_disease_is_cross_area.sum()),
        usable_chemical_support_ge04=int(b.usable_chemical_support_ge04.sum()),
        neural_full_fit_top20=int(b.ensemble_drug_to_target_logit_rank_within_drug_384.le(20).sum()),
        exact_evidence_overlap=len(set(d.pair_id)&set(obs.pair_id)),
        bdb_validation_shared_drugs=len(set(b.ligand_inchikey)&set(bdb.ligand_inchikey)),
        bdb_validation_shared_targets=len(set(b.target_chembl_id)&set(bdb.target_chembl_id)),
        bdb_strict_no_prior_relation=dict(pairs=len(novel),positive=int(novel.binary_label.sum()),weak_negative=int((novel.binary_label==0).sum())),
        bindingdb_spr_cluster_bootstrap=ci,dates=date_stats,models_retrained=False,baseline_modified=False,
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [baseline,final,AUDIT/'PAIR_LABELS_AND_FROZEN_SCORES.csv.gz',AUDIT/'LABELED_OBSERVATIONS.csv.gz']},
        limitations=['BindingDB-covered cohorts retain existing cross-source labels/QC; not exclusively BindingDB-derived labels.',
            'September release is a cumulative snapshot, not a prospectively isolated September experimental dataset.',
            'Rank-band outcomes describe measured pairs only; not estimated hit rates for the frozen384.',
            'No local training pair does not imply cold drug, cold target, cold scaffold or pretraining-leakage-free.',
            'Neural agreement is descriptive; not independently calibrated or used to reselect candidates.'])
    assert len(d)==384 and set(d.pair_id)==set(b.pair_id)
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='input_sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
