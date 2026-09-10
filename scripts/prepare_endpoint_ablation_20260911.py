#!/usr/bin/env python3
"""Freeze two endpoint arms and common evaluation panels, without fitting models."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from biomaster.endpoint_ablation import ARMS,combine
from biomaster.portable_ranker_v2 import digest

DATA=ROOT/'data/processed/biomaster_training_full_20260910_v1'
OUT=ROOT/'outputs/biomaster_endpoint_ablation_20260911'
FEATURES=OUT/'features'
BUNDLE=ROOT/'outputs/biomaster_best_model_20260906/retargetmap_selected_v1'

def write_json(p,value):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)

def main():
    OUT.mkdir(exist_ok=True);FEATURES.mkdir(exist_ok=True)
    source_manifest=ROOT/'outputs/training_dataset_full_20260910_v1/MANIFEST.json'
    protocol=dict(status='FROZEN_BEFORE_TRAINING',arms=ARMS,seeds=[20260921,20260922,20260923],
        source_manifest_sha256=digest(source_manifest),initialization='identical random initialization per seed; no parent supervised weights',
        architecture='molecular_control/global drugclip_morgan, width192; production architecture',
        frozen_encoders='DrugCLIP fold0 CLS512 + Morgan2048 + atom-summary40 + availability; ESM2 t33 650M full-sequence mean1280',
        batch_size=1024,steps=6000,learning_rate=3e-4,weight_decay=1e-4,ema_decay=.995,
        loss='class-balanced measured-pair BCE on both readouts; no unmeasured negative pairs',
        score='mean of the two readout logits; no retrieval loss, so no directional specialization claim',
        schedule='cosine to 20% of initial LR; final EMA only; same 6000 optimizer steps and 6144000 sampled rows per arm',
        split='reuse fixed scaffold/connectivity splits and wetlab/previous benchmark reserves',
        inactivity='All accepted explicit inactivity, including endpoint-bearing annotations, enters both arms as pair negatives; conflicting positives in any task quarantine the inactivity pair in both arms',
        all_endpoint_pooling='One binary label per pair; selected-task positive/negative or within-task conflicts quarantined; Kd/Ki arm ignores other numeric endpoints except shared inactivity contradiction QC',
        primary_evaluation='identical TEST affinity Kd/Ki panel; IC50, EC50, union and pure inactivity panels secondary',
        calibration='Platt fitted on common validation Kd/Ki labels only; threshold maximizing validation F1; test never used for choice',
        test_policy='test scores generated only after all six fits complete; all seeds reported, no best-seed selection',
        confidence_intervals='paired scaffold bootstrap of seed-mean predictions; seed-level results also retained',
        limitations=['Retrospective scaffold holdout, not prospective SPR or protein-homology cold start.',
                    'Main scaffold splits may share documents; evaluate a separate test subset without train/validation citations.',
                    'Public encoder pretraining relation/scaffold exposure is not certified.',
                    'All-endpoint pooling tests the requested mixed-label hypothesis, not equivalence of Kd, Ki, IC50 and EC50.'],
        production_model_sha256=digest(BUNDLE/'model.pt'))
    if (OUT/'PROTOCOL.json').exists():assert json.loads((OUT/'PROTOCOL.json').read_text())==protocol,'protocol drift'
    else:write_json(OUT/'PROTOCOL.json',protocol)
    if (OUT/'DATA_MANIFEST.json').exists():
        old=json.loads((OUT/'DATA_MANIFEST.json').read_text())
        for name,h in old['files'].items():assert digest(OUT/name)==h,name
        print('Data preparation already complete and verified',flush=True);return
    labels=pd.read_parquet(DATA/'ALL_PAIR_TASK_LABELS.parquet')
    main=labels[labels.split.isin(['train','validation','test'])].copy()
    obs=pd.read_parquet(DATA/'OBSERVATIONS_WITH_QC.parquet',columns=['molecule_id','target_id','explicit_inactive','record_qc'])
    obs=obs[obs.record_qc.eq('ACCEPTED') & obs.explicit_inactive.eq(1)]
    inactive=set(obs.molecule_id+'__'+obs.target_id);del obs
    rows=[];required_d=set();required_t=set();files=[]
    combined={}
    for arm,tasks in ARMS.items():
        audit=combine(main,inactive,tasks);combined[arm]=audit
        p=OUT/(arm+'_PAIR_AUDIT.parquet');audit.to_parquet(p,index=False,compression='zstd');files.append(p)
        for split in ['train','validation']:
            f=audit[audit.eligible & audit.split.eq(split)].copy()
            p=OUT/(arm+'_'+split.upper()+'.parquet');f.to_parquet(p,index=False,compression='zstd');files.append(p)
            required_d.update(f.drug_feature_index);required_t.update(f.target_feature_index)
            rows.append(dict(arm=arm,split=split,pairs=len(f),positive=int(f.binary_label.sum()),negative=int(f.binary_label.eq(0).sum()),
                explicit_inactive_pairs=int(f.explicit_inactive.sum()),inactivity_added=int(f.inactivity_added_without_numeric_negative.sum()),
                molecules=f.molecule_id.nunique(),targets=f.target_id.nunique(),quarantined_conflicts=int((audit.split.eq(split)&audit.conflict).sum())))
    # Common panels never depend on either model's output.
    inactive_conflicts=set(combined['all_inactive'].loc[combined['all_inactive'].inactive_positive_conflict,'pair_id'])
    citations=pd.read_parquet(DATA/'PAIR_CITATIONS.parquet')
    traindocs=set(citations.loc[citations.split.isin(['train','validation']),'citation'])
    shared=set(citations.loc[citations.citation.isin(traindocs),'pair_id'])
    cited=set(citations.pair_id)
    for split in ['validation','test']:
        panels=[]
        for task in ['AFFINITY_KD_KI','ACTIVITY_IC50','ACTIVITY_EC50']:
            f=main[main.split.eq(split)&main.task.eq(task)&main.eligible_binary&~main.pair_id.isin(inactive_conflicts)].copy()
            f['panel']=task;panels.append(f)
        union=combined['all_inactive'];f=union[union.eligible & union.split.eq(split)].copy();f['panel']='ALL_ENDPOINT_UNION';panels.append(f)
        f=f[f.explicit_inactive].copy();f['panel']='EXPLICIT_INACTIVE';panels.append(f)
        panel=pd.concat(panels,ignore_index=True)
        panel['document_disjoint']=panel.pair_id.isin(cited)&~panel.pair_id.isin(shared)
        p=OUT/('COMMON_'+split.upper()+'.parquet');panel.to_parquet(p,index=False,compression='zstd');files.append(p)
        required_d.update(panel.drug_feature_index);required_t.update(panel.target_feature_index)
        assert not panel.duplicated(['panel','pair_id']).any()
    for name,ids in [('DRUG',required_d),('TARGET',required_t)]:
        p=OUT/('REQUIRED_'+name+'_IDS.npy');np.save(p,np.array(sorted(ids),np.int64));files.append(p)
    counts=pd.DataFrame(rows);counts.to_csv(OUT/'ARM_DATA_COUNTS.csv',index=False);files.append(OUT/'ARM_DATA_COUNTS.csv')
    a=combined['kdki_inactive'];b=combined['all_inactive']
    ai=set(a.loc[a.eligible & a.split.eq('train') & a.explicit_inactive,'pair_id'])
    bi=set(b.loc[b.eligible & b.split.eq('train') & b.explicit_inactive,'pair_id']);assert ai==bi
    trainpairs=set().union(*(set(v.loc[v.eligible&v.split.eq('train'),'pair_id']) for v in combined.values()))
    testpairs=set(pd.read_parquet(OUT/'COMMON_TEST.parquet').pair_id);assert not trainpairs&testpairs
    alltrain=main[main.pair_id.isin(trainpairs)];alltest=main[main.pair_id.isin(testpairs)]
    assert not set(alltrain.split_group)&set(alltest.split_group)
    assert not alltrain.wetlab_candidate.any() and not alltrain.wetlab_reference_control.any()
    frozen_dir=ROOT/'outputs/spr384_final_experiment_table_20260910'
    frozen={str(p.relative_to(ROOT)):digest(p) for p in [BUNDLE/'model.pt',frozen_dir/'SPR384_FINAL_EXPERIMENT_TABLE.csv',frozen_dir/'SPR384_FINAL_DETAILED.csv',frozen_dir/'SPR112_REFERENCE_CONTROLS.csv']}
    manifest=dict(status='COMPLETE',required_molecules=len(required_d),required_sequences=len(required_t),shared_explicit_inactive_training_pairs=len(ai),
        counts=rows,files={str(p.relative_to(OUT)):digest(p) for p in files},frozen_inputs=frozen,protocol_sha256=digest(OUT/'PROTOCOL.json'))
    write_json(OUT/'DATA_MANIFEST.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k not in ['files','frozen_inputs']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
