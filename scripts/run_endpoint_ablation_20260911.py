#!/usr/bin/env python3
"""Six matched-budget research fits; test scoring only after every fit is frozen."""
import argparse,copy,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,precision_recall_curve
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from prepare_endpoint_ablation_20260911 import DATA,OUT,FEATURES,BUNDLE,write_json
from biomaster.endpoint_ablation import ARMS,CyclingRows
from biomaster.model_registry import build_model
from biomaster.best_model_training import WeightAverage
from biomaster.portable_ranker_v2 import digest

def log(**v):
    write_json(OUT/'TRAIN_STATUS.json',v);print(json.dumps(v),flush=True)

class Bank:
    def __init__(self):
        m=json.loads((FEATURES/'MANIFEST.json').read_text());assert m['all_required_rows_prepared']
        self.clip=torch.from_numpy(np.load(FEATURES/'DRUG_CLIP.npy')).cuda()
        self.graph=torch.from_numpy(np.load(FEATURES/'GRAPH.npy')).cuda()
        self.fp=torch.from_numpy(np.load(FEATURES/'MORGAN.npy')).cuda()
        self.available=torch.from_numpy(np.load(FEATURES/'AVAILABLE.npy')).cuda().float()
        self.target=torch.from_numpy(np.load(FEATURES/'TARGET.npy')).cuda()
    def batch(self,d,t):
        d=torch.as_tensor(d,dtype=torch.long,device='cuda');t=torch.as_tensor(t,dtype=torch.long,device='cuda')
        return dict(drug_global=torch.cat([self.clip[d],self.fp[d].float()],-1),drug_graph_mean=self.graph[d],
                    pretrained_available=self.available[d],target_global=self.target[t])

@torch.inference_mode()
def score(model,bank,frame):
    model.eval();result=[]
    for start in range(0,len(frame),2048):
        f=frame.iloc[start:start+2048]
        with torch.autocast('cuda',dtype=torch.bfloat16):s=model(bank.batch(f.drug_feature_index.to_numpy(),f.target_feature_index.to_numpy()))
        result.append(s.float().mean(1).cpu().numpy())
    result=np.concatenate(result);assert np.isfinite(result).all();return result

def calibrate(frame,scores):
    use=frame.panel.eq('AFFINITY_KD_KI').to_numpy();y=frame.loc[use,'binary_label'].to_numpy(int)
    model=LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000,random_state=20260911).fit(scores[use,None],y)
    assert model.coef_[0,0]>0,'negative validation calibration slope'
    probabilities=model.predict_proba(scores[use,None])[:,1]
    p,r,thresholds=precision_recall_curve(y,probabilities)
    f1=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-12);best=int(np.argmax(f1))
    return dict(slope=float(model.coef_[0,0]),intercept=float(model.intercept_[0]),threshold=float(thresholds[best]),validation_f1=float(f1[best]),fit_panel='VALIDATION_AFFINITY_KD_KI')

def basic(y,s,prob,threshold):
    y=np.asarray(y,int);s=np.asarray(s,float);prob=np.asarray(prob,float);pred=prob>=threshold
    tp=int((pred & (y==1)).sum());fp=int((pred & (y==0)).sum());pos=int(y.sum());neg=len(y)-pos
    both=pos>0 and neg>0
    r=dict(pairs=len(y),positive=pos,negative=neg,prevalence=float(y.mean()),
        auroc=float(roc_auc_score(y,s)) if both else None,ap=float(average_precision_score(y,s)) if both else None,
        brier=float(brier_score_loss(y,prob)),precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/pos if pos else None,false_positive_rate=fp/neg if neg else None,threshold=float(threshold),mean_probability=float(prob.mean()))
    order=np.argsort(-s,kind='stable')
    for fraction,name in [(.01,'top1pct'),(.05,'top5pct'),(.1,'top10pct')]:
        n=max(1,int(math.ceil(len(y)*fraction)));precision=float(y[order[:n]].mean())
        r[name+'_n']=n;r[name+'_precision']=precision;r[name+'_enrichment']=precision/float(y.mean()) if pos else None
    return r

def report_metrics(frame,scores,cal,arm,seed):
    f=frame.copy();f['score']=scores;f['probability']=expit(cal['slope']*scores+cal['intercept']);rows=[];queryrows=[]
    for panel,g in f.groupby('panel'):
        scopes=[('all',g),('document_disjoint',g[g.document_disjoint])]
        if 'target_training_coverage' in g:
            scopes.extend(('targets_'+str(k),sub) for k,sub in g.groupby('target_training_coverage'))
        for scope,sub in scopes:
            if sub.empty:continue
            rows.append(dict(arm=arm,seed=seed,panel=panel,scope=scope,**basic(sub.binary_label,sub.score,sub.probability,cal['threshold'])))
        for group,direction in [('target_id','within_target'),('molecule_id','within_drug')]:
            for key,sub in g.groupby(group):
                if len(sub)<10 or sub.binary_label.nunique()<2:continue
                b=basic(sub.binary_label,sub.score,sub.probability,cal['threshold'])
                queryrows.append(dict(arm=arm,seed=seed,panel=panel,direction=direction,query=key,**b))
    return rows,queryrows,f

def fit(arm,seed,bank,protocol,config):
    run=OUT/f'{arm}_seed_{seed}';run.mkdir(exist_ok=True)
    identity=dict(protocol=digest(OUT/'PROTOCOL.json'),data=digest(OUT/'DATA_MANIFEST.json'),features=digest(FEATURES/'MANIFEST.json'),
        producer=digest(Path(__file__)),rules=digest(ROOT/'biomaster/endpoint_ablation.py'),
        model_code=digest(ROOT/'biomaster/unified_interaction.py'),model_input_code=digest(ROOT/'biomaster/molecular_controls.py'))
    if (run/'RESULT.json').exists():
        r=json.loads((run/'RESULT.json').read_text());assert r['identity']==identity and digest(run/'model.pt')==r['checkpoint_sha256'];return r
    frame=pd.read_parquet(OUT/(arm+'_TRAIN.parquet'));validation=pd.read_parquet(OUT/'COMMON_VALIDATION.parquet')
    d=frame.drug_feature_index.to_numpy();t=frame.target_feature_index.to_numpy();y=frame.binary_label.to_numpy(np.float32)
    torch.manual_seed(seed);np.random.seed(seed);model=build_model('molecular_control',config).cuda()
    initial_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    ema=WeightAverage(model,protocol['ema_decay']);opt=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    stream=CyclingRows(len(frame),seed);prevalence=float(y.mean());trace=hashlib.sha256();history=[];startstep=0
    if (run/'RESUME.pt').exists():
        state=torch.load(run/'RESUME.pt',map_location='cpu',weights_only=False);assert state['identity']==identity
        model.load_state_dict(state['model']);ema.model.load_state_dict(state['ema']);ema.updates=state['ema_updates'];opt.load_state_dict(state['optimizer'])
        stream.order=state['order'];stream.offset=state['offset'];stream.passes=state['passes'];stream.rng.bit_generator.state=state['rng']
        torch.cuda.set_rng_state(state['cuda_rng']);history=state['history'];startstep=state['step'];del state
    started=time.monotonic();losses=[];model.train()
    log(stage='training_started',arm=arm,seed=seed,pairs=len(frame),positive=int(y.sum()),negative=int(len(y)-y.sum()),steps=protocol['steps'],resumed_step=startstep)
    for step in range(startstep,protocol['steps']):
        ix=stream.take(protocol['batch_size']);trace.update(ix.astype('<i8').tobytes())
        rate=protocol['learning_rate']*(.2+.8*(1+math.cos(math.pi*step/protocol['steps']))/2)
        for g in opt.param_groups:g['lr']=rate
        opt.zero_grad(set_to_none=True);yb=torch.tensor(y[ix],device='cuda');w=torch.where(yb.bool(),.5/prevalence,.5/(1-prevalence))
        with torch.autocast('cuda',dtype=torch.bfloat16):
            s=model(bank.batch(d[ix],t[ix])).float();loss=(F.binary_cross_entropy_with_logits(s,yb[:,None].expand(-1,2),reduction='none')*w[:,None]).mean()
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step();ema.update(model);losses.append(float(loss.detach()))
        if (step+1)%500==0:
            row=dict(step=step+1,loss=float(np.mean(losses)),complete_passes=stream.passes,seconds=round(time.monotonic()-started,1),learning_rate=rate)
            history.append(row);losses=[];write_json(run/'HISTORY.json',history);log(stage='training',arm=arm,seed=seed,**row)
        if (step+1)%2000==0:
            tmp=run/'RESUME.pt.tmp';torch.save(dict(identity=identity,model=model.state_dict(),ema=ema.model.state_dict(),ema_updates=ema.updates,
                optimizer=opt.state_dict(),order=stream.order,offset=stream.offset,passes=stream.passes,rng=stream.rng.bit_generator.state,
                cuda_rng=torch.cuda.get_rng_state(),history=history,step=step+1),tmp);tmp.replace(run/'RESUME.pt')
    assert stream.passes>=1
    scores=score(ema.model,bank,validation);cal=calibrate(validation,scores);write_json(run/'CALIBRATION.json',cal)
    rows,_,_=report_metrics(validation,scores,cal,arm,seed);pd.DataFrame(rows).to_csv(run/'VALIDATION_METRICS.csv',index=False)
    torch.save(dict(architecture='molecular_control',config=config,model={k:v.detach().cpu() for k,v in ema.model.state_dict().items()},
        role='RESEARCH_ENDPOINT_ABLATION_NOT_DEPLOYED',arm=arm,seed=seed,identity=identity),run/'model.pt')
    result=dict(status='COMPLETE',arm=arm,seed=seed,identity=identity,pairs=len(frame),positive=int(y.sum()),negative=int(len(y)-y.sum()),
        explicit_inactive_pairs=int(frame.explicit_inactive.sum()),initialization_sha256=initial_hash,trainable_parameters=sum(p.numel() for p in model.parameters()),
        optimizer_steps=protocol['steps'],sampled_rows=protocol['steps']*protocol['batch_size'],complete_training_passes=stream.passes,
        checkpoint_sha256=digest(run/'model.pt'),sampling_trace_this_process=trace.hexdigest(),seconds=time.monotonic()-started,
        test_used_for_fitting=False,production_replaced=False)
    write_json(run/'RESULT.json',result);log(stage='fit_complete',**result)
    del model,ema,opt;torch.cuda.empty_cache();return result

def bootstrap(frame,a,b,reps=500):
    y=frame.binary_label.to_numpy(int);groups=list(frame.groupby('split_group').indices.values());rng=np.random.default_rng(20260911)
    auc=[];ap=[]
    for _ in range(reps):
        ix=np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))])
        if np.unique(y[ix]).size<2:continue
        auc.append(roc_auc_score(y[ix],b[ix])-roc_auc_score(y[ix],a[ix]));ap.append(average_precision_score(y[ix],b[ix])-average_precision_score(y[ix],a[ix]))
    return dict(groups=len(groups),valid_replicates=len(auc),auroc_delta_ci95=np.quantile(auc,[.025,.975]).tolist(),ap_delta_ci95=np.quantile(ap,[.025,.975]).tolist(),
        auroc_delta=float(roc_auc_score(y,b)-roc_auc_score(y,a)),ap_delta=float(average_precision_score(y,b)-average_precision_score(y,a)),direction='all_inactive minus kdki_inactive')

def evaluate(bank,protocol):
    frozen=[]
    for seed in protocol['seeds']:
        hashes=[]
        for arm in ARMS:
            run=OUT/f'{arm}_seed_{seed}';r=json.loads((run/'RESULT.json').read_text());assert digest(run/'model.pt')==r['checkpoint_sha256']
            frozen.append(r);hashes.append(r['initialization_sha256'])
        assert len(set(hashes))==1
    write_json(OUT/'ALL_FITS_FROZEN.json',dict(runs=frozen,test_not_scored_before_all_fits=True))
    test=pd.read_parquet(OUT/'COMMON_TEST.parquet').reset_index(drop=True)
    seen={arm:set(pd.read_parquet(OUT/(arm+'_TRAIN.parquet'),columns=['target_id']).target_id) for arm in ARMS}
    sa=test.target_id.isin(seen['kdki_inactive']);sb=test.target_id.isin(seen['all_inactive'])
    test['target_training_coverage']=np.select([sa&sb,~sa&sb,sa&~sb],['both_arms','only_all','only_kdki'],default='neither')
    metrics=[];queries=[];predictions=test[['panel','pair_id','molecule_id','target_id','split_group','binary_label','document_disjoint','target_training_coverage']].copy()
    for seed in protocol['seeds']:
        for arm in ARMS:
            run=OUT/f'{arm}_seed_{seed}';state=torch.load(run/'model.pt',map_location='cpu',weights_only=True)
            model=build_model(state['architecture'],state['config']).cuda();model.load_state_dict(state['model']);s=score(model,bank,test)
            cal=json.loads((run/'CALIBRATION.json').read_text());rows,q,f=report_metrics(test,s,cal,arm,seed);metrics.extend(rows);queries.extend(q)
            predictions[f'{arm}_{seed}_score']=s;predictions[f'{arm}_{seed}_prob']=f.probability.to_numpy();del model;torch.cuda.empty_cache()
    pd.DataFrame(metrics).to_csv(OUT/'TEST_METRICS.csv',index=False);pd.DataFrame(queries).to_csv(OUT/'TEST_QUERY_METRICS.csv.gz',index=False)
    predictions.to_parquet(OUT/'TEST_PREDICTIONS.parquet',index=False,compression='zstd')
    ci=[]
    for panel in ['AFFINITY_KD_KI','ACTIVITY_IC50','ACTIVITY_EC50','ALL_ENDPOINT_UNION']:
        for scope in ['all','document_disjoint']:
            mask=predictions.panel.eq(panel)
            if scope=='document_disjoint':mask &= predictions.document_disjoint
            f=predictions[mask].reset_index(drop=True)
            if f.binary_label.nunique()<2:continue
            a=f[[f'kdki_inactive_{seed}_prob' for seed in protocol['seeds']]].mean(1).to_numpy()
            b=f[[f'all_inactive_{seed}_prob' for seed in protocol['seeds']]].mean(1).to_numpy()
            ci.append(dict(panel=panel,scope=scope,pairs=len(f),**bootstrap(f,a,b)))
    write_json(OUT/'PAIRED_SCAFFOLD_BOOTSTRAP.json',ci)
    mf=pd.DataFrame(metrics);mf.groupby(['arm','panel','scope'])[['auroc','ap','brier','top1pct_precision','top5pct_precision','false_positive_rate']].agg(['mean','std']).to_csv(OUT/'TEST_SUMMARY.csv')
    main=mf[mf.panel.eq('AFFINITY_KD_KI') & mf.scope.eq('all')];primary=next(c for c in ci if c['panel']=='AFFINITY_KD_KI' and c['scope']=='all')
    data=json.loads((OUT/'DATA_MANIFEST.json').read_text())
    for name,h in data['frozen_inputs'].items():assert digest(ROOT/name)==h,name
    summary=dict(status='COMPLETE_RESEARCH_COMPARISON',runs=6,primary_panel='COMMON_TEST_AFFINITY_KD_KI',primary_metrics=main.to_dict('records'),
        primary_paired_bootstrap=primary,train_counts=data['counts'],equal_initialization_and_optimizer_budget=True,
        all_training_rows_exposed=True,production_replaced=False,wetlab_unchanged=True,protocol_sha256=digest(OUT/'PROTOCOL.json'),
        feature_manifest_sha256=digest(FEATURES/'MANIFEST.json'),limitations=protocol['limitations'])
    write_json(OUT/'SUMMARY.json',summary);log(stage='complete',runs=6,primary_ap_delta=primary['ap_delta'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['train','evaluate','all'],default='all');a=p.parse_args()
    protocol=json.loads((OUT/'PROTOCOL.json').read_text());torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    state=torch.load(BUNDLE/'model.pt',map_location='cpu',weights_only=True);config=state['config'];del state
    bank=Bank()
    if a.stage in ['train','all']:
        for seed in protocol['seeds']:
            for arm in ARMS:fit(arm,seed,bank,protocol,config)
    if a.stage in ['evaluate','all']:evaluate(bank,protocol)

if __name__=='__main__':main()
