#!/usr/bin/env python3
"""Thirty resumable assay/balance/adaptation fits; test gate after all converge."""
import argparse
import fcntl
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from prepare_assay_aware_20260911 import OUT, PARENT, SOURCE, DATA, FEATURES, VARIANTS, prepare, write_json
from run_endpoint_ablation_20260911 import Bank, calibrate, report_metrics
from run_endpoint_multitask_20260911 import now, save_torch, parameters_hash, predict, Validation, QueryMetrics, regression_metrics
from biomaster.endpoint_multitask import AuxiliaryInteraction, ENDPOINTS, ValidationPlateau, endpoint_huber, stream_state, restore_stream
from biomaster.endpoint_ablation import CyclingRows
from biomaster.assay_aware_training import (balanced_target_weights, RotatingTargetClassRows, row_stream_state,
    restore_row_stream, AssayPairs, assay_difference_huber)
from biomaster.best_model_training import WeightAverage
from biomaster.portable_ranker_v2 import digest


def status(**value):
    value=dict(updated_utc=now(),pid=os.getpid(),**value)
    write_json(OUT/'STATUS.json',value);print(json.dumps(value,ensure_ascii=False),flush=True)


class ArmData:
    def __init__(self,arm,p):
        self.arm=arm;self.endpoints=ENDPOINTS[arm]
        self.frame=pd.read_parquet(SOURCE/f'{arm}_TRAIN.parquet')
        self.d=self.frame.drug_feature_index.to_numpy(np.int64);self.t=self.frame.target_feature_index.to_numpy(np.int64)
        self.y=self.frame.binary_label.to_numpy(np.float32);self.prevalence=float(self.y.mean())
        self.weights=balanced_target_weights(self.frame,p['target_count_smoothing'],p['maximum_class_relative_gain'])
        reg=pd.read_parquet(PARENT/f'{arm}_TRAIN_REGRESSION.parquet')
        self.reg=[reg[reg.endpoint.eq(e)].reset_index(drop=True) for e in self.endpoints]
        self.reg_arrays=[(f.drug_feature_index.to_numpy(np.int64),f.target_feature_index.to_numpy(np.int64),
                         f.median_p_activity_unique.to_numpy(np.float32)) for f in self.reg]
        self.means=np.array([f.median_p_activity_unique.mean() for f in self.reg],np.float32)
        self.scales=np.array([f.median_p_activity_unique.std(ddof=0) for f in self.reg],np.float32)
        self.valreg=pd.read_parquet(PARENT/f'{arm}_VALIDATION_REGRESSION.parquet')
        assays=pd.read_parquet(OUT/f'{arm}_ASSAY_TRAIN.parquet')
        self.assays=[AssayPairs(assays[assays.endpoint.eq(e)],p['assay_minimum_pair_difference']) for e in self.endpoints]
        tasks=pd.read_parquet(DATA/'TRAIN.parquet',columns=['pair_id','task'])
        self.kdki=self.frame.pair_id.isin(tasks.loc[tasks.task.eq('AFFINITY_KD_KI'),'pair_id']).to_numpy()
        self.inactive=self.frame.explicit_inactive.to_numpy(bool)
        pd.DataFrame([dict(arm=arm,group=name,pairs=int(mask.sum()),raw_fraction=float(mask.mean()),
            soft_bce_coefficient_fraction=float(self.weights[mask].sum()/self.weights.sum())) for name,mask in [
                ('positive',self.y==1),('negative',self.y==0),('numeric_KdKi_present',self.kdki),
                ('explicit_inactive',self.inactive)]]).to_csv(OUT/f'{arm}_SUPERVISION_WEIGHTS.csv',index=False)


class Streams:
    def __init__(self,data,seed,variant,p):
        self.binary=(RotatingTargetClassRows(data.frame,seed,p['rotating_cap']) if variant=='rotate150'
                     else CyclingRows(len(data.frame),seed))
        self.reg=[CyclingRows(len(f),seed+200000+i) for i,f in enumerate(data.reg)]
        self.assay=np.random.default_rng(seed+300000);self.trace='00'*32
        self.seen=np.zeros(len(data.frame),bool)

    def state(self):
        return dict(binary=row_stream_state(self.binary),reg=[stream_state(s) for s in self.reg],
            assay=self.assay.bit_generator.state,trace=self.trace,seen=np.packbits(self.seen))

    def restore(self,state):
        restore_row_stream(self.binary,state['binary'])
        for s,v in zip(self.reg,state['reg'],strict=True):restore_stream(s,v)
        self.assay.bit_generator.state=state['assay'];self.trace=state['trace']
        self.seen=np.unpackbits(state['seen'])[:len(self.seen)].astype(bool)


def step_loss(model,bank,data,streams,variant,p,scales):
    v=VARIANTS[variant];ix=streams.binary.take(p['batch_size']);streams.seen[ix]=True
    streams.trace=hashlib.sha256(bytes.fromhex(streams.trace)+ix.astype('<i8').tobytes()).hexdigest()
    ds,ts=[data.d[ix]],[data.t[ix]];offset=len(ix);regs=[];pairs=[]
    for e,((d,t,y),stream) in enumerate(zip(data.reg_arrays,streams.reg)) if v['regression'] else []:
        j=stream.take(p['regression_rows_per_endpoint']);ds.append(d[j]);ts.append(t[j])
        regs.append((offset,offset+len(j),e,y[j]));offset+=len(j)
    for e,pool in enumerate(data.assays) if v['assay'] else []:
        d,t,y=pool.sample(streams.assay,p['assay_pairs_per_endpoint']);ds.append(d.ravel());ts.append(t.ravel())
        pairs.append((offset,offset+d.size,e,y));offset+=d.size
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits,pred=model(bank.batch(np.concatenate(ds),np.concatenate(ts)))
    logits,pred=logits.float(),pred.float();y=torch.as_tensor(data.y[ix],device='cuda')
    if variant=='rotate150':
        prevalence=streams.binary.positive_fraction
        w=torch.where(y.bool(),.5/prevalence,.5/(1-prevalence))
    else:w=torch.as_tensor(data.weights[ix],device='cuda')
    bce=(torch.nn.functional.binary_cross_entropy_with_logits(logits[:len(ix)],y[:,None].expand(-1,2),reduction='none')*w[:,None]).mean()
    losses={'binary':bce};total=bce
    if regs:
        rpred=torch.cat([pred[a:b] for a,b,_,_ in regs]);truth=torch.as_tensor(np.concatenate([x[3] for x in regs]),device='cuda')
        ids=torch.as_tensor(np.concatenate([np.full(b-a,e) for a,b,e,_ in regs]),device='cuda',dtype=torch.long)
        losses['regression']=endpoint_huber(rpred,truth,ids,*scales)
        total=total+p['auxiliary_weights']['regression']*losses['regression']
    if pairs:
        terms=[assay_difference_huber(pred[a:b,e].reshape(-1,2),torch.as_tensor(y,device='cuda'),data.scales[e]) for a,b,e,y in pairs]
        losses['assay_difference']=torch.stack(terms).mean()
        total=total+p['auxiliary_weights']['assay_difference']*losses['assay_difference']
    if not torch.isfinite(total):raise FloatingPointError('Nonfinite objective')
    return total,losses,dict(binary=len(ix),regression=sum(b-a for a,b,_,_ in regs),assay=sum(b-a for a,b,_,_ in pairs))


def create_model(data,seed,config,parent=None):
    torch.manual_seed(seed);np.random.seed(seed)
    model=AuxiliaryInteraction(config,data.endpoints).cuda()
    if parent:
        state=torch.load(parent,map_location='cpu',weights_only=True)
        model.base.load_state_dict({k[5:]:v for k,v in state['model'].items() if k.startswith('base.')})
        with torch.no_grad():
            for e,endpoint in enumerate(data.endpoints):
                j=state['endpoints'].index(endpoint)
                ratio=state['regression_scales'][j]/float(data.scales[e])
                model.regression.weight[e].copy_(state['model']['regression.weight'][j]*ratio)
                model.regression.bias[e].copy_((state['model']['regression.bias'][j]*state['regression_scales'][j]
                    +state['regression_means'][j]-float(data.means[e]))/float(data.scales[e]))
    initial=parameters_hash(model.base);torch.manual_seed(seed+400000)
    return model,initial


def run_path(arm,variant,seed):return OUT/f'{arm}__{variant}__seed_{seed}'


def fit(data,variant,seed,bank,validation,p,config,identity,index,parent_arm=None):
    name=variant if parent_arm is None else ('adapt_from_B' if parent_arm=='all_inactive' else 'continue_A')
    run=run_path(data.arm,name,seed);run.mkdir(exist_ok=True)
    parent=run_path(parent_arm,'assay',seed)/'model.pt' if parent_arm else None
    lineage=dict(parent_path=str(parent.relative_to(ROOT)) if parent else None,parent_sha256=digest(parent) if parent else None)
    if (run/'RESULT.json').exists():
        result=json.loads((run/'RESULT.json').read_text())
        assert result['identity']==identity and result['lineage']==lineage and result['checkpoint_sha256']==digest(run/'model.pt')
        assert result['status']=='CONVERGED_VALIDATION_PLATEAU';return result
    model,initial=create_model(data,seed,config,parent);ema=WeightAverage(model,p['ema_decay'])
    rate=p['adaptation']['learning_rate'] if parent else p['learning_rate']
    opt=torch.optim.AdamW(model.parameters(),lr=rate,weight_decay=p['weight_decay']);streams=Streams(data,seed,variant,p)
    rule=p['convergence'];controller=ValidationPlateau(rate,rule['minimum_lr'],rule['improvement_delta'],
        rule['lr_reduce_every_stale_checks'],rule['stale_checks_to_stop'],rule['minimum_complete_passes'])
    interval=max(1000,math.ceil(2*len(data.frame)/p['batch_size']))
    scales=(torch.tensor(data.means,device='cuda'),torch.tensor(data.scales,device='cuda'))
    history=[];startstep=0;previous_seconds=0.;counts_total=dict(binary=0,regression=0,assay=0)
    if (run/'RESUME.pt').exists():
        state=torch.load(run/'RESUME.pt',map_location='cpu',weights_only=False)
        assert state['identity']==identity and state['lineage']==lineage
        model.load_state_dict(state['model']);ema.model.load_state_dict(state['ema']);ema.updates=state['ema_updates']
        opt.load_state_dict(state['optimizer']);streams.restore(state['streams']);controller.__dict__.update(state['controller'])
        torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
        history=state['history'];startstep=state['step'];previous_seconds=state['seconds'];counts_total=state['counts'];del state
    write_json(run/'REGRESSION_SCALING.json',dict(endpoints=data.endpoints,means=data.means.tolist(),scales=data.scales.tolist(),fit_split='TRAIN_ONLY'))
    started=time.monotonic();last_log=started;model.train();losses={}
    converged=bool(history and history[-1]['converged']);finalstep=startstep
    status(stage='training_started',arm=data.arm,variant=name,seed=seed,fit_index=index,total_fits=p['total_fits'],
        resumed_step=startstep,train_pairs=len(data.frame),lineage=lineage)
    for step in range(startstep,p['emergency_max_steps']) if not converged else []:
        lr=controller.lr*min(1.,(step+1)/p['warmup_steps'])
        for g in opt.param_groups:g['lr']=lr
        opt.zero_grad(set_to_none=True);loss,components,counts=step_loss(model,bank,data,streams,variant,p,scales)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step();ema.update(model)
        for k,x in components.items():losses.setdefault(k,[]).append(float(x.detach()))
        for k,x in counts.items():counts_total[k]+=x
        finalstep=step+1
        if finalstep%500==0:
            status(stage='training',arm=data.arm,variant=name,seed=seed,fit_index=index,total_fits=p['total_fits'],step=finalstep,
                nominal_full_pool_passes=counts_total['binary']/len(data.frame),sampler_rounds=streams.binary.passes,
                unique_binary_fraction=float(streams.seen.mean()),learning_rate=lr,
                seconds=previous_seconds+time.monotonic()-started,steps_per_second=500/(time.monotonic()-last_log))
            last_log=time.monotonic()
        if finalstep%interval!=0 and finalstep!=p['emergency_max_steps']:continue
        valstart=time.monotonic()
        metrics=validation.evaluate(ema.model,bank,data,'regression' if VARIANTS[variant]['regression'] else 'binary')
        exposure=counts_total['binary']/len(data.frame)
        event=controller.observe(metrics['selection_score'],exposure if streams.seen.all() else 0.)
        converged=event['converged']
        row=dict(step=finalstep,nominal_full_pool_passes=exposure,sampler_rounds=streams.binary.passes,
            unique_binary_fraction=float(streams.seen.mean()),learning_rate=lr,seconds=previous_seconds+time.monotonic()-started,
            validation_seconds=time.monotonic()-valstart,training_losses={k:float(np.mean(v)) for k,v in losses.items()},
            stale_validation_checks=controller.stale,**event,**metrics)
        history.append(row);losses={};write_json(run/'HISTORY.json',history)
        if event['best']:save_torch(run/'BEST.pt',dict(model=ema.model.state_dict(),step=finalstep,validation=metrics,identity=identity,lineage=lineage))
        save_torch(run/'RESUME.pt',dict(identity=identity,lineage=lineage,model=model.state_dict(),ema=ema.model.state_dict(),
            ema_updates=ema.updates,optimizer=opt.state_dict(),streams=streams.state(),controller=controller.__dict__.copy(),
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),history=history,step=finalstep,
            seconds=previous_seconds+time.monotonic()-started,counts=counts_total))
        status(stage='validation',arm=data.arm,variant=name,seed=seed,fit_index=index,total_fits=p['total_fits'],**row)
        if converged:break
        model.train();last_log=time.monotonic()
    if not converged:
        write_json(run/'NOT_CONVERGED.json',dict(step=finalstep,status='NOT_CONVERGED_NEEDS_EXTENSION'))
        raise RuntimeError(f'{run.name}: guard reached without convergence')
    best=torch.load(run/'BEST.pt',map_location='cpu',weights_only=False);ema.model.load_state_dict(best['model'])
    scores,_=predict(ema.model,bank,validation.frame);cal=calibrate(validation.frame,scores);write_json(run/'CALIBRATION.json',cal)
    save_torch(run/'model.pt',dict(architecture='assay_aware_research',config=config,endpoints=data.endpoints,
        model={k:v.detach().cpu() for k,v in ema.model.state_dict().items()},arm=data.arm,variant=name,seed=seed,identity=identity,
        regression_means=data.means.tolist(),regression_scales=data.scales.tolist(),lineage=lineage,
        score='mean of two binary logits; endpoint regressions remain auxiliary'))
    result=dict(status='CONVERGED_VALIDATION_PLATEAU',arm=data.arm,variant=name,seed=seed,identity=identity,lineage=lineage,
        initial_base_sha256=initial,best_step=best['step'],optimizer_steps=finalstep,best_validation=best['validation'],
        nominal_full_pool_passes=counts_total['binary']/len(data.frame),unique_binary_fraction=float(streams.seen.mean()),
        sample_counts=counts_total,sampling_trace=streams.trace,checkpoint_sha256=digest(run/'model.pt'),
        seconds=previous_seconds+time.monotonic()-started,test_used_for_training=False,production_replaced=False)
    write_json(run/'RESULT.json',result);status(stage='fit_complete',arm=data.arm,variant=name,seed=seed,fit_index=index,
        total_fits=p['total_fits'],seconds=result['seconds'],best_step=best['step'],optimizer_steps=finalstep)
    del model,ema,opt;torch.cuda.empty_cache();return result


def benchmark(bank,datasets,validation,p,config):
    rows=[]
    for variant in VARIANTS:
        for arm,data in datasets.items():
            model,_=create_model(data,p['seeds'][0],config);model.train();ema=WeightAverage(model,p['ema_decay'])
            opt=torch.optim.AdamW(model.parameters(),lr=p['learning_rate'],weight_decay=p['weight_decay'])
            streams=Streams(data,p['seeds'][0],variant,p);scales=(torch.tensor(data.means,device='cuda'),torch.tensor(data.scales,device='cuda'))
            for step in range(120):
                if step==20:torch.cuda.synchronize();start=time.monotonic()
                opt.zero_grad(set_to_none=True);loss,_,_=step_loss(model,bank,data,streams,variant,p,scales)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step();ema.update(model)
            torch.cuda.synchronize();perstep=(time.monotonic()-start)/100
            start=time.monotonic();validation.evaluate(ema.model,bank,data,'regression' if VARIANTS[variant]['regression'] else 'binary')
            valtime=time.monotonic()-start;interval=max(1000,math.ceil(2*len(data.frame)/p['batch_size']))
            row=dict(arm=arm,variant=variant,seconds_per_step=perstep,validation_seconds=valtime)
            for passes in [30,60,90]:
                steps=math.ceil(passes*len(data.frame)/p['batch_size'])
                row[f'seconds_at_{passes}_passes']=steps*perstep+math.ceil(steps/interval)*valtime
            rows.append(row);status(stage='benchmark',**row);del model,ema,opt;torch.cuda.empty_cache()
    eta=dict(rows=rows,total_fits=p['total_fits'],benchmark_weights_discarded=True,
        caveat='Measured throughput scenarios; convergence and rotating sampler setup can change actual duration. Add preparation, checkpoint I/O and final evaluation.')
    a=next(r for r in rows if r['arm']=='kdki_inactive' and r['variant']=='assay')
    for passes in [30,60,90]:eta[f'total_seconds_at_{passes}_passes']=len(p['seeds'])*(sum(r[f'seconds_at_{passes}_passes'] for r in rows)+2*a[f'seconds_at_{passes}_passes'])
    write_json(OUT/'THROUGHPUT_ETA.json',eta)


def evaluate_all(bank,datasets,p,config,results,identity):
    assert len(results)==p['total_fits'] and all(r['status']=='CONVERGED_VALIDATION_PLATEAU' for r in results)
    for r in results:assert digest(run_path(r['arm'],r['variant'],r['seed'])/'model.pt')==r['checkpoint_sha256']
    for seed in p['seeds']:
        fresh=[r['initial_base_sha256'] for r in results if r['seed']==seed and r['lineage']['parent_path'] is None]
        assert len(set(fresh))==1
    write_json(OUT/'ALL_FITS_FROZEN.json',dict(created_utc=now(),runs=results,no_test_scoring_before_all_converged=True))
    test=pd.read_parquet(SOURCE/'COMMON_TEST.parquet').reset_index(drop=True)
    predictions=test[['panel','pair_id','molecule_id','target_id','split_group','binary_label','document_disjoint']].copy()
    regression=pd.read_parquet(DATA/'TEST_REGRESSION.parquet');regression=regression[regression.pair_id.isin(test.pair_id)]
    rows=[];queryrows=[];regrows=[];summary=[]
    for r in results:
        arm,variant,seed=r['arm'],r['variant'],r['seed'];run=run_path(arm,variant,seed)
        model=AuxiliaryInteraction(config,ENDPOINTS[arm]).cuda();state=torch.load(run/'model.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state['model']);scores,_=predict(model,bank,test);cal=json.loads((run/'CALIBRATION.json').read_text())
        context=pd.read_parquet(OUT/f'{arm}_TEST_CONTEXT.parquet')
        assert context[['pair_id','panel']].equals(test[['pair_id','panel']])
        pairmetrics,queries,_=report_metrics(test,scores,cal,arm,seed)
        rows.extend(dict(variant=variant,**x) for x in pairmetrics);queryrows.extend(dict(variant=variant,**x) for x in queries)
        key=f'{arm}__{variant}__{seed}';predictions[key+'_score']=scores
        predictions[key+'_prob']=expit(cal['slope']*scores+cal['intercept'])
        for scope in ['source_assay_disjoint','source_assay_and_document_disjoint']:
            mask=context[scope].to_numpy()
            if mask.any():
                rr,_,_=report_metrics(test[mask],scores[mask],cal,arm,seed)
                rows.extend(dict(variant=variant,**{**x,'scope':scope}) for x in rr if x['scope']=='all')
        kd=test.panel.eq('AFFINITY_KD_KI').to_numpy();s=scores[kd];sub=test[kd].reset_index(drop=True)
        q={name:QueryMetrics(sub,col).evaluate(s) for name,col in [('target','target_id'),('drug','molecule_id')]}
        for name,metric in q.items():queryrows.append(dict(arm=arm,variant=variant,seed=seed,panel='AFFINITY_KD_KI',direction='macro_'+name,query='ALL_ELIGIBLE',**metric))
        inactive=test.panel.eq('EXPLICIT_INACTIVE').to_numpy();prob=expit(cal['slope']*scores+cal['intercept'])
        summary.append(dict(arm=arm,variant=variant,seed=seed,ap=average_precision_score(sub.binary_label,s),
            auroc=roc_auc_score(sub.binary_label,s),target_macro_ap=q['target']['macro_ap'],drug_macro_ap=q['drug']['macro_ap'],
            balanced_query_ap=(q['target']['macro_ap']+q['drug']['macro_ap'])/2,target_p5=q['target']['p5'],drug_p5=q['drug']['p5'],
            explicit_inactive_fpr=float((prob[inactive]>=cal['threshold']).mean()),
            validation_selection_score=r['best_validation']['selection_score']))
        if variant not in ['balanced','rotate150']:
            f=regression[regression.endpoint.isin(ENDPOINTS[arm])].reset_index(drop=True);_,rp=predict(model,bank,f)
            regrows.extend(dict(arm=arm,variant=variant,seed=seed,endpoint=e,**x) for e,x in regression_metrics(f,rp,datasets[arm]).items())
        status(stage='test_evaluation',arm=arm,variant=variant,seed=seed)
        del model,state;torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(OUT/'TEST_METRICS.csv',index=False)
    pd.DataFrame(queryrows).to_csv(OUT/'TEST_QUERY_METRICS.csv.gz',index=False)
    pd.DataFrame(regrows).to_csv(OUT/'TEST_REGRESSION_METRICS.csv',index=False)
    predictions.to_parquet(OUT/'TEST_PREDICTIONS.parquet',index=False,compression='zstd')
    comparisons=pd.DataFrame(summary);comparisons.to_csv(OUT/'OBJECTIVE_COMPARISON_PER_SEED.csv',index=False)
    metriccols=[c for c in comparisons if c not in ['arm','variant','seed']]
    grouped=comparisons.groupby(['arm','variant'])[metriccols].agg(['mean','std']);grouped.columns=['_'.join(x) for x in grouped.columns]
    grouped.reset_index().to_csv(OUT/'OBJECTIVE_COMPARISON.csv',index=False)
    pd.DataFrame([{k:r[k] for k in ['arm','variant','seed','optimizer_steps','best_step','seconds','nominal_full_pool_passes','unique_binary_fraction']} for r in results]).to_csv(OUT/'CONVERGENCE_SUMMARY.csv',index=False)
    # Compare to existing converged binary baselines without fitting or selecting on test.
    baseline=pd.read_csv(PARENT/'OBJECTIVE_COMPARISON.csv');baseline=baseline[baseline.variant.eq('binary')].copy()
    baseline.to_csv(OUT/'EXISTING_BINARY_BASELINES.csv',index=False)
    manifest=json.loads((OUT/'DATA_MANIFEST.json').read_text())
    for path,h in {**manifest['inputs'],**manifest['frozen_inputs']}.items():assert digest(ROOT/path)==h,path
    write_json(OUT/'SUMMARY.json',dict(status='COMPLETE_30_VALIDATION_CONVERGED_FITS',completed_utc=now(),
        fits=len(results),inputs_unchanged=True,production_replaced=False,limitations=p['limitations']))
    status(stage='complete',fits=len(results))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--benchmark-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True);lock=(OUT/'RUN.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write_json(OUT/'PROCESS.json',dict(pid=os.getpid(),started_utc=now(),command=sys.argv))
    try:
        status(stage='preparing');prepare();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
        p=json.loads((OUT/'PROTOCOL.json').read_text());config=json.loads((OUT/'MODEL_CONFIG.json').read_text())
        files=[Path(__file__),ROOT/'scripts/prepare_assay_aware_20260911.py',ROOT/'biomaster/assay_aware_training.py',
            ROOT/'biomaster/endpoint_multitask.py',ROOT/'biomaster/endpoint_ablation.py',ROOT/'biomaster/molecular_controls.py',
            ROOT/'biomaster/unified_interaction.py',ROOT/'biomaster/model_registry.py',ROOT/'biomaster/best_model_training.py',
            ROOT/'scripts/run_endpoint_multitask_20260911.py',ROOT/'scripts/run_endpoint_ablation_20260911.py',
            OUT/'PROTOCOL.json',OUT/'DATA_MANIFEST.json',OUT/'MODEL_CONFIG.json']
        identity={str(f.relative_to(ROOT)):digest(f) for f in files}
        status(stage='loading_feature_bank');bank=Bank();validation=Validation();datasets={}
        for arm in ENDPOINTS:status(stage='loading_arm',arm=arm);datasets[arm]=ArmData(arm,p)
        if not (OUT/'THROUGHPUT_ETA.json').exists():benchmark(bank,datasets,validation,p,config)
        if args.benchmark_only:status(stage='benchmark_complete_not_training');return
        results=[]
        for seed in p['seeds']:
            for variant in VARIANTS:
                for arm,data in datasets.items():
                    results.append(fit(data,variant,seed,bank,validation,p,config,identity,len(results)+1))
                    write_json(OUT/'QUEUE_PROGRESS.json',dict(updated_utc=now(),completed_fits=len(results),total_fits=p['total_fits'],
                        runs=[{k:r[k] for k in ['arm','variant','seed','status','seconds','optimizer_steps']} for r in results]))
            for parent in ['kdki_inactive','all_inactive']:
                results.append(fit(datasets['kdki_inactive'],'assay',seed,bank,validation,p,config,identity,len(results)+1,parent))
                write_json(OUT/'QUEUE_PROGRESS.json',dict(updated_utc=now(),completed_fits=len(results),total_fits=p['total_fits'],
                    runs=[{k:r[k] for k in ['arm','variant','seed','status','seconds','optimizer_steps']} for r in results]))
        evaluate_all(bank,datasets,p,config,results,identity)
    except Exception:
        write_json(OUT/'ERROR.json',dict(time_utc=now(),traceback=traceback.format_exc()))
        status(stage='failed',error_file=str(OUT/'ERROR.json'));raise


if __name__=='__main__':main()
