#!/usr/bin/env python3
"""Fresh chronological V3/J training. This entrypoint never reads test labels.

Development: <=2020 fit, 2021-2022 first-seen validation. Final: fresh refit
through 2022 with development-selected epoch counts and unchanged schedules.
The current all-year production checkpoints are never loaded here.
"""
from argparse import Namespace
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from train_biomaster_selectivity_v3 import Runtime as BaseRuntime, train_batch, write_json, now
from biomaster.odti_support_data_v3 import SupportBatch
from biomaster.selectivity_training_v3 import QueryBatchStream, support_summary_features
from biomaster.odti_v3_incremental import IncrementalConfig, IncrementalV3
from biomaster.old_relation_retrieval import known_relation_loss
from biomaster.ranking_audit import risk_set_ranking
from build_biomaster_odti_v4_features import sha256

OUT = ROOT/'outputs/biomaster_v3_kirhub_temporal_20260906/temporal'
CONFIG = ROOT/'configs/biomaster_v3_temporal_20260906.json'


def identity():
    names = ['scripts/train_v3_temporal_20260906.py', 'scripts/train_biomaster_selectivity_v3.py',
             'biomaster/odti_v2.py', 'biomaster/odti_support_v3.py', 'biomaster/odti_support_data_v3.py',
             'biomaster/selectivity_training_v3.py', 'biomaster/odti_v3_incremental.py',
             'biomaster/old_relation_retrieval.py', 'biomaster/ranking_audit.py']
    return {'protocol_sha256': sha256(OUT/'PROTOCOL.json'), 'data_manifest_sha256': sha256(OUT/'DATA_MANIFEST.json'),
            'source_sha256': {n:sha256(ROOT/n) for n in names}}


def log(row):
    row = {'utc': now(), **row}
    write_json(OUT/'TRAIN_STATUS.json', row)
    print(json.dumps(row), flush=True)


class Stage:
    def __init__(self, name, manifest):
        self.name, self.folder = name, OUT/name
        self.cutoff = 2020 if name == 'development' else 2022
        self.train = pd.read_csv(OUT/('DEVELOPMENT_TRAIN.csv.gz' if name == 'development' else 'FINAL_TRAIN.csv.gz'))
        assert self.train.max_document_year.max() <= self.cutoff
        self.pool = pd.read_csv(OUT/'QUERY_POOL.csv.gz')  # pair identities only; no future labels
        self.old = pd.read_csv(OUT/'OLD_DRUG_INDEX.csv')
        self.targets = pd.read_csv(OUT/'TARGET_INDEX.csv.gz')
        self.molecules = pd.read_csv(OUT/'MOLECULES.csv.gz')
        self.index = pd.MultiIndex.from_frame(self.pool[['drug_feature_index','target_feature_index']])
        self.positions = self.rows(self.train)
        self.data = self.pool.copy()
        self.data['binary_label'] = np.nan
        self.data.loc[self.positions, 'binary_label'] = self.train.binary_label.to_numpy()
        self.families = sorted(self.targets.target_assay_family.unique())
        self.target_families = self.targets.target_assay_family.map({s:i for i,s in enumerate(self.families)}).to_numpy(int)
        self.data['family_index'] = self.target_families[self.pool.target_feature_index]
        self.data['murcko_scaffold'] = self.molecules.murcko_scaffold.to_numpy()[self.pool.drug_feature_index]
        self.data['entity_key'] = self.molecules.entity_key.to_numpy()[self.pool.drug_feature_index]
        self.labels = self.data.binary_label.to_numpy(np.float32)
        self.supports = {scope:SupportBatch(**{key:np.load(path,mmap_mode='r') for key,path in files.items()})
                         for scope,files in manifest['supports'][name].items() if scope in ['pool','old']}
        self.axes = {'pool':(self.pool.drug_feature_index.to_numpy(int),self.pool.target_feature_index.to_numpy(int)),
                     'old':(np.repeat(self.old.drug_feature_index.to_numpy(int),384),np.tile(np.arange(384),720))}
        self.stats = {key:support_summary_features(s.similarities,s.mask) for key,s in self.supports.items()}
        self.known_train = self.known(self.train)
        self.risk = np.load(OUT/'RISK_SETS.npz')[name]
        self.val = self.val_rows = self.known_val = self.val_groups = None
        if name == 'development':
            self.val = pd.read_csv(OUT/'DEVELOPMENT_VALIDATION_NEW.csv.gz')
            assert self.val.min_document_year.min() >= 2021 and self.val.max_document_year.max() <= 2022
            self.val_rows = self.rows(self.val)
            assert not np.intersect1d(self.positions,self.val_rows).size
            self.known_val = self.known(self.val)
            assert not (self.known_val & ~self.risk).any()
            y = self.val.binary_label.to_numpy()
            self.val_groups = [[np.asarray(pos) for pos in self.val.groupby(key,sort=False).indices.values()
                                if np.unique(y[pos]).size == 2]
                               for key in ['drug_feature_index','target_feature_index']]
            assert all(len(g) for g in self.val_groups)
        # Actual cached support identities must be same-target training labels,
        # with the query molecule excluded. Audit all cached entries vectorially.
        lookup = set(zip(self.train.drug_feature_index,self.train.target_feature_index,self.train.binary_label))
        valid_codes=np.sort(np.array([(a*384+b)*2+c for a,b,c in lookup],dtype=np.int64))
        for scope,support in self.supports.items():
            d,t = self.axes[scope]
            for start in range(0,len(d),8192):
                ix=np.asarray(support.indices[start:start+8192]); mask=np.asarray(support.mask[start:start+8192])
                assert not (mask & (ix == d[start:start+8192,None,None])).any()
                # Compare encoded triples with a precomputed sorted key array below.
                flat_d=ix[mask]; flat_t=np.broadcast_to(t[start:start+8192,None,None],ix.shape)[mask]
                flat_y=np.broadcast_to(np.arange(2)[None,:,None],ix.shape)[mask]
                codes=(flat_d*384+flat_t)*2+flat_y
                positions=np.searchsorted(valid_codes,codes)
                assert (positions < len(valid_codes)).all() and np.array_equal(valid_codes[positions],codes)
        log({'stage':name,'event':'support_audit_passed','train_rows':len(self.train),
             'train_old_positive_pairs':int(self.known_train.sum()),'cutoff':self.cutoff})

    def rows(self, frame):
        values=self.index.get_indexer(pd.MultiIndex.from_frame(frame[['drug_feature_index','target_feature_index']]))
        assert (values >= 0).all()
        return values

    def known(self, frame):
        out=np.zeros((720,384),bool)
        part=frame.loc[frame.binary_label.eq(1)&frame.drug_feature_index.isin(self.old.drug_feature_index)]
        mapping=self.old.set_index('drug_feature_index').old_drug_index
        out[part.drug_feature_index.map(mapping).to_numpy(int),part.target_feature_index.to_numpy(int)]=True
        return out

    def base(self, seed):
        torch.manual_seed(seed)
        args=Namespace(features=OUT/'MORGAN.npy',target_features=OUT/'PROTBERT.npy',target_aux=OUT/'ESM2_LEGACY.npy',
                       width=192,pair_hidden=256,dropout=.12,device='cuda',precision='fp32',support='both',local=False,
                       support_dropout=.2,support_item_dropout=.1,scaffold_dropout=.2,eval_batch_size=1024,
                       micro_batch_size=256,observed_weight=.25,rank_weight=0.,rank_loss='pairwise',grad_clip=5.)
        return BaseRuntime(args,self.data,{'families':self.families},None,self.supports['pool'])

    def measured_validation(self, scores):
        if scores.ndim == 1:scores=np.repeat(scores[:,None],2,axis=1)
        y=self.val.binary_label.to_numpy()
        return {direction:float(np.mean([average_precision_score(y[g],scores[g,i]) for g in self.val_groups[i]]))
                for i,direction in enumerate(['d2t','t2d'])}


def train_base(stage, seed, p, ident, fixed_epoch=None):
    folder=stage.folder/'base'/f'seed_{seed}'
    if (folder/'RESULT.json').exists():
        result=json.loads((folder/'RESULT.json').read_text()); assert result['identity']==ident
        return result
    if folder.exists() and any(folder.iterdir()):raise FileExistsError(f'incomplete run {folder}')
    folder.mkdir(parents=True)
    rt=stage.base(seed)
    optimizer=torch.optim.AdamW(rt.parameters(),lr=p['base_lr'],weight_decay=.0001)
    stream=QueryBatchStream(rt.drugs,rt.labels,stage.positions,64,16)
    best=-float('inf'); best_epoch=0; history=[]; started=time.monotonic()
    epochs=p['base_max_epochs'] if fixed_epoch is None else fixed_epoch
    for epoch in range(1,epochs+1):
        rng=np.random.default_rng(seed+(epoch-1)*104729)
        evidence_rng=np.random.default_rng(seed+(epoch-1)*104729+11)
        torch.manual_seed(seed+(epoch-1)*104729)
        coverage=rng.permutation(stage.positions)
        queries=stream.batches(seed+(epoch-1)*104729+7)
        lr=p['base_lr']*(.1+.9*(1+math.cos(math.pi*(epoch-1)/p['base_max_epochs']))/2)
        for group in optimizer.param_groups:group['lr']=lr
        exposure=hashlib.sha256();losses=[]
        batches=math.ceil(len(coverage)/p['base_batch_size'])
        for step in range(batches):
            rows=coverage[step*p['base_batch_size']:(step+1)*p['base_batch_size']]
            query_rows=next(queries)
            for part in [rows,query_rows]:exposure.update(part.astype('<i8').tobytes())
            cov=train_batch(rt,rows,optimizer,False,evidence_rng,False)
            query=train_batch(rt,query_rows,optimizer,True,evidence_rng,False)
            losses.append((cov[0]+query[0])/2)
            if (step+1)%150 == 0:
                log({'stage':stage.name,'kind':'base','seed':seed,'epoch':epoch,'step':step+1,'steps':batches})
        validation=None
        if fixed_epoch is None:
            scores,_=rt.predict(stage.val_rows)
            validation=stage.measured_validation(scores)
            value=float(np.mean(list(validation.values())))
        else:value=float(epoch)
        if value > best:
            best,best_epoch=value,epoch
            torch.save({'model_state':{k:v.detach().cpu().clone() for k,v in rt.model.state_dict().items()},
                        'base_config':asdict(rt.base_config),'support_config':asdict(rt.support_config),
                        'family_count':len(stage.families),'epoch':epoch,'seed':seed,'cutoff':stage.cutoff,
                        'identity':ident,'initialization':'fresh_random','validation':validation},folder/'BEST.pt')
        row={'stage':stage.name,'kind':'base','seed':seed,'epoch':epoch,'validation':validation,
             'mean_bce':float(np.mean(losses)),'selected_epoch':best_epoch,'coverage_rows':len(coverage),
             'exposure_sha256':exposure.hexdigest(),'seconds':time.monotonic()-started}
        history.append(row);write_json(folder/'HISTORY.json',history);log(row)
    result={'identity':ident,'stage':stage.name,'seed':seed,'selected_epoch':best_epoch,
            'checkpoint_sha256':sha256(folder/'BEST.pt'),'seconds':time.monotonic()-started,
            'train_rows':len(stage.train),'train_max_year':stage.cutoff,'test_used':False}
    write_json(folder/'RESULT.json',result)
    del rt,optimizer;gc.collect();torch.cuda.empty_cache()
    return result


def build_caches(stage,p,ident):
    folder=stage.folder/'cache';folder.mkdir(exist_ok=True)
    if (folder/'MANIFEST.json').exists():
        assert json.loads((folder/'MANIFEST.json').read_text())['identity']==ident
        return
    checkpoints=[stage.folder/'base'/f'seed_{seed}'/'BEST.pt' for seed in p['base_seeds']]
    hidden={s:np.lib.format.open_memmap(folder/f'{s}_HIDDEN.npy',mode='w+',dtype='float32',shape=(len(d),384))
            for s,(d,t) in stage.axes.items()}
    logits={s:np.zeros(len(d),np.float32) for s,(d,t) in stage.axes.items()}
    for member,(seed,checkpoint) in enumerate(zip(p['base_seeds'],checkpoints)):
        rt=stage.base(seed);state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['identity']==ident and state['cutoff']==stage.cutoff
        rt.model.load_state_dict(state['model_state']);rt.mode(False)
        with torch.no_grad():
            for scope,(d,t) in stage.axes.items():
                for start in range(0,len(d),1024):
                    ix=np.arange(start,min(start+1024,len(d)))
                    result=rt.model(**rt.inputs(d[ix],t[ix],stage.target_families[t[ix]],stage.supports[scope],ix))
                    logits[scope][ix] += result['final_logit'].cpu().numpy()/len(checkpoints)
                    hidden[scope][ix,member*192:(member+1)*192]=result['interaction_hidden'].cpu().numpy()
                hidden[scope].flush()
                log({'stage':stage.name,'event':'base_cache','seed':seed,'scope':scope,'rows':len(d)})
        del rt,state;gc.collect();torch.cuda.empty_cache()
    for scope,values in logits.items():np.save(folder/f'{scope}_BASE.npy',values)
    for scope,values in stage.stats.items():np.save(folder/f'{scope}_SUPPORT.npy',values)
    write_json(folder/'MANIFEST.json',{'identity':ident,'checkpoints':{str(f.relative_to(ROOT)):sha256(f) for f in checkpoints},
               'files':{f.name:sha256(f) for f in folder.glob('*.npy')}})


class AdapterRuntime:
    def __init__(self,stage):
        self.stage=stage
        self.banks={scope:{name:torch.from_numpy(np.load(stage.folder/'cache'/f'{scope}_{name}.npy')).cuda()
                           for name in ['BASE','HIDDEN','SUPPORT']} for scope in ['pool','old']}
        self.axes={scope:tuple(torch.as_tensor(v,device='cuda') for v in pair) for scope,pair in stage.axes.items()}
        self.drug=torch.from_numpy(np.load(OUT/'BERMOL.npy')).cuda()
        self.target=torch.from_numpy(np.load(OUT/'ESM2_COMPLETE.npy')).cuda()
        self.y=torch.from_numpy(stage.labels).cuda()
        self.base_validation=self.validation(None) if stage.name=='development' else None

    def forward(self,model,scope,rows):
        rows=torch.as_tensor(rows,device='cuda');bank=self.banks[scope];d,t=self.axes[scope]
        return model(bank['BASE'][rows],bank['HIDDEN'][rows],bank['SUPPORT'][rows],self.drug[d[rows]],self.target[t[rows]])

    @torch.no_grad()
    def predict(self,model,scope,rows=None):
        if rows is None:rows=np.arange(len(self.banks[scope]['BASE']))
        if model is None:
            base=self.banks[scope]['BASE'][torch.as_tensor(rows,device='cuda')].cpu().numpy()
            return np.repeat(base[:,None],2,axis=1)
        model.eval()
        return np.concatenate([self.forward(model,scope,rows[start:start+4096]).cpu().numpy() for start in range(0,len(rows),4096)])

    def validation(self,model):
        stage=self.stage
        old=self.predict(model,'old').reshape(720,384,2)
        measured=stage.measured_validation(self.predict(model,'pool',stage.val_rows))
        result={}
        for i,direction in enumerate(['d2t','t2d']):
            y,s,risk=stage.known_val,old[:,:,i],stage.risk
            q,c=stage.old.ligand_inchikey.to_numpy(),stage.targets.target_chembl_id.to_numpy()
            if i:y,s,risk,q,c=y.T,s.T,risk.T,c,q
            metrics,_,_=risk_set_ranking(y,s,risk,q,c)
            result[direction]={'ap':metrics['macro_ap'],'r20':metrics['macro_recall_20'],
                               'positive_queries':metrics['positive_queries'],'measured_ap':measured[direction]}
        result['selection']=sum(.35*result[d]['ap']+.15*result[d]['r20'] for d in ['d2t','t2d'])
        return result


def train_adapter(rt,seed,p,ident,fixed_epoch=None):
    stage=rt.stage;folder=stage.folder/'adapter'/f'seed_{seed}'
    if (folder/'RESULT.json').exists():
        result=json.loads((folder/'RESULT.json').read_text());assert result['identity']==ident
        return result
    if folder.exists() and any(folder.iterdir()):raise FileExistsError(f'incomplete run {folder}')
    folder.mkdir(parents=True)
    torch.manual_seed(seed);config=IncrementalConfig(evidence=True,pretrained=True)
    model=IncrementalV3(config).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=p['adapter_lr'],weight_decay=.0001)
    known=stage.known_train;queries=[np.flatnonzero(known.any(1)),np.flatnonzero(known.any(0))]
    steps=[math.ceil(len(q)/8) for q in queries]
    assert min(steps)>0
    best=rt.base_validation;best_epoch=0;history=[];started=time.monotonic()
    epochs=p['adapter_max_epochs'] if fixed_epoch is None else fixed_epoch

    def save(epoch):
        torch.save({'config':asdict(config),'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    'epoch':epoch,'seed':seed,'identity':ident,'cutoff':stage.cutoff,'validation':best,
                    'initialization':'fresh_zero_residual','retrieval_labels':'dated_train_experimental_positives'},folder/'BEST.pt')

    save(0)
    if best is not None:history.append({'epoch':0,'validation':best,'eligible':True})
    for epoch in range(1,epochs+1):
        rng=np.random.default_rng(seed+epoch*104729);torch.manual_seed(seed+epoch*104729)
        model.train();losses=[];exposure=hashlib.sha256()
        lr=p['adapter_lr']*(.2+.8*(1+math.cos(math.pi*(epoch-1)/p['adapter_max_epochs']))/2)
        for group in optimizer.param_groups:group['lr']=lr

        def update(loss):
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite adapter loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step();model.constrain();losses.append(float(loss.detach()))

        coverage=rng.permutation(stage.positions)
        for start in range(0,len(coverage),4096):
            ix=coverage[start:start+4096];exposure.update(ix.astype('<i8').tobytes())
            scores=rt.forward(model,'pool',ix)
            yy=rt.y[torch.as_tensor(ix,device='cuda')][:,None].expand(-1,2)
            update(.25*F.binary_cross_entropy_with_logits(scores,yy))
        for direction,q in enumerate(queries):
            order=rng.permutation(q)
            for start in range(0,len(order),8):
                query=order[start:start+8]
                if direction==0:rows=query[:,None]*384+np.arange(384)[None,:];y=known[query]
                else:rows=np.arange(720)[None,:]*384+query[:,None];y=known[:,query].T
                exposure.update(rows.astype('<i8').tobytes())
                scores=rt.forward(model,'old',rows.reshape(-1))[:,direction].reshape(rows.shape)
                loss=known_relation_loss(scores,torch.as_tensor(y,device='cuda'),2.)
                update(.5*(sum(steps)/2/steps[direction])*loss)
        validation=rt.validation(model) if fixed_epoch is None else None
        eligible=(fixed_epoch is not None or all(validation[d]['measured_ap']>=rt.base_validation[d]['measured_ap']-.01 for d in ['d2t','t2d']))
        if fixed_epoch is not None or (eligible and validation['selection']>best['selection']+1e-12):
            best,best_epoch=validation,epoch;save(epoch)
        row={'stage':stage.name,'kind':'adapter','seed':seed,'epoch':epoch,'validation':validation,
             'eligible':eligible,'selected_epoch':best_epoch,'mean_loss':float(np.mean(losses)),
             'exposure_sha256':exposure.hexdigest(),'seconds':time.monotonic()-started}
        history.append(row);write_json(folder/'HISTORY.json',history);log(row)
    result={'stage':stage.name,'seed':seed,'selected_epoch':best_epoch,'validation':best,'baseline_validation':rt.base_validation,
            'identity':ident,'checkpoint_sha256':sha256(folder/'BEST.pt'),'seconds':time.monotonic()-started,
            'known_training_pairs':int(known.sum()),'train_max_year':stage.cutoff,'test_used':False}
    write_json(folder/'RESULT.json',result)
    del model,optimizer;gc.collect();torch.cuda.empty_cache()
    return result


def score_final(rt,p,ident):
    folder=rt.stage.folder/'scores';folder.mkdir(exist_ok=True)
    files={};members=[]
    for seed in p['adapter_seeds']:
        checkpoint=rt.stage.folder/'adapter'/f'seed_{seed}'/'BEST.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['identity']==ident
        model=IncrementalV3(IncrementalConfig(**state['config'])).cuda();model.load_state_dict(state['model'])
        member={scope:rt.predict(model,scope) for scope in ['pool','old']};members.append(member)
        del model,state
    estimator=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500,C=1.,random_state=0))
    stage=rt.stage;estimator.fit(stage.stats['pool'][stage.positions],stage.labels[stage.positions])
    for scope in ['pool','old']:
        scores={'temporal_v3':rt.predict(None,scope),
                'temporal_J':np.mean([m[scope] for m in members],axis=0),
                'positive_nearest':np.repeat(stage.stats[scope][:,1,None],2,axis=1),
                'pn_logistic':np.repeat(estimator.decision_function(stage.stats[scope])[:,None],2,axis=1)}
        path=folder/f'{scope.upper()}_SCORES.npz';np.savez_compressed(path,**scores);files[path.name]=sha256(path)
    write_json(folder/'BASELINE_FIT.json',{'fit_max_year':2022,'fit_rows':len(stage.positions),
        'mean':estimator[0].mean_.tolist(),'scale':estimator[0].scale_.tolist(),
        'coef':estimator[1].coef_.tolist(),'intercept':estimator[1].intercept_.tolist()})
    write_json(folder/'MANIFEST.json',{'identity':ident,'files':files,'test_labels_read':False})


def main():
    torch.set_num_threads(4)
    p=json.loads(CONFIG.read_text());manifest=json.loads((OUT/'DATA_MANIFEST.json').read_text())
    assert p['base_max_epochs']==6 and p['adapter_max_epochs']==24
    assert json.loads((OUT/'PROTOCOL.json').read_text())['config_sha256']==sha256(CONFIG)
    if (OUT/'TEST_RELEASE.json').exists():raise RuntimeError('test already released; refitting forbidden')
    ident=identity()
    for path,digest in manifest['files'].items():assert sha256(ROOT/path)==digest,path
    frozen=OUT/'TRAINING_IDENTITY.json'
    if frozen.exists():assert json.loads(frozen.read_text())==ident
    else:write_json(frozen,ident)
    dev=Stage('development',manifest)
    bases=[train_base(dev,seed,p,ident) for seed in p['base_seeds']]
    build_caches(dev,p,ident)
    rt=AdapterRuntime(dev)
    adapters=[train_adapter(rt,seed,p,ident) for seed in p['adapter_seeds']]
    selection={'status':'FROZEN_BEFORE_FINAL_REFIT_AND_TEST','utc':now(),'identity':ident,
               'bases':bases,'adapters':adapters,'test_labels_read':False,
               'base_epochs':{str(r['seed']):r['selected_epoch'] for r in bases},
               'adapter_epochs':{str(r['seed']):r['selected_epoch'] for r in adapters}}
    if (OUT/'SELECTION.json').exists():
        old_selection=json.loads((OUT/'SELECTION.json').read_text())
        assert old_selection['identity']==ident and old_selection['base_epochs']==selection['base_epochs'] and old_selection['adapter_epochs']==selection['adapter_epochs']
    else:write_json(OUT/'SELECTION.json',selection)
    del rt,dev;gc.collect();torch.cuda.empty_cache()
    final=Stage('final',manifest)
    for seed in p['base_seeds']:train_base(final,seed,p,ident,selection['base_epochs'][str(seed)])
    build_caches(final,p,ident)
    rt=AdapterRuntime(final)
    for seed in p['adapter_seeds']:train_adapter(rt,seed,p,ident,selection['adapter_epochs'][str(seed)])
    score_final(rt,p,ident)
    checkpoints=list((OUT/'final').glob('base/*/BEST.pt'))+list((OUT/'final').glob('adapter/*/BEST.pt'))
    release={'status':'FROZEN_MODELS_TEST_MAY_BE_EVALUATED','utc':now(),'identity':ident,
             'selection_sha256':sha256(OUT/'SELECTION.json'),'scores_manifest_sha256':sha256(OUT/'final/scores/MANIFEST.json'),
             'checkpoints':{str(f.relative_to(ROOT)):sha256(f) for f in checkpoints},'test_labels_read':False,
             'train_max_year':2022,'fresh_base_and_adapter_refit':True}
    write_json(OUT/'TEST_RELEASE.json',release);log(release)


if __name__=='__main__':main()
