#!/usr/bin/env python3
"""Audited incremental research refit with drug/document-purged holdout and replay control.

All interaction-network parameters train; pretrained molecular/protein features stay
frozen. This is NOT a from-scratch fit of the entire BindingDB archive. Production
and the wet-lab 384 are immutable. No automatic model promotion is performed.
"""
import copy
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biomaster.anchored_interaction import AnchoredFeatureBank
from biomaster.best_model_training import WeightAverage
from biomaster.model_registry import build_model
from biomaster.portable_ranker_v2 import CatalogRanker, digest
from scripts.extend_biomaster_matrix_20260910 import BUNDLE, OUT as MATRIX, BASE, FROZEN, write_json
from scripts.evaluate_affinity_refresh_predictor_20260910 import classify, load_training

OUT = ROOT/'outputs/biomaster_bindingdb_incremental_20260910'
SOURCE = ROOT/'outputs/biomaster_v3_kirhub_temporal_20260906/temporal'
PARENT = ROOT/'outputs/biomaster_best_model_20260906/data/fullfit_2025/TRAIN.csv.gz'
SUPPLEMENT = ROOT/'outputs/affinity_predictor_audit_20260910/SUPPLEMENTAL_KD_KI_LABEL_CANDIDATES.csv'
DB = ROOT/'data/processed/biomaster_affinity_evidence_20260910.sqlite'
QC = ROOT/'data/external/affinity_refresh_20260910/SOURCE_RECORD_QC_OVERRIDES_20260910.csv'


def holdout_group(key):
    return int(hashlib.sha256(('20260910:'+key[:14]).encode()).hexdigest()[:8],16)%5 == 0


def prepare():
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.warning'); RDLogger.DisableLog('rdApp.error')
    a = pd.read_csv(SUPPLEMENT)
    original = a.copy()
    old = pd.read_csv(SOURCE/'OLD_DRUG_INDEX.csv')
    core = pd.read_csv(SOURCE/'TARGET_INDEX.csv.gz')
    targets = pd.read_csv(MATRIX/'TARGET_INDEX.csv.gz')
    train = pd.read_csv(PARENT)
    parent_exact = train.merge(old[['drug_feature_index','ligand_inchikey']], on='drug_feature_index').merge(
        core[['target_feature_index','target_chembl_id']], on='target_feature_index')
    parent_exact = set(parent_exact.ligand_inchikey+'__'+parent_exact.target_chembl_id)
    # Actual current parent membership, including stereoisomer/connectivity equivalents.
    key_cache = OUT/'PARENT_MOLECULE_KEYS.csv.gz'
    if key_cache.exists():
        keys = pd.read_csv(key_cache, keep_default_na=False)
    else:
        molecules = pd.read_csv(SOURCE/'MOLECULES.csv.gz')
        molecules = molecules[molecules.drug_feature_index.isin(train.drug_feature_index)]
        rows = []
        for i, r in enumerate(molecules.itertuples()):
            mol = Chem.MolFromSmiles(r.model_ligand_smiles)
            if mol is None: raise ValueError('invalid parent molecule')
            key = Chem.MolToInchiKey(mol)
            if not key: raise ValueError('parent molecule has no InChIKey')
            rows.append(dict(drug_feature_index=r.drug_feature_index, ligand_inchikey=key))
            if (i+1)%50000==0: print(json.dumps(dict(stage='parent_identity', done=i+1, total=len(molecules))), flush=True)
        keys = pd.DataFrame(rows); keys.to_csv(key_cache, index=False)
    assert set(keys.drug_feature_index)==set(train.drug_feature_index)
    parent_connect = train.merge(keys,on='drug_feature_index').merge(core[['target_feature_index','target_chembl_id']],on='target_feature_index')
    parent_connect = set(parent_connect.ligand_inchikey.str[:14]+'__'+parent_connect.target_chembl_id)
    _, previous_connect, _, _, prior_manifest = load_training()
    identity = pd.read_parquet(ROOT/'outputs/joint_screen_720x384_20260909/FULL_276480_PAIR_AUDIT.parquet',
                              columns=['ligand_inchikey','identity_scope_pass']).drop_duplicates()
    assert identity.ligand_inchikey.is_unique and len(identity)==720
    a = a.drop(columns=['identity_scope_pass']).merge(identity, on='ligand_inchikey', validate='many_to_one')
    a = a.merge(old[['ligand_inchikey','drug_feature_index','old_drug_index','model_ligand_smiles']],on='ligand_inchikey',validate='many_to_one')
    a = a.merge(targets[['target_chembl_id','matrix_target_index','core_model_target']],on='target_chembl_id',validate='many_to_one')
    a['structure_matches_model'] = [Chem.MolToInchiKey(Chem.MolFromSmiles(s))==k for s,k in zip(a.model_ligand_smiles,a.ligand_inchikey)]
    a['current_parent_exact_pair'] = a.pair_id.isin(parent_exact)
    connections = a.ligand_inchikey.str[:14]+'__'+a.target_chembl_id
    a['current_parent_connectivity_pair'] = connections.isin(parent_connect)
    a['previous_training_connectivity_pair'] = connections.isin(previous_connect)
    final = pd.read_csv(FROZEN/'SPR384_FINAL_DETAILED.csv')
    frozen_pairs = set(final['药物完整InChIKey']+'__'+final['新靶点ChEMBL编号'])
    frozen_connect = set(final['药物完整InChIKey'].str[:14]+'__'+final['新靶点ChEMBL编号'])
    a['wetlab_pair'] = connections.isin(frozen_connect)
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    records = pd.read_sql_query("select * from reviewed_comparable_numeric_evidence where endpoint in ('Kd','Ki')", con)
    assays = pd.read_sql_query('select l.reactant_set_id, a.assay_name, a.description from bindingdb_assay_links l join bindingdb_assays a using(entry_assay_id)', con)
    con.close()
    records['pair_id'] = records.ligand_key+'__'+records.target_chembl_id
    records = records[records.pair_id.isin(a.pair_id)].copy()
    records['observation_label'] = [classify(v,r) for v,r in zip(records.value_nM,records.relation)]
    descriptions = assays.fillna('').groupby('reactant_set_id').apply(
        lambda g: '\n'.join(sorted(set(g.assay_name+' '+g.description))), include_groups=False).to_dict()
    records['assay_description'] = [descriptions.get(str(r).split(':')[0],'') if s=='BindingDB_202609' else ''
                                  for r,s in zip(records.record_id,records.source)]
    # A flag is a quarantine trigger, not proof of a mutation. Domains without mutant flags remain contextual.
    mutation = r'(?i)\bmutant\b|\bmutations?\b|\bchimera\b|\bfusion\b|\b[A-Z]\d{2,4}[A-Z]\b'
    records['construct_flag'] = records.target_name.fillna('').str.contains(mutation,regex=True)
    # Assay descriptions can discuss multiple panel targets: retain separately, quarantine conservatively.
    records['assay_mutation_flag'] = records.assay_description.str.contains(mutation,regex=True)
    docs = defaultdict(set); bdb_states = defaultdict(set); all_states = defaultdict(set); flags = set()
    for r in records.itertuples():
        all_states[r.pair_id].add(r.observation_label)
        if r.source != 'BindingDB_202609': continue
        if r.construct_flag or r.assay_mutation_flag: flags.add(r.pair_id)
        bdb_states[r.pair_id].add(r.observation_label)
        m = json.loads(r.metadata_json)
        for prefix, val in [('patent',m.get('Patent Number','')),('doi',m.get('Article DOI','') or r.doi),('pmid',r.pmid)]:
            if pd.notna(val) and str(val).strip(): docs[r.pair_id].add(prefix+':'+str(val).lower().strip().removesuffix('.0'))
    a['bdb_decisive_same_label'] = [
        ('POSITIVE' if y else 'WEAK_NEGATIVE') in bdb_states[p] for p,y in zip(a.pair_id,a.binary_label)]
    a['source_conflict'] = [ {'POSITIVE','WEAK_NEGATIVE'} <= all_states[p] for p in a.pair_id]
    a['construct_or_assay_mutation_hold'] = a.pair_id.isin(flags)
    a['documents'] = a.pair_id.map(lambda p:';'.join(sorted(docs[p])))
    a['has_citation'] = a.pair_id.map(lambda p:bool(docs[p]))
    rules = dict(identity_hold=~a.identity_scope_pass, model_structure_mismatch=~a.structure_matches_model,
                 current_parent_overlap=a.current_parent_connectivity_pair,
                 previous_training_overlap=a.previous_training_connectivity_pair, wetlab_frozen=a.wetlab_pair,
                 no_bindingdb_decisive_label=~a.bdb_decisive_same_label, source_conflict=a.source_conflict,
                 construct_hold=a.construct_or_assay_mutation_hold, no_citation=~a.has_citation)
    a['exclusion_reasons'] = [';'.join(k for k,v in rules.items() if v.iloc[i]) for i in range(len(a))]
    a['eligible'] = a.exclusion_reasons.eq('')
    eligible = a[a.eligible].copy()
    eligible['split'] = np.where(eligible.ligand_inchikey.map(holdout_group), 'holdout', 'train')
    held_docs = set().union(*(docs[p] for p in eligible.loc[eligible.split.eq('holdout'),'pair_id']))
    purge = eligible.split.eq('train') & eligible.pair_id.map(lambda p:bool(docs[p]&held_docs))
    eligible.loc[purge,'split'] = 'purged_shared_holdout_document'
    tr = eligible[eligible.split.eq('train')]; te = eligible[eligible.split.eq('holdout')]
    assert len(tr)>0 and len(te)>0 and tr.binary_label.nunique()==te.binary_label.nunique()==2
    assert not set(tr.ligand_inchikey.str[:14]) & set(te.ligand_inchikey.str[:14])
    assert not set().union(*(docs[p] for p in tr.pair_id)) & held_docs
    assert not set(eligible.pair_id) & frozen_pairs
    a.to_csv(OUT/'LABEL_ELIGIBILITY_AUDIT.csv',index=False)
    eligible.to_csv(OUT/'FROZEN_SPLIT.csv',index=False)
    records.to_csv(OUT/'SOURCE_RECORD_AUDIT.csv.gz',index=False)
    summary = dict(original_supplemental_pairs=len(original), original_label_counts=original.binary_label.value_counts().to_dict(),
                   current_parent_exact_overlap=int(a.current_parent_exact_pair.sum()),
                   current_parent_connectivity_overlap=int(a.current_parent_connectivity_pair.sum()),
                   exclusions_nonexclusive={k:int(v.sum()) for k,v in rules.items()}, eligible_pairs=len(eligible),
                   eligible_label_counts=eligible.binary_label.value_counts().to_dict(),
                   split_counts=eligible.groupby(['split','binary_label']).size().unstack(fill_value=0).to_dict('index'),
                   wetlab_pairs_used=0, prior_training_sources=prior_manifest,
                   citation_purged_between_increment_and_holdout=True,
                   holdout_role='Retrospective development holdout, NOT an untouched prospective validation; parent historical document exposure is not certified.')
    write_json(OUT/'DATA_AUDIT.json',summary)
    return eligible, train, targets, old


def measures(frame, scores):
    result = []; y = frame.binary_label.to_numpy(int)
    for scope, mask in [('all',np.ones(len(frame),bool)), ('core384',frame.core_model_target.to_numpy(bool)),
                        ('extended506',~frame.core_model_target.to_numpy(bool))]:
        for h,name in enumerate(['drug_to_target','target_to_drug']):
            f = frame[mask].copy(); v = scores[mask,h]; yy = y[mask]
            row = dict(scope=scope,head=name,pairs=len(f),positive=int(yy.sum()),weak_negative=int(len(yy)-yy.sum()))
            row.update(auroc=float(roc_auc_score(yy,v)) if len(set(yy))==2 else None,
                       average_precision=float(average_precision_score(yy,v)) if len(set(yy))==2 else None)
            f['score'] = v; macro = []
            group = 'ligand_inchikey' if h==0 else 'target_chembl_id'
            for _, g in f.groupby(group):
                if g.binary_label.nunique()==2: macro.append(average_precision_score(g.binary_label,g.score))
            row['macro_query_ap'] = float(np.mean(macro)) if macro else None
            row['both_class_queries'] = len(macro); result.append(row)
    return result


@torch.inference_mode()
def score_frame(model, bank, frame):
    model.eval(); out = []
    for start in range(0,len(frame),512):
        f = frame.iloc[start:start+512]
        out.append(model(bank.batch(f.drug_feature_index.to_numpy(),f.matrix_target_index.to_numpy(),'global')).float().cpu().numpy())
    scores = np.concatenate(out); assert np.isfinite(scores).all()
    return scores


def paired_ci(frame, baseline, new, reps=1000):
    groups = list(frame.groupby(frame.ligand_inchikey.str[:14]).indices.values())
    rng = np.random.default_rng(20260910); y = frame.binary_label.to_numpy(int); auc=[]; ap=[]
    for _ in range(reps):
        ids = np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))])
        if len(np.unique(y[ids]))<2: continue
        auc.append(roc_auc_score(y[ids],new[ids])-roc_auc_score(y[ids],baseline[ids]))
        ap.append(average_precision_score(y[ids],new[ids])-average_precision_score(y[ids],baseline[ids]))
    return dict(auroc_delta_ci95=np.quantile(auc,[.025,.975]).tolist(),
                ap_delta_ci95=np.quantile(ap,[.025,.975]).tolist(),valid_reps=len(auc))


def main():
    OUT.mkdir(exist_ok=True); started=time.monotonic()
    frozen = json.loads((MATRIX/'FROZEN_INPUTS.json').read_text())
    for p,h in frozen.items(): assert digest(ROOT/p)==h
    protocol = dict(status='FIXED_BEFORE_NEW_FITS', seeds=[20260921,20260922,20260923], steps=500,
        learning_rate=1e-5, weight_decay=1e-4, replay_batch=256, extra_batch=16,
        extra_loss_weight=.25, ema_decay=.99, class_balanced_bce=True,
        initialization='frozen selected fullfit 2025', trainable='all interaction-network parameters',
        frozen='ESM2, DrugCLIP, Morgan features', arms=['replay_control','bindingdb_increment'],
        split='Drug connectivity hash modulo 5 == 0 is held out; remove all increment-training pairs sharing any holdout DOI/PMID/patent',
        selection='Fixed final EMA; no holdout tuning, seed selection, early stopping, or automatic deployment',
        interpretation='Versioned research incremental refit; not a full-archive from-scratch retraining',
        parent_sha256=digest(BUNDLE/'model.pt'), parent_train_sha256=digest(PARENT),
        supplemental_sha256=digest(SUPPLEMENT), qc_sha256=digest(QC), producer_sha256=digest(Path(__file__)))
    path=OUT/'PROTOCOL.json'
    if path.exists() and json.loads(path.read_text()) != protocol: raise ValueError('protocol drift; choose new output version')
    write_json(path,protocol)
    eligible, parent, targets, old = prepare()
    protocol_hash=digest(path); split_hash=digest(OUT/'FROZEN_SPLIT.csv')
    write_json(OUT/'FITTING_INPUTS.json',dict(protocol_sha256=protocol_hash,split_sha256=split_hash))
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    bank=AnchoredFeatureBank(BASE,ROOT/'outputs/biomaster_best_model_20260906/data/supplemental_features',SOURCE,'drugclip_morgan',local=False)
    bank.target_global=torch.tensor(np.load(MATRIX/'TARGET_GLOBAL.npy'),device='cuda')
    core=pd.read_csv(SOURCE/'TARGET_INDEX.csv.gz'); target_map=targets.set_index('uniprot_accession').matrix_target_index
    remap=core.set_index('target_feature_index').uniprot_accession.map(target_map)
    parent=parent.copy(); parent['matrix_target_index']=parent.target_feature_index.map(remap).astype(int)
    assert bank.required[torch.tensor(parent.drug_feature_index.to_numpy(),device='cuda')].all()
    state=torch.load(BUNDLE/'model.pt',map_location='cpu',weights_only=True)
    baseline=build_model(state['architecture'],state['config']).cuda(); baseline.load_state_dict(state['model']); baseline.eval()
    # Compare every eligible pair to completed matrix before training; catches drug/global bank mapping errors.
    original_scores=score_frame(baseline,bank,eligible)
    expected=np.load(MATRIX/'DIRECTIONAL_LOGITS.npy')[eligible.old_drug_index.to_numpy(int),eligible.matrix_target_index.to_numpy(int)]
    difference=float(np.abs(original_scores-expected).max()); assert difference<2e-4, difference
    held=eligible[eligible.split.eq('holdout')].copy().reset_index(drop=True)
    increment=eligible[eligible.split.eq('train')].copy().reset_index(drop=True)
    base_scores=score_frame(baseline,bank,held)
    rows=[dict(model='frozen_parent',seed=0,**r) for r in measures(held,base_scores)]
    all_rows=[dict(model='frozen_parent',seed=0,**r) for r in measures(eligible,original_scores)]
    pd.DataFrame(all_rows).to_csv(OUT/'BASELINE_ALL_ELIGIBLE_METRICS.csv',index=False)
    predictions=held[['pair_id','ligand_inchikey','target_chembl_id','binary_label','core_model_target']].copy()
    predictions['parent_d2t']=base_scores[:,0]; predictions['parent_t2d']=base_scores[:,1]
    results={}; history=[]
    # Fixed replay exposure across both arms, independent new-label random generator.
    py=parent.binary_label.to_numpy(np.float32); iy=increment.binary_label.to_numpy(np.float32)
    def loss(model, frame, labels, prevalence):
        scores=model(bank.batch(frame.drug_feature_index.to_numpy(),frame.matrix_target_index.to_numpy(),'global'))
        y=torch.tensor(labels,device='cuda'); w=torch.where(y.bool(),.5/prevalence,.5/(1-prevalence))
        return (F.binary_cross_entropy_with_logits(scores,y[:,None].expand(-1,2),reduction='none')*w[:,None]).mean()
    for seed in protocol['seeds']:
        for arm in protocol['arms']:
            run=OUT/f'{arm}_seed_{seed}'; run.mkdir(exist_ok=True)
            torch.manual_seed(seed); model=copy.deepcopy(baseline); model.train()
            opt=torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
            ema=WeightAverage(model,protocol['ema_decay']); replay_rng=np.random.default_rng(seed); extra_rng=np.random.default_rng(seed+1)
            trace=hashlib.sha256(); losses=[]
            for step in range(protocol['steps']):
                ids=replay_rng.choice(len(parent),protocol['replay_batch'],replace=False)
                extra=extra_rng.choice(len(increment) if arm=='bindingdb_increment' else len(parent),protocol['extra_batch'],replace=False)
                trace.update(ids.astype('<i8').tobytes()); opt.zero_grad(set_to_none=True)
                old_loss=loss(model,parent.iloc[ids],py[ids],float(py.mean()))
                if arm=='bindingdb_increment': extra_loss=loss(model,increment.iloc[extra],iy[extra],float(iy.mean()))
                else: extra_loss=loss(model,parent.iloc[extra],py[extra],float(py.mean()))
                objective=old_loss+protocol['extra_loss_weight']*extra_loss
                if not torch.isfinite(objective): raise FloatingPointError('nonfinite training objective')
                objective.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
                opt.step(); ema.update(model); losses.append(float(objective.detach()))
                if (step+1)%100==0:
                    status=dict(stage='training',arm=arm,seed=seed,step=step+1,steps=protocol['steps'])
                    write_json(OUT/'STATUS.json',status);print(json.dumps(status),flush=True)
            scores=score_frame(ema.model,bank,held); results[(arm,seed)]=scores
            rows.extend(dict(model=arm,seed=seed,**r) for r in measures(held,scores))
            for h,name in enumerate(['d2t','t2d']): predictions[f'{arm}_{seed}_{name}']=scores[:,h]
            torch.save(dict(architecture=state['architecture'],config=state['config'],model={k:v.cpu() for k,v in ema.model.state_dict().items()},
                            role='RESEARCH_INCREMENTAL_MODEL_NOT_DEPLOYED',protocol_sha256=protocol_hash,split_sha256=split_hash,seed=seed,arm=arm),run/'model.pt')
            history.append(dict(arm=arm,seed=seed,steps=protocol['steps'],mean_loss=float(np.mean(losses)),replay_exposure_sha256=trace.hexdigest(),
                                checkpoint_sha256=digest(run/'model.pt')))
            write_json(OUT/'HISTORY.json',history); pd.DataFrame(rows).to_csv(OUT/'HOLDOUT_METRICS.csv',index=False)
            del model,opt,ema
    for seed in protocol['seeds']:
        hs=[h['replay_exposure_sha256'] for h in history if h['seed']==seed]; assert len(set(hs))==1
    predictions.to_csv(OUT/'HOLDOUT_PREDICTIONS.csv',index=False)
    cis=[]
    for seed in protocol['seeds']:
        new=results[('bindingdb_increment',seed)]
        for ref_name,ref in [('frozen_parent',base_scores),('replay_control',results[('replay_control',seed)])]:
            for h,name in enumerate(['drug_to_target','target_to_drug']):
                cis.append(dict(seed=seed,reference=ref_name,head=name,**paired_ci(held,ref[:,h],new[:,h])))
    write_json(OUT/'PAIRED_DRUG_BOOTSTRAP.json',cis)
    for p,h in frozen.items(): assert digest(ROOT/p)==h, p
    assert digest(path)==protocol_hash and digest(OUT/'FROZEN_SPLIT.csv')==split_hash
    summary=dict(status='COMPLETE_RESEARCH_INCREMENTAL_COMPARISON',production_replaced=False,
        wetlab_baseline_unchanged=True,training_runs=6,trainable_parameters=sum(p.numel() for p in baseline.parameters()),
        replay_source_pairs=len(parent),increment_train_pairs=len(increment),holdout_pairs=len(held),
        feature_score_compatibility_max_abs_difference=difference,metrics=rows,
        limitations=['This is incremental interaction-network training, not a from-scratch full BindingDB fit.',
                     'Previously examined retrospective data; no prospective hit-rate claim.',
                     'Drug and citation separation applies to NEW increment versus holdout; frozen parent/pretrained encoder source exposure is not fully certified.',
                     'Only explicit decisive Kd/Ki labels; weak negatives mean >=10 uM, not proof of no binding.',
                     'Construct text filtering is conservative automated QC, not exhaustive paper-by-paper verification.',
                     'No seed is selected or automatically promoted from these holdout results.'],
        seconds=round(time.monotonic()-started,1))
    write_json(OUT/'SUMMARY.json',summary);write_json(OUT/'STATUS.json',dict(status=summary['status'],seconds=summary['seconds']))
    print(json.dumps({k:v for k,v in summary.items() if k!='metrics'}),flush=True)


if __name__=='__main__': main()
