#!/usr/bin/env python3
"""Evaluate frozen recommended A/B weights on the exact historical SPR panels.

No training, tuning, candidate selection, production updates, or DTIAM TEST access.
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
from biomaster.training_dataset_v1 import molecule_identity

OUT = ROOT / 'outputs/palinova_ab_same_task_20260914'
DATA = ROOT / 'data/processed/biomaster_training_full_20260910_v1'
AB = ROOT / 'outputs/biomaster_endpoint_ablation_20260911'
FEATURES = AB / 'features'
CAT = ROOT / 'outputs/biomaster_matrix_720x890_20260910'
BUNDLE = ROOT / 'outputs/biomaster_best_model_20260906/retargetmap_selected_v1'
OLD = ROOT / 'outputs/old_drug_target_sota_v1/drug_centric_ranker_v1'
REFRESH = ROOT / 'outputs/affinity_predictor_audit_20260910'
BDB = ROOT / 'outputs/spr384_bindingdb_quality_review_20260910'
REGISTRY = ROOT / 'outputs/biomaster_model_consolidation_20260911/RECOMMENDED_MODELS.json'
SCORE = 'independent_validation_rank_score'
ROLES = {'Palinova_A_binary': 'binding', 'Palinova_B_binary': 'activity',
         'Palinova_A_joint_reference': 'ranking'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def metric(frame, scores):
    y = frame.binary_label.to_numpy(int)
    both = len(y) > 0 and 0 < y.sum() < len(y)
    result = dict(pairs=len(y), positive=int(y.sum()), negative=int(len(y)-y.sum()),
                  random_ap_reference=float(y.mean()) if len(y) else None,
                  ap=float(average_precision_score(y, scores)) if both else None,
                  auroc=float(roc_auc_score(y, scores)) if both else None)
    for direction, key in [('drug', 'ligand_inchikey'), ('target', 'target_chembl_id')]:
        vals, aucs, p5 = [], [], []
        for ids in frame.groupby(key, sort=True).indices.values():
            a, b = y[ids], scores[ids]
            # Same both-class query inclusion as the old S5 and refresh report.
            if not 0 < a.sum() < len(a):
                continue
            vals.append(average_precision_score(a, b)); aucs.append(roc_auc_score(a, b))
            if len(a) >= 5:
                p5.append(a[np.argsort(-b, kind='stable')[:5]].mean())
        result.update({direction+'_queries_both_classes': len(vals),
                       direction+'_macro_ap': float(np.mean(vals)) if vals else None,
                       direction+'_macro_auroc': float(np.mean(aucs)) if aucs else None,
                       direction+'_p5': float(np.mean(p5)) if p5 else None,
                       direction+'_p5_queries': len(p5)})
    return result


@torch.inference_mode()
def predict(model, arrays, di, ti):
    result = []
    for start in range(0, len(di), 4096):
        d = torch.tensor(di[start:start+4096], device='cuda', dtype=torch.long)
        t = torch.tensor(ti[start:start+4096], device='cuda', dtype=torch.long)
        batch = {k: v[t if k == 'target_global' else d] for k, v in arrays.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = model(batch)
        result.append(logits.float().mean(1).cpu().numpy())
    result = np.concatenate(result)
    assert np.isfinite(result).all()
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    registry = json.loads(REGISTRY.read_text())
    paths = [Path(__file__), REGISTRY, OLD/'BIOMASTER_DRUG_TO_TARGET_S5_TEST_V1.csv.gz',
             OLD/'BIOMASTER_DRUG_TO_TARGET_SUMMARY_V1.json',
             REFRESH/'PAIR_LABELS_AND_FROZEN_SCORES.csv.gz',
             BDB/'BINDINGDB_COVERED_EVALUATION_PAIRS.csv', BDB/'COHORT_METRICS.csv',
             CAT/'DRUG_INDEX.csv', CAT/'TARGET_INDEX.csv.gz', CAT/'TARGET_GLOBAL.npy',
             AB/'DATA_MANIFEST.json', FEATURES/'MANIFEST.json', AB/'COMMON_TEST.parquet',
             ROOT/'outputs/biomaster_endpoint_multitask_20260911/TEST_PREDICTIONS.parquet',
             DATA/'MOLECULES.parquet', DATA/'TARGETS.parquet',
             ROOT/'outputs/joint_screen_720x384_20260909/FULL_276480_PAIR_AUDIT.parquet',
             ROOT/'outputs/spr384_final_experiment_table_20260910/SPR384_FINAL_DETAILED.csv']
    for arm in ['kdki_inactive', 'all_inactive']:
        paths.extend(AB/f'{arm}_{split}.parquet' for split in ['TRAIN', 'VALIDATION'])
    for role in ROLES.values():
        p = ROOT/registry['roles'][role]['checkpoint']
        assert sha(p) == registry['roles'][role]['checkpoint_sha256']
        paths.append(p)
    paths.extend((BUNDLE/'features').glob('*.npy'))
    frozen = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status='FROZEN_BEFORE_NEW_MODEL_INFERENCE', input_hashes=frozen,
        models={name: registry['roles'][role] for name, role in ROLES.items()},
        score='mean of two binary logits; GPU BF16, batch4096, TF32 enabled as training evaluation',
        primary_input='Exact molecule and sequence mapping; prepared A/B feature rows take precedence; otherwise preserved catalog features. No feature cache mutation.',
        secondary_input='Original catalog features, same weights and precision; sensitivity only, never chosen by outcome',
        panels='Exact S5 2556, merged refresh 488, BindingDB-covered 479 and identity/chemistry subset382; original labels retained, no new endpoint pooling',
        overlap='Exact and connectivity+exact-sequence membership in A/B train AND validation. Strict shared subset excludes either arm exposure; separate unseen split-group subset when mapped.',
        queries='Both labels required; no minimum size for macro AP/AUROC, matching historical S5/refresh; P5 only when >=5 measured pairs',
        catalog_rank='Same original720x384 axis, descending stable original-row tie order, before known-relationship exclusion; no reselection',
        uncertainty='1000 paired drug-cluster bootstraps; difference versus frozen SPR score',
        training=False, deployment=False, wetlab_reselection=False, dtiam_test_access=False)
    write_json('PROTOCOL.json', protocol)

    drugs = pd.read_csv(CAT/'DRUG_INDEX.csv')
    targets = pd.read_csv(CAT/'TARGET_INDEX.csv.gz', usecols=['target_chembl_id','sequence_sha256'])
    assert drugs.drug_id.is_unique and targets.target_chembl_id.is_unique
    mol = pd.read_parquet(DATA/'MOLECULES.parquet').set_index('molecule_id')
    proteins = pd.read_parquet(DATA/'TARGETS.parquet').set_index('sequence_sha256')
    for d in drugs.itertuples():
        ident, reason = molecule_identity(d.smiles)
        assert ident and ident['molecule_id'] == d.drug_id, (d.drug_id, reason)
    drugs['training_feature_index'] = drugs.drug_id.map(mol.drug_feature_index)
    drugs['split_group'] = drugs.drug_id.map(mol.split_group)
    targets['training_feature_index'] = targets.sequence_sha256.map(proteins.target_feature_index)
    targets['target_id'] = 'SEQ:' + targets.sequence_sha256
    original = {k: np.load(BUNDLE/'features'/f'{k}.npy').copy()
                for k in ['drug_global', 'drug_graph_mean', 'pretrained_available']}
    original['target_global'] = np.load(CAT/'TARGET_GLOBAL.npy')
    primary = {k: v.copy() for k, v in original.items()}
    mask = drugs.training_feature_index.notna()
    mask.loc[mask] = np.load(FEATURES/'DRUG_DONE.npy')[drugs.loc[mask, 'training_feature_index'].astype(int)]
    ids = drugs.loc[mask, 'training_feature_index'].astype(int).to_numpy()
    primary['drug_global'][mask, :512] = np.load(FEATURES/'DRUG_CLIP.npy', mmap_mode='r')[ids]
    primary['drug_global'][mask, 512:] = np.load(FEATURES/'MORGAN.npy', mmap_mode='r')[ids]
    primary['drug_graph_mean'][mask] = np.load(FEATURES/'GRAPH.npy', mmap_mode='r')[ids]
    primary['pretrained_available'][mask] = np.load(FEATURES/'AVAILABLE.npy')[ids]
    drugs['feature_source'] = np.where(mask, 'prepared_A_B_cache', 'preserved_catalog_cache')
    tmask = targets.training_feature_index.notna()
    tmask.loc[tmask] = np.load(FEATURES/'TARGET_DONE.npy')[targets.loc[tmask, 'training_feature_index'].astype(int)]
    ids = targets.loc[tmask, 'training_feature_index'].astype(int).to_numpy()
    tf = np.load(FEATURES/'TARGET.npy', mmap_mode='r')[ids]
    assert np.array_equal(tf, original['target_global'][tmask])
    primary['target_global'][tmask] = tf
    feature_audit = {}
    for k, arr in primary.items():
        delta = np.abs(arr.astype(float) - original[k].astype(float))
        changed = (delta > 0).any(1) if arr.ndim == 2 else delta > 0
        feature_audit[k] = dict(shape=list(arr.shape), rows_changed=int(changed.sum()),
                                max_abs_difference=float(delta.max()))
        np.save(OUT/f'INPUT_{k}.npy', arr)
    drugs.to_csv(OUT/'DRUG_INPUT_AUDIT.csv', index=False)
    write_json('FEATURE_AUDIT.json', feature_audit)

    s5 = pd.read_csv(OLD/'BIOMASTER_DRUG_TO_TARGET_S5_TEST_V1.csv.gz').rename(columns={'parent_standard_inchi_key':'ligand_inchikey'})
    s5['pair_id'] = s5.ligand_inchikey+'__'+s5.target_chembl_id
    s5['cohort'] = 'S5_HISTORICAL_TEST'
    refresh = pd.read_csv(REFRESH/'PAIR_LABELS_AND_FROZEN_SCORES.csv.gz', low_memory=False)
    refresh = refresh[refresh.endpoint_group.eq('Kd_Ki') & refresh.refresh_endpoint_covered &
                      ~refresh.local_training_connectivity_pair & refresh.scored & refresh.binary_label.notna()].copy()
    assert len(refresh) == 488
    refresh['cohort'] = 'MERGED_REFRESH_488'
    bdb_ids = set(pd.read_csv(BDB/'BINDINGDB_COVERED_EVALUATION_PAIRS.csv').pair_id)
    assert len(bdb_ids) == 479 and bdb_ids <= set(refresh.pair_id)
    frame = pd.concat([s5, refresh], ignore_index=True)
    assert not frame.duplicated(['cohort','pair_id']).any()
    frame['binary_label'] = frame.binary_label.astype(int)
    frame['is_bindingdb479'] = frame.cohort.eq('MERGED_REFRESH_488') & frame.pair_id.isin(bdb_ids)
    frame['di'] = frame.ligand_inchikey.map(pd.Series(np.arange(len(drugs)), index=drugs.drug_id))
    frame['ti'] = frame.target_chembl_id.map(pd.Series(np.arange(len(targets)), index=targets.target_chembl_id))
    assert frame[['di', 'ti']].notna().all().all()
    frame['molecule_id'] = frame.ligand_inchikey
    frame['target_id'] = frame.target_chembl_id.map(targets.set_index('target_chembl_id').target_id)
    frame['normalized_pair_id'] = frame.molecule_id+'__'+frame.target_id
    frame['connectivity_pair'] = frame.molecule_id.str[:14]+'__'+frame.target_id
    frame['split_group'] = frame.molecule_id.map(mol.split_group)
    frame['feature_source'] = frame.molecule_id.map(drugs.set_index('drug_id').feature_source)
    overlap_columns, group_columns = [], []
    for arm, short in [('kdki_inactive', 'A'), ('all_inactive', 'B')]:
        for split in ['TRAIN', 'VALIDATION']:
            f = pd.read_parquet(AB/f'{arm}_{split}.parquet', columns=['pair_id','molecule_id','target_id','split_group','binary_label'])
            cp = set(f.molecule_id.str[:14]+'__'+f.target_id)
            prefix = f'{short}_{split.lower()}'
            frame[prefix+'_exact_pair_seen'] = frame.normalized_pair_id.isin(set(f.pair_id))
            frame[prefix+'_connectivity_pair_seen'] = frame.connectivity_pair.isin(cp)
            frame[prefix+'_split_group_seen'] = frame.split_group.isin(set(f.split_group))
            labels = frame.normalized_pair_id.map(f.set_index('pair_id').binary_label)
            frame[prefix+'_label_conflict'] = labels.notna() & labels.ne(frame.binary_label)
            overlap_columns.append(prefix+'_connectivity_pair_seen')
            group_columns.append(prefix+'_split_group_seen')
    frame['both_AB_train_validation_pair_unseen'] = ~frame[overlap_columns].any(axis=1)
    frame['both_AB_train_validation_group_unseen'] = (frame.both_AB_train_validation_pair_unseen &
        frame.split_group.notna() & ~frame[group_columns].any(axis=1))
    core = pd.read_parquet(ROOT/'outputs/joint_screen_720x384_20260909/FULL_276480_PAIR_AUDIT.parquet',
                          columns=['pair_id','ligand_inchikey','target_chembl_id'])
    assert len(core) == 276480 and core.pair_id.is_unique
    cdi = core.ligand_inchikey.map(pd.Series(np.arange(len(drugs)),index=drugs.drug_id)).to_numpy(int)
    cti = core.target_chembl_id.map(pd.Series(np.arange(len(targets)),index=targets.target_chembl_id)).to_numpy(int)
    tensors = {k:torch.as_tensor(v,device='cuda') for k,v in primary.items()}
    secondary = {k:torch.as_tensor(v,device='cuda') for k,v in original.items()}
    model_paths = {}
    for name, role in ROLES.items():
        spec = registry['roles'][role]
        state = torch.load(ROOT/spec['checkpoint'], map_location='cpu', weights_only=True)
        model = AuxiliaryInteraction(state['config'], state['endpoints']).cuda().eval()
        model.load_state_dict(state['model'], strict=True)
        frame[name+'_score'] = predict(model, tensors, frame.di.to_numpy(int), frame.ti.to_numpy(int))
        frame[name+'_catalog_input_score'] = predict(model, secondary, frame.di.to_numpy(int), frame.ti.to_numpy(int))
        core[name+'_score'] = predict(model, tensors, cdi, cti)
        core[name+'_rank384'] = core.groupby('ligand_inchikey')[name+'_score'].rank(method='first',ascending=False).astype(int)
        frame[name+'_rank384'] = frame.pair_id.map(core.set_index('pair_id')[name+'_rank384'])
        model_paths[name] = spec['checkpoint']
        print('SCORED', name, len(frame), 'panel rows;', len(core), 'catalog pairs', flush=True)
        del model, state
    core.to_parquet(OUT/'CORE720x384_RECOMMENDED_SCORES.parquet', index=False, compression='zstd')
    frame.to_csv(OUT/'ALL_PANEL_PREDICTIONS_AND_OVERLAP.csv', index=False, encoding='utf-8-sig')

    # Validate that preserved samples, labels and old score recreate published results.
    old_summary = json.loads((OLD/'BIOMASTER_DRUG_TO_TARGET_SUMMARY_V1.json').read_text())
    old_s5 = next(x for x in old_summary['s5_test']['metrics'] if x['model']=='INDEPENDENT_VALIDATION_RANK')
    s5_metric = metric(s5, s5[SCORE].to_numpy())
    assert abs(s5_metric['ap']-old_s5['micro_auprc']) < 1e-12
    assert abs(s5_metric['drug_macro_ap']-old_s5['drug_macro_auprc']) < 1e-12
    expected = pd.read_csv(BDB/'COHORT_METRICS.csv')
    for name, mask in [('MERGED_REFRESH', frame.cohort.eq('MERGED_REFRESH_488')),
                       ('BINDINGDB_COVERED_MERGED_QC',frame.is_bindingdb479)]:
        f = frame[mask]
        m = metric(f,f[SCORE].to_numpy()); e = expected[expected.cohort.eq(name)&expected.model.eq('SPR_current')].iloc[0]
        assert len(f) == e.n_pairs and abs(m['ap']-e.average_precision)<1e-12
        assert abs(m['auroc']-e.auroc)<1e-12

    cohort_masks = {'S5_HISTORICAL_TEST':frame.cohort.eq('S5_HISTORICAL_TEST'),
        'MERGED_REFRESH_488':frame.cohort.eq('MERGED_REFRESH_488'),
        'BINDINGDB_479':frame.is_bindingdb479,
        'BINDINGDB_IDENTITY_CHEMISTRY_382':frame.is_bindingdb479 & frame.identity_scope_pass.eq(True) & frame.chemistry_policy_pass.eq(True)}
    assert int(cohort_masks['BINDINGDB_IDENTITY_CHEMISTRY_382'].sum()) == 382
    models = {'original_SPR_ranker':SCORE, **{n:n+'_score' for n in ROLES}}
    rows, overlap, ci, bands, sensitivity = [], [], [], [], []
    for cohort, mask in cohort_masks.items():
        full = frame[mask].copy()
        overlap.append(dict(cohort=cohort, pairs=len(full), positive=int(full.binary_label.sum()),
                            **{col:int(full[col].sum()) for col in frame if col.endswith(('_seen','_conflict','_unseen'))}))
        for scope, use in [('all_original_members', np.ones(len(full),bool)),
                          ('both_AB_train_validation_pair_unseen', full.both_AB_train_validation_pair_unseen.to_numpy()),
                          ('both_AB_train_validation_group_unseen', full.both_AB_train_validation_group_unseen.to_numpy())]:
            sub = full[use].reset_index(drop=True)
            for name, col in models.items():
                rows.append(dict(cohort=cohort, scope=scope, model=name, **metric(sub,sub[col].to_numpy())))
            if sub.binary_label.nunique()==2 and min(sub.binary_label.sum(),len(sub)-sub.binary_label.sum())>=10:
                y=sub.binary_label.to_numpy(int); groups=list(sub.groupby('ligand_inchikey',sort=True).indices.values())
                vals={name:[] for name in ROLES}; rng=np.random.default_rng(20260914)
                old=sub[SCORE].to_numpy(); scores={name:sub[name+'_score'].to_numpy() for name in ROLES}
                for _ in range(1000):
                    ix=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])
                    yy=y[ix]
                    if yy.min()==yy.max():continue
                    ap0=average_precision_score(yy,old[ix]); auc0=roc_auc_score(yy,old[ix])
                    for name,s in scores.items():
                        vals[name].append([average_precision_score(yy,s[ix])-ap0,roc_auc_score(yy,s[ix])-auc0])
                for name, v in vals.items():
                    a=np.asarray(v); q=np.quantile(a,[.025,.975],axis=0)
                    ci.append(dict(cohort=cohort,scope=scope,model=name,drug_clusters=len(groups),replicates=len(a),
                        ap_delta=float(average_precision_score(y,scores[name])-average_precision_score(y,old)),
                        ap_delta_ci95_low=float(q[0,0]),ap_delta_ci95_high=float(q[1,0]),
                        auroc_delta=float(roc_auc_score(y,scores[name])-roc_auc_score(y,old)),
                        auroc_delta_ci95_low=float(q[0,1]),auroc_delta_ci95_high=float(q[1,1])))
        for scope, sub in [('all_original_members', full),
                           ('both_AB_train_validation_pair_unseen',full[full.both_AB_train_validation_pair_unseen])]:
            for name in ROLES:
                sensitivity.append(dict(cohort=cohort,scope=scope,model=name,input='original_catalog',
                                        **metric(sub,sub[name+'_catalog_input_score'].to_numpy())))
            if cohort != 'S5_HISTORICAL_TEST':
                for name in models:
                    col='rank_SPR_current' if name=='original_SPR_ranker' else name+'_rank384'
                    for lo,hi in [(1,10),(11,20),(21,384)]:
                        f=sub[sub[col].between(lo,hi)]
                        bands.append(dict(cohort=cohort,scope=scope,model=name,rank_band=f'{lo}-{hi}',measured_pairs=len(f),
                                          positive=int(f.binary_label.sum()),negative=int(len(f)-f.binary_label.sum()),
                                          measured_positive_fraction=float(f.binary_label.mean()) if len(f) else None))
        print('EVALUATED',cohort,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'SAME_PANEL_METRICS.csv',index=False)
    pd.DataFrame(overlap).to_csv(OUT/'TRAIN_VALIDATION_OVERLAP.csv',index=False)
    pd.DataFrame(ci).to_csv(OUT/'PAIRED_DRUG_BOOTSTRAP.csv',index=False)
    pd.DataFrame(bands).to_csv(OUT/'RANK_BAND_MEASURED_OUTCOMES.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'CATALOG_INPUT_SENSITIVITY.csv',index=False)
    # Reproduce already-published neural A/B TEST scores; do not access DTIAM's TEST.
    val=pd.read_parquet(AB/'COMMON_TEST.parquet').iloc[:4096].copy()
    did=val.drug_feature_index.to_numpy(int);tid=val.target_feature_index.to_numpy(int)
    vb={'drug_global':np.concatenate([np.load(FEATURES/'DRUG_CLIP.npy',mmap_mode='r')[did],
        np.load(FEATURES/'MORGAN.npy',mmap_mode='r')[did].astype(np.float32)],1),
        'drug_graph_mean':np.load(FEATURES/'GRAPH.npy',mmap_mode='r')[did],
        'pretrained_available':np.load(FEATURES/'AVAILABLE.npy')[did].astype(np.float32),
        'target_global':np.load(FEATURES/'TARGET.npy',mmap_mode='r')[tid]}
    vt={k:torch.as_tensor(v,device='cuda') for k,v in vb.items()}; parity=[]
    for name,role in ROLES.items():
        run=(ROOT/model_paths[name]).parent
        state=torch.load(run/'model.pt',map_location='cpu',weights_only=True)
        model=AuxiliaryInteraction(state['config'],state['endpoints']).cuda().eval();model.load_state_dict(state['model'])
        pred=predict(model,vt,np.arange(len(val)),np.arange(len(val)))
        spec=registry['roles'][role]
        scorecol=f"{spec['arm']}__{spec['variant']}__{spec['seed']}_score"
        saved=pd.read_parquet(ROOT/'outputs/biomaster_endpoint_multitask_20260911/TEST_PREDICTIONS.parquet',
                              columns=['panel','pair_id',scorecol]).iloc[:len(val)]
        assert saved[['panel','pair_id']].equals(val[['panel','pair_id']])
        maxdiff=float(np.abs(pred-saved[scorecol].to_numpy()).max())
        assert maxdiff==0, (name,maxdiff)
        parity.append(dict(model=name,finite=bool(np.isfinite(pred).all()),rows=len(val),
                           max_abs_difference_from_frozen_test=maxdiff,
                           prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest()))
        del model,state
    for path,h in frozen.items():assert sha(ROOT/path)==h,path
    write_json('SUMMARY.json',dict(status='COMPLETE_FROZEN_SAME_PANEL_AUDIT',
        models=model_paths,metrics=rows,overlap=overlap,feature_audit=feature_audit,
        validations=dict(original_metrics_reproduced=True,panel_labels_unchanged=True,all_inputs_unchanged=True,
            finite_prediction_checks=parity),
        limitations=['Same original panels are retrospective and have previously been examined.',
            'Pair-unseen does not certify document-, drug-, target-, scaffold-, or public-pretraining independence.',
            'Historical S5 includes mixed endpoint/activity labels; not exclusively Kd/Ki.',
            'Refresh labels retain original cross-source QC even if later A/B labels disagree.',
            'Measured rank-band fractions are not SPR384 hit-rate estimates; no candidates were reselected.',
            'Original SPR predictions use their historical feature pipeline; new scores use documented A/B cache precedence.',
            'One preselected representative per role; not three-seed averages, not a newly tuned ensemble.'],
        training=False,production_modified=False,wetlab_modified=False,dtiam_test_access=False,
        finished_utc=datetime.now(timezone.utc).isoformat()))
    print(pd.DataFrame(rows).query("scope == 'all_original_members'")[['cohort','model','pairs','ap','auroc','drug_macro_ap']].to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
