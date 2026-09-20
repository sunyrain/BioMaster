#!/usr/bin/env python3
"""Register pinned local model assets, public source metadata, and missing downloads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dti_research_preparation_20260920'
META=OUT/'source_metadata'
STORE=ROOT/'data/research/dti_reliability_20260920_v1/model_assets'
REPOS={
    'ConPLex':('samsledje/ConPLex','third_party/ConPLex'),
    'DrugCLIP':('THU-ATOM/Drug-The-Whole-Genome','third_party/sota_dti_2026/Drug-The-Whole-Genome'),
    'DTIAM':('CSUBioGroup/DTIAM','third_party/sota_dti_2026/DTIAM'),
    'ProbeMatchDTI':('developer-hq/ProbeMatchDTI','.external/ProbeMatchDTI'),
    'DTBind':('liqy09/DTBind','.external/DTBind'),
    'nesso':('recursionpharma/nesso','.external/nesso'),
    'EviDTI':('zhaoyanpeng208/EviDTI','third_party/sota_dti_2026/EviDTI'),
    'TAPB':('GaomingL1n/TAPB','third_party/sota_dti_2026/TAPB'),
    'DrugBAN':('peizhenbai/DrugBAN','third_party/DrugBAN'),
    'DeepDTA':('hkmztrk/DeepDTA','third_party/DeepDTA'),
    'ScopeDTI_light':('Yigang-Chen/Lightweight-SCOPE-DTI-for-Inference','.external/scope_dti_lightweight')}


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def refresh_repo(item):
    name,(repo,_) = item
    try:
        r=requests.get(f'https://api.github.com/repos/{repo}',timeout=30);r.raise_for_status();data=r.json()
        c=requests.get(f'https://api.github.com/repos/{repo}/commits/{data["default_branch"]}',timeout=30);c.raise_for_status();rev=c.json()['sha']
        t=requests.get(f'https://api.github.com/repos/{repo}/git/trees/{rev}?recursive=1',timeout=30);t.raise_for_status();tree=t.json()
        save(META/f'{name}_TREE.json',dict(commit=rev,truncated=tree.get('truncated'),tree=tree.get('tree',[])))
        record=dict(model=name,repo=repo,default_branch=data['default_branch'],license=data.get('license'),
                    pushed_at=data.get('pushed_at'),commit=rev,retrieved_utc=datetime.now(timezone.utc).isoformat())
        record['weight_candidates']=[dict(path=x['path'],bytes=x.get('size'),blob=x['sha']) for x in tree.get('tree',[]) if x['type']=='blob'
            and (x['path'].lower().endswith(('.pt','.pth','.ckpt','.safetensors','.h5','.pkl','.bin')) or 'all_model' in x['path'].lower())
            and not any(term in x['path'].lower() for term in ['dataset','sample_test','pre_filter','pre_transform'])]
        for x in tree.get('tree',[]):
            p=x['path']
            if x['type']=='blob' and (p.lower() in ['readme.md','license','license.md','license.txt'] or p.lower().endswith('model_weights_license.md')):
                rr=requests.get(f'https://raw.githubusercontent.com/{repo}/{rev}/{p}',timeout=30)
                if rr.status_code==200:(META/f'{name}_{p.replace("/","_")}').write_text(rr.text)
        save(META/f'{name}_REPO.json',record);return dict(model=name,status='METADATA_COLLECTED')
    except requests.RequestException as e:
        result=dict(model=name,status='SOURCE_METADATA_UNAVAILABLE',detail=str(e));save(META/f'{name}_FETCH_ERROR.json',result);return result


def specs():
    rows=[]
    def add(asset_id,family,path,role='task_weights',url='',expected='',status='AVAILABLE_REQUIRES_PARITY',license_note='See model registry'):
        rows.append(dict(asset_id=asset_id,family=family,path=path,role=role,official_download_url=url,
                         expected_sha256=expected,qualification=status,license_note=license_note))
    add('retargetmap_deployed','ReTargetMap','outputs/biomaster_best_model_20260906/retargetmap_selected_v1/model.pt',status='DEPLOYED',
        expected='ca1ad1de8364806282bacac993fd477711c59498c42248982c441eafcc254f76')
    legacy=pd.read_csv(ROOT/'outputs/biomaster_model_consolidation_20260911/MODEL_LEDGER.csv')
    for r in legacy.drop_duplicates('checkpoint').itertuples():
        add(r.model_id,'Palinova_AB',r.checkpoint,role='internal_trained_weights',expected=r.checkpoint_sha256,status='EXPLORATORY_TRAINED')
    for arm in ['kdki_inactive','all_inactive']:
        for seed in [20260921,20260922,20260923]:
            add(f'dtiam_{arm}_{seed}','DTIAM',f'outputs/biomaster_dtiam_ab_20260912/{arm}__seed_{seed}/predictor',
                role='internal_trained_bundle',status='EXPLORATORY_TRAINED')
    add('conplex_bindingdb','ConPLex','third_party/ConPLex/models/BindingDB_ExperimentalValidModel.pt',
        url='https://cb.csail.mit.edu/cb/conplex/data/models/BindingDB_ExperimentalValidModel.pt',status='DEPLOYED',
        expected='c28f0e64907e2cec786f2fdefd0ee75ac991b971671a49f896611dc192e6d39c')
    for fold in range(6):
        add(f'drugclip_fold{fold}','DrugCLIP',f'third_party/sota_dti_2026/Drug-The-Whole-Genome/data/model_weights/6_folds/fold_{fold}.pt',
            status='DEPLOYED_ENSEMBLE_MEMBER',license_note='Author specifies CC-BY-NC-4.0 for weights and outputs; code/database CC-BY-4.0')
    frontier=json.loads((ROOT/'outputs/frontier_dti_20260916/MODEL_ASSETS.json').read_text())
    digest_map={r['path']:r['sha256'] for r in frontier['files']}
    probe='.external/ProbeMatchDTI/model/All_Model'
    add('probematch_all','ProbeMatchDTI',probe,
        url='https://raw.githubusercontent.com/developer-hq/ProbeMatchDTI/2b9c0dcb40bcc4e5482d6ea98aa1419538ecdd12/model/All_Model',
        expected=digest_map[probe],status='DEPLOYED',license_note='No explicit license found in checked repository; availability does not establish redistribution permission')
    for head in ['occurrence','affinity','site']:
        path=f'.external/DTBind/models/{head}_model.pth'
        add('dtbind_'+head,'DTBind',path,
            url=f'https://raw.githubusercontent.com/liqy09/DTBind/08983e476760fbe5b5dc62be06fd2e087b31ac0c/models/{head}_model.pth',
            expected=digest_map.get(path,''),status='DEPLOYED' if head=='occurrence' else 'WEIGHTS_AVAILABLE_INPUT_REQUIREMENTS_DIFFER',
            license_note='No explicit repository license located; record dataset/weight terms separately')
    for r in frontier['files']:
        if '/nesso/' in r['path']:
            filename=Path(r['path']).name
            add('nesso_'+filename,'nesso',r['path'],role='model_config' if filename.endswith('.json') else 'task_weights',
                url=f'https://huggingface.co/recursionpharma/nesso/resolve/1896c84c7186c506c7efd79051480809d51098bf/v1.0.0/{filename}',
                expected=r['sha256'],status='DEPLOYED')
    for task in ['drugbank','davis','kiba']:
        add('evidti_'+task,'EviDTI',f'third_party/sota_dti_2026/EviDTI/runs/{task}_model/checkpoint.pth',
            url='https://zenodo.org/records/14056305',status='COLLECTED_NOT_CURRENT_CATALOG_QUALIFIED',
            license_note='README states Apache-2.0; Zenodo record terms captured separately')
    for path in sorted((ROOT/'.external/scope_dti_lightweight/models_path').glob('**/*.pth')):
        add('scope_'+path.parent.name+'_'+path.stem,'ScopeDTI_light',str(path.relative_to(ROOT)),
            status='COLLECTED_FIXED_TARGET_UNIVERSE',license_note='Local source LICENSE is GPL-3.0; task-weight applicability checked separately')
    # Encoder-only availability is not promoted to availability of a fitted DTI predictor.
    encoder_paths=[('bermol_encoder','DTIAM','third_party/sota_dti_2026/DTIAM/code/BerMolModel_base.pkl'),
                   ('esm2_650m','shared_encoders','/root/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt'),
                   ('protbert_bfd','ProbeMatchDTI','.cache/frontier_dti/prot_bert_bfd/pytorch_model.bin'),
                   ('smiles_roberta','ProbeMatchDTI','.cache/frontier_dti/PubChem10M_SMILES_BPE_450k/pytorch_model.bin')]
    for name,family,path in encoder_paths:add(name,family,path,role='pretrained_encoder',status='ENCODER_ONLY')
    for path in sorted((ROOT/'.model_cache/huggingface/hub/models--ibm--MoLFormer-XL-both-10pct/snapshots').glob('*/model.safetensors')):
        add('molformer_'+path.parent.name,'TAPB',str(path.relative_to(ROOT)),role='pretrained_encoder',status='ENCODER_ONLY')
    for path in sorted((ROOT/'third_party/ConPLex/models/huggingface/transformers/models--Rostlab--prot_bert/snapshots').glob('*/model.safetensors')):
        add('protbert_'+path.parent.name,'ConPLex',str(path.relative_to(ROOT)),role='pretrained_encoder',status='ENCODER_ONLY')
    return rows


def inventory(fetch_missing=False):
    OUT.mkdir(parents=True,exist_ok=True);META.mkdir(exist_ok=True);STORE.mkdir(parents=True,exist_ok=True)
    for item in REPOS.items():
        if not (META/f'{item[0]}_REPO.json').exists():print(refresh_repo(item),flush=True)
    rows=specs();expected_files=json.loads((ROOT/'outputs/biomaster_dtiam_ab_20260912/TRAINED_ARTIFACTS.json').read_text())['files']
    files=[];missing=[]
    for index,row in enumerate(rows):
        path=ROOT/row['path']
        if not path.exists() and fetch_missing and row['official_download_url'] and '/records/' not in row['official_download_url']:
            path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_suffix(path.suffix+'.part')
            with requests.get(row['official_download_url'],stream=True,timeout=(15,60)) as response:
                response.raise_for_status()
                with tmp.open('wb') as f:
                    for block in response.iter_content(4*1024*1024):f.write(block)
            if row['expected_sha256'] and sha(tmp)!=row['expected_sha256']:raise ValueError('Downloaded digest mismatch: '+row['asset_id'])
            tmp.replace(path)
        if not path.exists():
            row['status']='MISSING';missing.append(row);continue
        members=sorted(q for q in path.rglob('*') if q.is_file()) if path.is_dir() else [path]
        fingerprint=[];total=0
        for member in members:
            digest=sha(member)
            key=str(member.relative_to(ROOT)) if member.is_relative_to(ROOT) else str(member)
            expected=expected_files.get(key,'')
            if path.is_file():expected=row['expected_sha256'] or expected
            if expected and digest!=expected:raise ValueError('Frozen asset changed: '+key)
            size=member.stat().st_size
            if path.is_file() and size<1024:raise ValueError('Suspiciously small model asset: '+key)
            files.append(dict(asset_id=row['asset_id'],family=row['family'],path=key,bytes=size,sha256=digest,
                              prior_manifest_match=bool(expected),role=row['role']))
            fingerprint.append(str(member.relative_to(path))+'\0'+digest if path.is_dir() else digest);total+=size
        row.update(status='VERIFIED_LOCAL',files=len(members),bytes=total,
                   sha256=hashlib.sha256('\n'.join(fingerprint).encode()).hexdigest() if path.is_dir() else fingerprint[0],
                   digest_kind='directory_manifest_sha256' if path.is_dir() else 'file_sha256')
        link=STORE/row['asset_id']
        if link.is_symlink():assert link.resolve()==path.resolve()
        elif link.exists():raise ValueError('Refuse to replace existing asset store item')
        else:link.symlink_to(os.path.relpath(path,link.parent),target_is_directory=path.is_dir())
        row['registered_link']=str(link.relative_to(ROOT))
        if index%10==0:print('Registered weights',index+1,'/',len(rows),flush=True)
    registry=[]
    statuses={'TAPB':'ENCODERS_AND_TRAINING_CODE; NO_PUBLISHED_TASK_CHECKPOINT_FOUND',
              'DrugBAN':'TRAINING_CODE; LOCAL_ONE_EPOCH_DEMO_NOT_A_QUALIFIED_PUBLIC_MODEL',
              'DeepDTA':'TRAINING_CODE; NO_PUBLISHED_TASK_CHECKPOINT_FOUND',
              'EviDTI':'THREE_TASK_CHECKPOINTS_COLLECTED; ADAPTER_PARITY_PENDING',
              'ScopeDTI_light':'TASK_CHECKPOINTS_COLLECTED; FIXED_TARGET_UNIVERSE_NOT_ARBITRARY_COLD_TARGET_MODEL'}
    for name,(repo,folder) in REPOS.items():
        local=ROOT/folder;metadata=json.loads((META/f'{name}_REPO.json').read_text()) if (META/f'{name}_REPO.json').exists() else {}
        has_git=(local/'.git').exists()
        revision=subprocess.check_output(['git','-C',str(local),'rev-parse','HEAD'],text=True).strip() if has_git else None
        code_files=[p for p in local.rglob('*.py') if not any(x in p.parts for x in ['.git','__pycache__','dataset','datasets','models_path'])]
        code_fingerprint=hashlib.sha256('\n'.join(str(p.relative_to(local))+'\0'+sha(p) for p in sorted(code_files)).encode()).hexdigest()
        current=[r for r in rows if r['family']==name and r.get('status')=='VERIFIED_LOCAL' and r['role']!='pretrained_encoder']
        registry.append(dict(model=name,official_repo='https://github.com/'+repo,local_code=folder,
            local_git_revision=revision,vendored_without_independent_git=not has_git,code_sha256=code_fingerprint,
            upstream_checked_revision=metadata.get('commit'),upstream_changed=bool(revision and metadata.get('commit') and revision!=metadata['commit']),
            code_license_spdx=(metadata.get('license') or {}).get('spdx_id','NOT_IDENTIFIED'),
            task_asset_entries=len(current),status=statuses.get(name,'DEPLOYED_OR_TRAINED_REGISTERED'),
            weights_license='CC-BY-NC-4.0' if name=='DrugCLIP' else ('NOT_IDENTIFIED' if name in ['ProbeMatchDTI','DTBind'] else 'SEE_PINNED_SOURCE_METADATA'),
            qualification='File verification is not scientific benchmark parity or independent test validity.'))
    pd.DataFrame(rows).to_csv(OUT/'WEIGHT_REGISTRY.csv',index=False)
    pd.DataFrame(files).to_csv(OUT/'WEIGHT_FILES.csv',index=False)
    pd.DataFrame(registry).to_csv(OUT/'PUBLIC_MODEL_REGISTRY.csv',index=False)
    save(OUT/'PUBLIC_MODEL_REGISTRY.json',registry)
    save(OUT/'MISSING_ASSETS.json',missing)
    summary=dict(status='COLLECTED_AND_LOCAL_HASH_VERIFIED',registered_asset_entries=len(rows),missing_asset_entries=len(missing),
        public_code_repositories=len(registry),verified_file_references=len(files),
        unique_files=len({r['path'] for r in files}),referenced_GiB=sum({r['path']:r['bytes'] for r in files}.values())/2**30,
        newly_duplicated_weight_bytes=0,storage='Relative symlinks to existing weights; check hashes before every run.',
        public_task_checkpoint_search_gaps=['TAPB','DrugBAN','DeepDTA'],
        source_metadata_utc=datetime.now(timezone.utc).isoformat(),no_models_loaded_for_hash_audit=True,
        licensing='Publicly downloadable weights are not automatically permissively open-source. Code and weight terms are separate fields.')
    save(OUT/'ASSET_SUMMARY.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--refresh-sources',action='store_true');p.add_argument('--fetch-missing',action='store_true');args=p.parse_args()
    META.mkdir(parents=True,exist_ok=True)
    if args.refresh_sources:
        with ThreadPoolExecutor(max_workers=3) as pool:
            for value in pool.map(refresh_repo,REPOS.items()):print(value,flush=True)
    inventory(args.fetch_missing)
