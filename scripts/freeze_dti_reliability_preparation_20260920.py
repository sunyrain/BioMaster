#!/usr/bin/env python3
"""Freeze the preparation handoff and inventory locally available external measurements."""
from datetime import datetime, timezone
from importlib import metadata
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dti_research_preparation_20260920'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def freeze():
    preflight=json.loads((OUT/'PREFLIGHT_FULL.json').read_text())
    assert preflight['status']=='PASS' and preflight['all_data_and_feature_hashes_checked']
    records=json.loads((ROOT/'data/external/affinity_refresh_20260910/DOWNLOAD_MANIFEST.json').read_text())
    files=[]
    for record in records:
        if 'file' not in record:continue
        path=ROOT/record['file']
        if not path.is_file():continue
        digest=sha(path);expected=record.get('sha256')
        assert not expected or digest==expected,path
        files.append(dict(path=record['file'],official_url=record.get('url',''),bytes=path.stat().st_size,
            sha256=digest,previous_manifest_match=bool(expected),independent_test_status='NOT_CERTIFIED_BY_DOWNLOAD'))
    pd.DataFrame(files).to_csv(OUT/'EXTERNAL_SOURCE_FILES.csv',index=False)
    pd.DataFrame([
        dict(source='CACHE3',priority=1,local_glob='data/external/affinity_refresh_20260910/CACHE_*CACHE3*.xlsx',
            endpoint='SPR binding, censored/failed fits separate',role='Same-target measured ordering and artifact sensitivity',
            remaining='Construct/identity mapping, author QC, repeated-measure aggregation, training overlap; few targets'),
        dict(source='OpenBind',priority=1,local_glob='data/external/affinity_refresh_20260910/OpenBind_*.csv',
            endpoint='GCI Kd, author accepted subset',role='External assay/source and viral-target distribution shift',
            remaining='Viral construct differs from human training; native model applicability and exposure audit'),
        dict(source='CACHE1',priority=2,local_glob='data/external/affinity_refresh_20260910/CACHE_all*.xlsx',
            endpoint='Dose fit plus single-concentration and counter-screen',role='Assay validity and experimental selection utility',
            remaining='Separate response from quantitative affinity; assay/construct and selectivity audit'),
        dict(source='BindingDB_202609',priority=2,local_glob='data/external/affinity_refresh_20260910/BindingDB_*',
            endpoint='Kd/Ki/IC50/EC50 separate',role='Training and source-purged retrospective test panels',
            remaining='Already used in training; release month is not first-experiment date; curate dense measured panels'),
        dict(source='HiQBind/Supercharge/RAF_MEK',priority=3,local_glob='See existing affinity import manifest',
            endpoint='Reused structure labels / apparent Kd / response',role='Structural or assay-context sensitivity only',
            remaining='Not automatically independent new labels or interchangeable pure-protein Kd')
    ]).to_csv(OUT/'EXTERNAL_PANEL_PLAN.csv',index=False)
    distributions=sorted({(x.metadata['Name'],x.version) for x in metadata.distributions() if x.metadata['Name']})
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
    env=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
        preparation_environment_packages=[dict(name=name,version=version) for name,version in distributions],
        gpu=gpu.stdout.strip(),gpu_query_status=gpu.returncode,
        scope='Preparation environment only; public-model and DTIAM native environments retained in their original manifests.',
        captured_utc=datetime.now(timezone.utc).isoformat())
    save(OUT/'ENVIRONMENT.json',env)
    package=[]
    for folder in [ROOT/'configs/dti_reliability_20260920',OUT]:
        for path in sorted(folder.glob('*')):
            if path.is_file() and path.suffix in ['.json','.csv'] and path.name!='PACKAGE_MANIFEST.json':package.append(path)
    package.extend(ROOT/'scripts'/name for name in [
        'prepare_dti_reliability_data_20260920.py','collect_dti_research_assets_20260920.py',
        'build_dti_reliability_protocol_20260920.py','preflight_dti_reliability_20260920.py',
        'freeze_dti_reliability_preparation_20260920.py'])
    package.extend([ROOT/'docs/protocols/DTI_RELIABILITY_PROTOCOL_20260920_ZH.md',
                    ROOT/'docs/BIOMASTER_DTI_RESEARCH_PREPARATION_20260920_ZH.md'])
    package.extend(sorted((OUT/'source_metadata').glob('*.json')))
    save(OUT/'PACKAGE_MANIFEST.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        parent_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in package},
        excluded_from_git='Large local model/data/feature bytes, logs and upstream README copies; retrieve by pinned sources and hashes.',
        status='PREPARATION_FROZEN_NO_NEW_TRAINING',external_registered=False))
    print('Frozen package files:',len(package),'External source files verified:',len(files))


if __name__=='__main__':freeze()
