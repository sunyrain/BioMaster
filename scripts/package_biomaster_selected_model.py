#!/usr/bin/env python3
"""Export, document, validate and archive the single frozen deployment model."""
import argparse
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import write_json
OUT=ROOT/'outputs/biomaster_best_model_20260906'


def state(status,**kwargs):
    write_json(OUT/'PACKAGE_STATUS.json',dict(status=status,utc=datetime.now(timezone.utc).isoformat(),**kwargs))


def call(script,args):
    state('RUNNING',script=script)
    with (OUT/'logs'/('package_'+Path(script).stem+'.log')).open('a') as log:
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script),*map(str,args)],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,check=True)


def main(wait=False):
    lock=(OUT/'PACKAGE_CONTROLLER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if wait:
        while True:
            p=OUT/'DELIVERY_STATUS.json';prior=json.loads(p.read_text()) if p.exists() else {}
            if prior.get('status')=='READY_FOR_EXPORT':break
            if prior.get('status')=='FAILED':raise ValueError('selected delivery queue failed')
            state('WAITING_FOR_SELECTED_FULLFIT',delivery_status=prior.get('status'));time.sleep(10)
    selection=OUT/'DEPLOYMENT_SELECTION.json';s=json.loads(selection.read_text())
    if s['status']!='SELECTED':raise ValueError('deployment checkpoint not selected')
    checkpoint=Path(s['checkpoint']['path'])
    if file_identity(checkpoint)!=s['checkpoint']:raise ValueError('selected checkpoint drift')
    bundle=OUT/'retargetmap_selected_v1'
    if not bundle.exists():
        call('export_biomaster_catalog_model_v2.py',['--checkpoint',checkpoint,'--destination',bundle,'--selection',selection])
    else:
        meta=json.loads((bundle/'metadata.json').read_text())
        if meta['source_checkpoint']!=s['checkpoint']:raise ValueError('existing bundle contains different weights')
    if not (bundle/'MODEL_CARD.md').exists():call('document_biomaster_selected_model.py',[bundle])
    validation=OUT/(bundle.name+'_RELEASE_VALIDATION.json')
    if not validation.exists():call('verify_biomaster_selected_release.py',[bundle])
    v=json.loads(validation.read_text())
    if v['status']!='PASS' or v['smoke_only'] or v['bundle']!=file_identity(bundle/'MANIFEST.json'):
        raise ValueError('selected bundle has not passed its current manifest validation')
    if v['source_checkpoint']!=s['checkpoint']:raise ValueError('validated different weights')
    archive=OUT/(bundle.name+'.zip');manifest=json.loads((bundle/'MANIFEST.json').read_text())
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in sorted([*manifest['files'],'MANIFEST.json']):z.write(bundle/name,bundle.name+'/'+name)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('archive CRC failed')
        if len(z.namelist())!=len(manifest['files'])+1:raise ValueError('archive file count differs')
    config=dict(status='VALIDATED_LOCAL_MODEL',release='retargetmap_selected_v1',
        bundle=str(bundle.relative_to(ROOT)),archive=str(archive.relative_to(ROOT)),
        manifest_sha256=file_identity(bundle/'MANIFEST.json')['sha256'],checkpoint_sha256=s['checkpoint']['sha256'],
        downstream_training_cutoff=2025,scope=dict(drugs=720,targets=384),
        interface='retargetmap.CatalogRanker',cli=str((bundle/'infer.py').relative_to(ROOT)),default_precision='fp32',
        source_selection=str(selection.relative_to(ROOT)),validation=str(validation.relative_to(ROOT)),
        evaluated_temporal_weights_cutoff=2022,fullfit_has_independent_test_score=False)
    write_json(ROOT/'configs/biomaster_selected_catalog_20260906.json',config)
    result=dict(status='COMPLETE',bundle=str(bundle),archive=file_identity(archive),
        model=file_identity(bundle/'model.pt'),manifest=file_identity(bundle/'MANIFEST.json'),
        validation=file_identity(validation),selection=file_identity(selection),
        configuration=file_identity(ROOT/'configs/biomaster_selected_catalog_20260906.json'),
        uncompressed_bytes=sum(p['size_bytes'] for p in manifest['files'].values()),
        model_card=file_identity(bundle/'MODEL_CARD.md'),archive_crc='PASS',
        one_checkpoint=True,independent_inference=True,driver=file_identity(Path(__file__)))
    write_json(OUT/'DELIVERED_MODEL.json',result);state('COMPLETE',delivered=file_identity(OUT/'DELIVERED_MODEL.json'))
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wait-for-delivery',action='store_true');a=p.parse_args()
    try:main(a.wait_for_delivery)
    except BlockingIOError:raise
    except BaseException as e:state('FAILED',error=str(e));raise
