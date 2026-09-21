#!/usr/bin/env python3
"""After fast GPU jobs, retry resource-limited native features before Nesso."""
import json,time,subprocess
from dti_official_runtime_20260921 import ROOT,OUT,dump
WAIT=['MAMMAL_pKd','GraphBAN','EviDTI']
while True:
 states={m:json.loads((OUT/m/'STATUS.json').read_text()).get('state') for m in WAIT}
 dump(OUT/'GPU_FRONT_QUEUE_STATUS.json',{'state':'WAITING','dependencies':states})
 if all(s.startswith('COMPLETE') for s in states.values()):break
 time.sleep(30)
checks=[]
folder=OUT/'EviDTI';before=json.loads((folder/'STATUS.json').read_text());failures=json.loads((folder/'PROTT5_FAILURES.json').read_text())
if any('out of memory' in x.get('reason','').lower() for x in failures):
 old_count=len(list((folder/'protein').glob('*.npy')))
 command=[str(ROOT/'.venvs/balm/bin/python'),str(ROOT/'scripts/prepare_dti_evidti_inputs_20260921.py'),'protein']
 with (OUT/'EviDTI_RESOURCE_RETRY.log').open('a') as log:
  p=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  checks.append({'stage':'native_protein_retry_with_free_GPU','exit_code':p.returncode,'before_targets':old_count,'after_targets':len(list((folder/'protein').glob('*.npy')))})
  if p.returncode==0 and len(list((folder/'protein').glob('*.npy')))>old_count:
   q=subprocess.run(['python',str(ROOT/'scripts/run_dti_official_evidti_20260921.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
   checks.append({'stage':'inference_on_recovered_targets','exit_code':q.returncode})
   if q.returncode:
    before['recovery_error']='See EviDTI_RESOURCE_RETRY.log; original completed partial scope retained'
    dump(folder/'STATUS.json',before)
dump(OUT/'GPU_FRONT_QUEUE_FINISHED.json',{'state':'FINISHED','dependencies':states,'resource_retry':checks})
dump(OUT/'GPU_FRONT_QUEUE_STATUS.json',{'state':'FINISHED','resource_retry':checks})
