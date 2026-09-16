#!/usr/bin/env python3
"""Run released occurrence checkpoint with publisher protein graphs and exact sequences."""
import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from rdkit import RDLogger
from torch_geometric.data import Batch

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'.external/DTBind'
OUT=ROOT/'outputs/frontier_dti_20260916/dtbind'

def atomic_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def main():
    OUT.mkdir(parents=True,exist_ok=True);drug_dir=OUT/'drug_graph';drug_dir.mkdir(exist_ok=True)
    torch.set_num_threads(3);torch.manual_seed(20260916);RDLogger.DisableLog('rdApp.warning')
    # Keep the released graph featurizer, including its bond-index behavior;
    # altering it would require retraining, rather than comparing released weights.
    script=REPO/'data_process/graph_construction/drug_gra.py'
    tree=ast.parse(script.read_text());tree.body=[n for n in tree.body if not isinstance(n,ast.Expr)]
    module={};exec(compile(tree,str(script),'exec'),module)
    sys.path.insert(0,str(REPO/'script/occurrence'))
    from dti_model import DTISite,params
    torch.autograd.set_detect_anomaly(False)
    checkpoint=REPO/'models/occurrence_model.pth'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=DTISite(params).cpu().eval();model.load_state_dict(payload['model_state_dict'],strict=True)
    threshold=float(payload.get('best_threshold',0.5))
    def score(protein,drug):
        with torch.inference_mode():
            probability=model(Batch.from_data_list([protein]).cpu(),Batch.from_data_list([drug]).cpu()).reshape(-1)
        assert probability.numel()==1
        # DTISite.classifier already ends in Sigmoid and training uses BCELoss.
        # The published dti_test.py accidentally applies a second Sigmoid.
        # Keep that wrapper output only for exact reproduction, never as our score.
        value=float(probability.item())
        if not np.isfinite(value):raise ValueError('nonfinite output')
        return value
    smoke=[]
    for r in pd.read_csv(REPO/'sample_test/occurrence/sample_data.tsv',sep='\t').itertuples(index=False):
        pg=torch.load(REPO/f'sample_test/occurrence/protein_graph/{r.protein}.pt',map_location='cpu',weights_only=False)
        dg=torch.load(REPO/f'sample_test/occurrence/ligand_graph/{r.drug}.pt',map_location='cpu',weights_only=False)
        value=score(pg,dg)
        smoke.append(dict(drug_id=r.drug,protein_id=r.protein,label=r.label,score=value,official_wrapper_score=float(torch.sigmoid(torch.tensor(value)).item())))
    pd.DataFrame(smoke).to_csv(OUT/'OFFICIAL_SMOKE.csv',index=False)
    expected=pd.read_csv(REPO/'logs/occurrence/prediction_results_2025-10-08_10-22-37.csv')
    reproduction=pd.DataFrame(smoke).merge(expected,on=['drug_id','protein_id'],validate='one_to_one')
    assert len(reproduction)==5 and np.allclose(reproduction.official_wrapper_score,reproduction.prediction_prob,atol=1e-7,rtol=0)
    data=pd.read_csv(OUT.parent/'UNIQUE_PAIRS.csv')
    seqs={r.id:str(r.seq) for r in SeqIO.parse(REPO/'Data/dti/biosnap_protein_seq.fasta','fasta')}
    drugs={};proteins={};predictions={};started=time.time()
    # Resume only the current score contract, identified by its model-output smoke.
    prior=OUT/'PREDICTIONS.csv'
    if prior.exists():
        for r in pd.read_csv(prior).to_dict('records'):
            if r['status']=='completed' or r.get('reason') in ('publisher_sequence_mismatch','publisher_protein_graph_not_available'):
                predictions[r['pair_id']]=r
    for row in data.drop_duplicates('drug_id').itertuples(index=False):
        try:
            path=drug_dir/(row.drug_id+'.pt')
            if not path.exists():module['Mol2Graph'](row.smiles,row.drug_id,str(drug_dir))
            drugs[row.drug_id]=torch.load(path,map_location='cpu',weights_only=False)
        except Exception as exc:drugs[row.drug_id]=f'{type(exc).__name__}: {exc}'
    def flush(state):
        temp=OUT/'PREDICTIONS.tmp';pd.DataFrame(list(predictions.values()),columns=['pair_id','status','score','reason']).to_csv(temp,index=False);temp.replace(OUT/'PREDICTIONS.csv')
        atomic_json(OUT/'STATUS.json',dict(state=state,pid=os.getpid(),completed=sum(x['status']=='completed' for x in predictions.values()),attempted=len(predictions),total=len(data),elapsed_seconds=time.time()-started))
    while True:
        fetch_path=OUT/'FETCH_STATUS.json'
        try: fetching=json.loads(fetch_path.read_text()).get('status')!='completed'
        except (FileNotFoundError,json.JSONDecodeError):fetching=True
        fetch_failed=False
        if fetching:
            try:
                fetch_pid=int((OUT.parent/'fetch_dtbind.pid').read_text())
                fetch_alive=Path(f'/proc/{fetch_pid}/stat').read_text().split()[2]!='Z'
            except (OSError,ValueError):fetch_alive=False
            if not fetch_alive and time.time()-started>120:
                fetching=False;fetch_failed=True
        for row in data.itertuples(index=False):
            if row.pair_id in predictions:continue
            result=dict(pair_id=row.pair_id,status='unavailable',score=None,reason='')
            if row.uniprot_id not in seqs:result['reason']='publisher_protein_graph_not_available'
            elif seqs[row.uniprot_id]!=row.sequence:result['reason']='publisher_sequence_mismatch'
            elif isinstance(drugs[row.drug_id],str):result['reason']='drug_graph_error: '+drugs[row.drug_id]
            else:
                path=OUT/f'protein_graph/{row.uniprot_id}.pt'
                if not path.exists():
                    if fetching:continue
                    result['reason']='feature_download_incomplete' if fetch_failed else 'publisher_protein_graph_not_available'
                else:
                    try:
                        if row.uniprot_id not in proteins:
                            graph=torch.load(path,map_location='cpu',weights_only=False)
                            if graph.x.shape!=(len(row.sequence),1024):raise ValueError('protein embedding/sequence shape mismatch')
                            if graph.surface_x.shape!=(len(row.sequence),10):raise ValueError('surface/sequence shape mismatch')
                            proteins[row.uniprot_id]=graph
                        result.update(status='completed',score=score(proteins[row.uniprot_id],drugs[row.drug_id]))
                    except Exception as exc:result.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
            predictions[row.pair_id]=result
            if len(predictions)%10==0:flush('inference')
        flush('waiting_graphs' if fetching else 'completed_with_missing' if fetch_failed else 'completed')
        print('attempted',len(predictions),'scored',sum(x['status']=='completed' for x in predictions.values()),'fetching',fetching,flush=True)
        if not fetching:break
        time.sleep(30)
    atomic_json(OUT/'ADAPTER.json',dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),source_revision='08983e476760fbe5b5dc62be06fd2e087b31ac0c',task='occurrence',score='model output, already sigmoid; no second sigmoid',official_threshold=threshold,strict_weight_load=True,device='cpu',training='none',sequence_match='exact publisher FASTA sequence and graph dimensions; isoform mismatch excluded',protein_features='official BioSnap/AlphaFold, ProtTrans 1024 and surface 10 dimensions; not zero-filled',drug_features='official drug_gra.py unmodified atom/bond features; retains upstream bond-zero omission',smoke_samples=len(smoke),official_example_reproduction='all five wrapper outputs match publisher CSV within 1e-7',wrapper_fix='Removed duplicate sigmoid in official test wrapper; original wrapper values retained in OFFICIAL_SMOKE.csv; underlying model unchanged.',caveat='Public training overlap unknown. No complex affinity predictions without complex coordinates.'))

if __name__=='__main__':main()
