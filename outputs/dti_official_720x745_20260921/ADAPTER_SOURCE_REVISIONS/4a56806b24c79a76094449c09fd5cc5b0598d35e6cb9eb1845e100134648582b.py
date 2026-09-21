#!/usr/bin/env python3
"""Restore EviDTI DrugBank native three-input inference on frozen identities."""
import sys,time,copy
import numpy as np,pandas as pd,torch
from dti_official_runtime_20260921 import ROOT,inputs,directory,dump,sha,status,contract,save_scores,read_scores

def main():
 name='EviDTI';out=directory(name);start=time.time()
 while not all((out/f).exists() for f in ['PROTEIN_READY.json','GEOMETRY_READY.json','DRUG_MGBERT.parquet']):
  status(name,'WAITING_NATIVE_FEATURES',scored_pairs=0);time.sleep(30)
 from run_evidti_case_cross_rank import install_evidti_compat,patch_loaded_model,load_case_dataset
 install_evidti_compat()
 from torch_geometric.data import Data,Batch
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 checkpoint=ROOT/'third_party/sota_dti_2026/EviDTI/runs/drugbank_model/checkpoint.pth'
 model=patch_loaded_model(torch.load(checkpoint,map_location='cpu')).cuda().eval();drugs,targets=inputs();df=pd.read_parquet(out/'DRUG_MGBERT.parquet');mg=dict(zip(df.drug_id,df.embedding))
 geom={d.drug_id:np.load(out/'geometry'/f'{d.drug_id}.npy',allow_pickle=True).item() for d in drugs.itertuples() if (out/'geometry'/f'{d.drug_id}.npy').exists() and d.drug_id in mg}
 proteins={t.target_id:np.load(out/'protein'/f'{t.target_id}.npy') for t in targets.itertuples() if (out/'protein'/f'{t.target_id}.npy').exists()}
 def graph(d,t):
  v=geom[d.drug_id];atom=np.stack([v[k] for k in ['atomic_num','chiral_tag','degree','explicit_valence','formal_charge','hybridization','implicit_valence']]);bond=np.stack([v[k] for k in ['bond_dir','bond_type','is_in_ring','bond_length']])
  x=Data(x=torch.tensor(atom).T,edge_index=torch.tensor(v['edges'].T,dtype=torch.long),edge_attr=bond.T)
  x.bag=Data(x=torch.tensor(bond).T,edge_index=torch.tensor(v['BondAngleGraph_edges'].T,dtype=torch.long),edge_attr=v['bond_angle'])
  x.d_2D_feature=mg[d.drug_id];x.t_1D_feature=proteins[t.target_id];x.l=0
  x.metadata={'id':t.target_id,'length':len(t.sequence),'sequence':t.sequence,'d_id':d.drug_id,'smiles':d.smiles};return x
 def predict(items):
  b=Batch.from_data_list(items);lengths=b.metadata['length'].cuda();mask=torch.arange(lengths.max(),device='cuda')[None,:]<lengths[:,None]
  with torch.inference_mode():
   a=model(b,mask=mask,sequence_lengths=lengths[:,None]);v=(a/a.sum(1,keepdim=True))[:,1].cpu().numpy()
  assert np.isfinite(v).all();return v
 # Hold published 3D geometry fixed when checking restored 2D/protein encoders.
 from rdkit import Chem
 drug_by_key={d.drug_id:d for d in drugs.itertuples()};target_by_seq={t.sequence:t for t in targets.itertuples()};published=[];restored=[]
 for item in load_case_dataset():
  meta=item.metadata;sm=meta.get('smiles') or meta.get('smile');key=Chem.MolToInchiKey(Chem.MolFromSmiles(sm));t=target_by_seq.get(meta['sequence'])
  if key not in mg or t is None or t.target_id not in proteins:continue
  c=copy.deepcopy(item);c.d_2D_feature=mg[key];c.t_1D_feature=proteins[t.target_id];published.append(item);restored.append(c)
  if len(published)==4:break
 assert len(published)==4
 original=np.concatenate([predict([x]) for x in published]);new=np.concatenate([predict([x]) for x in restored]);delta=float(np.max(abs(original-new)));assert delta<1e-3,delta
 ds=[d for d in drugs.itertuples() if d.drug_id in geom];ts=[t for t in targets.itertuples() if t.target_id in proteins]
 smoke=[graph(d,t) for d,t in zip(ds[:3],ts[:3])];single=np.concatenate([predict([x]) for x in smoke]);batch=predict(smoke);batchdelta=float(np.max(abs(single-batch)));assert batchdelta<1e-4,batchdelta
 dump(out/'REPLAY_CHECK.json',{'passed':True,'restored_encoders_native_score_max_delta':delta,'native_single_vs_batch_max_delta':batchdelta,'comparison':'Public 3D held fixed; test encoder restoration separately from allowed conformer regeneration'})
 contract(name,checkpoint_sha256=sha(checkpoint),adapter_sha256=sha(__file__),head='DrugBank evidential alpha1/(alpha0+alpha1)',inputs='Replayed MG-BERT global node; author MMFF10 geometry; full ProtT5 per residue FP16',precision='FP32 task model, native FP16 ProtT5; checkpoint compatibility uses existing validated PyG adapter',native_source='Local EviDTI official repo and public reference features',new_geometry='Native stochastic conformer generation; cached exact inputs and RDKit seed retained')
 frame=read_scores(name);failures=[];generated=0;began=time.time()
 for t in ts:
  done=set(frame.loc[frame.target_id.eq(t.target_id),'drug_id']);pending=[d for d in ds if d.drug_id not in done];rows=[]
  if not pending:continue
  for i in range(0,len(pending),8):
   b=pending[i:i+8]
   try:
    v=predict([graph(d,t) for d in b]);rows.extend(dict(drug_id=d.drug_id,target_id=t.target_id,score=float(s)) for d,s in zip(b,v))
   except RuntimeError as e:
    torch.cuda.empty_cache()
    for d in b:
     try:rows.append(dict(drug_id=d.drug_id,target_id=t.target_id,score=float(predict([graph(d,t)])[0])))
     except RuntimeError as error:failures.append(dict(drug_id=d.drug_id,target_id=t.target_id,reason=str(error)[:400]));torch.cuda.empty_cache()
   generated+=len(b)
  frame=pd.concat([frame,pd.DataFrame(rows)],ignore_index=True);save_scores(name,frame);dump(out/'PAIR_FAILURES.json',failures)
  status(name,'INFERENCE',scored_pairs=len(frame),target=t.gene,elapsed_seconds=time.time()-start,eta_seconds=(len(ds)*len(ts)-len(frame))*(time.time()-began)/max(generated,1))
 status(name,'COMPLETE' if len(frame)==536400 else 'COMPLETE_WITH_NATIVE_INPUT_FAILURES',scored_pairs=len(frame),elapsed_seconds=time.time()-start)
if __name__=='__main__':main()
