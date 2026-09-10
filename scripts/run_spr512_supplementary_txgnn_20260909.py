"""Frozen Explorer inference only; encode once, use original decoder, all diseases."""
import os,json,time,random
from pathlib import Path
import numpy as np,pandas as pd,torch,dgl
from txgnn import TxData,TxGNN
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/spr512_integrated_disease_20260909'
def main():
 torch.set_num_threads(8);torch.manual_seed(0);np.random.seed(0);random.seed(0)
 print('load existing full_graph split',flush=True)
 data=TxData(data_folder_path=str(ROOT/'data/raw/txgnn'));data.prepare_split(split='full_graph',seed=42)
 model=TxGNN(data=data,weight_bias_track=False,device='cpu');print('load weights; no training',flush=True)
 model.load_pretrained(str(ROOT/'data/raw/txgnn/TxGNNExplorer'));model.model.eval()
 print('encode once',flush=True)
 with torch.no_grad():h=model.retrieve_embedding()
 # Retain exact encoder result to allow reproducible decoder-only re-runs.
 # Encoder is regenerated from the unchanged weights and verified against native predict.
 c=pd.read_csv(OUT/'DRUG_COVERAGE.csv');c=c[c.status.eq('SUPPLEMENTARY_INFERENCE')].copy()
 kg=pd.read_csv(ROOT/'data/raw/txgnn/kg_directed.csv',dtype={'x_id':str,'y_id':str})
 nodes=pd.concat([kg[['x_type','x_id','x_idx']].set_axis(['type','id','idx'],axis=1),kg[['y_type','y_id','y_idx']].set_axis(['type','id','idx'],axis=1)]).drop_duplicates();nodes['id']=nodes.id.str.replace(r'\.0$','',regex=True)
 lookup=nodes[nodes.type.eq('drug')].set_index('id').idx
 c['drug_idx']=c.graph_drug_id.map(lookup).astype(int);c=c.sort_values('drug_idx');c.to_csv(OUT/'TXGNN_SCORE_DRUG_ORDER.csv',index=False)
 ds=pd.read_csv(ROOT/'outputs/txgnn_biopathnet_comparison_20260909/COMMON_DISEASE_UNIVERSE.csv').sort_values('idx');ds.to_csv(OUT/'TXGNN_SCORE_DISEASE_ORDER.csv',index=False)
 sizes={k:model.G.num_nodes(k) for k in model.G.ntypes};etype=('drug','indication','disease')
 def decode(drugidx,diseaseidx):
  out={e:(torch.empty(0,dtype=torch.int64),torch.empty(0,dtype=torch.int64)) for e in model.G.canonical_etypes}
  out[etype]=(torch.tensor(drugidx),torch.tensor(diseaseidx));q=dgl.heterograph(out,num_nodes_dict=sizes)
  with torch.no_grad():scores,_=model.model.pred(q,model.G,h,pretrain_mode=False,mode='test_pos')
  return scores[etype].detach().numpy()
 # Native predict comparison on seen and unseen disease queries, all selected drug nodes.
 x=np.tile(c.drug_idx.values,3);y=np.repeat(ds.idx.values[[0,len(ds)//2,len(ds)-1]],len(c))
 fast=decode(x,y)
 with torch.no_grad():native=model.predict(pd.DataFrame({'x_idx':x,'y_idx':y,'relation':'indication'}))[etype].numpy()
 delta=float(np.max(np.abs(fast-native)));assert np.allclose(fast,native,atol=1e-5,rtol=1e-5),delta
 (OUT/'TXGNN_DECODER_VERIFICATION.json').write_text(json.dumps({'native_comparison_rows':len(x),'max_abs_error':delta,'seed':0,'no_training':True},indent=2))
 scores=np.lib.format.open_memmap(str(OUT/'TXGNN_INDICATION_LOGITS.npy'),mode='w+',dtype='float32',shape=(len(c),len(ds)))
 for start in range(0,len(ds),64):
  ids=ds.idx.values[start:start+64];x=np.tile(c.drug_idx.values,len(ids));y=np.repeat(ids,len(c))
  v=decode(x,y).reshape(len(ids),len(c)).T;assert np.isfinite(v).all();scores[:,start:start+len(ids)]=v
  if start%1024==0: scores.flush();print('diseases',start,'/',len(ds),flush=True)
 scores.flush()
 (OUT/'TXGNN_RUN_COMPLETE.json').write_text(json.dumps({'drugs':len(c),'diseases':len(ds),'scores':int(scores.size),'native_max_abs_error':delta,'relation':'indication','weights_changed':False,'training':False,'score_semantics':'raw_logit_not_calibrated_probability','seed':0},indent=2))
 print('COMPLETE',scores.shape,flush=True)
if __name__=='__main__':main()
