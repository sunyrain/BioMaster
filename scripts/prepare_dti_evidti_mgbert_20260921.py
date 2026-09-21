#!/usr/bin/env python3
"""Require published EviDTI feature parity before extending its MG-BERT encoder."""
import sys,os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np,pandas as pd,tensorflow as tf
from dti_official_runtime_20260921 import ROOT,inputs,directory,dump,sha
sys.path.insert(0,str(ROOT/'.external/Molecular-graph-BERT'))
from model import Encoder
from utils import smiles2adjoin
from dataset import str2num
out=directory('EviDTI');repo=ROOT/'third_party/sota_dti_2026/EviDTI';tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(2)
model=Encoder(num_layers=6,d_model=256,num_heads=8,dff=512,input_vocab_size=17,maximum_position_encoding=8000)
def encode(s):
 atoms,adj=smiles2adjoin(s,explicit_hydrogens=True);ids=np.array([[16]+[str2num.get(a,14) for a in atoms]],dtype=np.int64);ad=np.ones((len(atoms)+1,len(atoms)+1));ad[1:,1:]=adj
 return model(tf.constant(ids),training=False,mask=tf.cast(ids==0,tf.float32)[:,None,None,:],adjoin_matrix=tf.constant(((1-ad)*-1e9)[None,:,:],dtype=tf.float32)).numpy()[0,0]
encode('CC');weight=repo/'models/bert_weights_encoderMedium_10.h5';model.load_weights(weight)
case=pd.read_csv(repo/'dataset/case_drugbank/drugbank_total.csv');features=np.load(repo/'dataset/case_drugbank/drug_2d_feature.npy');rows=[]
for idx,d in case.drop_duplicates('cid').iterrows():
 actual=encode(d.smile);expected=features[idx];rows.append(dict(drug_id=d.cid,max_abs_delta=float(np.max(np.abs(actual-expected))),cosine=float(actual@expected/np.linalg.norm(actual)/np.linalg.norm(expected))))
maxdelta=max(r['max_abs_delta'] for r in rows);passed=maxdelta<1e-3
dump(out/'MGBERT_REPLAY_CHECK.json',{'passed':passed,'max_abs_delta':maxdelta,'tolerance':1e-3,'published_drugs':rows,'encoder_sha256':sha(weight),'upstream_source':'zhang-xuan1314/Molecular-graph-BERT@cf801e1875dfeef77a90247d505687f0c292761a','restored_missing_readout':'global node before classifier; qualification depends on published feature replay'})
if not passed:raise RuntimeError('Published feature parity failed: do not use guessed EviDTI input pipeline')
drugs,_=inputs();allvalues=[];failed=[]
for d in drugs.itertuples():
 try:allvalues.append(dict(drug_id=d.drug_id,embedding=encode(d.smiles)))
 except Exception as e:failed.append(dict(drug_id=d.drug_id,reason=str(e)))
pd.DataFrame(allvalues).to_parquet(out/'DRUG_MGBERT.parquet',index=False);dump(out/'MGBERT_FAILURES.json',failed)
