#!/usr/bin/env python3
"""Exact author descriptor extractor; import only the required Karate Club classes."""
import sys,os,types,importlib.util
import numpy as np,pandas as pd
from rdkit import Chem
from pathlib import Path
from dti_official_runtime_20260921 import ROOT,inputs,directory,dump,sha
out=directory('ADME-DTI');sys.path.insert(0,str(ROOT/'.external/ADME-DTI'))
# karateclub.__init__ eagerly imports unrelated optional graph algorithms.
# Load its unmodified Estimator and LDP source without these unused dependencies.
spec=importlib.util.find_spec('karateclub');root=Path(spec.origin).parent
pkg=types.ModuleType('karateclub');pkg.__path__=[str(root)];sys.modules['karateclub']=pkg
for name,path in [('karateclub.estimator',root/'estimator.py'),('karateclub.graph_embedding.ldp',root/'graph_embedding/ldp.py')]:
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m)
pkg.Estimator=sys.modules['karateclub.estimator'].Estimator;pkg.LDP=sys.modules['karateclub.graph_embedding.ldp'].LDP
from utils.embedding.extractor import drug_extractor as extractor
# scikit-mol no longer makes every concrete transformer inherit the deprecated
# FpsTransformer alias. Restore only the author's isinstance dispatch.
extractor.FpsTransformer = tuple(getattr(extractor, n) for n in [
 'AtomPairFingerprintTransformer','AvalonFingerprintTransformer','MACCSKeysFingerprintTransformer',
 'MorganFingerprintTransformer','RDKitFingerprintTransformer','SECFingerprintTransformer',
 'TopologicalTorsionFingerprintTransformer'])
os.chdir(out);Path('data/embeddings/drug_embedding').mkdir(parents=True,exist_ok=True)
d,_=inputs();failures=[]
for cls in [*extractor.FpsTransformer,extractor.LDP]:
 name=cls.__name__;path=Path('data/embeddings/drug_embedding')/(name+'.parquet')
 if path.exists():continue
 try:extractor.__drug_transformer(d.smiles.to_numpy(),cls())
 except Exception:
  if cls is extractor.LDP:raise
  valid=[];arrays=[];transformer=cls()
  for drug in d.itertuples():
   try:
    v=transformer.transform([Chem.MolFromSmiles(drug.smiles)])[0];valid.append(drug.smiles);arrays.append(v)
   except Exception as e:failures.append(dict(drug_id=drug.drug_id,descriptor=name,reason=str(e)[:400]))
  x=np.stack(arrays);x=(x-x.min())/(x.max()-x.min())
  pd.DataFrame({'drug':valid,'embedding':list(x)}).to_parquet(path,index=False)
dump(out/'DRUG_DESCRIPTOR_FAILURES.json',failures)
dump(out/'DRUG_DESCRIPTOR_CONTRACT.json',{'pool':'Fixed 720 drugs, fresh native descriptor cache; native global min/max per transformer','scikit_mol_version':'0.4.6','compatibility':'Concrete-class isinstance dispatch for deprecated FpsTransformer alias; descriptor math unchanged','karateclub_version':'1.3.3','ldp_source_sha256':sha(root/'graph_embedding/ldp.py'),'ldp_fit':'Non-parametric graph histogram descriptor; no supervised fitting'})
