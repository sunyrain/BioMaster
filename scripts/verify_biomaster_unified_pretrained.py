#!/usr/bin/env python3
"""Compare custom atom input/encoder extraction with the unchanged DrugCLIP API."""
from pathlib import Path
import pickle
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import lmdb
import torch
from torch.utils.data import DataLoader
from prepare_biomaster_unified_interaction import (load_pretrained,conformer,heavy_record,
                                                  pretrained_batch,pretrained_tokens,write_json,OUTPUT)


def main():
    torch.set_num_threads(4)
    task,model=load_pretrained()
    records=[conformer((i,s))[1] for i,s in enumerate(['CCO','N[C@@H](C)C(=O)O','Clc1ccc(CC(=O)O)cc1','[2H]C([2H])([2H])OC'])]
    own_records=[heavy_record(r) for r in records]
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'molecules.lmdb'
        env=lmdb.open(str(path),subdir=False,map_size=32*1024**2)
        with env.begin(write=True) as txn:
            for i,r in enumerate(records):
                txn.put(str(i).encode(),pickle.dumps(dict(atoms=r['atoms'],coordinates=[r['coordinates']],smi=r['smiles'])))
        env.close()
        dataset=task.load_mols_dataset_dtwg(str(path),'atoms','coordinates',dataset_type=1)
        official=next(iter(DataLoader(dataset,batch_size=len(records),collate_fn=dataset.collater)))['net_input']
        own=pretrained_batch(own_records,task.dictionary);audit={}
        keys=['mol_src_tokens','mol_src_distance','mol_src_edge_type']
        for j,key in enumerate(keys):
            value=official[key].cuda()
            value=value[:,:own[j].shape[1]] if j==0 else value[:,:own[j].shape[1],:own[j].shape[1]]
            error=float((own[j]-value).abs().max());audit[key+'_max_error']=error
            assert error<2e-5,(key,error)
        with torch.inference_mode():
            reference=pretrained_tokens(model,tuple(official[k].cuda() for k in keys))
            actual=pretrained_tokens(model,own)
            torch.testing.assert_close(actual,reference[:,:actual.shape[1]],atol=2e-4,rtol=2e-4)
            audit['pretrained_token_max_error']=float((actual-reference[:,:actual.shape[1]]).abs().max())
        audit.update(status='PASS',comparison='unmodified upstream DrugCLIP dataset and encoder',
                     molecule_count=len(records),explicit_isotope_hydrogen_removal_tested=True)
        write_json(OUTPUT/'PRETRAINED_COMPATIBILITY.json',audit)
        print(audit)


if __name__=='__main__':main()
