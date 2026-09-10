#!/usr/bin/env python3
"""Export a standalone catalog ranker; requires a selection record outside smoke runs."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from biomaster.portable_ranker import digest
from prepare_biomaster_unified_interaction import SOURCE,OUTPUT,write_json
from biomaster.odti_pockets_v3 import file_identity


def export(checkpoint,destination,selection=None,smoke=False):
    checkpoint=Path(checkpoint);destination=Path(destination)
    if destination.exists():raise FileExistsError('use a new versioned bundle directory')
    record=None
    if not smoke:
        if selection is None:raise ValueError('final export requires an explicit selection record')
        record=json.loads(Path(selection).read_text())
        if record.get('status')!='SELECTED' or record.get('checkpoint_sha256')!=digest(checkpoint):
            raise ValueError('checkpoint not authorized by selection record')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    refined=bool(state.get('refinement'))
    variant=state['config']['variant'];local=variant in ['sequence','site','geometry']
    destination.mkdir(parents=True);feature=destination/'features';feature.mkdir()
    package=destination/'retargetmap';package.mkdir()
    names=['unified_interaction.py','refined_interaction.py','portable_ranker.py']
    for name in names:
        text=(ROOT/'biomaster'/name).read_text()
        if name=='refined_interaction.py':text=text.replace('from biomaster.unified_interaction import','from .unified_interaction import')
        (package/name).write_text(text)
    (package/'__init__.py').write_text('from .portable_ranker import CatalogRanker\n\n__all__ = ["CatalogRanker"]\n')
    old=pd.read_csv(SOURCE/'OLD_DRUG_INDEX.csv').sort_values('old_drug_index')
    targets=pd.read_csv(SOURCE/'TARGET_INDEX.csv.gz').sort_values('target_feature_index')
    d=old.drug_feature_index.to_numpy(int);t=targets.target_feature_index.to_numpy(int)
    if not old.ligand_inchikey.is_unique or not targets.uniprot_accession.is_unique:
        raise ValueError('packaged catalog keys must be unique')
    pd.DataFrame(dict(drug_id=old.ligand_inchikey,name=old.drug_names,
        smiles=old.model_ligand_smiles,native_feature_index=d)).to_csv(destination/'drugs.csv.gz',index=False)
    pd.DataFrame(dict(target_id=targets.uniprot_accession,gene_symbol=targets.gene_symbol,
        chembl_id=targets.target_chembl_id,native_feature_index=t)).to_csv(destination/'targets.csv.gz',index=False)
    def save(name,value):np.save(feature/(name+'.npy'),value,allow_pickle=False)
    index=np.load(OUTPUT/'ATOM_INDEX.npz')
    save('drug_global',np.load(OUTPUT/'MOLECULE_GLOBAL.npy',mmap_mode='r')[d])
    save('drug_graph_mean',index['graph_mean'][d])
    save('pretrained_available',np.load(OUTPUT/'PRETRAINED_AVAILABLE.npy')[d].astype(np.float32))
    save('target_global',np.load(OUTPUT/'TARGET_GLOBAL_MEAN.npy')[t])
    if local:
        lengths=index['lengths'][d];offsets=np.r_[0,np.cumsum(lengths)].astype(np.int64)
        rows=np.concatenate([np.arange(index['offsets'][i],index['offsets'][i+1]) for i in d])
        save('atom_lengths',lengths);save('atom_offsets',offsets)
        for name,source in [('atom_tokens','ATOM_TOKENS'),('atom_chemistry','ATOM_CHEMISTRY'),
                            ('neighbors','ATOM_NEIGHBORS'),('bond','ATOM_BOND'),('stereo','ATOM_STEREO')]:
            save(name,np.load(OUTPUT/(source+'.npy'),mmap_mode='r')[rows])
        mode='SEQUENCE' if variant=='sequence' else 'SITE'
        save('residue_tokens',np.load(OUTPUT/f'TARGET_{mode}_TOKENS.npy',mmap_mode='r')[t])
        save('residue_indices',np.load(OUTPUT/f'TARGET_{mode}_INDICES.npy')[t])
        coverage=pd.read_csv(OUTPUT/'TARGET_COVERAGE.csv').sort_values('target_feature_index')
        save('protein_lengths',coverage.length.to_numpy()[t])
        for name,source in [('ca','TARGET_CA'),('quality','TARGET_QUALITY'),('geometry_mask','TARGET_GEOMETRY_MASK')]:
            save(name,np.load(OUTPUT/(source+'.npy'))[t])
    torch.save(dict(architecture='refined' if refined else 'unified',config=state['config'],model=state['model']),destination/'model.pt')
    write_json(destination/'metadata.json',dict(format_version=1,
        status='SMOKE_ONLY_NOT_SELECTED' if smoke else 'SELECTED_CATALOG_MODEL',drugs=len(d),targets=len(t),
        training_cutoff=state['cutoff'],seed=state['seed'],epoch=state['epoch'],variant=variant,
        scores='two directional logits; not calibrated binding probabilities',
        scope='packaged 720-drug / 384-target core; new entities require a separately validated feature extension',
        default_rank_denominator=dict(drug_to_target=len(t),target_to_drug=len(d)),
        pretrained_scope='DrugCLIP mol encoder and ESM2 features frozen; pretraining relation overlap/chronology not certified',
        structure_fallback='missing CA/quality/mask disables geometry; sequence/site tokens remain',
        selection_record=record,source_checkpoint=file_identity(checkpoint)))
    (destination/'requirements.txt').write_text('numpy>=1.24\npandas>=2.0\ntorch>=2.1\n')
    (destination/'infer.py').write_text('''import argparse
from pathlib import Path
from retargetmap import CatalogRanker

parser=argparse.ArgumentParser(description="Rank the packaged old-drug/target catalog")
query=parser.add_mutually_exclusive_group(required=True)
query.add_argument("--drug",help="Full InChIKey from drugs.csv.gz")
query.add_argument("--target",help="UniProt accession from targets.csv.gz")
parser.add_argument("--top-k",type=int,default=20)
parser.add_argument("--device",default="cpu")
args=parser.parse_args()
model=CatalogRanker(Path(__file__).resolve().parent,device=args.device)
result=model.rank_targets(args.drug,top_k=args.top_k) if args.drug else model.rank_drugs(args.target,top_k=args.top_k)
print(result.to_csv(index=False),end="")
''')
    (destination/'README.md').write_text(f'''# ReTargetMap catalog model

Status: {'SMOKE TEST ONLY — this is not a selected model.' if smoke else 'Selected checkpoint; consult metadata.json and the accompanying model card for its evaluation scope.'}

This directory is self-contained. It ranks exactly {len(d)} packaged drugs and {len(t)} packaged targets using one neural checkpoint and cached label-free encoder features. No training scripts, support stores, internet download or optimizer state are needed at inference.

Install `requirements.txt`, then run:

```sh
python infer.py --drug {old.ligand_inchikey.iloc[0]} --top-k 20
python infer.py --target {targets.uniprot_accession.iloc[0]} --top-k 20
```

The Python interface is `from retargetmap import CatalogRanker`; initialize it with this directory. `score_pairs` returns both directional logits. `rank_targets` and `rank_drugs` accept catalog ID candidate lists and report the actual candidate count. Drugs use full InChIKeys; targets use UniProt accessions. Unknown IDs and duplicate ranking candidates are rejected. Scores are not calibrated probabilities or measured affinities. The 745-target registry is not a common ranking denominator.

Default inference uses FP32 and CPU; `--device cuda` is optional. Hashes are verified on load. Optional receptor geometry can be absent; the loader then disables geometric messages. The public DrugCLIP source/weights used for cached representations have research/noncommercial license constraints; see provenance.json. Downstream date cutoffs do not certify public pretraining chronology or absence of overlap.
''')
    provenance=json.loads((OUTPUT.parent/'ENCODER_PROVENANCE.json').read_text())
    write_json(destination/'provenance.json',dict(encoder=provenance,
        input_manifests=[file_identity(OUTPUT/'ATOM_MANIFEST.json'),file_identity(OUTPUT/'TARGET_MANIFEST.json')],
        exporter=file_identity(Path(__file__)),code=[file_identity(ROOT/'biomaster'/n) for n in names]))
    files={}
    for path in sorted(destination.rglob('*')):
        if not path.is_file():continue
        name=str(path.relative_to(destination));files[name]=dict(sha256=digest(path),size_bytes=path.stat().st_size)
        if name in ['features/ca.npy','features/quality.npy','features/geometry_mask.npy']:
            files[name]['optional_geometry']=True
    write_json(destination/'MANIFEST.json',dict(format_version=1,files=files))
    print(json.dumps(dict(status='EXPORTED_SMOKE' if smoke else 'EXPORTED',directory=str(destination),
        size_bytes=sum(r['size_bytes'] for r in files.values()),files=len(files))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--selection',type=Path);parser.add_argument('--smoke',action='store_true')
    a=parser.parse_args();export(a.checkpoint,a.destination,a.selection,a.smoke)
