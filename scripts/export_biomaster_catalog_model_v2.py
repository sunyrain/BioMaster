#!/usr/bin/env python3
"""Export only the selected network family, its weights and compact catalog inputs."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from biomaster.portable_ranker_v2 import digest
from biomaster.model_registry import infer_family
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import SOURCE,OUTPUT,write_json

NETWORKS={
 'unified':('unified_interaction','UnifiedConfig','UnifiedInteraction',['unified_interaction']),
 'refined':('refined_interaction','RefinedConfig','RefinedInteraction',['unified_interaction','refined_interaction']),
 'anchored':('anchored_interaction','AnchoredConfig','AnchoredInteraction',['unified_interaction','refined_interaction','anchored_interaction']),
 'chemical':('chemical_representation','ChemicalConfig','ChemicalInteraction',['unified_interaction','chemical_representation']),
 'molecular_control':('molecular_controls','MolecularControlConfig','MolecularControlInteraction',['unified_interaction','molecular_controls']),
}


def export(checkpoint,destination,selection=None,smoke=False):
    checkpoint=Path(checkpoint);destination=Path(destination)
    if destination.exists():raise FileExistsError('use a new versioned output directory')
    record=None
    if not smoke:
        if selection is None:raise ValueError('selected checkpoint record required')
        record=json.loads(Path(selection).read_text())
        if record.get('status')!='SELECTED' or record.get('checkpoint_sha256')!=digest(checkpoint):
            raise ValueError('checkpoint is not the selected release')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);family=infer_family(state)
    module,cfg_class,model_class,sources=NETWORKS[family]
    variant=state['config']['variant'];local=variant in ['sequence','site','geometry']
    representation=state['config'].get('drug_representation','drugclip')
    uses_clip=local or representation in ['drugclip','drugclip_morgan']
    uses_bermol=representation in ['bermol','bermol_morgan']
    uses_morgan=representation in ['morgan','drugclip_morgan','bermol_morgan']
    destination.mkdir(parents=True);feature=destination/'features';feature.mkdir();package=destination/'retargetmap';package.mkdir()
    for name in sources:
        text=(ROOT/'biomaster'/(name+'.py')).read_text()
        for classname in ['ChemicalFeatureBank','MolecularControlBank','AnchoredFeatureBank']:
            text=text.split('\nclass '+classname+':')[0].split('\nclass '+classname+'(')[0]
        text='\n'.join(line for line in text.splitlines() if not line.startswith('from biomaster.best_model_training import'))+'\n'
        text=text.replace('from biomaster.unified_interaction import','from .unified_interaction import')
        text=text.replace('from biomaster.refined_interaction import','from .refined_interaction import')
        if 'biomaster.' in text or 'SupplementedFeatureBank' in text:raise ValueError('training/project dependency survived export')
        (package/(name+'.py')).write_text(text)
    (package/'portable_ranker.py').write_text((ROOT/'biomaster/portable_ranker_v2.py').read_text())
    (package/'model_registry.py').write_text(f'''from .{module} import {cfg_class}, {model_class}

def build_model(family, config):
    if family != {family!r}:
        raise ValueError("This bundle contains one selected architecture")
    return {model_class}({cfg_class}(**config))
''')
    (package/'__init__.py').write_text('from .portable_ranker import CatalogRanker\n\n__all__=["CatalogRanker"]\n')
    old=pd.read_csv(SOURCE/'OLD_DRUG_INDEX.csv').sort_values('old_drug_index')
    targets=pd.read_csv(SOURCE/'TARGET_INDEX.csv.gz').sort_values('target_feature_index')
    d=old.drug_feature_index.to_numpy(int);t=targets.target_feature_index.to_numpy(int)
    assert old.ligand_inchikey.is_unique and targets.uniprot_accession.is_unique
    pd.DataFrame(dict(drug_id=old.ligand_inchikey,name=old.drug_names,smiles=old.model_ligand_smiles,native_feature_index=d)).to_csv(destination/'drugs.csv.gz',index=False)
    pd.DataFrame(dict(target_id=targets.uniprot_accession,gene_symbol=targets.gene_symbol,chembl_id=targets.target_chembl_id,native_feature_index=t)).to_csv(destination/'targets.csv.gz',index=False)
    def save(name,value):np.save(feature/(name+'.npy'),value,allow_pickle=False)
    index=np.load(OUTPUT/'ATOM_INDEX.npz')
    available=np.ones(len(d),np.float32)
    if representation in ['drugclip','drugclip_morgan']:
        drug=np.load(OUTPUT/'MOLECULE_GLOBAL.npy',mmap_mode='r')[d]
        available=np.load(OUTPUT/'PRETRAINED_AVAILABLE.npy')[d].astype(np.float32)
    elif uses_bermol:drug=np.load(SOURCE/'BERMOL.npy',mmap_mode='r')[d]
    elif representation=='morgan':drug=np.load(SOURCE/'MORGAN.npy',mmap_mode='r')[d].astype(np.float32)
    else:raise ValueError('unknown global drug representation')
    if representation.endswith('_morgan'):
        drug=np.concatenate([drug,np.load(SOURCE/'MORGAN.npy',mmap_mode='r')[d].astype(np.float32)],axis=1)
    save('drug_global',drug);save('drug_graph_mean',index['graph_mean'][d]);save('pretrained_available',available)
    save('target_global',np.load(OUTPUT/'TARGET_GLOBAL_MEAN.npy')[t])
    if local:
        lengths=index['lengths'][d];offsets=np.r_[0,np.cumsum(lengths)].astype(np.int64)
        rows=np.concatenate([np.arange(index['offsets'][i],index['offsets'][i+1]) for i in d])
        save('atom_lengths',lengths);save('atom_offsets',offsets)
        for name,source in [('atom_tokens','ATOM_TOKENS'),('atom_chemistry','ATOM_CHEMISTRY'),('neighbors','ATOM_NEIGHBORS'),('bond','ATOM_BOND'),('stereo','ATOM_STEREO')]:
            save(name,np.load(OUTPUT/(source+'.npy'),mmap_mode='r')[rows])
        mode='SEQUENCE' if variant=='sequence' else 'SITE'
        save('residue_tokens',np.load(OUTPUT/f'TARGET_{mode}_TOKENS.npy',mmap_mode='r')[t])
        save('residue_indices',np.load(OUTPUT/f'TARGET_{mode}_INDICES.npy')[t])
        save('protein_lengths',pd.read_csv(OUTPUT/'TARGET_COVERAGE.csv').sort_values('target_feature_index').length.to_numpy()[t])
        for name,source in [('ca','TARGET_CA'),('quality','TARGET_QUALITY'),('geometry_mask','TARGET_GEOMETRY_MASK')]:save(name,np.load(OUTPUT/(source+'.npy'))[t])
    torch.save(dict(architecture=family,config=state['config'],model=state['model']),destination/'model.pt')
    metadata=dict(format_version=2,status='SMOKE_ONLY_NOT_SELECTED' if smoke else 'SELECTED_CATALOG_MODEL',
        drugs=len(d),targets=len(t),training_cutoff=state['cutoff'],seed=state['seed'],epoch=state['epoch'],
        family=family,variant=variant,drug_representation=representation,local_interactions=local,
        pretrained_inputs=dict(esm2=True,drugclip=uses_clip,bermol=uses_bermol),morgan_fingerprints=uses_morgan,
        scores='two directional logits; not calibrated probabilities or measured affinities',
        scope='packaged 720-drug / 384-target core; new entities require validated feature extension',
        default_rank_denominator=dict(drug_to_target=len(t),target_to_drug=len(d)),
        temporal_scope='downstream training cutoff only; encoder training overlap/chronology not certified',
        structure_fallback='missing optional CA/quality/mask disables geometry and preserves site/sequence interactions' if local else 'global model has no receptor structure dependency',
        selection_record=record,source_checkpoint=file_identity(checkpoint))
    write_json(destination/'metadata.json',metadata)
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
    status='SMOKE TEST ONLY. These weights have not been selected.' if smoke else 'Selected catalog checkpoint. See MODEL_CARD.md for development selection and retrospective evaluation.'
    (destination/'README.md').write_text(f'''# ReTargetMap catalog model

{status}

One neural checkpoint ranks {len(d)} drugs and {len(t)} targets. This directory includes the selected network, cached inputs and catalog mappings. It runs independently of the research repository and does not download encoder weights.

Install `requirements.txt`, then run:

```sh
python infer.py --drug {old.ligand_inchikey.iloc[0]} --top-k 20
python infer.py --target {targets.uniprot_accession.iloc[0]} --top-k 20
```

The Python API is `CatalogRanker(directory)` from `retargetmap`. `score_pairs` returns both directional scores; `rank_targets` and `rank_drugs` rank catalog IDs or an explicit candidate subset. Ranking output includes the actual candidate count and readable gene/drug names. Unknown IDs, duplicate candidates and invalid arguments are rejected.

Default inference is FP32 on CPU; `--device cuda` enables GPU inference. Scores are logits, not binding probabilities or experimentally measured affinities. Default target ranking uses the 384-target core; the 745-target registry is not one rank denominator. The current bundle supports the listed catalog, not arbitrary new SMILES or protein sequences.

Global drug representation: `{representation}`. Local interaction refinement: `{local}`. Training supervision cutoff: `{state['cutoff']}`. Hashes are checked on load. Optional geometry can be absent in a local bundle; geometric messages are then disabled. Public encoder chronology and relation overlap are not certified. Provenance and the upstream terms for the representations actually used are recorded in `provenance.json`.
''')
    provenance=dict(source_checkpoint=file_identity(checkpoint),exporter=file_identity(Path(__file__)),
        required_pretrained_inputs=metadata['pretrained_inputs'],source_axes=[file_identity(SOURCE/'OLD_DRUG_INDEX.csv'),file_identity(SOURCE/'TARGET_INDEX.csv.gz')],
        code=[file_identity(ROOT/'biomaster'/(name+'.py')) for name in sources+['portable_ranker_v2']],
        graph_feature_manifest=file_identity(OUTPUT/'ATOM_MANIFEST.json'),target_feature_manifest=file_identity(OUTPUT/'TARGET_MANIFEST.json'))
    if uses_clip:provenance['drugclip']=json.loads((OUTPUT.parent/'ENCODER_PROVENANCE.json').read_text())
    if uses_bermol or uses_morgan:provenance['chemical_inputs']=json.loads((OUTPUT.parent.parent/'biomaster_best_model_20260906/CHEMICAL_FEATURE_IDENTITY.json').read_text())
    if uses_bermol:provenance['bermol_encoder']=file_identity(ROOT/'third_party/sota_dti_2026/DTIAM/code/BerMolModel_base.pkl')
    write_json(destination/'provenance.json',provenance)
    files={}
    for path in sorted(destination.rglob('*')):
        if not path.is_file():continue
        name=str(path.relative_to(destination));files[name]=dict(sha256=digest(path),size_bytes=path.stat().st_size)
        if name in ['features/ca.npy','features/quality.npy','features/geometry_mask.npy']:files[name]['optional_geometry']=True
    write_json(destination/'MANIFEST.json',dict(format_version=2,files=files))
    print(json.dumps(dict(status='EXPORTED_SMOKE' if smoke else 'EXPORTED',family=family,representation=representation,
        directory=str(destination),files=len(files),size_bytes=sum(x['size_bytes'] for x in files.values()))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--selection',type=Path);parser.add_argument('--smoke',action='store_true')
    a=parser.parse_args();export(a.checkpoint,a.destination,a.selection,a.smoke)
