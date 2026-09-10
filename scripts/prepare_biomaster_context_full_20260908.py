#!/usr/bin/env python3
"""Resume audited real-global and ligand-blind predicted-pocket preparation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import string
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT/'outputs/biomaster_pocket_precision_20260906/structural_data/training_2020'
OUT = ROOT/'outputs/biomaster_context_full_20260908/data'
PRANK = ROOT/'tools/p2rank_2.5.1/prank'
JAVA = ROOT/'.conda_envs/pocket_tools/lib/jvm/bin/java'


def write_json(path, data):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n'); tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def predict_record(args):
    record, out = args; out = Path(out); sid = record['system_id']
    done = out/'prediction_index'/(sid+'.json')
    if done.exists(): return json.loads(done.read_text())
    from biomaster.structural_complex import mapped_receptor, region_record
    from biomaster.odti_pockets_v3 import THREE_TO_ONE
    started = time.monotonic()
    folder = BASE.parent/'complexes_2020/systems'/sid
    sequences, atoms, _ = mapped_receptor(folder)
    chains = sorted(sequences)
    if len(chains) > 62 or len(atoms) > 99999 or max(map(len, sequences.values())) > 9999:
        raise ValueError('PDB representation budget exceeded; use mmCIF, never truncate: '+sid)
    chain_ids = dict(zip(chains, string.ascii_uppercase+string.ascii_lowercase+string.digits))
    reverse = {v:k for k,v in chain_ids.items()}
    aa3 = {v:k for k,v in THREE_TO_ONE.items() if len(k)==3 and v!='X'}
    lines = []
    for serial, ((chain, index, name), (element, xyz, occupancy)) in enumerate(sorted(atoms.items()), 1):
        # AlphaFold P2Rank configuration ignores B-factors. Protein atoms only;
        # labels/bound ligand never enter prediction or input pocket selection.
        lines.append(f'ATOM  {serial:5d} {name:>4s} {aa3.get(sequences[chain][index], "UNK"):>3s} '
                     f'{chain_ids[chain]}{index+1:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}'
                     f'{occupancy:6.2f}{0.:6.2f}          {element:>2s}\n')
    raw = ''.join(lines)+'END\n'; digest = hashlib.sha256(raw.encode()).hexdigest()
    work = out/'p2rank'/sid; work.mkdir(parents=True, exist_ok=True)
    receptor = work/'receptor.pdb'; receptor.write_text(raw)
    prediction = work/'result/receptor.pdb_predictions.csv'
    if not prediction.exists():
        command = [str(JAVA), '-Xmx2048m', '-cp', str(PRANK.parent/'bin/p2rank.jar')+':'+str(PRANK.parent/'bin/lib/*'),
                   'cz.siret.prank.program.Main', 'predict', '-f', str(receptor), '-o', str(work/'result'),
                   '-c', 'alphafold', '-threads', '1', '-visualizations', '0']
        with (work/'P2RANK.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    frame = pd.read_csv(prediction); frame.columns = frame.columns.str.strip()
    candidates, rejected = [], []
    for row in frame.sort_values(['probability','score'], ascending=False).to_dict('records'):
        if float(row['probability']) < .2: continue
        try:
            keys = sorted({(reverse[token.split('_')[0]], int(token.split('_')[1])-1)
                           for token in str(row['residue_ids']).split()})
            if not 5 <= len(keys) <= 256: raise ValueError('residue_budget_no_truncation')
            region = region_record(atoms, sequences, keys)
            if len(region['atoms']) > 2048: raise ValueError('heavy_atom_budget_no_truncation')
            current = set(keys)
            if any(len(current & set(map(tuple, p['keys'])))/len(current | set(map(tuple, p['keys']))) >= .5
                   for p in candidates): continue
            candidates.append(dict(pocket_id=str(row['name']).strip(), probability=float(row['probability']),
                                   keys=keys, residues=len(keys), atoms=len(region['atoms'])))
        except ValueError as e:
            rejected.append(dict(pocket_id=str(row['name']).strip(), reason=str(e)))
    result = dict(system_id=sid, receptor_sha256=digest, source='experimental_p2rank',
                  ligand_used_for_pocket_selection=False, pockets=candidates, rejected=rejected,
                  seconds=time.monotonic()-started, prediction_sha256=sha(prediction))
    write_json(done, result)
    return result


def encode_all(records, out):
    import torch
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from scripts.prepare_biomaster_unified_interaction import load_pretrained, pretrained_batch, pretrained_tokens
    from scripts.prepare_biomaster_pocket_precision import encode_pocket, geometry_features
    from biomaster.structural_complex import mapped_receptor, region_record, minimum_distances
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    task, clip = load_pretrained()
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    began = time.monotonic(); rows = []
    with torch.inference_mode():
        for i, record in enumerate(records):
            sid = record['system_id']; dest = out/'encoded_index'/(sid+'.json')
            if dest.exists(): rows.append(json.loads(dest.read_text())); continue
            if shutil.disk_usage(out).free < 8*1024**3: raise RuntimeError('disk reserve below 8 GiB')
            source = BASE/'mapped'/(sid+'.pkl'); raw = source.read_bytes(); item = pickle.loads(raw)
            old = torch.load(BASE/record['file'], map_location='cpu', weights_only=False)
            if hashlib.sha256(raw).hexdigest() != old['source_sha256']: raise ValueError('mapped identity changed')
            prediction = json.loads((out/'prediction_index'/(sid+'.json')).read_text())
            sequence_states = {}
            for sequence in set(item['sequences'].values()):
                digest = hashlib.sha256(sequence.encode()).hexdigest()
                states = np.load(BASE/'esm2'/(digest+'.npy'), mmap_mode='r')
                if states.shape != (len(sequence),1280): raise ValueError('full chain ESM2 missing')
                sequence_states[digest] = states
            # All distinct full chains, residue-weighted. Identical homomer
            # copies do not multiply sequence context. No ligand-defined crop.
            total = sum(len(s) for s in sequence_states.values())
            target = sum(s.sum(0, dtype=np.float32) for s in sequence_states.values())/total
            h = pretrained_tokens(clip, pretrained_batch([item['ligand']], task.dictionary))[0]
            aligned = torch.nn.functional.normalize(clip.mol_project(h[None,0]), dim=-1)[0].cpu().numpy()
            if not np.allclose(aligned, old['drug_aligned'], atol=2e-5, rtol=2e-5):
                raise ValueError('molecule encoder/atom identity mismatch')
            # Match the task fingerprint's active-moiety/uncharge policy while
            # preserving the experimentally mapped atom/charge order locally.
            mol=Chem.MolFromSmiles(item['ligand']['smiles'])
            parent=max(Chem.GetMolFrags(mol,asMols=True,sanitizeFrags=True),key=lambda m:m.GetNumHeavyAtoms())
            parent=rdMolStandardize.Uncharger().uncharge(parent)
            fp = fpgen.GetFingerprintAsNumPy(parent).astype(np.float32)
            gp = out/'global'/(sid+'.npz')
            np.savez_compressed(gp, drug_global=np.concatenate([h[0].cpu().numpy(),fp]),
                drug_graph_mean=item['ligand']['atom_chemistry'].mean(0), pretrained_available=np.float32(1),
                target_global=target)
            sequences, atoms, _ = mapped_receptor(BASE.parent/'complexes_2020/systems'/sid)
            pockets, covered = [], set()
            native = item['regions'][0]
            positive_res = {
                (ch,int(j)) for ch,j,positive in zip(native['residue_chains'],native['residue_indices'],
                np.any(item['distance_labels'][0]<4.5,axis=0)) if positive}
            for p in prediction['pockets']:
                keys = list(map(tuple,p['keys'])); region = region_record(atoms,sequences,keys)
                tokens, aligned = encode_pocket(clip, task, region)
                rid = region['atom_residue']; nr = len(region['ca'])
                pooled = np.stack([tokens[rid==j].mean(0) for j in range(nr)]).astype(np.float16)
                residues = np.stack([sequence_states[digest][position] for digest,position in
                                    zip(region['sequence_hashes'],region['residue_indices'])])
                distance, geometry, quality = geometry_features(region)
                # Bound coordinates enter here, AFTER ligand-blind selection.
                truth = minimum_distances(item['ligand']['bound_coordinates'],region)
                pockets.append(dict(residue_tokens=residues,pocket_residue_tokens=pooled,
                    residue_distance=distance,residue_geometry=geometry,pocket_aligned=aligned,
                    pocket_metadata=np.array([p['probability'],quality.mean()],np.float32),distance_labels=truth))
                covered.update(set(keys)&positive_res)
            payload = {k:v for k,v in old.items() if k not in ['pockets','native_pocket_index']}
            payload.update(pockets=pockets,native_pocket_index=0,source='experimental_p2rank')
            pp = out/'predicted'/(sid+'.pt')
            tmp = pp.with_suffix('.tmp'); torch.save(payload,tmp); tmp.replace(pp)
            row = dict(**record, global_file=str(gp.relative_to(out)),predicted_file=str(pp.relative_to(out)),
                global_sha256=sha(gp),predicted_sha256=sha(pp),mapped_sha256=old['source_sha256'],
                predicted_pockets=len(pockets),positive_residues=len(positive_res),covered_positive_residues=len(covered),
                input_chains=len(sequences),distinct_sequence_chains=len(sequence_states),
                rejected_pockets=len(prediction['rejected']),
                prediction_index_sha256=sha(out/'prediction_index'/(sid+'.json')),
                receptor_sha256=prediction['receptor_sha256'],p2rank_predictions_sha256=prediction['prediction_sha256'])
            write_json(dest,row); rows.append(row)
            if (i+1)%25==0:
                write_json(out/'STATUS.json',dict(status='ENCODING_REAL_CONTEXT_AND_PREDICTED_POCKETS',
                    completed=len(rows),total=len(records),seconds=time.monotonic()-began,
                    utc=datetime.now(timezone.utc).isoformat()))
                print(json.dumps(dict(encoded=len(rows),seconds=round(time.monotonic()-began,1))),flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--limit',type=int,default=0,help='Engineering test only; cannot create a formal manifest')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    for directory in ['prediction_index','p2rank','encoded_index','global','predicted']:(out/directory).mkdir(exist_ok=True)
    settings=dict(base_manifest_sha256=sha(BASE/'MANIFEST.json'),producer_sha256=sha(Path(__file__)),
        p2rank_jar_sha256=sha(PRANK.parent/'bin/p2rank.jar'),p2rank_config_sha256=sha(PRANK.parent/'config/alphafold.groovy'),
        min_probability=.2,jaccard=.5,max_residues=256,max_atoms=2048,limit=args.limit)
    settings['dependencies']={str(path):sha(path) for path in [ROOT/'biomaster/structural_complex.py',
        ROOT/'biomaster/odti_local_features_v3.py',ROOT/'scripts/prepare_biomaster_unified_interaction.py',
        ROOT/'scripts/prepare_biomaster_pocket_precision.py',
        ROOT/'third_party/sota_dti_2026/Drug-The-Whole-Genome/data/model_weights/6_folds/fold_0.pt']}
    if (out/'IDENTITY.json').exists() and json.loads((out/'IDENTITY.json').read_text())!=settings:
        raise ValueError('preparation source/config changed; use a new directory')
    write_json(out/'IDENTITY.json',settings)
    base=json.loads((BASE/'MANIFEST.json').read_text());records=base['records']
    if args.limit:records=records[:args.limit]
    start=time.monotonic()
    write_json(out/'STATUS.json',dict(status='PREDICTING_LIGAND_BLIND_POCKETS',total=len(records),completed=0))
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i,result in enumerate(pool.map(predict_record,[(r,str(out)) for r in records],chunksize=1)):
                if (i+1)%25==0 or i+1==len(records):
                    state=dict(status='PREDICTING_LIGAND_BLIND_POCKETS',completed=i+1,total=len(records),
                               seconds=time.monotonic()-start,utc=datetime.now(timezone.utc).isoformat())
                    write_json(out/'STATUS.json',state);print(json.dumps(state),flush=True)
        rows=encode_all(records,out)
        result=dict(status='ENGINEERING_SUBSET_COMPLETE' if args.limit else 'FULL_DATA_READY',
            base=str(BASE),identity=settings,records=rows,counts={s:sum(r['split']==s for r in rows) for s in ['train','validation']},
            real_global_context_required=True,global_sequence_pooling='residue-weighted distinct full receptor chains',
            pocket_selection='P2Rank alphafold configuration; probability >= .2; all diverse pockets; no ligand input',
            source_axes=['experimental_native','experimental_p2rank','alphafold_p2rank'],
            structural_supervision_receptor='experimental_holo',actual_apo_or_AF_contact_supervision=False,
            no_prediction_count=sum(r['predicted_pockets']==0 for r in rows),
            predicted_pockets=sum(r['predicted_pockets'] for r in rows),
            rejected_pockets=sum(r['rejected_pockets'] for r in rows),
            seconds=time.monotonic()-start,full_data=not bool(args.limit))
        write_json(out/'MANIFEST.json',result);write_json(out/'STATUS.json',{k:v for k,v in result.items() if k!='records'})
    except Exception as e:
        write_json(out/'STATUS.json',dict(status='PREPARATION_FAILED',error=type(e).__name__+': '+str(e),
                   seconds=time.monotonic()-start));raise


if __name__=='__main__':main()
