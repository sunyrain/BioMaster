#!/usr/bin/env python3
"""Fetch selected PLINDER members via HTTP ranges, without full archive copies."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import fsspec
import pandas as pd
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import write_json

DATA = ROOT / 'outputs/biomaster_pocket_precision_20260906/structural_data'
OUT = DATA / 'complexes_2020'


def chemistry_key(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('invalid chemical identity')
    return Chem.MolToInchiKey(Chem.RemoveHs(mol)).split('-')[0], mol


def selection():
    RDLogger.DisableLog('rdApp.*')
    source = DATA / 'CANDIDATE_INDEX_LE_2020.parquet'
    frame = pd.read_parquet(source)
    old = pd.read_csv(ROOT / 'outputs/biomaster_v3_kirhub_temporal_20260906/temporal/OLD_DRUG_INDEX.csv')
    keys = {chemistry_key(s)[0] for s in old.model_ligand_smiles}
    records, reasons = [], Counter()
    for r in frame.to_dict('records'):
        if r['project_exact_uniprot_overlap']:
            reasons['project_target_uniprot_overlap'] += 1; continue
        try:
            key, mol = chemistry_key(r['ligand_rdkit_canonical_smiles'])
        except ValueError:
            reasons['invalid_ligand_smiles'] += 1; continue
        if key in keys:
            reasons['project_drug_connectivity_overlap'] += 1; continue
        if any(a.GetAtomicNum() not in {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53} for a in mol.GetAtoms()):
            reasons['metal_or_unsupported_element_special_system'] += 1; continue
        if len(Chem.GetMolFrags(mol)) != 1:
            reasons['multiple_disconnected_ligand_fragments'] += 1; continue
        r['ligand_connectivity_key'] = key
        records.append(r)
    selected = pd.DataFrame(records)
    # Cluster sizes are extremely skewed (one component contains > half the
    # records). Select whole clusters deterministically with a 10% size budget;
    # no contact labels or model outputs enter this metadata-only partition.
    counts = selected.cluster_for_val_split.value_counts().to_dict()
    budget = round(0.10 * len(selected)); heldout, used = set(), 0
    for cluster in sorted(counts, key=lambda c: hashlib.sha256(('20260906:' + c).encode()).hexdigest()):
        if used + counts[cluster] <= budget:
            heldout.add(cluster); used += counts[cluster]
    selected['internal_split'] = selected.cluster_for_val_split.map(lambda c: 'validation' if c in heldout else 'train')
    path = OUT / 'SELECTED.parquet'
    if path.exists():
        previous = pd.read_parquet(path)
        if previous.system_id.tolist() != selected.system_id.tolist():
            raise ValueError('download selection changed')
    selected.to_parquet(path, index=False)
    write_json(OUT / 'SELECTION.json', dict(status='FROZEN_DOWNLOAD_SELECTION', source=file_identity(source),
               selected=len(selected), exclusions=dict(reasons), splits=selected.internal_split.value_counts().to_dict(),
               cluster_partition='whole clusters ordered by sha256(20260906:cluster); fill a 10% system budget without splitting clusters',
               partition_amendment='metadata size balance before any structural training or model evaluation; original hash-modulo gave 52% validation',
               project_drug_exclusion='InChIKey connectivity block, conservative across stereoisomers',
               project_target_exclusion='exact UniProt now; sequence search after fetching chains',
               atom_types='organic small molecules only; metal/special systems excluded explicitly',
               max_release_year=2020, downstream_test_labels_used=False, producer=file_identity(Path(__file__))))
    return selected


def fetch_group(url, records):
    results = []
    with fsspec.open(url, mode='rb', block_size=512*1024, cache_type='blockcache') as stream:
        with zipfile.ZipFile(stream) as archive:
            by_system = {}
            for info in archive.infolist():
                parts = Path(info.filename).parts
                if len(parts) < 2 or '..' in parts or info.is_dir():
                    continue
                if parts[-1] in ['receptor.cif', 'sequences.fasta', 'chain_mapping.json'] or (len(parts) == 3 and parts[1] == 'ligand_files' and parts[-1].endswith('.sdf')):
                    by_system.setdefault(parts[0], []).append(info)
            for record in records:
                sid = record['system_id']; folder = OUT / 'systems' / sid
                manifest_path = folder / 'DOWNLOAD.json'
                if manifest_path.exists():
                    m = json.loads(manifest_path.read_text())
                    if all((folder / x['name']).is_file() and (folder / x['name']).stat().st_size == x['bytes'] for x in m['files']):
                        results.append(dict(system_id=sid, status='CACHED', bytes=sum(x['bytes'] for x in m['files']))); continue
                members = by_system.get(sid, [])
                names = {Path(i.filename).name for i in members}
                if not {'receptor.cif', 'sequences.fasta'}.issubset(names) or not any(n.endswith('.sdf') for n in names):
                    results.append(dict(system_id=sid, status='FAILED', reason='required_archive_members_missing')); continue
                files = []
                for info in members:
                    target = OUT / 'systems' / info.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    content = archive.read(info)  # zipfile checks member CRC
                    tmp = target.with_suffix(target.suffix + '.tmp'); tmp.write_bytes(content); tmp.replace(target)
                    files.append(dict(name=str(target.relative_to(folder)), bytes=len(content),
                                      sha256=hashlib.sha256(content).hexdigest(), crc32=info.CRC))
                write_json(manifest_path, dict(system_id=sid, archive=url, files=files))
                results.append(dict(system_id=sid, status='DOWNLOADED', bytes=sum(x['bytes'] for x in files)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=64)
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    frame = selection(); start = time.monotonic(); results = []
    groups = [(url, part.to_dict('records')) for url, part in frame.groupby('structure_archive', sort=True)]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(fetch_group, url, records): (url, records) for url, records in groups}
        for i, future in enumerate(as_completed(jobs), 1):
            url, records = jobs[future]
            try:
                results.extend(future.result())
            except Exception as error:
                results.extend(dict(system_id=r['system_id'], status='FAILED', reason=repr(error)[:300]) for r in records)
            event = dict(status='DOWNLOADING', groups_completed=i, groups=len(groups), records=len(results), expected=len(frame),
                         counts=dict(Counter(r['status'] for r in results)), seconds=round(time.monotonic()-start, 1))
            write_json(OUT / 'STATUS.json', event)
            print(json.dumps(event), flush=True)
    pd.DataFrame(results).to_csv(OUT / 'DOWNLOAD_RESULTS.csv', index=False)
    failed = sum(r['status'] == 'FAILED' for r in results)
    event.update(status='COMPLETE' if not failed else 'INCOMPLETE_RETRY_REQUIRED', failed=failed)
    write_json(OUT / 'STATUS.json', event)


if __name__ == '__main__':
    main()
