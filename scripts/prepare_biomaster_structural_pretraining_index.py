#!/usr/bin/env python3
"""Select real-complex pretraining candidates from pinned official PLINDER data.

Produces candidate metadata, NOT a training-ready contact dataset. Coordinate
mapping, stereochemistry, project overlap, and cluster audits remain explicit.
"""
import hashlib
import json
from pathlib import Path
import re
import sys

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import write_json

OUT = ROOT / 'outputs/biomaster_pocket_precision_20260906/structural_data'
PREFIX = 'https://storage.googleapis.com/plinder/2024-06/v2/'


def download(relative, name):
    path = OUT / name
    source = OUT / (name + '.source.json')
    if not path.exists():
        tmp = path.with_suffix('.part')
        with requests.get(PREFIX + relative, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            with tmp.open('wb') as handle:
                for block in response.iter_content(4 * 1024**2):
                    handle.write(block)
        tmp.replace(path)
    identity = file_identity(path)
    if source.exists() and json.loads(source.read_text())['identity'] != identity:
        raise ValueError('pinned structural source changed')
    write_json(source, dict(url=PREFIX + relative, release='2024-06', iteration='v2', identity=identity))
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    annotation = download('index/annotation_table.parquet', 'annotation_table.parquet')
    split_path = download('splits/split.parquet', 'split.parquet')
    cols = ['system_id', 'entry_release_date', 'entry_resolution', 'system_pocket_UniProt',
            'system_proper_num_protein_chains', 'system_proper_num_ligand_chains',
            'system_proper_num_covalent_ligands', 'ligand_rdkit_canonical_smiles', 'ligand_num_heavy_atoms',
            'ligand_is_covalent', 'ligand_is_artifact', 'ligand_is_invalid', 'ligand_matches_smiles_atom_num',
            'system_ligand_validation_average_rscc', 'system_ligand_validation_average_occupancy',
            'system_ligand_validation_num_unresolved_heavy_atoms', 'uniqueness']
    data = pd.read_parquet(annotation, columns=cols)
    splits = pd.read_parquet(split_path)
    if splits.system_id.duplicated().any():
        raise ValueError('ambiguous official split assignment')
    data = data.merge(splits[['system_id', 'split', 'cluster', 'cluster_for_val_split',
                              'system_pass_validation_criteria', 'system_pass_statistics_criteria']],
                      on='system_id', how='left', validate='many_to_one')
    data['release_date'] = pd.to_datetime(data.entry_release_date, errors='coerce')
    filters = [
        ('official_train_split', data.split.eq('train')),
        ('official_quality', data.system_pass_validation_criteria.eq(True)),
        ('official_statistics', data.system_pass_statistics_criteria.eq(True)),
        ('known_release_date', data.release_date.notna()),
        ('resolution_at_most_2_5A', data.entry_resolution.gt(0) & data.entry_resolution.le(2.5)),
        ('noncovalent', data.ligand_is_covalent.eq(False) & data.system_proper_num_covalent_ligands.eq(0)),
        ('nonartifact_valid_ligand', data.ligand_is_artifact.eq(False) & data.ligand_is_invalid.eq(False)),
        ('resolved_atom_count_matches', data.ligand_matches_smiles_atom_num.eq(True)),
        ('one_proper_ligand', data.system_proper_num_ligand_chains.eq(1)),
        ('small_molecule_5_to_128_heavy_atoms', data.ligand_num_heavy_atoms.between(5, 128)),
        ('density_rscc_at_least_0_8', data.system_ligand_validation_average_rscc.ge(0.8)),
        ('occupancy_at_least_0_9', data.system_ligand_validation_average_occupancy.ge(0.9)),
        ('no_unresolved_ligand_heavy_atoms', data.system_ligand_validation_num_unresolved_heavy_atoms.eq(0)),
    ]
    keep = pd.Series(True, index=data.index)
    audit = []
    for name, mask in filters:
        keep &= mask.fillna(False)
        audit.append(dict(filter=name, remaining_ligand_rows=int(keep.sum()), remaining_systems=int(data.loc[keep, 'system_id'].nunique())))
    candidate = data.loc[keep].copy()
    # These are candidate-overlap flags, never a claim that sequence/pocket/
    # scaffold similarity to project evaluation has been removed.
    source = ROOT / 'outputs/biomaster_v3_kirhub_temporal_20260906/temporal'
    target = pd.read_csv(source / 'TARGET_INDEX.csv.gz')
    accession_columns = [c for c in target.columns if 'accession' in c]
    accessions = set()
    for col in accession_columns:
        for value in target[col].dropna().astype(str):
            accessions.update(re.split(r'[;,| ]+', value))
    candidate['project_exact_uniprot_overlap'] = candidate.system_pocket_UniProt.fillna('').apply(
        lambda x: bool(set(re.split(r'[;,| ]+', x)) & accessions))
    counts = {}
    for cutoff in [2018, 2020, 2022]:
        part = candidate[candidate.release_date.dt.year.le(cutoff)].copy()
        # Keep deterministic representatives of the release's redundancy groups.
        part = part.sort_values(['release_date', 'system_id']).drop_duplicates('system_id')
        part = part.drop_duplicates('uniqueness')
        part['training_admission'] = 'PENDING_COORDINATE_AND_PROJECT_OVERLAP_AUDIT'
        part['structure_archive'] = part.system_id.str[1:3].apply(lambda code: PREFIX + 'systems/' + code + '.zip')
        part.to_parquet(OUT / f'CANDIDATE_INDEX_LE_{cutoff}.parquet', index=False)
        counts[str(cutoff)] = dict(systems=len(part), protein_clusters=int(part.cluster.nunique()),
                                   project_exact_uniprot_overlap=int(part.project_exact_uniprot_overlap.sum()),
                                   multiple_protein_chains=int(part.system_proper_num_protein_chains.gt(1).sum()))
    manifest = dict(status='CANDIDATE_INDEX_COMPLETE_NOT_TRAINING_READY', official_release='2024-06/v2',
                    annotation_rows=len(data), official_split_systems=len(splits), official_split_counts=splits.split.value_counts().to_dict(),
                    quality_filter_cascade=audit, candidate_counts=counts, producer=file_identity(Path(__file__)),
                    annotation=file_identity(annotation), split=file_identity(split_path),
                    project_accession_columns=accession_columns,
                    pending=['download selected system structures', 'experimental ligand atom/bond/stereo identity mapping',
                             'protein residue mapping and alternate conformer policy', 'project pair/pocket/sequence/scaffold overlap audit',
                             'encode paired molecules/pockets and map observed contact labels', 'structural pretraining'],
                    structure_files_downloaded=False, structural_training_performed=False)
    write_json(OUT / 'MANIFEST.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
