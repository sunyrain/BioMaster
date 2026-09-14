"""Read-only source audit for pair-JEPA planning; never opens held-out labels."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biomaster.training_dataset_v1 import molecule_identity

OUT = ROOT / 'outputs/biomaster_pm_jepa_design_20260914'
DATA = ROOT / 'data/processed/biomaster_training_full_20260910_v1'
BASE = ROOT / 'outputs/biomaster_endpoint_ablation_20260911'
OLD = ROOT / 'outputs/old_drug_target_sota_v1/feature_store_v1'
GRAPH = ROOT / 'outputs/biomaster_odti_experimental_template_graph_features_v1'
SOURCES = {}


def track(path):
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    s = p.stat()
    SOURCES[str(p.relative_to(ROOT))] = {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}
    return p


def csv(path, **kw):
    return pd.read_csv(track(path), **kw)


def pq(path, **kw):
    return pd.read_parquet(track(path), **kw)


def arr(path):
    return np.load(track(path), mmap_mode='r')


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def truth(series):
    return series.fillna(False).astype(str).str.lower().isin(['true', '1', '1.0'])


def existing_path(value):
    if not isinstance(value, str) or not value:
        return False
    p = Path(value)
    return (p if p.is_absolute() else ROOT / p).is_file()


def identities(smiles):
    result = {}
    for s in pd.Series(smiles).dropna().unique():
        record, _ = molecule_identity(s)
        result[s] = record['molecule_id'] if record else None
    return result


def audit_structures():
    base = ROOT / 'outputs/biomaster_pocket_precision_20260906/structural_data/training_2020'
    ctx = ROOT / 'outputs/biomaster_context_full_20260908/data'
    manifest = json.loads(track(base / 'MANIFEST.json').read_text())
    cm = json.loads(track(ctx / 'MANIFEST.json').read_text())
    admitted = pq(base / 'ADMITTED.parquet')
    result = {'source': 'PLINDER official 2024-06/v2; existing project <=2020 structural branch', 'splits': {}}
    for split, group in admitted.groupby('split'):
        records = [r for r in manifest['records'] if r['split'] == split]
        cr = [r for r in cm['records'] if r['split'] == split]
        result['splits'][split] = {'systems': len(group), 'clusters': int(group.cluster.nunique()), 'largest_cluster_systems': int(group.cluster.value_counts().max()), 'mapped_pickle_present': sum((base / 'mapped' / f'{s}.pkl').is_file() for s in group.system_id), 'native_encoded_present': sum(Path(r['identity']['path']).is_file() for r in records), 'global_context_present': sum((ctx / r['global_file']).is_file() for r in cr), 'predicted_encoded_present': sum((ctx / r['predicted_file']).is_file() for r in cr)}
    w = track('outputs/biomaster_pocket_precision_20260906/structural_pretraining/cutoff_2020/seed_20260921/STRUCTURAL_PRETRAINED.pt')
    result['checkpoint'] = {'path': str(w.relative_to(ROOT)), 'exists': w.exists(), 'bytes': w.stat().st_size}
    result['caveats'] = ['Native and predicted encoded feature caches were cleaned; regenerate before training.', 'Mapped experimental contact/distance data and pooled inputs retained.', 'Original structural train split excluded old project identities; overlap with new A/B split is not certified.', 'Structural pretrained readout is not validated as a stronger affinity teacher.']
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / 'EXPERIMENTAL_STRUCTURE_ASSETS.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    molecules = pq(DATA / 'MOLECULES.parquet', columns=['drug_feature_index', 'molecule_id', 'smiles'])
    targets = pq(DATA / 'TARGETS.parquet', columns=['target_feature_index', 'target_id', 'sequence_sha256', 'sequence_length'])
    arms = {a: pq(BASE / f'{a}_TRAIN.parquet') for a in ['kdki_inactive', 'all_inactive']}
    current_by_smiles = dict(zip(molecules.smiles, molecules.drug_feature_index))
    old_t = csv(OLD / 'TARGET_FEATURE_INDEX_V1.csv.gz')
    extra_t = csv('outputs/biomaster_bindingdb_affinity_feature_package_v1/BINDINGDB_NEW_TARGET_FEATURE_INDEX_V1.csv.gz')
    seq_by_old = dict(zip(old_t.target_feature_index, old_t.protein_sequence))
    seq_by_old.update(zip(extra_t.target_feature_index, extra_t.protein_sequence))
    hash_to_id = dict(zip(targets.sequence_sha256, targets.target_feature_index))
    token_i = csv('outputs/biomaster_bindingdb_target_token_feature_package_v1/ESM2_650M_RESIDUE_INDEX_COMBINED_V1.csv.gz')
    token_a = arr('outputs/biomaster_bindingdb_target_token_feature_package_v1/ESM2_650M_RESIDUE_FLOAT16_COMBINED_V1.npy')
    token_ids = {hash_to_id[sha(seq_by_old[i])] for i in token_i.target_feature_index if i in seq_by_old and sha(seq_by_old[i]) in hash_to_id}
    assert (token_i.token_offset + token_i.token_length).max() == token_a.shape[0]
    pocket = csv(GRAPH / 'POCKET_GRAPH_INDEX_V1.csv.gz')
    pocket_ids = {hash_to_id[sha(seq_by_old[i])] for i in pocket.loc[truth(pocket.graph_available), 'target_feature_index'] if sha(seq_by_old[i]) in hash_to_id}
    pocket_pass_ids = {hash_to_id[sha(seq_by_old[i])] for i in pocket.loc[truth(pocket.graph_available) & pocket.reference_redocking_status.eq('PASS'), 'target_feature_index'] if sha(seq_by_old[i]) in hash_to_id}
    old_d = csv(OLD / 'DRUG_FEATURE_INDEX_V1.csv.gz')
    ligand = csv(GRAPH / 'LIGAND_GRAPH_INDEX_V1.csv.gz')
    good_ligands = old_d.loc[old_d.drug_feature_index.isin(ligand.loc[truth(ligand.graph_available), 'drug_feature_index'])]
    ligand_ids = {current_by_smiles[s] for s in good_ligands.model_ligand_smiles if s in current_by_smiles}
    for name in ['LIGAND_ATOM_FEATURES_FLOAT16_V1.npy', 'POCKET_ESM2_RESIDUE_FLOAT16_V1.npy', 'POCKET_RESIDUE_AUX_FLOAT16_V1.npy', 'POCKET_CA_COORD_FLOAT32_V1.npy']:
        arr(GRAPH / name)

    ready = arr(BASE / 'features/AVAILABLE.npy').astype(bool)
    b_done = arr('outputs/biomaster_dtiam_ab_20260912/features/BERMOL_DONE.npy')
    p_done = arr('outputs/biomaster_dtiam_ab_20260912/features/ESM2_DONE.npy')
    coverage = []
    regressions = {}
    for a, df in arms.items():
        reg = pq(f'outputs/biomaster_endpoint_multitask_20260911/{a}_TRAIN_REGRESSION.parquet')
        ep = reg.groupby('pair_id').endpoint.nunique()
        assay = pq(f'outputs/biomaster_assay_aware_20260911/{a}_ASSAY_TRAIN.parquet')
        assert reg.pair_id.isin(df.pair_id).all()
        regressions[a] = {'rows': len(reg), 'unique_pairs': reg.pair_id.nunique(), 'endpoint_counts': reg.endpoint.value_counts().to_dict(), 'pairs_with_at_least_2_endpoints': int(ep.ge(2).sum()), 'assay_rows': len(assay), 'assay_groups': assay.assay_group.nunique()}
        d = df.drug_feature_index.to_numpy(); t = df.target_feature_index.to_numpy()
        row = dict(arm=a, pairs=len(df), molecules=df.molecule_id.nunique(), targets=df.target_id.nunique(), positives=int(df.binary_label.sum()), explicit_inactive=int(df.explicit_inactive.sum()), basic_embeddings_complete_pairs=int((b_done[d] & p_done[t]).sum()), drugclip_available_pairs=int(ready[d].sum()), esm2_truncated_pairs=int(targets.set_index('target_feature_index').sequence_length.loc[t].gt(1022).sum()), residue_token_target_pairs=int(df.target_feature_index.isin(token_ids).sum()), template_pocket_target_pairs=int(df.target_feature_index.isin(pocket_ids).sum()), template_pocket_pass_target_pairs=int(df.target_feature_index.isin(pocket_pass_ids).sum()), existing_ligand_graph_pairs=int(df.drug_feature_index.isin(ligand_ids).sum()), existing_both_graph_pairs=int((df.target_feature_index.isin(pocket_ids) & df.drug_feature_index.isin(ligand_ids)).sum()))
        row['existing_both_graph_pass_pairs'] = int((df.target_feature_index.isin(pocket_pass_ids) & df.drug_feature_index.isin(ligand_ids)).sum())
        coverage.append(row)

    print('Training and token/graph overlap complete', flush=True)
    boltz = []
    boltz_pairs = set()
    protocol_hash = {}
    for p in (ROOT / 'outputs/strict_receptor_protocol_338_v1/targets').glob('*/protocol.json'):
        j = json.loads(p.read_text()); protocol_hash[j['target_chembl_id']] = j['canonical_sequence_sha256']
    boltz_paths = ['outputs/boltz2_calibration_338_v1/evaluation/BOLTZ2_CALIBRATION_PAIR_EVIDENCE_V1.csv.gz', 'outputs/boltz2_discovery_conditional_v1/evaluation/BOLTZ2_CONDITIONAL_DISCOVERY_EVIDENCE_V1.csv.gz']
    for path in boltz_paths:
        df = csv(path)
        ids = identities(df.canonicalSmiles)
        df['molecule_id'] = df.canonicalSmiles.map(ids)
        df['target_id'] = df.target_chembl_id.map(protocol_hash).map(lambda x: 'SEQ:' + x if isinstance(x, str) else None)
        df['mapped_pair_id'] = df.molecule_id + '__' + df.target_id
        available = truth(df.boltzCompleted) & df.boltzCifPath.map(existing_path)
        boltz_pairs.update(df.loc[available, 'mapped_pair_id'].dropna())
        row = dict(source=path, rows=len(df), completed=int(truth(df.boltzCompleted).sum()), cif_files_present=int(available.sum()), confidence_files_present=int(df.boltzConfidencePath.map(existing_path).sum()), affinity_files_present=int(df.boltzAffinityPath.map(existing_path).sum()), unique_project_pair_ids=int(df.loc[available,'pairId'].nunique()), mapped_pairs_current_registry=int(df.loc[available & df.molecule_id.isin(molecules.molecule_id) & df.target_id.isin(targets.target_id),'mapped_pair_id'].nunique()))
        for a, adf in arms.items():
            row[a+'_overlap'] = df.loc[available & df.mapped_pair_id.isin(adf.pair_id), 'mapped_pair_id'].nunique()
        if 'boltz_evidence_tier' in df:
            row['historical_tier_counts'] = df.boltz_evidence_tier.value_counts().to_dict()
        boltz.append(row)
    print('Boltz paths and canonical-identity overlap complete', flush=True)

    cp = csv('data/external/lincs_cmap/compoundinfo_beta.txt', sep='\t')
    with h5py.File(track('data/external/lincs_cmap/level5_beta_trt_cp_n720216x12328.gctx'), 'r') as f:
        gctx_shape = list(f['0/DATA/0/matrix'].shape)
    lincs = dict(compound_metadata_rows=len(cp), unique_inchikeys=cp.inchi_key.nunique(), gctx_matrix_shape=gctx_shape, raw_bytes=track('data/external/lincs_cmap/level5_beta_trt_cp_n720216x12328.gctx').stat().st_size, match_policy='exact supplied full InChIKey; no connectivity/tautomer expansion; overlap of compound metadata, not QC-passed signatures')
    for a, df in arms.items():
        lincs[a+'_molecules'] = int(df.loc[df.molecule_id.isin(cp.inchi_key), 'molecule_id'].nunique())
        lincs[a+'_pairs_with_molecule_metadata'] = int(df.molecule_id.isin(cp.inchi_key).sum())
    hq = csv('outputs/recent_affinity_sources_20260910/hiqbind_sm_metadata.csv')
    hiq = dict(metadata_rows=len(hq), unique_pdbs=hq.PDBID.nunique(), endpoint_rows=hq['Binding Affinity Measurement'].value_counts().to_dict(), full_structure_bundle_local=False, exact_pair_training_overlap='not_certified: metadata has UniProt IDs but no construct sequences')
    cache_specs = {}
    for rel in ['outputs/biomaster_dtiam_ab_20260912/features/BERMOL.npy','outputs/biomaster_dtiam_ab_20260912/features/ESM2.npy','outputs/biomaster_endpoint_ablation_20260911/features/DRUG_CLIP.npy','outputs/biomaster_endpoint_ablation_20260911/features/GRAPH.npy','outputs/biomaster_endpoint_ablation_20260911/features/MORGAN.npy']:
        x=arr(rel); cache_specs[rel]={'shape':list(x.shape),'dtype':str(x.dtype),'bytes':x.nbytes}
    resources={}
    for name in ['cpu.max','memory.max','memory.current','cpuset.cpus.effective']:
        p=Path('/sys/fs/cgroup')/name
        resources[name]=p.read_text().strip() if p.exists() else None
    resources['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
    for p in [Path('/'),ROOT]:
        s=os.statvfs(p);resources[str(p)+'_free_bytes']=s.f_bavail*s.f_frsize
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),coverage=coverage,regression=regressions,boltz=boltz,lincs=lincs,hiqbind=hiq,cache_specs=cache_specs,resources=resources,token_index_rows=len(token_i),token_shape=list(token_a.shape),existing_template_pocket_graphs=int(truth(pocket.graph_available).sum()),existing_template_redock_pass_graphs=int((truth(pocket.graph_available)&pocket.reference_redocking_status.eq('PASS')).sum()),notes=['Counts refer to TRAIN only; no held-out labels loaded.', 'Old ligand-graph overlap uses exact canonical SMILES text and is a conservative count.', 'Boltz overlap matches curated molecule identity and protocol canonical protein SHA256; actual modeled fragment/mutations/pose quality still require audit.', 'Completed Boltz scores and CIF files do not establish a cached latent representation or experimental binding.', 'Template pockets concern reference ligands, not candidate drug poses.', 'LINCS signature count is not unique compounds, and drug-cell response is not pair-specific affinity.', 'Array allocation size is not the number of available entities.', 'This script performs an inventory, not training or a speed benchmark.'],source_files=SOURCES)
    result['boltz_union_training_overlap'] = {a: len(boltz_pairs.intersection(df.pair_id)) for a, df in arms.items()}
    result['storage_arithmetic'] = {'B_512d_fp16_latents_GiB': len(arms['all_inactive'])*512*2/1024**3, 'all_registry_full_length_1280d_fp16_residue_tokens_GiB': int(targets.sequence_length.sum())*1280*2/1024**3, 'note': 'Theoretical tensor bytes only, excluding indices, activations, optimizer and checkpoints; not a measured runtime.'}
    audit_structures()
    (OUT/'RESOURCE_MODALITY_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    pd.DataFrame(coverage).to_csv(OUT/'TRAIN_MODALITY_COVERAGE.csv',index=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_files','cache_specs']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    if '--structure-only' in sys.argv:
        print(json.dumps(audit_structures(), indent=2))
    else:
        main()
