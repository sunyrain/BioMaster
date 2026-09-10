#!/usr/bin/env python3
"""Prepare audited query-first training indices; no encoder refit or test scoring."""
import hashlib
import json
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biomaster.best_model_training import RollingStage
from scripts.audit_biomaster_spr_candidates_20260908 import identity
from scripts.prepare_biomaster_pocket_precision import SOURCE, BASE, SUPPLEMENT, OUTPUT

OUT = ROOT / 'outputs/biomaster_training_preparation_20260908/context_pocket'
DATA = ROOT / 'outputs/biomaster_best_model_20260906/data'
STRUCTURAL = ROOT / 'outputs/biomaster_pocket_precision_20260906/structural_data/training_2020'
STRUCTURAL_CHECKPOINT = ROOT / 'outputs/biomaster_pocket_precision_20260906/structural_pretraining/cutoff_2020/seed_20260921/STRUCTURAL_PRETRAINED.pt'
CONFIG = ROOT / 'configs/biomaster_context_pocket_20260908.json'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text())
    cutoff, seed = config['cutoff'], config['seed']
    stage = RollingStage(DATA, SOURCE, cutoff)
    structural = json.loads((STRUCTURAL / 'MANIFEST.json').read_text())
    if structural['max_release_year'] > cutoff or not structural['project_overlap_audit_pass']:
        raise ValueError('structural time/overlap admission failed')
    feature_manifest = json.loads((OUTPUT / 'MANIFEST.json').read_text())
    if feature_manifest['status'] != 'COMPLETE' or feature_manifest['labels_used']:
        raise ValueError('invalid local feature source')
    required = set()
    for source in [BASE, SUPPLEMENT]:
        required.update(np.load(source / 'REQUIRED_MOLECULE_IDS.npy').tolist())
    if not set(stage.train.drug_feature_index).issubset(required):
        raise ValueError('training row lacks prepared molecular source')
    old_ids = np.flatnonzero(stage.train.is_project_old_drug.to_numpy(bool))
    general_ids = np.flatnonzero(~stage.train.is_project_old_drug.to_numpy(bool))
    np.savez_compressed(OUT / 'TRAINING_INDICES.npz', old_rows=old_ids, general_rows=general_ids,
                        known=stage.known, validation_known=stage.val_known, risk=stage.risk)
    # Full candidates are derived from the frozen registry, not chosen by score.
    rng = np.random.default_rng(seed)
    queries = [rng.permutation(q).tolist() for q in stage.positive_queries]
    schedule = []
    for d, t in zip_longest(*queries):
        for head, query in [(0, d), (1, t)]:
            if query is not None:
                schedule.append(dict(step=len(schedule), head=head, query=int(query),
                                     candidates=stage.nt if head == 0 else len(stage.old)))
    pd.DataFrame(schedule).to_csv(OUT / 'QUERY_CYCLE.csv', index=False)
    assay_path = DATA / f'ASSAY_CONTRASTS_{cutoff}.csv.gz'
    assay = pd.read_csv(assay_path)
    if assay.max_document_year.max() > cutoff:
        raise ValueError('future assay label')
    feature_ok = assay.drug_feature_index.isin(required)
    dropped = assay[~feature_ok].copy()
    usable = assay[feature_ok].copy()
    groups = usable.groupby('assay_group').binary_label.nunique()
    usable = usable[usable.assay_group.isin(groups[groups.eq(2)].index)]
    validation_pairs = pd.MultiIndex.from_frame(stage.val[['drug_feature_index', 'target_feature_index']])
    overlap = pd.MultiIndex.from_frame(usable[['drug_feature_index', 'target_feature_index']]).isin(validation_pairs)
    if overlap.any():
        raise ValueError('same-assay supervision overlaps validation relation')
    usable.to_csv(OUT / 'ASSAY_TRAIN.csv.gz', index=False)
    dropped.to_csv(OUT / 'ASSAY_MISSING_FEATURES.csv.gz', index=False)
    if usable.empty:
        raise ValueError('no usable measured assay contrasts')
    train_records = [r for r in structural['records'] if r['split'] == 'train']
    val_records = [r for r in structural['records'] if r['split'] == 'validation']
    if {r['cluster'] for r in train_records} & {r['cluster'] for r in val_records}:
        raise ValueError('structural cluster split overlap')
    by_cluster = {}
    for r in val_records:
        by_cluster.setdefault(r['cluster'], []).append(r)
    n = config['preflight']['structural_validation_complexes']
    chosen_clusters = rng.choice(sorted(by_cluster), min(n, len(by_cluster)), replace=False)
    probe = [by_cluster[c][int(rng.integers(len(by_cluster[c])))] for c in chosen_clusters]
    (OUT / 'STRUCTURAL_PROBE.json').write_text(json.dumps(probe, indent=2)+'\n')
    # Re-audit frozen prospective candidates against this newer training window.
    candidate_path = ROOT / 'outputs/retargetmap_spr512_ours_frozen_20260904/RETARGETMAP_SPR512_OURS_FROZEN_INTERNAL_MASTER_V2.csv'
    candidates = pd.read_csv(candidate_path).query("lab_row_type == 'BLINDED_CANDIDATE'")
    targets = pd.read_csv(SOURCE / 'TARGET_INDEX.csv.gz')
    mapped = candidates.merge(stage.old[['ligand_inchikey', 'drug_feature_index']], on='ligand_inchikey', how='left')
    mapped = mapped.merge(targets[['uniprot_accession', 'target_feature_index']], on='uniprot_accession', how='left')
    mapped.to_csv(OUT / 'FROZEN_SPR_CANDIDATE_MAPPING.csv', index=False)
    training_overlap = mapped.merge(stage.train[['drug_feature_index', 'target_feature_index']],
                                    on=['drug_feature_index', 'target_feature_index'])
    assay_overlap = mapped.merge(usable[['drug_feature_index', 'target_feature_index']].drop_duplicates(),
                                 on=['drug_feature_index', 'target_feature_index'])
    if len(training_overlap) or len(assay_overlap):
        raise ValueError('frozen prospective candidate has measured training supervision')
    selection = json.loads((DATA.parent / 'GLOBAL_PARENT_SELECTION.json').read_text())
    parent = next(r for r in selection['parents'] if r['cutoff'] == cutoff and r['seed'] == seed)
    if identity(Path(parent['checkpoint']['path'])) != parent['checkpoint']:
        raise ValueError('global initialization checkpoint identity mismatch')
    identities = [identity(p) for p in [CONFIG, DATA/f'roll_{cutoff}/TRAIN.csv.gz', DATA/f'roll_{cutoff}/VALIDATION.csv.gz',
                  DATA/f'roll_{cutoff}/RISK.npy', assay_path, SOURCE/'OLD_DRUG_INDEX.csv', SOURCE/'TARGET_INDEX.csv.gz',
                  OUTPUT/'MANIFEST.json', STRUCTURAL/'MANIFEST.json', STRUCTURAL_CHECKPOINT, candidate_path,
                  ROOT/'biomaster/context_pocket.py', Path(__file__)]]
    paths = {k: str(v) for k, v in dict(data=DATA, source=SOURCE, global_features=BASE,
             supplemental_features=SUPPLEMENT, pocket_features=OUTPUT, structural_data=STRUCTURAL,
             structural_checkpoint=STRUCTURAL_CHECKPOINT, parent_checkpoint=Path(parent['checkpoint']['path'])).items()}
    result = dict(status='DATA_PREPARED_PREFLIGHT_REQUIRED', utc=datetime.now(timezone.utc).isoformat(),
                  cutoff=cutoff, validation_years=[cutoff+1, cutoff+2], seed=seed, paths=paths,
                  training_rows=len(stage.train), old_observed_rows=len(old_ids), general_observed_rows=len(general_ids),
                  query_cycle_updates=len(schedule), d2t_queries=len(queries[0]), t2d_queries=len(queries[1]),
                  full_candidate_counts=dict(d2t=stage.nt, t2d=len(stage.old)),
                  assay_source_rows=len(assay), assay_consumed_rows=len(usable), assay_consumed_groups=usable.assay_group.nunique(),
                  assay_missing_molecular_sources=len(dropped), assay_rows_lost_with_single_class_groups=len(assay)-len(dropped)-len(usable),
                  structural_train=len(train_records), structural_validation=len(val_records), structural_probe=len(probe),
                  structural_sampling='uniform cluster, then uniform complex within cluster',
                  prospective_pairs_mapped=int(mapped[['drug_feature_index', 'target_feature_index']].notna().all(axis=1).sum()),
                  prospective_pairs_outside_current_registry=int(mapped.target_feature_index.isna().sum()),
                  prospective_supervised_training_overlap=0, prospective_labels_used=False,
                  validation_scope='retrospective development; public encoder chronology not certified',
                  quality_protocol='occupancy/pLDDT/P2Rank separate fields; legacy quality-product geometry channel replaced by valid-pair indicator',
                  unresolved_domain_shift='experimental ligand-cropped structural pockets versus AlphaFold/P2Rank deployment pockets; apo/predicted paired augmentation not prepared',
                  long_training_ready=False, model_selected=False, identities=identities,
                  prepared_artifacts=[identity(OUT / name) for name in ['TRAINING_INDICES.npz', 'QUERY_CYCLE.csv', 'ASSAY_TRAIN.csv.gz', 'STRUCTURAL_PROBE.json']])
    (OUT / 'MANIFEST.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['paths', 'identities', 'prepared_artifacts']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
