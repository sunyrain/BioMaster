import csv
import hashlib
import json
from pathlib import Path

import pytest
from biomaster.explorer_comprehensive_spr import DESIGN, REGISTRY, load_comprehensive
from biomaster.explorer_experiments import load_experiments

ROOT = Path(__file__).resolve().parents[1]


def test_real_snapshot_origins_and_separate_controls():
    if not (ROOT / DESIGN).exists():
        pytest.skip('Comprehensive snapshot not installed')
    registry = list(csv.DictReader((ROOT / REGISTRY).open()))
    drugs = {r['ligand_inchikey']: {} for r in registry}
    targets = {r['target_chembl_id']: {} for r in csv.DictReader((ROOT / DESIGN / 'TARGET_ROSTER.csv').open())}
    result = load_comprehensive(ROOT, drugs, targets)
    rows = [r for t in result['targets'].values() for r in t['experiments']]
    candidates = [r for r in rows if not r['is_control']]
    assert len(candidates) == 384 and len(rows) == 496
    assert sum(r['cross_area'] for r in candidates) == 195
    assert all(r['original_indications'] and r['original_indication_sources'] and r['proposed_indication'] for r in candidates)
    original = {r['ligand_inchikey']: r for r in registry}
    for r in candidates:
        assert r['original_target_ids'] == original[r['drug_id']]['known_target_chembl_ids']
    first = next(r for r in rows if r['id'] == 'C384-001')
    assert first['drug_name'] == 'tecovirimat' and first['gene_symbol'] == 'AR'
    assert first['proposed_indication'] == 'HER2 positive breast carcinoma'
    assert all(r['cross_area'] is None and not r['original_indication_sources'] for r in rows if r['is_control'])
    assert result['counts']['spr_not_released_pairs'] == 496
    json.dumps(result, allow_nan=False)


def test_partial_new_snapshot_never_silently_falls_back(tmp_path):
    (tmp_path / DESIGN).mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        load_experiments(tmp_path, {}, {})


def test_snapshot_tampering_rejected(tmp_path):
    dest = tmp_path / DESIGN
    dest.mkdir(parents=True)
    (dest / 'VALIDATION.json').write_text('{"all_pass":true,"checks":{"unique":true}}')
    (dest / 'MANIFEST.json').write_text(json.dumps({'output_sha256': {'RECOMMENDED_CANDIDATES_384.csv': hashlib.sha256(b'original').hexdigest()}}))
    (dest / 'RECOMMENDED_CANDIDATES_384.csv').write_text('changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_comprehensive(tmp_path, {}, {})
