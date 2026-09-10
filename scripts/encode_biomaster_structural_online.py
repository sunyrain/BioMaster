#!/usr/bin/env python3
"""Encode fully mapped inputs while download/CPU mapping continue."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from biomaster.structural_encoding import StructuralEncoder
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import write_json
from prepare_biomaster_structural_training import OUT


def main():
    encoder = StructuralEncoder(OUT)
    completed = {p.stem for p in (OUT / 'encoded').glob('*.pt')}
    started = time.monotonic()
    while True:
        records = []
        for path in sorted((OUT / 'mapped').glob('*.json')):
            if path.stem not in completed:
                r = json.loads(path.read_text())
                if r['status'] == 'MAPPED':
                    records.append(r)
        for i, record in enumerate(records, 1):
            encoder.complex(record); completed.add(record['system_id'])
            if i % 20 == 0 or i == len(records):
                status = dict(status='ENCODING_MAPPED_INPUTS', completed=len(completed), pending_snapshot=len(records)-i,
                              training_admission='NOT_YET_ADMITTED; sequence and ligand overlap audit runs before training',
                              seconds=time.monotonic()-started,
                              producer=file_identity(ROOT / 'biomaster/structural_encoding.py'))
                write_json(OUT / 'ONLINE_ENCODING_STATUS.json', status); print(json.dumps(status), flush=True)
        manifest_path = OUT / 'MAPPING_MANIFEST.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            ids = {r['system_id'] for r in manifest['records'] if r['status'] == 'MAPPED'}
            if manifest['status'] == 'COMPLETE' and ids.issubset(completed):
                status.update(status='COMPLETE', mapped_encoded=len(ids))
                write_json(OUT / 'ONLINE_ENCODING_STATUS.json', status); return
        time.sleep(10)


if __name__ == '__main__':
    main()
