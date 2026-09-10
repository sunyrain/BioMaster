#!/usr/bin/env python3
"""Fetch not-yet-started tail members in staging; publish without overwriting.

The running original downloader and its selection are left untouched. Atomic
directory rename only succeeds while the original destination remains absent.
If the original worker has started the same system, its files are retained.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import errno
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import fetch_biomaster_structural_complexes as fetch
from biomaster.odti_pockets_v3 import file_identity
from prepare_biomaster_unified_interaction import write_json


def main():
    destination = fetch.OUT
    stage = destination.parent / 'tail_staging_20260906'
    stage.mkdir(exist_ok=True)
    frame = pd.read_parquet(destination / 'SELECTED.parquet')
    selected = [r for r in frame.to_dict('records') if not (destination / 'systems' / r['system_id']).exists()]
    if len(selected) > 64:
        raise ValueError('tail assistant is limited to 64 unstarted systems')
    fetch.OUT = stage
    def one(record):
        sid = record['system_id']
        result = fetch.fetch_group(record['structure_archive'], [record])
        if len(result) != 1 or result[0]['status'] not in ['DOWNLOADED', 'CACHED']:
            raise ValueError(f'tail fetch failed: {result}')
        folder = stage / 'systems' / sid
        manifest = json.loads((folder / 'DOWNLOAD.json').read_text())
        for item in manifest['files']:
            if file_identity(folder / item['name'])['sha256'] != item['sha256']:
                raise ValueError('tail file content changed')
        try:
            folder.rename(destination / 'systems' / sid)
            return dict(system_id=sid, status='PUBLISHED_ATOMICALLY')
        except OSError as error:
            if error.errno not in [errno.EEXIST, errno.ENOTEMPTY]:
                raise
            return dict(system_id=sid, status='ORIGINAL_WORKER_STARTED_RETAINED_ORIGINAL')
    results = []; started = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(one, r) for r in reversed(selected)]):
            try:
                results.append(future.result())
            except Exception as error:
                results.append(dict(status='FAILED', error=repr(error)))
            write_json(stage / 'STATUS.json', dict(status='ASSISTING', selected=len(selected),
                       completed=len(results), results=results, seconds=time.monotonic()-started))
            print(json.dumps(results[-1]), flush=True)
    write_json(stage / 'RESULT.json', dict(status='COMPLETE' if all(r['status'] != 'FAILED' for r in results) else 'PARTIAL_FAILURE',
               utc=datetime.now(timezone.utc).isoformat(), selected=len(selected), results=results,
               selection=file_identity(destination / 'SELECTED.parquet'), producer=file_identity(Path(__file__)),
               seconds=time.monotonic()-started, original_sources_or_selection_modified=False))


if __name__ == '__main__':
    main()
