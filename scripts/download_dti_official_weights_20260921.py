#!/usr/bin/env python3
"""Download pinned author task bundles onto the data disk, resume and verify.

Existing project weights are inventoried separately and reused, not duplicated.
This only obtains public files; it does not execute downloaded model code.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dti_official_weights_download_20260921'
STORE = ROOT / 'data/research/dti_official_weights_20260921'
LOCK = threading.Lock()
PROGRESS = {}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def progress(key, **fields):
    with LOCK:
        PROGRESS.setdefault(key, {}).update(fields)
        save(OUT / 'STATUS.json', {'updated_utc': stamp(), 'files': PROGRESS})


def digest(path, algorithm='sha256'):
    with path.open('rb') as f:
        return hashlib.file_digest(f, algorithm).hexdigest()


def make_manifest():
    remote = json.loads((ROOT / 'outputs/dti_final_model_matrix_20260920/TASK_WEIGHT_REMOTE_METADATA.json').read_text())
    audit = json.loads((ROOT / 'outputs/dti_final_model_matrix_20260920/NEW_MODEL_SOURCE_AUDIT.json').read_text())
    items = []
    for model in ['MAMMAL', 'BALM', 'CheMLT-F']:
        meta = remote[model]
        repo = meta['url'].split('/api/models/')[1].split('?')[0]
        for file in meta['files']:
            name = file['rfilename']
            if name == '.gitattributes':
                continue
            lfs = file.get('lfs', {})
            items.append(dict(model=model, name=name, bytes=file['size'], revision=meta['revision'],
                url=f'https://huggingface.co/{repo}/resolve/{meta["revision"]}/{quote(name)}?download=true',
                expected_sha256=lfs.get('sha256'), expected_git_blob=file.get('blobId') if not lfs else None,
                destination=str((STORE / model / name).relative_to(ROOT))))
    for file in remote['GraphBAN']['files']:
        items.append(dict(model='GraphBAN', name=file['key'], bytes=file['size'], revision='zenodo:14813233',
            url=file['links']['self'], expected_md5=file['checksum'].split(':')[1],
            destination=str((STORE / 'GraphBAN' / file['key']).relative_to(ROOT))))
    meta = audit['ADME-DTI']
    tree = json.loads((OUT / 'ADME_TREE.json').read_text())
    assert not tree.get('truncated')
    repo = meta['repo'].split('github.com/')[1]
    for file in tree['tree']:
        name = file['path']
        if file['type'] != 'blob' or not (name.startswith('saved_models/') or
            (file.get('size', 0) < 500000 and name.endswith(('.py', '.json', '.yaml', '.yml', '.md', '.txt'))
             and not name.startswith(('dataset', 'data/', 'embeddings/')))):
            continue
        # Resolve any Git LFS pointer before registering expected size/digest.
        url = f'https://raw.githubusercontent.com/{repo}/{meta["revision"]}/{quote(name)}'
        row = dict(model='ADME-DTI', name=name, bytes=file['size'], revision=meta['revision'], url=url,
                   expected_git_blob=file['sha'], destination=str((STORE / 'ADME-DTI' / name).relative_to(ROOT)))
        if name.startswith('saved_models/') and name.endswith(('.pt', '.pth', '.pkl', '.bin', '.safetensors')) and file['size'] < 1024:
            r = requests.get(url, timeout=30); r.raise_for_status()
            if r.text.startswith('version https://git-lfs.github.com/spec/v1'):
                lines = r.text.splitlines()
                row.update(url=f'https://media.githubusercontent.com/media/{repo}/{meta["revision"]}/{quote(name)}',
                    expected_sha256=next(x.split('sha256:')[1] for x in lines if x.startswith('oid ')),
                    bytes=int(next(x.split()[1] for x in lines if x.startswith('size '))), expected_git_blob=None)
        items.append(row)
    assert len({x['destination'] for x in items}) == len(items)
    save(OUT / 'DOWNLOAD_MANIFEST.json', dict(created_utc=stamp(), store=str(STORE.relative_to(ROOT)),
        scope='Pinned main task bundles plus all three GraphBAN case-study archives, both CheMLT-F task variants/encoders, and all ADME saved models. No ablation dump or new training.',
        files=items, total_bytes=sum(x['bytes'] for x in items)))
    return items


def verify(path, row):
    if path.stat().st_size != row['bytes']:
        raise ValueError('Size mismatch: ' + row['destination'])
    hashes = {'sha256': hashlib.sha256(), 'md5': hashlib.md5(), 'git_blob': hashlib.sha1()}
    hashes['git_blob'].update(f'blob {row["bytes"]}\0'.encode())
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            for h in hashes.values():
                h.update(chunk)
    values = {k: h.hexdigest() for k, h in hashes.items()}
    for key in ['sha256', 'md5', 'git_blob']:
        if row.get('expected_' + key) and row['expected_' + key] != values[key]:
            raise ValueError(key + ' mismatch: ' + row['destination'])
    return values


def ranged_transfer(row, path, part, key):
    """Bounded parallel HTTP ranges; persisted completion map supports restart."""
    bitmap = part.with_name(part.name + '.ranges.json')
    if bitmap.exists():
        state = json.loads(bitmap.read_text())
        assert state['bytes'] == row['bytes'] and state['url'] == row['url']
    else:
        prefix = part.stat().st_size if part.exists() else 0
        assert prefix <= row['bytes']
        state = dict(bytes=row['bytes'], url=row['url'], prefix=prefix, ranges=[
            dict(start=start, end=min(start + 64 * 1024**2, row['bytes']) - 1, complete=False)
            for start in range(prefix, row['bytes'], 64 * 1024**2)])
        save(bitmap, state)
    state_lock = threading.Lock()
    fd = os.open(part, os.O_CREAT | os.O_RDWR, 0o644)
    def fetch(segment):
        if segment['complete']:
            return
        for attempt in range(6):
            try:
                start, end = segment['start'], segment['end']
                url = row['url'] + ('&' if '?' in row['url'] else '?') + f'chunk={start}_{end}&attempt={attempt}'
                with requests.get(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
                                  stream=True, timeout=(30, 90)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{row["bytes"]}':
                        raise ValueError('Range response does not match requested segment')
                    pos = start
                    for chunk in response.iter_content(2 * 1024**2):
                        if pos + len(chunk) > end + 1:
                            raise ValueError('Range too long')
                        if chunk:
                            written = os.pwrite(fd, chunk, pos)
                            if written != len(chunk):
                                raise OSError('Short file write')
                            pos += written
                    if pos != end + 1:
                        raise ValueError('Incomplete range')
                os.fsync(fd)
                with state_lock:
                    segment['complete'] = True
                    save(bitmap, state)
                    n = state['prefix'] + sum(s['end']-s['start']+1 for s in state['ranges'] if s['complete'])
                    progress(key, status='DOWNLOADING_RANGES', downloaded_bytes=n)
                return
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(5 + attempt * 3)
    try:
        n = state['prefix'] + sum(s['end']-s['start']+1 for s in state['ranges'] if s['complete'])
        progress(key, status='DOWNLOADING_RANGES', expected_bytes=row['bytes'], downloaded_bytes=n)
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(fetch, state['ranges']))
    finally:
        os.close(fd)
    progress(key, status='VERIFYING', downloaded_bytes=row['bytes'])
    hashes = verify(part, row)
    part.replace(path)
    bitmap.unlink()
    return hashes


def download(row):
    key = row['model'] + '/' + row['name']
    path = ROOT / row['destination']
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    progress(key, status='VERIFYING' if path.exists() else 'QUEUED', expected_bytes=row['bytes'])
    if path.exists():
        hashes = verify(path, row)
        progress(key, status='VERIFIED', downloaded_bytes=row['bytes'], **hashes)
        return dict(row, status='VERIFIED', **hashes)
    if row['bytes'] > 128 * 1024**2:
        try:
            hashes = ranged_transfer(row, path, part, key)
            progress(key, status='VERIFIED', downloaded_bytes=row['bytes'], **hashes)
            print('VERIFIED', key, row['bytes'], flush=True)
            return dict(row, status='VERIFIED', **hashes)
        except Exception as e:
            progress(key, status='FAILED', error_type=type(e).__name__)
            return dict(row, status='FAILED', error_type=type(e).__name__)
    for attempt in range(6):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset < row['bytes']:
                if shutil.disk_usage(STORE).free < 3 * 1024**3:
                    raise RuntimeError('Data disk reserve below 3 GiB')
                headers = {'Range': f'bytes={offset}-', 'Accept-Encoding': 'identity'} if offset else {'Accept-Encoding': 'identity'}
                progress(key, status='DOWNLOADING', downloaded_bytes=offset, attempt=attempt + 1)
                # Cache-bust only retry requests; keep canonical source URL in manifest.
                url = row['url']
                if attempt:
                    url += ('&' if '?' in url else '?') + 'resume_at=' + str(int(time.time()))
                with requests.get(url, headers=headers, stream=True, timeout=(30, 90)) as response:
                    response.raise_for_status()
                    if offset and response.status_code != 206:
                        offset = 0
                    if response.status_code == 206 and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Unexpected Content-Range')
                    last = time.monotonic()
                    with part.open('ab' if offset else 'wb') as f:
                        for chunk in response.iter_content(4 * 1024**2):
                            if not chunk:
                                continue
                            f.write(chunk); offset += len(chunk)
                            if offset > row['bytes']:
                                raise ValueError('Response exceeds registered size')
                            if time.monotonic() - last > 15:
                                progress(key, downloaded_bytes=offset); last = time.monotonic()
            progress(key, status='VERIFYING', downloaded_bytes=part.stat().st_size)
            hashes = verify(part, row)
            part.replace(path)
            progress(key, status='VERIFIED', downloaded_bytes=row['bytes'], **hashes)
            print('VERIFIED', key, row['bytes'], flush=True)
            return dict(row, status='VERIFIED', **hashes)
        except Exception as e:
            # Do not log redirected signed URLs from HTTP exceptions.
            progress(key, status='RETRYING', error_type=type(e).__name__, attempt=attempt + 1)
            if isinstance(e, ValueError) and part.exists():
                # Keep damaged transfer for diagnosis; never overwrite a verified artifact.
                part.replace(part.with_name(part.name + '.invalid'))
            if attempt == 5:
                progress(key, status='FAILED', error_type=type(e).__name__)
                return dict(row, status='FAILED', error_type=type(e).__name__)
            time.sleep(min(5 * (attempt + 1), 25))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); STORE.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / 'DOWNLOAD_MANIFEST.json'
    items = json.loads(manifest_path.read_text())['files'] if manifest_path.exists() else make_manifest()
    def pending_bytes(row):
        path = ROOT / row['destination']
        if path.exists():
            return 0
        bitmap = path.with_name(path.name + '.part.ranges.json')
        if bitmap.exists():
            state = json.loads(bitmap.read_text())
            complete = state['prefix'] + sum(s['end']-s['start']+1 for s in state['ranges'] if s['complete'])
        else:
            part = path.with_name(path.name + '.part')
            complete = part.stat().st_size if part.exists() else 0
        return max(0, row['bytes']-complete)
    remaining = sum(pending_bytes(x) for x in items)
    free = shutil.disk_usage(STORE).free
    print(json.dumps(dict(files=len(items), total_GiB=sum(x['bytes'] for x in items)/2**30,
        remaining_GiB=remaining/2**30, free_GiB=free/2**30)), flush=True)
    if args.prepare_only:
        return
    if remaining + 3 * 1024**3 > free:
        raise RuntimeError('Insufficient data disk space for all registered artifacts plus reserve')
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, row) for row in items]
        for future in as_completed(futures):
            results.append(future.result())
            save(OUT / 'VERIFIED_FILES.json', sorted(results, key=lambda x: x['destination']))
    failed = [x for x in results if x['status'] != 'VERIFIED']
    save(OUT / 'DOWNLOAD_SUMMARY.json', dict(completed_utc=stamp(), status='PASS' if not failed else 'INCOMPLETE',
        registered_files=len(items), verified_files=len(results)-len(failed), failed_files=len(failed),
        verified_bytes=sum(x['bytes'] for x in results if x['status']=='VERIFIED'),
        new_training_started=False, model_forward_validation_completed=False))
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
