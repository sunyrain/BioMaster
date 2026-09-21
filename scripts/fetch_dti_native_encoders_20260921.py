#!/usr/bin/env python3
"""Download pinned base encoders required by native official task checkpoints."""
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
import requests
import download_dti_official_weights_20260921 as transfer


def main():
    root = transfer.ROOT
    transfer.OUT = root / 'outputs/dti_official_720x745_20260921/encoder_download'
    transfer.STORE = root / 'data/research/dti_native_encoders_20260921'
    transfer.OUT.mkdir(exist_ok=True)
    sources = json.loads((transfer.STORE / 'ENCODER_SOURCES.json').read_text())
    rows = []
    for spec in sources:
        repo, rev = spec['repo_id'], spec['revision']
        response = requests.get(f'https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true', timeout=60)
        response.raise_for_status()
        meta = response.json()
        has_safe = any(f['rfilename'] == 'model.safetensors' for f in meta['siblings'])
        for f in meta['siblings']:
            name = f['rfilename']
            if not (name.endswith(('.json', '.txt')) or name == ('model.safetensors' if has_safe else 'pytorch_model.bin')):
                continue
            lfs = f.get('lfs', {})
            rows.append(dict(model=repo.split('/')[-1], name=name, bytes=f['size'], revision=rev,
                             url=f'https://huggingface.co/{repo}/resolve/{rev}/{quote(name)}?download=true',
                             expected_sha256=lfs.get('sha256'), expected_git_blob=f.get('blobId') if not lfs else None,
                             destination=str((transfer.STORE / repo.split('/')[-1] / name).relative_to(root))))
    transfer.save(transfer.OUT / 'MANIFEST.json', dict(files=rows))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(transfer.download, rows))
    transfer.save(transfer.OUT / 'VERIFIED_FILES.json', results)
    assert all(r['status'] == 'VERIFIED' for r in results)


if __name__ == '__main__':
    main()
