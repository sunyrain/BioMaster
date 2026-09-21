#!/usr/bin/env python3
"""Fetch pinned native code, without copying large training data/checkpoints."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    audit = json.loads((ROOT / 'outputs/dti_final_model_matrix_20260920/NEW_MODEL_SOURCE_AUDIT.json').read_text())
    for name in ['MAMMAL', 'BALM', 'GraphBAN', 'CheMLT-F', 'ADME-DTI']:
        spec = audit[name]
        slug = spec['repo'].removeprefix('https://github.com/')
        rev = spec['revision']
        dest = ROOT / '.external' / name
        dest.mkdir(exist_ok=True)
        response = requests.get(f'https://api.github.com/repos/{slug}/git/trees/{rev}?recursive=1', timeout=90)
        response.raise_for_status()
        tree = response.json()
        assert not tree.get('truncated'), name
        files = [r for r in tree['tree'] if r['type'] == 'blob' and r.get('size', 0) < 3_000_000
                 and (Path(r['path']).suffix in ['.py', '.yaml', '.yml', '.toml', '.md', '.ipynb', '.cfg']
                      or Path(r['path']).name in ['requirements.txt', 'setup.py', 'LICENSE', 'MANIFEST.in'])]

        def fetch(row):
            path = dest / row['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_file():
                data = path.read_bytes()
                digest = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
                if digest == row['sha']:
                    return dict(path=row['path'], git_blob_sha1=digest)
            response = requests.get(f'https://raw.githubusercontent.com/{slug}/{rev}/{row["path"]}', timeout=90)
            response.raise_for_status()
            data = response.content
            digest = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
            assert digest == row['sha'], row['path']
            path.write_bytes(data)
            return dict(path=row['path'], git_blob_sha1=digest)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(fetch, files))
        (dest / 'SOURCE_PROVENANCE.json').write_text(json.dumps(dict(repo=spec['repo'], revision=rev, files=results), indent=2))
        print(name, len(results), 'source files verified', flush=True)


if __name__ == '__main__':
    main()
