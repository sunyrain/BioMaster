#!/usr/bin/env python3
"""Rehash existing official task assets and link them into the new collection."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dti_official_weights_download_20260921'
STORE = ROOT / 'data/research/dti_official_weights_20260921/existing'
FAMILIES = {'ConPLex', 'DrugCLIP', 'ProbeMatchDTI', 'DTBind', 'nesso', 'EviDTI', 'ScopeDTI_light'}


def main():
    source = ROOT / 'outputs/dti_resource_catalog_20260920/WEIGHT_LIST.csv'
    rows = list(csv.DictReader(source.open(encoding='utf-8-sig')))
    result = []
    for row in rows:
        if row['模型系列'] not in FAMILIES or row['资产类型'] not in ['任务检查点', '模型配置', '预训练编码器']:
            continue
        path = ROOT / row['本地位置']
        assert path.is_file(), path
        with path.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        assert digest == row['SHA256'], row['资产编号']
        link = STORE / row['模型系列'] / row['资产编号']
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(os.path.relpath(path, link.parent))
        assert link.resolve() == path.resolve()
        result.append(dict(model=row['模型系列'], asset_id=row['资产编号'], role=row['资产类型'],
            source_path=str(path.relative_to(ROOT)), collection_link=str(link.relative_to(ROOT)),
            bytes=path.stat().st_size, sha256=digest, official_source=row['官方获取地址'],
            status='EXISTING_FILE_HASH_VERIFIED', inference_qualification='UNCHANGED_FROM_PRIOR_REGISTRY'))
    with (OUT / 'EXISTING_OFFICIAL_ASSETS.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result[0].keys()); writer.writeheader(); writer.writerows(result)
    summary = dict(checked_utc=datetime.now(timezone.utc).isoformat(), status='PASS',
        families=sorted(FAMILIES), asset_files=len(result),
        bytes_reused=sum(x['bytes'] for x in result), duplicated_weight_bytes=0)
    (OUT / 'EXISTING_AUDIT.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
