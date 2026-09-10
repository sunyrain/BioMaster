"""Validate and export the current live SPR review snapshot without changing baseline."""
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from biomaster.explorer_spr_reviews import BASELINE_PATH, REVIEW_DIR, load_spr_reviews

baseline = ROOT / BASELINE_PATH
with baseline.open() as handle:
    candidates = list(csv.DictReader(handle))
reviews = load_spr_reviews(ROOT, baseline, candidates)
rows = [reviews[r['pair_id']] for r in candidates if r['pair_id'] in reviews]
summary = {'updated_at': datetime.now(timezone.utc).isoformat(), 'baseline_sha256': hashlib.sha256(baseline.read_bytes()).hexdigest(),
    'reviewed': len(rows), 'pending': 384-len(rows), 'verdict_counts': dict(Counter(r['verdict'] for r in rows)),
    'all_complete': len(rows)==384, 'experimental_release': False}
output = ROOT / REVIEW_DIR
for name, value in [('SUMMARY.json', summary), ('REVIEWS_384.json', {**summary, 'reviews': rows})]:
    temp = output / (name+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(output / name)
fields = ['candidate_id', 'pair_id', 'modeled_entity_name', 'gene_symbol', 'origin_summary', 'recommended_disease', 'verdict', 'verdict_label', 'summary', 'support', 'concern', 'next_step', 'evidence_scope', 'source_notes', 'source_urls', 'search_log']
with (output/'REVIEWS_EXPORT.csv.tmp').open('w', encoding='utf-8-sig', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields);writer.writeheader()
    by_pair = {r['pair_id']: r for r in candidates}
    for review in rows:
        row = {**by_pair[review['pair_id']], **review}
        writer.writerow({k: json.dumps(row.get(k),ensure_ascii=False) if isinstance(row.get(k),(dict,list)) else row.get(k,'') for k in fields})
(output/'REVIEWS_EXPORT.csv.tmp').replace(output/'REVIEWS_EXPORT.csv')
print(json.dumps(summary, ensure_ascii=False))
