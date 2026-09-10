"""Build a read-only, reviewable outputs retirement plan; never delete files."""
from pathlib import Path
import csv
import gzip
import json
import os
import re
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/disk_reclamation_audit_20260910'


def main():
    draft = json.loads((OUT / 'ADDITIONAL_SELECTION_DRAFT.json').read_text())
    rows = {r['path']: dict(r, action='RETIRE_HISTORICAL_ARTIFACT', keep_path='') for r in draft}
    duplicates = json.loads((OUT / 'OUTPUT_EXACT_BINARY_DUPLICATES.json').read_text())
    # Retain source features / formal checkpoints. Preserve existing references via deduplication.
    plans = [
        ('outputs/biomaster_odti_local_graph_features_v1/LIGAND_ATOM_FEATURES_FLOAT16_V1.npy',
         'outputs/biomaster_odti_experimental_template_graph_features_v1/LIGAND_ATOM_FEATURES_FLOAT16_V1.npy'),
        ('outputs/biomaster_comprehensive_training_v1/MORGAN2048_UINT8_COMPREHENSIVE_V1.npy',
         'outputs/retrain_20260901/comprehensive_training_v1/MORGAN2048_UINT8_COMPREHENSIVE_V1.npy'),
        ('outputs/biomaster_unified_interaction_20260906/features/TARGET_SITE_TOKENS.npy',
         'outputs/biomaster_best_model_20260906/package_smoke_geometry_v1/features/residue_tokens.npy'),
        ('outputs/biomaster_unified_interaction_20260906/features/TARGET_SITE_TOKENS.npy',
         'outputs/biomaster_best_model_20260906/package_smoke_anchored_v2/features/residue_tokens.npy'),
        ('outputs/biomaster_pocket_precision_20260906/training/structurally_pretrained/cutoff_2020/seed_20260921/BEST.pt',
         'outputs/biomaster_pocket_precision_20260906/downstream_diagnostics/epoch1_fp32_20260907/EPOCH1_SNAPSHOT.pt'),
    ]
    already_shared=[]
    for keep, path in plans:
        proof = next(g for g in duplicates['groups'] if keep in g['paths'] and path in g['paths'])
        a=(ROOT/keep).stat();b=(ROOT/path).stat()
        if (a.st_dev,a.st_ino)==(b.st_dev,b.st_ino):
            already_shared.append(dict(keep=keep,alias=path,reason='ALREADY_HARDLINKED_ZERO_RECLAIM'))
            continue
        assert path not in rows
        rows[path] = dict(group='C4_EXACT_DUPLICATES_KEEP_PATHS', path=path,
            allocated_bytes=(ROOT/path).stat().st_blocks*512, action='DEDUP_PRESERVE_BOTH_PATHS',
            keep_path=keep, sha256=proof['sha256'])

    # Check static references. This is evidence for review, not complete dependency analysis.
    scopes = json.loads((OUT/'TRIAL_ROOTS.json').read_text()) + [
        'outputs/biomaster_pocket_precision_20260906/engineering',
        'outputs/biomaster_context_full_20260908/engineering_resume',
        'outputs/biomaster_context_full_20260908/engineering_continuous',
        'outputs/biomaster_context_full_20260908/final_driver_check',
        'outputs/biomaster_pocket_precision_20260906/structural_data/training_2020/esm2',
        'outputs/biomaster_pocket_precision_20260906/structural_data/training_2020/encoded',
    ]
    names = {p.rsplit('/',1)[-1]:p for p in scopes}
    names.pop('esm2',None); names.pop('encoded',None)  # Avoid generic-token false matches.
    pattern = re.compile('|'.join(re.escape(x) for x in sorted(names,key=len,reverse=True)))
    refs=[]
    for base in ['biomaster','scripts','configs']:
        for p in (ROOT/base).rglob('*'):
            if p.suffix not in {'.py','.json','.yaml','.yml','.sh'} or p.stat().st_size > 2*2**20:continue
            if p.name.startswith('audit_outputs_retirement_'):continue
            for number,line in enumerate(p.read_text(errors='replace').splitlines(),1):
                found=set(pattern.findall(line))
                for token in found:
                    live = p.name.startswith('explorer_') or p.name.startswith('portable_ranker') or p.name in {
                        'predict_biomaster_target_comparators_20260908.py','run_explorer.py'}
                    refs.append(dict(scope=names[token],source=str(p.relative_to(ROOT)),line=number,
                        reference_class='LIVE_ENTRYPOINT_REVIEW' if live else 'RESEARCH_OR_CONFIG_STATIC_REFERENCE'))
    with (OUT/'ADDITIONAL_STATIC_REFERENCES.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['scope','source','line','reference_class']);w.writeheader();w.writerows(refs)

    with gzip.open(OUT/'CANDIDATE_FILES.csv.gz','rt',encoding='utf-8-sig') as f: previous=list(csv.DictReader(f))
    previous_paths={r['relative_path'] for r in previous}
    assert not previous_paths.intersection(rows), 'Additional reclaim must not overlap previous inventory'
    for path,r in rows.items():
        p=ROOT/path;s=p.stat()
        assert not p.is_symlink() and s.st_nlink==1
        assert s.st_blocks*512==r['allocated_bytes']
        r.update(logical_bytes=s.st_size,mtime_ns=s.st_mtime_ns,device=s.st_dev,inode=s.st_ino)
    # Preserve all DTIAM folds in the recommended alternative.
    recommended=[]
    for r in previous:
        if r['group']=='B3_DTIAM_OLD_FOLD_PREDICTORS':continue
        recommended.append(dict(group=r['group'],path=r['relative_path'],allocated_bytes=int(r['allocated_bytes']),
            action='DEDUP_PRESERVE_BOTH_PATHS' if r['group']=='A1_DUPLICATE_BOLTZ_CACHE' else 'RETIRE_PER_PREVIOUS_AUDIT',
            keep_path='',logical_bytes=int(r['logical_bytes']),mtime_ns=int(r['mtime_ns']),device=int(r['device']),inode=int(r['inode'])))
    recommended.extend(rows.values())
    assert len({r['path'] for r in recommended})==len(recommended)
    selected={str(ROOT/r['path']) for r in recommended};opened=[];inaccessible=0
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            for fd in (proc/'fd').iterdir():
                try:
                    target=os.readlink(fd)
                    if target in selected:opened.append(dict(pid=int(proc.name),path=target))
                except OSError:pass
        except OSError:inaccessible+=1
    fields=['group','path','allocated_bytes','action','keep_path','sha256','logical_bytes','mtime_ns','device','inode']
    for name,data in [('ADDITIONAL_CANDIDATE_FILES.csv.gz',list(rows.values())),('RECOMMENDED_100GB_FILES.csv.gz',recommended)]:
        with gzip.open(OUT/name,'wt',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
    grouped=Counter()
    for r in rows.values():grouped[r['group']]+=r['allocated_bytes']
    old_sum=sum(int(r['allocated_bytes']) for r in previous);new_sum=sum(grouped.values())
    plan_sum=sum(r['allocated_bytes'] for r in recommended)
    summary=dict(status='AUDIT_ONLY_NO_DELETION',additional_groups_GiB={k:v/2**30 for k,v in grouped.items()},
        additional_total_GiB=new_sum/2**30,expanded_union_GiB=(old_sum+new_sum)/2**30,
        recommended_without_DTIAM_fold_deletion_GiB=plan_sum/2**30,
        recommended_without_DTIAM_fold_deletion_GB=plan_sum/1e9,
        selected_open_fds=opened,inaccessible_proc_fd_directories=inaccessible,
        already_shared_duplicate_pairs=already_shared,
        live_entrypoint_static_references=[r for r in refs if r['reference_class']=='LIVE_ENTRYPOINT_REVIEW'],
        caveats=['Static references do not cover every dynamically constructed path.',
                 'The plan preserves structures, labels, splits, source weights, formal checkpoint snapshots and result tables.',
                 'Retiring derived structural features prevents immediate resumption of the stopped structural/context training branches.',
                 'Original and additional candidate lists are disjoint; duplicate groups overlapping old trial weights are not double-counted.',
                 'Deduplication requires keeping both paths usable and rechecking hashes immediately before any later mutation.'])
    (OUT/'OUTPUTS_RETIREMENT_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
