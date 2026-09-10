"""Execute only the exact, user-approved 2026-09-10 reclamation manifest.

No recursive deletion. Default invocation validates only; --execute performs the
authorized retirement after validating every file, duplicate and retained archive.
"""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import stat
import time
import zipfile
import zlib
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT/'outputs/disk_reclamation_audit_20260910'
OUT = ROOT/'outputs/disk_reclamation_execution_20260910'
PLAN = AUDIT/'RECOMMENDED_100GB_FILES.csv.gz'
APPROVED_SHA256 = 'fe5112323cb03ba9b46194a57f53010983992e5ee357ccd34c40e1add8956903'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()


def save(name,value):
    p=OUT/name;tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));os.replace(tmp,p)


def signature(p):
    s=p.lstat()
    assert stat.S_ISREG(s.st_mode) and s.st_nlink==1, str(p)
    return (s.st_size,s.st_mtime_ns,s.st_dev,s.st_ino,s.st_blocks*512)


def expected(r):
    return tuple(int(r[k]) for k in ['logical_bytes','mtime_ns','device','inode','allocated_bytes'])


def protected_snapshot():
    hashes={};metadata={}
    for rel in ['outputs/spr384_final_experiment_table_20260910',
                'outputs/spr384_final_experiment_table_before_fda_v2_20260910',
                'outputs/joint384_comprehensive_20260909',
                'outputs/biomaster_best_model_20260906/retargetmap_selected_v1']:
        for p in (ROOT/rel).rglob('*'):
            if p.is_file():hashes[str(p.relative_to(ROOT))]=sha(p)
    for rel in [
        'outputs/unified_pair_program_720x384_v1/UNIFIED_DTA_720_X_384_PAIR_MATRIX_V1.csv.gz',
        'outputs/old_drug_target_sota_v1/drug_centric_ranker_v1/BIOMASTER_DRUG_TO_TARGET_720X384_V1.csv.gz',
        'outputs/old_drug_target_sota_v1/public_retrained_v1/dtiam_720x384_deployment_v1/DTIAM_720X384_SCORES_V1.csv.gz',
        'outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv']:
        hashes[rel]=sha(ROOT/rel)
    rel='outputs/old_drug_target_sota_v1/public_retrained_v1/dtiam_same_data_compatible_v1'
    for p in (ROOT/rel).rglob('*'):
        if p.is_file():
            s=p.stat();metadata[str(p.relative_to(ROOT))]=[s.st_size,s.st_mtime_ns,s.st_dev,s.st_ino]
    return dict(sha256=hashes,stat=metadata)


def main(execute):
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'EXECUTION_STARTED.json').exists(), 'Execution already started; inspect journal before retrying'
    assert sha(PLAN)==APPROVED_SHA256, 'Approved manifest changed'
    with gzip.open(PLAN,'rt',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    selected={r['path'] for r in rows};assert len(selected)==len(rows)==272494
    assert sum(int(r['allocated_bytes']) for r in rows)==101647089664
    for rel in selected:
        assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
        assert rel.startswith(('outputs/','.tmp/bindingdb_202608_full/segments/'))
        assert not any(x in rel for x in ['dtiam_same_data_compatible_v1','/spr_results/',
            '/spr384_final_experiment_table','/retargetmap_selected_v1/','/complexes_2020/'])
    protect={}
    with (AUDIT/'PROTECTED_WETLAB_BOLTZ.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            stems,batches=protect.setdefault(r['run'],(set(),set()))
            stems.add(r['boltz_stem']);batches.add(r['protected_processed_batch'])
    checked_parents=set()
    for i,r in enumerate(rows,1):
        p=ROOT/r['path'];assert signature(p)==expected(r), 'Changed file: '+str(p)
        if p.parent not in checked_parents:
            assert p.parent.resolve()==p.parent, 'Symlink parent: '+str(p.parent)
            checked_parents.add(p.parent)
        for run,(stems,batches) in protect.items():
            if r['path'].startswith(run+'/'):
                if r['group']=='B1_BOLTZ_ERROR_MATRICES':assert not stems.intersection(p.parts)
                if r['group']=='B2_BOLTZ_PROCESSED':assert not batches.intersection(p.parts)
        if i%50000==0:print('Preflight file metadata',i,'/',len(rows),flush=True)
    opened=[];denied=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            for fd in (proc/'fd').iterdir():
                try:
                    target=os.readlink(fd)
                    if target.startswith(str(ROOT)+'/') and target[len(str(ROOT))+1:] in selected:
                        opened.append(dict(pid=proc.name,path=target))
                except OSError:pass
        except OSError:denied.append(proc.name)
    assert not opened, opened

    duplicate_proofs={r['primary']:r for r in json.loads((AUDIT/'duplicate_hashes.json').read_text())}
    dedup={}
    for r in rows:
        if r['action']!='DEDUP_PRESERVE_BOTH_PATHS':continue
        if r['group']=='A1_DUPLICATE_BOLTZ_CACHE':
            proof=duplicate_proofs[r['path']];keep=Path(proof['duplicate']);digest=proof['sha256']
        else:keep=ROOT/r['keep_path'];digest=r['sha256']
        assert str(keep.relative_to(ROOT)) not in selected if keep.is_relative_to(ROOT) else True
        assert keep.is_file() and not keep.is_symlink()
        assert sha(keep)==sha(ROOT/r['path'])==digest, 'Duplicate mismatch'
        assert keep.stat().st_dev==(ROOT/r['path']).stat().st_dev
        dedup[r['path']]=dict(keep=str(keep),sha256=digest)
        print('Duplicate reverified:',r['path'],flush=True)

    segment=json.loads((AUDIT/'segments_duplicate_check.json').read_text());h=hashlib.sha256()
    for rel in segment['parts']:
        assert rel in selected
        with (ROOT/rel).open('rb') as f:
            for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    assert h.hexdigest()==sha(ROOT/segment['assembled_archive'])==segment['archive_sha256']
    archived=json.loads((AUDIT/'offline_duplicate_check.json').read_text())
    base=ROOT/archived['unpacked'];covered=set()
    with zipfile.ZipFile(ROOT/archived['zip']) as z:
        assert z.testzip() is None
        for info in z.infolist():
            if info.is_dir():continue
            # Archive stores the top-level package directory.
            rel=Path(info.filename)
            if rel.parts[0]==base.name:rel=Path(*rel.parts[1:])
            p=base/rel;crc=0;size=0
            with p.open('rb') as f:
                for b in iter(lambda:f.read(2**20),b''):crc=zlib.crc32(b,crc);size+=len(b)
            assert size==info.file_size and crc==info.CRC, str(p)
            covered.add(str(p.relative_to(ROOT)))
    extras={r['path'] for r in rows if r['group']=='A3_OFFLINE_UNPACKED'}-covered
    assert extras==set(archived['unarchived_files'])
    assert all('/__pycache__/' in p and p.endswith('.pyc') for p in extras)
    print('Retained BindingDB ZIP and offline archive reverified',flush=True)
    snapshot=protected_snapshot();save('PROTECTED_BEFORE.json',snapshot)
    shutil.copyfile(PLAN,OUT/'EXECUTED_PLAN.csv.gz')
    preflight=dict(status='PASS',files=len(rows),allocated_bytes=101647089664,plan_sha256=APPROVED_SHA256,
        protected_hashed_files=len(snapshot['sha256']),protected_DTIAM_metadata_files=len(snapshot['stat']),
        duplicates=dedup,open_selected_fds=opened,inaccessible_proc_fd_directories=denied,
        offline_archive_sha256=sha(ROOT/archived['zip']),bindingdb_archive_sha256=segment['archive_sha256'])
    save('PREFLIGHT.json',preflight)
    print('Preflight PASS',flush=True)
    if not execute:return

    start=shutil.disk_usage(ROOT);began=time.time()
    save('EXECUTION_STARTED.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        plan_sha256=APPROVED_SHA256,disk_free_bytes=start.free,status='IN_PROGRESS',
        authorization='User: 好的，清理这些 — approved recommended 101.65 GB file plan'))
    counts=Counter();released=Counter();parents=set();deleted=0;linked=0
    try:
        with (OUT/'FILE_ACTIONS.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f);writer.writerow(['group','path','action','allocated_bytes','keep_path'])
            for i,r in enumerate(rows,1):
                p=ROOT/r['path'];assert signature(p)==expected(r), 'Changed immediately before operation: '+str(p)
                if r['path'] in dedup:
                    keep=Path(dedup[r['path']]['keep']);tmp=p.with_name(p.name+'.reclaim-link-tmp')
                    assert not tmp.exists()
                    os.link(keep,tmp)
                    os.replace(tmp,p)
                    assert os.path.samefile(p,keep)
                    writer.writerow([r['group'],r['path'],'REPLACED_WITH_HARDLINK',r['allocated_bytes'],str(keep)])
                    linked+=1
                else:
                    p.unlink();deleted+=1;parents.add(p.parent)
                    writer.writerow([r['group'],r['path'],'UNLINKED',r['allocated_bytes'],''])
                counts[r['group']]+=1;released[r['group']]+=int(r['allocated_bytes'])
                if i%10000==0:
                    f.flush();os.fsync(f.fileno())
                    print('Retired/deduplicated',i,'/',len(rows),'GiB',round(sum(released.values())/2**30,2),flush=True)
            f.flush();os.fsync(f.fileno())
        # Only remove now-empty ancestors of actually deleted files, below output roots.
        empty=0;all_parents=set()
        for p in parents:
            while p.is_relative_to(ROOT) and len(p.relative_to(ROOT).parts)>=2:
                all_parents.add(p);p=p.parent
        for p in sorted(all_parents,key=lambda x:len(x.parts),reverse=True):
            try:p.rmdir();empty+=1
            except OSError:pass
        for r in rows:
            p=ROOT/r['path']
            if r['path'] in dedup:
                assert os.path.samefile(p,dedup[r['path']]['keep'])
            else:assert not p.exists(),str(p)
        for item in dedup.values():assert sha(Path(item['keep']))==item['sha256']
        after=protected_snapshot();save('PROTECTED_AFTER.json',after);assert after==snapshot,'Protected files changed'
        end=shutil.disk_usage(ROOT)
        save('RESULT.json',dict(status='COMPLETE',plan_sha256=APPROVED_SHA256,deleted_files=deleted,
            deduplicated_files=linked,empty_directories_removed=empty,group_counts=dict(counts),
            group_allocated_bytes=dict(released),planned_file_bytes=sum(released.values()),
            free_before_bytes=start.free,free_after_bytes=end.free,observed_free_gain_bytes=end.free-start.free,
            protected_snapshot_unchanged=True,elapsed_seconds=time.time()-began,
            note='Observed free-space gain also includes removed directory metadata and concurrent filesystem activity.'))
        print((OUT/'RESULT.json').read_text(),flush=True)
    except BaseException as e:
        save('ERROR.json',dict(status='INTERRUPTED',error=str(e),deleted_files=deleted,deduplicated_files=linked,
            group_counts=dict(counts),instruction='Inspect FILE_ACTIONS.csv and filesystem; do not blindly rerun.'))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true')
    main(parser.parse_args().execute)
