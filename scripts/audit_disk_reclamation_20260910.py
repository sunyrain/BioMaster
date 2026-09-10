"""Read-only disk reclamation inventory. No deletion or link mutation capability."""
from pathlib import Path
import os
import csv
import gzip
import json
import shutil
import sqlite3
import pandas as pd
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/disk_reclamation_audit_20260910'
RUNS=['outputs/boltz2_calibration_338_v1/formal_screen_run','outputs/boltz2_discovery_conditional_v1/screen_run']


def protected_boltz():
    """Preserve exact wet-lab pairs and entire processed batches containing them."""
    current=pd.read_csv(ROOT/'outputs/spr384_final_experiment_table_20260910/SPR384_FINAL_EXPERIMENT_TABLE.csv')
    pairs=set(current['药物完整InChIKey']+'__'+current['新靶点ChEMBL编号'])
    controls=pd.read_csv(ROOT/'outputs/spr384_final_experiment_table_20260910/SPR112_REFERENCE_CONTROLS.csv')
    pairs.update(controls['对照完整InChIKey']+'__'+controls['新靶点ChEMBL编号'])
    old=pd.read_csv(ROOT/'outputs/joint384_comprehensive_20260909/REFERENCE_CONTROLS_EXTRA.csv')
    pairs.update(old['ligand_inchikey']+'__'+old['target_chembl_id'])
    manifest=pd.read_csv(ROOT/'outputs/boltz2_calibration_338_v1/BOLTZ2_CALIBRATION_INPUT_MANIFEST_V1.csv')
    ids=manifest['drugId'].dropna().unique().tolist();mapping={}
    db=ROOT/'downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db'
    with sqlite3.connect(f'file:{db}?mode=ro',uri=True) as conn:
        for i in range(0,len(ids),500):
            part=ids[i:i+500]
            query='SELECT md.chembl_id, cs.standard_inchi_key FROM molecule_dictionary md JOIN compound_structures cs ON md.molregno=cs.molregno WHERE md.chembl_id IN ('+','.join('?' for _ in part)+')'
            mapping.update(conn.execute(query,part).fetchall())
    manifest['exact_pair']=manifest['drugId'].map(mapping)+'__'+manifest['target_chembl_id']
    # An unresolved identity is protected conservatively, rather than assumed unrelated.
    keep=manifest[manifest['exact_pair'].isin(pairs)|manifest['exact_pair'].isna()]
    calibration_ids=set(keep['pairId'])
    protected={};rows=[]
    for rel in RUNS:
        provenance=pd.read_csv(ROOT/rel/'result_provenance.csv')
        matches=provenance[provenance['pairId'].isin(calibration_ids if 'calibration' in rel else pairs)]
        protected[rel]=(set(matches['boltzStem']),set(matches['batch']))
        for row in matches.to_dict('records'):
            rows.append({'run':rel,'pair_id':row['pairId'],'boltz_stem':row['boltzStem'],'protected_processed_batch':row['batch']})
    pd.DataFrame(rows).to_csv(OUT/'PROTECTED_WETLAB_BOLTZ.csv',index=False,encoding='utf-8-sig')
    return protected,dict(wetlab_and_historical_control_unique_pairs=len(pairs),protected_provenance_rows=len(rows),
        unresolved_calibration_manifest_rows=int(manifest['exact_pair'].isna().sum()),
        matches_by_run={rel:len(stems) for rel,(stems,batches) in protected.items()})


def files(root):
    for directory, dirs, names in os.walk(root,followlinks=False):
        dirs[:]=[x for x in dirs if not (Path(directory)/x).is_symlink()]
        for name in names:
            p=Path(directory)/name
            if p.is_file() and not p.is_symlink():yield p


def main():
    OUT.mkdir(exist_ok=True)
    protected,protection_summary=protected_boltz()
    excluded=Counter()
    meta={
      'A1_DUPLICATE_BOLTZ_CACHE':('A_内容重复_须保持引用路径','三份Boltz缓存与共享缓存SHA256相同；保留共享实体，原路径需安全重定向，不能直接删断引用。'),
      'A2_DOWNLOAD_SEGMENTS':('A_重复下载分片','已组装完整zip；分片拼接SHA256相同后可移除分片，保留zip。'),
      'A3_OFFLINE_UNPACKED':('A_有完整归档的展开副本','保留旧版zip及校验文件；CRC与目录覆盖验证通过后可移除展开目录。'),
      'B1_BOLTZ_ERROR_MATRICES':('B_历史预测细节_有信息损失','PAE/PDE逐残基误差矩阵；保留CIF、affinity/confidence JSON和汇总CSV；失去精细误差重分析，恢复需重推理。'),
      'B2_BOLTZ_PROCESSED':('B_可重建预处理','历史Boltz processed中间文件；保留输入、源结构、模型、环境和预测结果；重跑需要重新预处理。'),
      'B3_DTIAM_OLD_FOLD_PREDICTORS':('B_历史折模型_复现成本高','仅非S5折的predictor目录；保留折级预测、汇总和训练输入；S5部署模型整套保留。删除将失去旧折模型的直接再推理能力，重训未必位级一致。'),
      'B4_UNPROMOTED_ATOM_TOKENS':('B_未晋升模型特征_可重建','旧unified轮ATOM_TOKENS.npy；保留LMDB、索引、特征清单、checkpoint和日志；旧轮复训/审计需重算特征。'),
      'B5_CONTEXT_DERIVED_INPUTS':('B_后续训练可能复用','未晋升context模型predicted与p2rank派生输入；保留原始结构、索引、全局特征、模型和训练记录；恢复需重做口袋和编码，不能称永远无用。'),
    }
    def selections():
        for name in ['boltz2_aff.ckpt','boltz2_conf.ckpt','mols.tar']:
            yield 'A1_DUPLICATE_BOLTZ_CACHE',ROOT/'outputs/boltz2_structure_affinity_v1/boltz_cache'/name
        for p in files(ROOT/'.tmp/bindingdb_202608_full/segments'):yield 'A2_DOWNLOAD_SEGMENTS',p
        for p in files(ROOT/'outputs/offline_release_20260909/BioMaster_Offline_Windows_x64'):yield 'A3_OFFLINE_UNPACKED',p
        for rel in RUNS:
            stems,batches=protected[rel]
            for p in files(ROOT/rel):
                parts=p.relative_to(ROOT/rel).parts
                if p.name.startswith(('pae_','pde_')) and p.suffix=='.npz':
                    if stems.intersection(parts):excluded['protected_matrix_bytes']+=p.stat().st_blocks*512
                    else:yield 'B1_BOLTZ_ERROR_MATRICES',p
                elif 'processed' in parts:
                    if batches.intersection(parts):excluded['protected_processed_bytes']+=p.stat().st_blocks*512
                    else:yield 'B2_BOLTZ_PROCESSED',p
        root=ROOT/'outputs/old_drug_target_sota_v1/public_retrained_v1/dtiam_same_data_compatible_v1'
        for run in root.iterdir():
            if run.is_dir() and not run.name.startswith('S5_'):
                for p in files(run/'predictor'):yield 'B3_DTIAM_OLD_FOLD_PREDICTORS',p
        yield 'B4_UNPROMOTED_ATOM_TOKENS',ROOT/'outputs/biomaster_unified_interaction_20260906/features/ATOM_TOKENS.npy'
        for name in ['predicted','p2rank']:
            for p in files(ROOT/'outputs/biomaster_context_full_20260908/data'/name):yield 'B5_CONTEXT_DERIVED_INPUTS',p
    allocated=Counter();logical=Counter();counts=Counter();seen=set();inodes=set();selected=set();hardlinks=[]
    with gzip.open(OUT/'CANDIDATE_FILES.csv.gz','wt',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['group','relative_path','logical_bytes','allocated_bytes','mtime_ns','device','inode','nlink'])
        for group,p in selections():
            assert p not in seen,p
            seen.add(p);selected.add(str(p));s=p.stat();identity=(s.st_dev,s.st_ino)
            if s.st_nlink>1:hardlinks.append(str(p))
            size=s.st_blocks*512 if identity not in inodes else 0
            inodes.add(identity);allocated[group]+=size;logical[group]+=s.st_size;counts[group]+=1
            writer.writerow([group,str(p.relative_to(ROOT)),s.st_size,size,s.st_mtime_ns,s.st_dev,s.st_ino,s.st_nlink])
    summaries=[]
    for group,(risk,reason) in meta.items():
        summaries.append(dict(group=group,risk=risk,files=counts[group],allocated_bytes=allocated[group],
            GiB=allocated[group]/2**30,decimal_GB=allocated[group]/1e9,reason=reason))
    with (OUT/'CLEANUP_CANDIDATES.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    opened=[];denied=0
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            for fd in (proc/'fd').iterdir():
                try:
                    target=os.readlink(fd)
                    if target in selected:opened.append(dict(pid=int(proc.name),path=target))
                except OSError:pass
        except OSError:denied+=1
    disk=shutil.disk_usage(ROOT)
    for name in ['duplicate_hashes','offline_duplicate_check','segments_duplicate_check']:
        path=Path('/tmp')/('biomaster_space_duplicate_hashes.json' if name=='duplicate_hashes' else f'biomaster_{name}.json')
        if path.exists():shutil.copyfile(path,OUT/(name+'.json'))
    result=dict(status='AUDIT_ONLY_NO_DELETIONS',size_unit='GiB = 2^30 bytes; GB = 10^9 bytes',
        disk_total_GiB=disk.total/2**30,disk_used_GiB=disk.used/2**30,disk_free_GiB=disk.free/2**30,
        candidate_total_GiB=sum(allocated.values())/2**30,candidate_total_decimal_GB=sum(allocated.values())/1e9,
        duplicate_or_archived_GiB=sum(v for k,v in allocated.items() if k.startswith('A'))/2**30,
        conditional_historical_GiB=sum(v for k,v in allocated.items() if k.startswith('B'))/2**30,
        candidates=summaries,hardlinked_candidate_files=hardlinks,open_candidate_fds=opened,inaccessible_proc_fd_directories=denied,
        wetlab_boltz_protection=protection_summary,protected_excluded_bytes=dict(excluded),
        caveats=['No claim of permanent future uselessness; B groups trade stored artifacts for recomputation or lost historical detail.',
          'Allocated bytes exclude directory metadata; not a guarantee of post-delete free space.',
          'Open descriptor scan is a momentary check, not proof of no runtime or future dependencies.',
          'No deletion or link replacement performed; full-file hashes are only computed for stated duplicate proofs.',
          'Keep source inputs, versioned models, environment definitions and metrics before reclaiming derived outputs.'])
    (OUT/'SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['candidates','caveats']},ensure_ascii=False,indent=2))
    for row in summaries:print(row['group'],round(row['GiB'],3),row['files'])


if __name__=='__main__':main()
