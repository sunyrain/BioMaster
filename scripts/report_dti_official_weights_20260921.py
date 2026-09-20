#!/usr/bin/env python3
"""Report per-model acquisition, without promoting downloads to inference parity."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dti_official_weights_download_20260921'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    manifest=json.loads((OUT/'DOWNLOAD_MANIFEST.json').read_text())
    verified=json.loads((OUT/'VERIFIED_FILES.json').read_text()) if (OUT/'VERIFIED_FILES.json').exists() else []
    by_path={r['destination']:r for r in verified if r['status']=='VERIFIED'}
    rows=[]
    for model in sorted({r['model'] for r in manifest['files']}):
        planned=[r for r in manifest['files'] if r['model']==model]
        done=[by_path[r['destination']] for r in planned if r['destination'] in by_path]
        for r in done:assert (ROOT/r['destination']).stat().st_size==r['bytes']
        rows.append(dict(model=model, origin='NEW_DOWNLOAD', planned_files=len(planned), verified_files=len(done),
            verified_GiB=sum(r['bytes'] for r in done)/2**30,
            status='DOWNLOADED_HASH_VERIFIED' if len(done)==len(planned) else 'IN_PROGRESS',
            local_path=f'data/research/dti_official_weights_20260921/{model}',
            inference_state='NATIVE_FORWARD_PARITY_NOT_YET_QUALIFIED'))
    existing=list(csv.DictReader((OUT/'EXISTING_OFFICIAL_ASSETS.csv').open(encoding='utf-8-sig')))
    for model in sorted({r['model'] for r in existing}):
        group=[r for r in existing if r['model']==model]
        rows.append(dict(model=model,origin='EXISTING_REUSED',planned_files=len(group),verified_files=len(group),
            verified_GiB=sum(int(r['bytes']) for r in group)/2**30,status='EXISTING_HASH_REVERIFIED',
            local_path=f'data/research/dti_official_weights_20260921/existing/{model}',
            inference_state='PREVIOUS_STATUS_RETAINED; EviDTI/SCOPE_NOT_CATALOG_QUALIFIED'))
    with (OUT/'MODEL_COLLECTION_STATUS.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    collection={r['model']:r for r in rows}
    aliases={'MAMMAL_pKd':'MAMMAL','DTBind_occurrence':'DTBind','Nesso-1':'nesso','SCOPE':'ScopeDTI_light'}
    resource=json.loads((ROOT/'configs/dti_official_weights_20260920/MODEL_MATRIX_v3.json').read_text())
    combined=[]
    for model in resource['models']:
        item=dict(model)
        item['previous_weights_note']=item.pop('weights')
        acquisition=collection.get(aliases.get(item['id'],item['id']))
        item['collection_status']=acquisition['status'] if acquisition else ('LOCAL_AUTHORIZED_FALLBACK' if item['id']=='DTIAM_A' else 'NOT_IN_OFFICIAL_DOWNLOAD_SCOPE')
        item['current_local_path']=acquisition['local_path'] if acquisition else (item['weight_source'] if item['id']=='DTIAM_A' else '')
        item['phase1_role']={'CURRENT':'CURRENT_SCORE_CHANNEL','PRIORITY_ADD':'NEXT_NATIVE_INFERENCE_ADAPTER','CONDITIONAL':'CONDITIONAL_INFERENCE_EXTENSION','EXCLUDED':'EXCLUDED'}.get(item['group'],'NOT_ACTIVE_IN_PHASE1')
        item['current_score_snapshot_included']=item['group']=='CURRENT'
        combined.append(item)
    with (OUT/'CURRENT_MODEL_RESOURCE_MATRIX.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=combined[0].keys());w.writeheader();w.writerows(combined)
    archives=[]
    for r in by_path.values():
        if r['model']=='GraphBAN' and r['name'].endswith('.zip'):
            with zipfile.ZipFile(ROOT/r['destination']) as z:
                members=z.infolist()
                for member in members:
                    assert not member.filename.startswith('/') and '..' not in Path(member.filename).parts
                archives.append(dict(archive=r['name'],zip_directory_readable=True,members=len(members),
                    pth_checkpoints=sum(m.filename.endswith('.pth') for m in members),
                    uncompressed_bytes=sum(m.file_size for m in members),
                    author_md5_matched=bool(r.get('expected_md5')==r.get('md5'))))
    complete=len(by_path)==len(manifest['files'])
    summary=dict(updated_utc=datetime.now(timezone.utc).isoformat(),
        status='COMPLETE_HASH_VERIFIED' if complete else 'DOWNLOADING',official_model_families=len(rows),
        new_verified_files=len(by_path),new_planned_files=len(manifest['files']),
        new_verified_bytes=sum(r['bytes'] for r in by_path.values()),
        reused_files=len(existing),reused_bytes=sum(int(r['bytes']) for r in existing),
        known_official_task_checkpoint_gaps=['DTIAM','3DICE','DrugCMF','GRAM-DTI','DrugLAMP','TAPB','DeepDTA','DTI-LM'],
        gap_note='Previously surveyed task-checkpoint gaps, not download failures. DTIAM localA is separate.',
        complete_download_not_native_inference_qualification=True,graphban_archives=archives,
        manifest_sha256=sha(OUT/'DOWNLOAD_MANIFEST.json'))
    (OUT/'COLLECTION_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# 官方DTI权重收集与校验','', '工作日期：2026-09-21。当前状态：**'+summary['status']+'**。',
        '',f'本轮覆盖{len(rows)}个有明确官方任务包的模型系列。新增文件{len(by_path)}/{len(manifest["files"])}个已完成校验；复用并重新哈希核验已有{len(existing)}个模型/配置/编码器文件。文件数量包含作者附属文件，不是模型数量。',
        '', 'DTIAM仍使用已授权的本地A版，它不计入官方任务权重系列。未找到公开下游任务权重的研究储备项保留来源缺口，不用编码器冒充任务模型。',
        '', '| 模型 | 本地状态 | 文件 | 校验体积GiB |', '|---|---|---:|---:|']
    lines += [f'| {r["model"]} | {r["status"]} | {r["verified_files"]}/{r["planned_files"]} | {r["verified_GiB"]:.3f} |' for r in rows]
    lines += ['', '新增主任务包包括MAMMAL pKd、指定BindingDB版BALM和GraphBAN；已有EviDTI三套权重复用。条件扩展收齐CheMLT-F两个任务版本及其编码器、ADME-DTI作者saved_models，已有SCOPE25套权重复用。',
        '', 'GraphBAN按作者发布的BindingDB、BioSNAP、KIBA三个完整案例ZIP收集。包内含逐epoch训练状态，不能把每个文件算成独立模型，也不根据本地测试选择epoch；正式使用版本仍须按作者流程及预先冻结规则核对。归档保留压缩状态，避免重复占用数十GB解压空间。',
        '', '所有新增文件存放在数据盘`data/research/dti_official_weights_20260921/`；已有文件通过相对链接复用。下载采用断点续传；大文件分段下载后整文件校验。HF LFS文件核对作者SHA256，GitHub普通文件核对Git blob SHA1并计算本地SHA256，Zenodo包核对作者MD5并计算本地SHA256。',
        '', '下载完成只说明指定来源文件完整到位。原生forward复现、特征/任务头选择、目录覆盖和全矩阵计算仍是下一步；本轮不新开训练。',
        '', '后台收尾程序在下载进程结束后检查完整清单；若存在失败项，最多自动续传三轮，再更新本报告。实时状态见[收尾状态](../outputs/dti_official_weights_download_20260921/FINALIZATION_STATUS.json)。作者模型包中原有Davis训练版本仅作资产溯源，不表示本轮使用Davis作评估数据集。',
        '', '资源来源：[MAMMAL任务模型](https://huggingface.co/ibm-research/biomed.omics.bl.sm.ma-ted-458m.dti_bindingdb_pkd)、[BALM任务模型](https://huggingface.co/BALM/bdb-cleaned-r-esm-lokr-chemberta-loha-cosinemse)、[GraphBAN案例包](https://zenodo.org/records/14813233)、[CheMLT-F](https://huggingface.co/BoulderyBoulder/CheMLT-F)、[ADME-DTI](https://github.com/tariqshaban/adme-dti)。',
        '', '登记文件：[当前完整模型资源矩阵](../outputs/dti_official_weights_download_20260921/CURRENT_MODEL_RESOURCE_MATRIX.csv)、[逐模型状态](../outputs/dti_official_weights_download_20260921/MODEL_COLLECTION_STATUS.csv)、[固定下载清单](../outputs/dti_official_weights_download_20260921/DOWNLOAD_MANIFEST.json)、[逐文件校验](../outputs/dti_official_weights_download_20260921/VERIFIED_FILES.json)、[已有资产复核](../outputs/dti_official_weights_download_20260921/EXISTING_OFFICIAL_ASSETS.csv)、[收集汇总](../outputs/dti_official_weights_download_20260921/COLLECTION_SUMMARY.json)。',
        '', '当前研究先做[双向排序一致性与推荐分歧](BIOMASTER_DTI_RANKING_SCOPE_20260921_ZH.md)，不使用Davis，不判断谁对谁错。']
    (ROOT/'docs/BIOMASTER_DTI_OFFICIAL_WEIGHTS_DOWNLOAD_20260921_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
