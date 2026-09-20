#!/usr/bin/env python3
"""Readable architecture, data and weight lists derived from the frozen preparation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
PREP=ROOT/'outputs/dti_research_preparation_20260920'
OUT=ROOT/'outputs/dti_resource_catalog_20260920'
DOC=ROOT/'docs/BIOMASTER_DTI_ARCHITECTURE_DATA_WEIGHTS_20260920_ZH.md'
DATA=ROOT/'data/research/dti_reliability_20260920_v1'
MASTER=ROOT/'data/processed/biomaster_training_full_20260910_v1'
KG=ROOT/'outputs/biomaster_disease_evidence_720x888_20260909'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write_csv(name,rows):
    frame=pd.DataFrame(rows);frame.to_csv(OUT/name,index=False,encoding='utf-8-sig');return frame


def local(path):
    path=Path(path)
    if not path.is_absolute():path=ROOT/path
    assert path.exists(),path
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def table(frame,cols):
    def cell(x):return str(x).replace('|',' / ').replace('\n','；')
    return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+
        ['| '+' | '.join(cell(v) for v in row)+' |' for row in frame[cols].itertuples(index=False,name=None)])


def architectures():
    rows=[]
    def add(name,group,inputs,mechanism,output,status,path,role,source=''):
        rows.append({'名称':name,'类别':group,'输入':inputs,'配对计算':mechanism,'输出':output,'当前状态':status,
            '主要用途':role,'代码或方案位置':local(path),'官方来源':source})
    audit='docs/BIOMASTER_SEVEN_MODEL_ARCHITECTURE_AUDIT_20260919_ZH.md'
    add('ReTargetMap 当前生产版','现用七模型','DrugCLIP512＋Morgan2048＋图均值40；ESM2全局1280',
        '两侧投影；[d,p,d×p,abs(d−p)]与残差MLP；当前global分支','双向排序logit','有生产权重',audit,'保留原生产基线')
    add('DTIAM 加强版A','现用七模型','BerMol768＋ESM21280','拼接后AutoGluon树/神经网络集成',
        'Kd/Ki＋明确失活二分类分数','A/B各3套；网站选A seed20260923',audit,'强表征＋学习器对照','https://github.com/CSUBioGroup/DTIAM')
    add('ConPLex','现用七模型','Morgan2048＋ProtBERT1024','Linear→ReLU双塔＋余弦',
        '相似度','1个官方任务检查点',audit,'双塔和并列秩诊断','https://github.com/samsledje/ConPLex')
    add('DrugCLIP','现用七模型','分子3D构象＋指定口袋3D原子','双UniMol式编码器＋归一化内积；六折均值',
        '检索相似度','6折权重已登记',audit,'结构口袋检索对照','https://github.com/THU-ATOM/Drug-The-Whole-Genome')
    add('ProbeMatchDTI','现用七模型','蛋白/SMILES token＋分子2D图','迭代探针、跨模态注意力、多路融合',
        '二分类分数','All_Model及编码器已登记',audit,'局部交互对照','https://github.com/developer-hq/ProbeMatchDTI')
    add('DTBind','现用七模型','ProtT5残基＋蛋白结构图/表面＋分子2D图','GNN＋双向交叉注意力＋门控',
        '当前occurrence二分类；affinity/site另分支','3种任务权重；当前启用occurrence',audit,'结构图交互对照','https://github.com/liqy09/DTBind')
    add('Nesso-1','现用七模型','ESM2残基＋分子3D构象','Pairformer、距离关系及口袋裁剪、分类与回归头',
        'binder；log10(IC50/μM)，连续头越低越强','权重及配置已登记；目录补算中',audit,'同主干不同头与适用区域','https://github.com/recursionpharma/nesso')
    add('Palinova A/B 历史重训','已有自训','DrugCLIP/Morgan/图均值＋ESM2全局',
        '原全局配对MLP；按变体增加排序与端点回归辅助','二分类及分端点回归',
        '6固定步数＋24多任务＋30 assay-aware拟合','biomaster/endpoint_multitask.py','复用目标函数、采样与种子证据')
    add('EviDTI','扩展模型','蛋白序列表征＋分子2D/3D特征','蛋白LightAttention、分子CNN/几何图网络与融合',
        '证据式分类；Davis/KIBA任务头分别解释','3套任务权重；目录适配待核验',
        'third_party/sota_dti_2026/EviDTI/drugbank_model.py','不确定性方法对照','https://github.com/zhaoyanpeng208/EviDTI')
    add('SCOPE 轻量版','扩展模型','分子3D图＋蛋白序列','分子GVP＋蛋白CNN＋双线性注意力',
        'DTI分类','25个检查点；现有入口使用固定目标库','.external/scope_dti_lightweight/models.py',
        '几何编码/注意力对照；任意新靶点适配未认证','https://github.com/Yigang-Chen/Lightweight-SCOPE-DTI-for-Inference')
    add('TAPB','扩展模型','MolFormer tokenizer的SMILES token＋ESM2蛋白表征',
        '分子Transformer＋交叉注意力解码器＋靶点先验干预','DTI分类','代码/编码器资源在；合格任务权重未确认',
        'third_party/sota_dti_2026/TAPB/models/tapb.py','靶点先验偏差对照；不把缓存MolFormer等同已训练TAPB','https://github.com/GaomingL1n/TAPB')
    add('DrugBAN','扩展模型','分子2D图＋蛋白序列','分子GCN＋蛋白CNN＋双线性注意力',
        'DTI分类','有代码；本地一轮演示模型不列合格基线','third_party/DrugBAN/models.py',
        '可重训的局部交互基线','https://github.com/peizhenbai/DrugBAN')
    add('DeepDTA','扩展模型','SMILES与蛋白序列字符','两侧1D-CNN＋池化拼接＋全连接',
        '连续任务回归','有代码；现成合格任务权重未确认','third_party/DeepDTA/source/run_experiments.py',
        '经典序列回归参照','https://github.com/hkmztrk/DeepDTA')
    active=ROOT/'configs/dti_reliability_20260920/ACTIVE_PROTOCOL.json'
    protocol_path=json.loads(active.read_text())['active_protocol'] if active.exists() else 'configs/dti_reliability_20260920/PROTOCOL.json'
    protocol=json.loads((ROOT/protocol_path).read_text())
    labels={'drug_only':('药物单侧','实体偏好'), 'protein_only':('蛋白单侧','实体偏好'), 'additive':('两侧加性','无配对交互参照'),
        'dual_cosine':('双塔余弦','低成本匹配'), 'bilinear':('低秩双线性','显式乘性交互'),
        'global_mlp':('全局交互MLP','受控配对主干'), 'lightgbm':('LightGBM','强树学习器')}
    for key in labels:
        cfg=protocol['architectures'][key]
        name,role=labels[key]
        add(name,'待训受控架构','＋'.join(cfg['inputs']),cfg['formula'],'二分类；目标干预组另加辅助',
            '协议已登记；新受控权重尚未训练',protocol_path,role)
    add('旧结构交互预训练教师','辅助路线','实验复合物原子/残基与接触/距离','局部结构交互编码＋几何读出',
        '接触与距离表征','结构预训练权重在；部分大编码缓存需重建',
        'docs/BIOMASTER_PM_JEPA_MODALITIES_AND_RESOURCES_20260914_ZH.md','为JEPA准备教师；未证实提升亲和')
    add('蛋白—分子组合JEPA','辅助路线','便宜的蛋白/分子嵌入；训练时部分昂贵教师视图','预测组合潜在表示z，再预测性质',
        '潜在表示及性质','设计阶段；无本方案已训练学生权重',
        'docs/BIOMASTER_PM_JEPA_DESIGN_20260914_ZH.md','后续表示学习实验')
    add('TxGNN','辅助路线','药物—疾病等异构知识图谱','异构图编码及关系解码',
        '药物—疾病适应证logit','官方模型本地存在；已做多方向推理',
        'scripts/run_txgnn_all_diseases_720_20260909.py','疾病假设支持，不作直接亲和真值')
    return write_csv('ARCHITECTURE_LIST.csv',rows)


def data_resources():
    rows=[]
    def add(name,group,paths,count,unit,content,use,status,notes='',date='2026-09-20'):
        paths=[paths] if isinstance(paths,(str,Path)) else paths
        rows.append({'资源':name,'层次':group,'规模':count,'计数单位':unit,'内容':content,'用途':use,'状态':status,
            '数据位置':';'.join(local(p) for p in paths),'数量依据日期':date,'边界':notes})
    add('ChEMBL37 原库','原始来源','downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db',1,'SQLite库',
        '原始activity/assay/target/structure/文献','回溯与重建','本地存在','记录与BindingDB可能重复')
    binding=sorted((ROOT/'data/external/affinity_refresh_20260910').glob('BindingDB*.zip'))
    add('BindingDB 202609归档','原始来源',binding,len(binding),'压缩归档',
        '全包、分来源包、assay与ID映射','训练证据及来源审计','已归档','各包不能相加当独立实验')
    for name,file,unit,content in [
        ('母库测量观测','OBSERVATIONS_WITH_QC.parquet','测量观测行','单位/删失/来源/身份QC，包含需按QC筛选的记录'),
        ('母库任务标签','ALL_PAIR_TASK_LABELS.parquet','配对×任务行','包含灰区/冲突及保护旗标'),
        ('分子登记','MOLECULES.parquet','分子登记行','SMILES/身份/固定特征索引'),
        ('蛋白登记','TARGETS.parquet','蛋白序列登记行','精确序列/标识/固定特征索引'),
        ('母库来源连接','PAIR_CITATIONS.parquet','配对—来源连接行','原论文/专利等溯源')]:
        path=MASTER/file
        add(name,'清洗母库',path,pq.read_metadata(path).num_rows,unit,content,'重建与审计','本地存在','母库行数不等于可训练唯一配对数')
    summary=json.loads((PREP/'DATA_PREPARATION_SUMMARY.json').read_text())
    add('研究唯一配对宇宙','研究成员',DATA/'CANONICAL_PAIRS.parquet',summary['canonical_pairs'],'唯一配对',
        '857332分子、4742蛋白；A/B全部split并集','统一身份连接','已生成并检查','包含验证/测试，不能整表作为训练')
    counts=pd.read_csv(PREP/'DATASET_COUNTS.csv')
    for arm in ['A','B']:
        for split,label in [('train','训练'),('validation','验证'),('test','测试')]:
            c=counts.query('arm == @arm and view == "scaffold_replay" and split == @split').iloc[0]
            add(f'{arm} 原骨架{label}','研究成员',DATA/f'{arm}_scaffold_replay_{split}.parquet',int(c.pairs),'配对',
                f'阳性{c.positive}；阴性{c.negative}；明确失活{c.explicit_inactive}',label,'已生成并检查',
                'A=Kd/Ki＋失活；B=全端点＋失活；失活是阴性子集')
    for view,label in [('source_purged','来源去重'),('cold_target','冷靶点'),('double_cold','双侧冷启动'),('temporal_conservative','保守时间')]:
        files=sorted(DATA.glob(f'*_{view}_*.parquet'))
        add(label,'研究成员',files,len(files),'成员表','两臂各train/validation/test；详细计数见DATASET_COUNTS.csv',
            '重新训练后检验泛化','已生成并检查','重新划分旧数据；旧任务权重不能作为独立测试')
    contrasts=pd.read_csv(PREP/'DATA_CONTRAST_COUNTS.csv')
    for c in contrasts.itertuples():
        add(c.name,'研究成员',DATA/(c.name+'.parquet'),int(c.pairs),'训练配对',
            f'阳性{c.positive}；明确失活{c.explicit_inactive}；目标{c.targets}',
            '控制A/B数量、目标及类别构成','已生成并检查','匹配/共同/增量是不同视图，不互相相加重复计数')
    for name,file,unit,use in [('精确端点回归','REGRESSION_EXACT.parquet','配对×端点','分端点回归辅助'),
        ('真实assay辅助','ASSAY_AUXILIARY.parquet','assay内配对行','过滤当前train后辅助排序'),
        ('蛋白同源簇','TARGET_CLUSTERS.parquet','蛋白登记行','同源簇划分'),
        ('来源组连接','PAIR_DOCUMENT_GROUPS.parquet','配对—源组连接','来源去重'),
        ('来源/日期QC','PAIR_SOURCE_QC.parquet','配对','时间与缺源审计')]:
        path=DATA/file
        add(name,'辅助监督及划分',path,pq.read_metadata(path).num_rows,unit,name,use,'已生成并检查',
            '辅助监督必须按当前train成员筛选；无精确值的失活不补造Kd')
    for f in json.loads((PREP/'FEATURE_REGISTRY.json').read_text()):
        available=f['required_rows']-f['zero_rows']
        add(f['name'],'冻结特征',f['path'],f['required_rows'],'本研究所需实体',
            f"数组{f['shape']}；{f['dtype']}；有效{available}；缺失{f['zero_rows']}",
            '输入表征','已核验',f['missing_policy']+' 数组分配行数不同于本研究实体数。')
    graph=ROOT/'outputs/biomaster_endpoint_ablation_20260911/features/GRAPH.npy'
    add('分子图均值40维','冻结特征',graph,996696,'数组分配行',str(np.load(graph,mmap_mode='r').shape),
        '旧Palinova输入','本地存在；非本轮统一主输入','有效行应结合成员索引与CHEM_DONE读取')
    affine=json.loads((ROOT/'outputs/affinity_evidence_refresh_20260910/SUMMARY.json').read_text())
    external=[('CACHE1','CACHE_all_data_with_structures_vs2.xlsx','响应与剂量拟合；不能全算Kd'),
        ('CACHE3','CACHE_1-All_data_CACHE3_Round1.xlsx','SPR；保留作者伪影/失败标记'),
        ('OpenBind_20260828','OpenBind_all_affinity_data_release_v1.csv','GCI Kd；病毒构建；区分作者QC'),
        ('RAF_MEK_2026','RAF_MEK_MOESM3.xlsx','kinobead响应；不是Kd')]
    for key,file,notes in external:
        paths=[ROOT/'data/external/affinity_refresh_20260910'/file]
        if key=='CACHE3':paths.append(ROOT/'data/external/affinity_refresh_20260910/CACHE_2-All_data_CACHE3_Round2.xlsx')
        if key=='RAF_MEK_2026':paths=list((ROOT/'data/external/affinity_refresh_20260910').glob('RAF_MEK_MOESM*.xlsx'))
        add(key,'外部测量资源',paths,affine['source_rows'][key],'9月10日索引测量/拟合/响应行',notes,
            '外部面板资格审计','原始文件及索引在；独立性待审','非全部原始工作表行数，非独立配对数','2026-09-10')
    for key,file,note in [('HiQBind_v3','hiqbind_sm_metadata.csv','只有元数据；结构附带既有标签，不是新测量'),
        ('Supercharge_Data6','SUPERCHARGE_Supplementary_Data6.xlsx','细胞/体系表观Kd，不能与纯蛋白Kd混用')]:
        add(key,'外部测量资源',ROOT/'outputs/recent_affinity_sources_20260910'/file,affine['source_rows'][key],
            '9月10日索引行',note,'结构或体系敏感性研究','本地存在；独立性待审',note,'2026-09-10')
    add('PLINDER 实验结构库存','辅助模态','outputs/biomaster_pocket_precision_20260906/structural_data',11126,'历史结构系统',
        '9903训练＋1223内部验证；接触/距离映射','结构教师准备','目录在；数量沿用9月14日逐文件审计',
        '大编码缓存需重建；新A/B重叠未认证；不是11126个亲和真值','2026-09-14')
    add('残基token缓存','辅助模态','outputs/biomaster_bindingdb_target_token_feature_package_v1/ESM2_650M_RESIDUE_FLOAT16_COMBINED_V1.npy',
        813,'历史靶点索引','483934×1280残基向量','局部交互输入','本地存在；数量依据9月14日',
        '映射新序列索引后才能使用；不是4742目标全覆盖','2026-09-14')
    add('实验参考口袋图','辅助模态','outputs/biomaster_odti_experimental_template_graph_features_v1/POCKET_GRAPH_INDEX_V1.csv.gz',
        326,'历史靶点模板','其中307个参考重对接PASS','局部教师输入','索引存在；数量依据9月14日',
        '参考配体模板并非当前候选已结合姿态','2026-09-14')
    add('LINCS L1000','辅助模态','data/external/lincs_cmap/level5_beta_trt_cp_n720216x12328.gctx',720216,'扰动签名',
        '12328维；化合物/细胞/条件相关','后续细胞效应模态','本地存在；数量依据9月14日',
        '不是720216药物，也不是药物—蛋白亲和标签','2026-09-14')
    tx=json.loads((KG/'TXGNN_RUN_COMPLETE.json').read_text())
    add('TxGNN 全疾病评分','图谱辅助',KG/'TXGNN_INDICATION_LOGITS.npy',tx['scores'],'预测药物—疾病对',
        f"{tx['drugs']}药×{tx['diseases']}疾病；全方向",'适应证假设','已有推理输出','模型预测，不是直接结合真值','2026-09-09')
    for name,file,note in [('OpenTargets 全方向','OPENTARGETS_ALL_TARGET_DISEASE_EVIDENCE.parquet','目标—疾病证据'),
        ('EC-KG 项目相关关系','ECKG_RELEVANT_RELATION_AUDIT.parquet','图谱关系与有效增量审计')]:
        path=KG/file;add(name,'图谱辅助',path,pq.read_metadata(path).num_rows,'证据行',note,'疾病/关系解释','本地存在',
            '不能直接替代Kd/Ki或实验失活','2026-09-09')
    add('七模型冻结分数快照','评价与实验',
        'outputs/model_interpretation_20260920/latest_results/SCORE_SNAPSHOT.csv.gz',276480,'核心目录配对',
        '720×384目录；该快照七模型完整共同矩形为50×719','排名分歧分析','已冻结快照','不是七模型全部覆盖；无记录不当阴性')
    add('SPR384冻结实验清单','评价与实验','outputs/spr384_final_experiment_table_20260910/SPR384_FINAL_EXPERIMENT_TABLE.csv',
        384,'候选配对','参考对照另计','湿实验接收与预选池评价','实验设计已冻结；结果访问状态未知',
        '清单不是384条实测标签')
    return write_csv('DATA_RESOURCE_LIST.csv',rows)


def weights():
    original=pd.read_csv(PREP/'WEIGHT_REGISTRY.csv').fillna('');rows=[];seen={}
    role_names={'task_weights':'任务检查点','internal_trained_weights':'自训检查点',
        'internal_trained_bundle':'自训模型目录','pretrained_encoder':'预训练编码器','model_config':'模型配置'}
    state_names={'DEPLOYED':'当前已用','EXPLORATORY_TRAINED':'历史已训练；测试已查看',
        'DEPLOYED_ENSEMBLE_MEMBER':'当前集成成员','WEIGHTS_AVAILABLE_INPUT_REQUIREMENTS_DIFFER':'有权重；分支输入适配待核验',
        'COLLECTED_NOT_CURRENT_CATALOG_QUALIFIED':'有权重；当前目录适配待核验',
        'COLLECTED_FIXED_TARGET_UNIVERSE':'有权重；固定目标库适用范围待核验','ENCODER_ONLY':'仅编码器'}
    for r in original.itertuples():
        path=ROOT/r.path;assert path.exists()
        if path.is_file():assert path.stat().st_size==r.bytes
        alias=seen.get(str(path.resolve()),'');seen.setdefault(str(path.resolve()),r.asset_id)
        rows.append({'范围':'核心DTI登记','资产编号':r.asset_id,'模型系列':r.family,'资产类型':role_names[r.role],
            '状态':state_names.get(r.qualification,r.qualification),'文件数':r.files,'大小MiB':round(r.bytes/2**20,3),
            '本地位置':local(path),'集中链接':local(r.registered_link),'SHA256':r.sha256,'哈希类型':r.digest_kind,
            '同一路径别名':alias,'官方获取地址':r.official_download_url,'许可备注':r.license_note,
            '核验口径':'9月20日冻结哈希登记；本轮复核存在/文件大小；模型目录未重复全量哈希'})
    auxiliary=[('txgnn_official','TxGNN','data/raw/txgnn/TxGNNExplorer/model.pt','疾病图谱任务；配套config.pkl和原图谱一起使用'),
        ('structural_teacher','结构预训练教师','outputs/biomaster_pocket_precision_20260906/structural_pretraining/cutoff_2020/seed_20260921/STRUCTURAL_PRETRAINED.pt',
         '几何任务教师；新训练数据对齐及部分编码缓存需恢复')]
    for aid,family,path,note in auxiliary:
        p=ROOT/path
        rows.append({'范围':'辅助路线另列','资产编号':aid,'模型系列':family,'资产类型':'辅助任务检查点','状态':note,
            '文件数':1,'大小MiB':round(p.stat().st_size/2**20,3),'本地位置':local(p),'集中链接':'',
            'SHA256':sha(p),'哈希类型':'file_sha256','同一路径别名':'','官方获取地址':'','许可备注':'沿原项目来源与许可记录',
            '核验口径':'本轮读取并计算SHA256；不冒称已验证本研究任务性能'})
    f=write_csv('WEIGHT_LIST.csv',rows)
    groups=[]
    for (scope,family,kind),part in f.groupby(['范围','模型系列','资产类型'],sort=False):
        groups.append({'范围':scope,'模型系列':family,'资产类型':kind,'登记条目':len(part),
            '不同路径数':part['本地位置'].nunique(),'大小MiB_按该组路径去重':round(part.drop_duplicates('本地位置')['大小MiB'].sum(),3),
            '状态':'；'.join(part['状态'].unique())})
    write_csv('WEIGHT_SUMMARY.csv',groups)
    return f


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    a=architectures();d=data_resources();w=weights()
    assert len(w)==117 and w['资产编号'].is_unique
    active=ROOT/'configs/dti_reliability_20260920/ACTIVE_PROTOCOL.json'
    current_note=''
    if active.exists():
        current_note='当前训练范围已按用户决定改为A主线：先30项共同输入/目标实验，主线含局部模型与冷靶点共45项，新增B为0。详见[A-only重训决策](BIOMASTER_DTI_A_ONLY_RETRAIN_PLAN_20260920_ZH.md)及[当前协议指针](../configs/dti_reliability_20260920/ACTIVE_PROTOCOL.json)。下列资产清单保留B历史资源，不表示继续安排B训练。'
    text=['# DTI研究架构、数据资源与已有权重清单','',
        '整理日期：2026-09-20。以已冻结的研究准备材料及本轮本地代码/文件核对为依据。新增训练状态仍为未开始；本清单不改动原冻结数据与协议。',
        '',current_note,
        '', '三个主表均为UTF-8 BOM CSV，可用Excel打开：', '',
        '- [架构列表](../outputs/dti_resource_catalog_20260920/ARCHITECTURE_LIST.csv)：输入、配对计算、输出、用途与实现状态。',
        '- [数据资源列表](../outputs/dti_resource_catalog_20260920/DATA_RESOURCE_LIST.csv)：来源、成员、辅助监督、特征、结构/细胞/图谱、实验清单。',
        '- [已有权重列表](../outputs/dti_resource_catalog_20260920/WEIGHT_LIST.csv)：逐资产路径、大小、SHA256、来源与当前状态。',
        '- [权重分类汇总](../outputs/dti_resource_catalog_20260920/WEIGHT_SUMMARY.csv)：按模型系列和资产类型统计。',
        '', '## 1. 架构列表', '']
    for category in a['类别'].unique():
        text += ['### '+category,'',table(a[a['类别'].eq(category)],['名称','输入','配对计算','输出','当前状态']),'']
    text += ['七模型实际启用分支依据[本地架构审计](BIOMASTER_SEVEN_MODEL_ARCHITECTURE_AUDIT_20260919_ZH.md)。',
        '现用七通道包括五个官方任务权重，以及本地ReTargetMap和DTIAM A；不能统称七个官方模型。训练集不同不妨碍用户场景比较，但限制架构归因及未见关系结论，详见[官方权重比较协议](BIOMASTER_OFFICIAL_WEIGHTS_COMPARISON_20260920_ZH.md)。',
        '扩展模型依据本地作者代码和官方说明：[TAPB](https://github.com/GaomingL1n/TAPB)、[SCOPE轻量版](https://github.com/Yigang-Chen/Lightweight-SCOPE-DTI-for-Inference)、[DrugBAN](https://github.com/peizhenbai/DrugBAN)。',
        'TAPB本地代码使用MolFormer tokenizer、独立分子Transformer和ESM2输入；已缓存MolFormer权重不表示已经训练出TAPB任务模型。',
        '', '## 2. 数据资源列表','',
        '以下资源存在派生、交集和重复来源关系，数量不可横向相加。母库观测行、任务标签行、唯一配对、结构系统、扰动签名、预测分数分别计数。', '']
    for group in d['层次'].unique():
        text += ['### '+group,'',table(d[d['层次'].eq(group)],['资源','规模','计数单位','用途','状态']),'']
    text += ['A/B原骨架训练分别337,570/1,116,270对；两臂各105,368个明确失活是同一批证据。研究宇宙1,386,225对包含训练、验证与测试。',
        '五种划分的完整数量见[DATASET_COUNTS.csv](../outputs/dti_research_preparation_20260920/DATASET_COUNTS.csv)。998,162行回归辅助与639,511行assay辅助必须按当前训练成员筛选。',
        '结构、token、模板和LINCS规模带有原审计日期；本轮确认入口存在，未重新逐项认证新版训练重叠。TxGNN/OpenTargets采用9月全方向结果，没有使用早期仅癌症的表冒充当前资源。',
        '', '## 3. 已有权重列表','',
        '**核心DTI登记仍为115项资产、222个唯一文件、约31.80 GiB；本清单另外列出TxGNN和结构预训练教师2项辅助检查点。117是登记条目数，包含配置、编码器、目录及别名，不是117个独立DTI模型。**','',
        table(pd.read_csv(OUT/'WEIGHT_SUMMARY.csv'),['范围','模型系列','资产类型','登记条目','不同路径数','状态']),'',
        '核心权重集中链接：`data/research/dti_reliability_20260920_v1/model_assets/`。直接使用逐行路径，不复制大权重。',
        'ReTargetMap生产文件也出现在Palinova历史登记中，已用“同一路径别名”字段标记，不能重复算一次训练。61项Palinova历史登记包括60次A/B拟合和1项生产旧基线；DTIAM另有6套A/B模型。',
        'DTIAM一个predictor目录内有多个学习器，不能将其每个文件都算独立实验；Nesso一组主干/双输出不重复登记为两个模型；DTBind三种任务检查点则分别列出。',
        'SCOPE的25个检查点按GPCR、IC、Kinase、NHR及Total五组各5个组织，任务适用范围仍需验证。TAPB、DrugBAN、DeepDTA未确认的合格任务权重不虚构为已有；新受控架构和JEPA学生当前也没有新权重。',
        '本表延续原公开来源/许可记录；文件存在和哈希登记不等于官方示例复现、独立测试成立或允许再分发。',
        '', '## 4. 使用顺序','',
        current_note or '先用BerMol＋统一ESM2和A原骨架成员完成21次受控架构对照，复用历史A/B及DTIAM结果；后续按有效协议推进。',
        '当前执行范围见[A主线方案](BIOMASTER_DTI_A_ONLY_RETRAIN_PLAN_20260920_ZH.md)，现成模型按[官方权重比较协议](BIOMASTER_OFFICIAL_WEIGHTS_COMPARISON_20260920_ZH.md)。[v1准备报告](BIOMASTER_DTI_RESEARCH_PREPARATION_20260920_ZH.md)及[v1协议](protocols/DTI_RELIABILITY_PROTOCOL_20260920_ZH.md)保留冻结历史；其中87项旧队列已由A主线替代。',
        '', '重建本清单：`OPENBLAS_NUM_THREADS=1 .venvs/frontier_dti/bin/python scripts/build_dti_resource_catalog_20260920.py`。该命令整理清单、复核路径，不训练或部署模型。','']
    DOC.write_text('\n'.join(text))
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'architecture_rows':len(a),'data_resource_rows':len(d),
        'weight_rows':len(w),'core_weight_rows':int(w['范围'].eq('核心DTI登记').sum()),'auxiliary_weight_rows':2,
        'core_weight_path_aliases':int(w['同一路径别名'].ne('').sum()),'path_checks':'PASS',
        'new_training_started':False,'upstream_protocol_unchanged':True,
        'inputs':{str(p.relative_to(ROOT)):sha(p) for p in [PREP/'WEIGHT_REGISTRY.csv',PREP/'DATASET_COUNTS.csv',PREP/'FEATURE_REGISTRY.json',ROOT/'configs/dti_reliability_20260920/PROTOCOL.json']},
        'outputs':{str(p.relative_to(ROOT)):sha(p) for p in [*OUT.glob('*.csv'),DOC]},
        'producer_sha256':sha(Path(__file__))}
    if active.exists():
        result['inputs'][str(active.relative_to(ROOT))]=sha(active)
        active_config=json.loads(active.read_text())
        for key in ['active_protocol','inference_comparison_addendum']:
            if key in active_config:
                result['inputs'][active_config[key]]=sha(ROOT/active_config[key])
    (OUT/'SUMMARY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['architecture_rows','data_resource_rows','weight_rows','core_weight_path_aliases','path_checks']},indent=2))


if __name__=='__main__':main()
