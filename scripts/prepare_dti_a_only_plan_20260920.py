#!/usr/bin/env python3
"""Record the user-directed A-only study amendment; do not start model training."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'configs/dti_reliability_20260920/PROTOCOL.json'
CONFIG=ROOT/'configs/dti_reliability_A_20260920_v2'
OUT=ROOT/'outputs/dti_reliability_A_plan_20260920'
PREP=ROOT/'outputs/dti_research_preparation_20260920'
DATA='data/research/dti_reliability_20260920_v1'
SEEDS=[101,202,303]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def csv(name,rows):
    f=pd.DataFrame(rows);f.to_csv(OUT/name,index=False,encoding='utf-8-sig');return f


def build():
    OUT.mkdir(parents=True,exist_ok=True);CONFIG.mkdir(parents=True,exist_ok=True)
    old=json.loads(BASE.read_text());protocol=copy.deepcopy(old)
    protocol.update(protocol_id='dti_reliability_A_20260920_v2',
        status='A_ONLY_RECOMMENDED_PLAN_NOT_STARTED_NOT_EXTERNALLY_PREREGISTERED',
        created_utc=datetime.now(timezone.utc).isoformat(),parent_protocol=str(BASE.relative_to(ROOT)),parent_sha256=sha(BASE),
        user_direction='Use A for the main training/validation/test study. No large-scale new B training.',
        objective='Explain pair selectivity and objective-induced disagreement on A, with targeted cold-target checks and existing public models.',
        allowed_new_training_arms=['A'],new_B_training_runs=0)
    protocol['labels'].pop('arm_B',None)
    protocol['labels']['regression']='Separate exact Kd and Ki heads only. Inactivity has no fabricated numeric affinity.'
    protocol['inputs']['scope']='The seven common-input controls share BerMol/ESM2. Native DrugBAN is a separately labeled input-and-architecture comparison.'
    protocol['architectures']['drugban_native']={
        'inputs':['molecular_2d_graph','protein_sequence_tokens'],'implementation':'third_party/DrugBAN/models.py',
        'model':'MolecularGCN + ProteinCNN + BANLayer + MLPDecoder','training':'Fresh full task model; vanilla supervised DrugBAN, no domain adaptation.',
        'domain_adaptation':False,'use_test_inputs_for_training':False,
        'claim':'Native-input system comparison. Cannot isolate local attention from encoder, input and training differences.',
        'preflight_required':['Map exact A molecule/sequence identities','Audit graph/token coverage and truncation',
            'Native forward/gradient smoke test','Profile 200 updates and full validation','Freeze model-specific implementation settings before fitting'],
        'hyperparameter_status':'Adapter and exact model-specific training settings pending; not execution-ready.'}
    protocol['inference_track']['scope']='Reuse frozen public/production checkpoints and historical A/B fits for exploratory references; no new B fits.'
    protocol['selection_protocol']['data_contrasts']='Large A/B matched/nested retraining deferred. Existing B results retained as historical supplementation, not proof of an endpoint-noise cause.'
    protocol['selection_protocol']['existing_dtiam_A']='Reuse three historical fits for common-member re-evaluation. Training protocol differs, so exclude them from strict same-budget architecture attribution.'
    protocol['selection_protocol']['cold_target']='Fresh supervised fits on A_cold_target; no reuse of original scaffold-trained task checkpoints as unseen-target models.'
    protocol['losses']['assay_scope']='Use only A membership and Kd/Ki rows. Original A scaffold train has 192469 exact endpoint rows; current min5 filtered assay rows are reported in counts.'
    protocol['statistical_plan']['primary_family']='Scaffold: global_mlp vs additive/dual_cosine/lightgbm, two directions (6); cold-target: global_mlp vs additive/lightgbm, two directions (4). Holm correction across 10 primary tests.'
    protocol['statistical_plan']['local_model']='DrugBAN native comparison is secondary, with matched evaluation members and explicit input/truncation differences.'
    protocol['execution'].update(stages=['A1 common-input architecture controls:21','A2 A-only objectives:9',
        'A3 native DrugBAN scaffold:3 after adapter qualification','A4 prespecified cold-target representatives:12'],
        new_training_started=False,training_runner_status='Planning amendment only; common trainer and DrugBAN adapter still need implementation.',
        first_stage_fit_count=30,main_fit_count=45,conditional_fit_count=15,
        conditional_rule='Source-purged, ReLU and TAPB hypotheses must be activated/frozen before examining their corresponding test results; they do not enter the default run queue.',
        new_training_eta='Not profiled. Counts are runs, not a claim of equal compute or duration.')
    save(CONFIG/'PROTOCOL.json',protocol)
    rows=[]
    def add(stage,view,arch,loss,seed,scope='main',variant='default',gate='Common-input trainer and speed/validation preflight'):
        run=f'A2_{stage}_{view}_{arch}_{loss}_{variant}_s{seed}'
        rows.append(dict(run_id=run,stage=stage,plan_scope=scope,arm='A',view=view,architecture=arch,loss=loss,seed=seed,
            variant=variant,feature_set='molecular_graph+sequence' if arch=='drugban_native' else ('TAPB_native' if arch.startswith('tapb') else 'bermol+esm2_dtiam'),
            train_members=f'{DATA}/A_{view}_train.parquet',validation_members=f'{DATA}/A_{view}_validation.parquet',
            test_members=f'{DATA}/A_{view}_test.parquet',status='PLANNED_NOT_STARTED' if scope=='main' else 'CONDITIONAL_NOT_QUEUED',
            requires=gate,output_dir=f'outputs/dti_reliability_A_runs_20260920/{run}',
            test_use='Final retrospective evaluation only; no test-based checkpoint or gate selection'))
    for arch in old['architectures']:
        for seed in SEEDS:add('A1','scaffold_replay',arch,'binary',seed)
    for loss in ['binary_rank','binary_regression','binary_both']:
        for seed in SEEDS:add('A2','scaffold_replay','global_mlp',loss,seed)
    drugban_gate='Exact graph/sequence coverage + native parity + non-DA adapter + model-specific config freeze + resource profile'
    for seed in SEEDS:add('A3','scaffold_replay','drugban_native','binary',seed,gate=drugban_gate)
    for arch in ['additive','global_mlp','lightgbm','drugban_native']:
        for seed in SEEDS:add('A4','cold_target',arch,'binary',seed,gate=drugban_gate if arch=='drugban_native' else 'Fresh cold-target fit; common-input trainer')
    for arch in ['additive','global_mlp']:
        for seed in SEEDS:add('C1','source_purged',arch,'binary',seed,'conditional',gate='Activate for source-dependence question before this test is viewed; account for reduced training size')
    for seed in SEEDS:add('C2','scaffold_replay','dual_cosine','binary',seed,'conditional','relu_projection',
        gate='Matched activation mechanism study; does not reproduce full native ConPLex')
    for intervention in ['enabled','disabled']:
        for seed in SEEDS:add('C3','scaffold_replay','tapb_native','binary',seed,'conditional',intervention,
            gate='Freeze target-prior hypothesis on development data; matched with/without-intervention pair; feature/config parity before training')
    all_runs=pd.DataFrame(rows);assert all_runs.run_id.is_unique and all_runs.arm.eq('A').all()
    main=csv('EXPERIMENT_MATRIX_A.csv',all_runs[all_runs.plan_scope.eq('main')]);cond=csv('CONDITIONAL_EXPERIMENTS_A.csv',all_runs[all_runs.plan_scope.eq('conditional')])
    assert len(main)==45 and len(cond)==15
    # Materialized members and initial labels are unchanged; inspect metadata only.
    manifest=json.loads((PREP/'DATA_MANIFEST.json').read_text())
    for path in set(all_runs.train_members)|set(all_runs.validation_members)|set(all_runs.test_members):
        assert path in manifest['prepared'] and (ROOT/path).exists(),path
        assert sha(ROOT/path)==manifest['prepared'][path]['sha256'],path
    counts=pd.read_csv(PREP/'DATASET_COUNTS.csv')
    counts=counts[counts.arm.eq('A') & counts.view.isin(['scaffold_replay','cold_target','source_purged']) & counts.split.ne('excluded')]
    csv('A_DATA_COUNTS.csv',counts)
    choices=[]
    def choice(name,action,why,core=0,optional=0,notes=''):
        choices.append({'模型或对照':name,'处置':action,'主线新拟合数':core,'条件拟合数':optional,'目的或理由':why,'解释边界':notes})
    choice('药物单侧/蛋白单侧','新训', '检验实体偏好；各3seed原骨架',6)
    choice('两侧加性','新训','无配对交互基线；原骨架和冷靶点各3seed',6)
    choice('双塔余弦（共同输入）','新训','匹配形式对照；GELU主线3seed；ReLU条件3seed',3,3,'共享BerMol/ESM2的双塔，不命名为完整原版ConPLex')
    choice('低秩双线性','新训','显式乘性交互参照；原骨架3seed',3)
    choice('Palinova式共同输入MLP','新训','原骨架4目标×3seed；冷靶点二分类3seed',15,
        notes='输入与维度按控制协议统一；不是旧生产权重的简单继续训练')
    choice('LightGBM（共同输入）','新训','强树学习器；原骨架和冷靶点各3seed',6,notes='单一树模型，不冒充完整DTIAM集成')
    choice('DrugBAN','适配后新训','局部交互代表；原骨架和冷靶点各3seed',6,
        notes='保留原2D图/序列路线，不启用使用测试数据的域适配；输入差异单列')
    choice('DTIAM A','复用已有3套','A原骨架3seed已完成；统一成员/指标重评',notes='历史选模/预算不同；若要报告DTIAM独立冷靶点成绩则需另行重训')
    choice('旧ReTargetMap / Palinova A','复用历史权重','保留生产及已完成A消融证据',notes='不将旧native输入和新共享输入的差别全归因于架构')
    choice('ConPLex官方权重','复用推理','与新共享输入双塔共同构成诊断',notes='原Morgan/ProtBERT与共同输入是不同系统')
    choice('DrugCLIP / Nesso','复用推理','保留口袋检索及原生双头；本轮不全量训练结构模型',notes='A标签不会自动补齐其训练所需结构/预处理')
    choice('ProbeMatchDTI / DTBind','复用推理；暂缓原版重训','首轮局部交互由DrugBAN承担；若要具体归因这两者还需单独训练',notes='不根据DrugBAN结果概括所有注意力/GNN模型')
    choice('EviDTI / SCOPE','先验证已有权重','先解决原生复现和A输入/目标覆盖',notes='有文件不代表统一A任务已经可比')
    choice('TAPB','条件升级','需要验证先验去偏时，同时训练有/无干预各3seed',optional=6,
        notes='不与全局MLP的一次比较直接当作去偏机制证明')
    choice('来源去重敏感性','条件升级','加性与共同输入MLP各3seed',optional=6,
        notes='训练规模变化也影响成绩，不能直接当成泄漏贡献')
    choice('DeepDTA / JEPA / TxGNN','不列新增DTI主线','DeepDTA暂缓；JEPA独立方法线；TxGNN保留疾病推理')
    choice('全部新增B及A/B混合扩量拟合','延期','遵循用户A为主线的决定；保留既有B数据、权重和结果',notes='不再把大规模B扩训作为得到本研究结论的前提')
    decisions=csv('MODEL_RETRAIN_DECISIONS.csv',choices)
    assert decisions['主线新拟合数'].sum()==45 and decisions['条件拟合数'].sum()==15
    old_runs=pd.read_csv(PREP/'EXPERIMENT_MATRIX.csv')
    csv('DEFERRED_B_AND_AB_RUNS.csv',old_runs[old_runs.arm.isin(['B','AB'])].assign(disposition='DEFERRED_BY_A_ONLY_USER_SCOPE'))
    # An additive pointer chooses the effective version without rewriting the frozen v1 package.
    save(ROOT/'configs/dti_reliability_20260920/ACTIVE_PROTOCOL.json',dict(
        active_protocol=str((CONFIG/'PROTOCOL.json').relative_to(ROOT)),active_protocol_sha256=sha(CONFIG/'PROTOCOL.json'),
        active_main_runs=str((OUT/'EXPERIMENT_MATRIX_A.csv').relative_to(ROOT)),main_runs_sha256=sha(OUT/'EXPERIMENT_MATRIX_A.csv'),
        conditional_runs=str((OUT/'CONDITIONAL_EXPERIMENTS_A.csv').relative_to(ROOT)),
        superseded_run_registry='outputs/dti_research_preparation_20260920/EXPERIMENT_MATRIX.csv',
        scope='A only; default excludes conditional experiments',new_training_started=False))
    save(OUT/'SUMMARY.json',dict(status='A_ONLY_PLAN_PREPARED_NOT_TRAINING',parent_protocol_sha256=sha(BASE),
        main_fits=len(main),first_phase_fits=30,conditional_fits=len(cond),new_B_fits=0,
        deferred_previous_B_AB_fits=int(old_runs.arm.isin(['B','AB']).sum()),
        main_by_stage=main.groupby('stage').size().to_dict(),all_members_in_existing_frozen_manifest=True,
        training_runner_still_required=True,native_adapter_still_required=True,existing_dtiam_A_fits=3,
        source_implementation_checks=['third_party/DrugBAN/models.py','third_party/sota_dti_2026/TAPB/models/tapb.py'],
        limitations=['Retrospective labels previously viewed','A does not represent all functional endpoints',
            'Measured drug-query coverage remains sparse','Native DrugBAN changes representation as well as architecture'],
        producer_sha256=sha(Path(__file__))))
    print((OUT/'SUMMARY.json').read_text())


if __name__=='__main__':build()
