from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(".")
OUT = ROOT / "outputs" / "report_scale" / "biomaster_full_paper_zh_cn.md"


def load_json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def csv_rows(path: str) -> int:
    with (ROOT / path).open(newline="", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def read_csv(path: str) -> list[dict[str, str]]:
    with (ROOT / path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compact_float(value: str, digits: int = 3) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return value


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(value.replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def main() -> int:
    drug_summary = {
        "rows": csv_rows("data/processed/drug_library_pubchem_chembl_mapped.csv"),
        "path": "data/processed/drug_library_pubchem_chembl_mapped.csv",
    }
    receptor_meta = load_json("data/processed/alphafold_receptor_manifest.metadata.json")
    opentargets_meta = load_json("data/processed/opentargets_target_disease_scores.metadata.json")
    string_meta = load_json("data/processed/string_human_filtered_edges.metadata.json")
    txgnn_meta = load_json("data/processed/txgnn_drug_disease_scores.metadata.json")
    stage5_ot_string_meta = load_json("outputs/report_scale/stage5_open_targets_string_ranked_candidates_915k.metadata.json")
    stage5_txgnn_meta = load_json("outputs/report_scale/stage5_txgnn_open_targets_string_ranked_candidates_915k.metadata.json")
    top1000_summary = load_json("outputs/report_scale/stage5_top1000_summary.json")
    top5000_summary = load_json("outputs/report_scale/stage5_top5000_summary.json")
    top10000_summary = load_json("outputs/report_scale/stage5_top10000_summary.json")
    stage6_meta = load_json("outputs/report_scale/stage6_top1000_consensus_candidates.metadata.json")
    stage6_summary = load_json("outputs/report_scale/stage6_top1000_report_summary.json")

    top_candidates = read_csv("outputs/report_scale/stage6_top100_consensus_candidates.csv")[:20]
    full_pid = ""
    pid_path = ROOT / "outputs/report_scale/diffdock_full_run/full_queue.pid"
    if pid_path.exists():
        full_pid = pid_path.read_text(encoding="utf-8").strip()
    full_score_files = len(list((ROOT / "outputs/report_scale/diffdock_full_run/scores").glob("*.scores.csv")))

    top_table = markdown_table(
        ["Stage6 rank", "Drug", "Target", "Stage5", "DiffDock confidence", "Stage6", "Status"],
        [
            [
                row.get("stage6_rank", ""),
                row.get("drug_name", ""),
                row.get("gene_name", ""),
                compact_float(row.get("stage5_final_priority_score", "")),
                compact_float(row.get("diffdock_confidence", ""), digits=2),
                compact_float(row.get("stage6_consensus_score", "")),
                row.get("structural_status", ""),
            ]
            for row in top_candidates
        ],
    )

    step_table = markdown_table(
        ["步骤", "目标", "关键真实输入", "主要产物", "当前结论"],
        [
            [
                "Step 1",
                "建立 FDA 已批准小分子库并补齐结构/ID",
                "FDA_approved_small_molecules_2005_2026_with_structures.xls；ChEMBL；PubChem",
                drug_summary["path"],
                f"{drug_summary['rows']} 个药物；SDF/SMILES/InChIKey/分子式均已补齐，PubChem CID 913/{drug_summary['rows']}，DrugBank ID 仍为 0/{drug_summary['rows']}",
            ],
            [
                "Step 2",
                "建立 1000 蛋白靶点与受体结构库",
                "UP000005640_9606_HUMAN_v6.tar；AlphaFold human proteome v6",
                "data/processed/protein_library_1000_alphafold_paths.csv",
                f"{receptor_meta['protein_rows']} 个蛋白；{receptor_meta['proteins_with_pdb_path']} 个有 PDB/CIF；缺失蛋白 {', '.join(receptor_meta['missing_proteins'])}",
            ],
            [
                "Step 3",
                "完成全量药物-蛋白亲和力预测",
                "915 个药物 × 1000 个蛋白",
                "outputs/report_scale/conplex_affinity_scores_915k.csv",
                "915000 个 pair 已完成 ConPLex 预测；这是当前亲和力层的主结果",
            ],
            [
                "Step 4",
                "形成结构可用和 AI 综合排序底座",
                "ConPLex 分数；受体 PDB；配体 SDF",
                "outputs/report_scale/stage4_affinity_candidates_915k.csv；outputs/report_scale/manifest_915k_diffdock_ready.csv",
                f"Stage4 全量 {csv_rows('outputs/report_scale/stage4_affinity_candidates_915k.csv')} 行；DiffDock-ready pair 为 {csv_rows('outputs/report_scale/manifest_915k_diffdock_ready.csv')}；全量 DiffDock 未作为 Step 1-5 完成条件",
            ],
            [
                "Step 5",
                "接入疾病证据并得到最终疾病优先级",
                "Open Targets；STRING v12.0 API；TxGNN",
                "outputs/report_scale/stage5_txgnn_open_targets_string_ranked_candidates_915k.csv",
                f"{stage5_txgnn_meta['rows']} 个 pair 完成疾病相关性排序；TxGNN 映射 {txgnn_meta['crosswalk']['mapped_rows']}/{txgnn_meta['crosswalk']['drug_rows']} 个药物",
            ],
        ],
    )

    topn_table = markdown_table(
        ["候选集合", "pair 数", "药物数", "蛋白数", "疾病证据", "TxGNN 映射"],
        [
            [
                "Top1000",
                str(top1000_summary["rows"]),
                str(top1000_summary["unique_drugs"]),
                str(top1000_summary["unique_proteins"]),
                json.dumps(top1000_summary["disease_evidence_status"], ensure_ascii=False),
                json.dumps(top1000_summary["txgnn_component_status"], ensure_ascii=False),
            ],
            [
                "Top5000",
                str(top5000_summary["rows"]),
                str(top5000_summary["unique_drugs"]),
                str(top5000_summary["unique_proteins"]),
                json.dumps(top5000_summary["disease_evidence_status"], ensure_ascii=False),
                json.dumps(top5000_summary["txgnn_component_status"], ensure_ascii=False),
            ],
            [
                "Top10000",
                str(top10000_summary["rows"]),
                str(top10000_summary["unique_drugs"]),
                str(top10000_summary["unique_proteins"]),
                json.dumps(top10000_summary["disease_evidence_status"], ensure_ascii=False),
                json.dumps(top10000_summary["txgnn_component_status"], ensure_ascii=False),
            ],
        ],
    )

    text = f"""# 基于 FDA 已批准小分子、疾病知识图谱与结构对接的肿瘤药物再定位候选筛选

生成时间：{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}

本稿定位：中文论文初稿兼项目交付报告。本文只陈述本地已有文件和已验证运行结果；未完成的全量 DiffDock 不写作已完成结果。

## 摘要

目的：构建一个可复现的多阶段计算筛选流程，从 915 个 FDA 已批准小分子和 1000 个候选人类蛋白靶点中识别与肿瘤相关的高优先级药物-靶点组合，并为后续机制分析和实验验证提供候选分层。

方法：首先对 FDA 已批准小分子进行结构和标识符补全，并使用 ChEMBL/PubChem 补齐 SMILES、InChIKey、分子式和 SDF。随后基于 AlphaFold human proteome v6 建立 1000 个蛋白的受体结构库。第三步使用 ConPLex 对 915000 个药物-蛋白 pair 进行亲和力预测。第四步形成结构可用的全量 pair manifest 和 DiffDock-ready manifest。第五步接入 Open Targets、STRING v12.0 API 和 TxGNN，将药物-蛋白亲和力、肿瘤疾病关联、蛋白网络邻近性和药物-疾病图模型分数合并为 Stage 5 疾病优先级。当前 Step 6 对 Stage 5 Top1000 候选进行 DiffDock 结构对接增强，并按 0.85 × Stage5 分数 + 0.15 × 归一化 DiffDock confidence 形成 Stage 6 共识排序。

结果：Step 1-5 主筛选已完成。药物库包含 {drug_summary['rows']} 个小分子，SDF、InChIKey 和分子式均为 {drug_summary['rows']}/{drug_summary['rows']}，PubChem CID 为 913/{drug_summary['rows']}。1000 个蛋白中 {receptor_meta['proteins_with_pdb_path']} 个获得 PDB/CIF 结构路径。ConPLex 完成 915000 个 pair 的亲和力预测。Open Targets 和 STRING 证据接入后，Stage 5 全量排序覆盖 {stage5_txgnn_meta['rows']} 个 pair，其中 TxGNN 药物映射覆盖 {stage5_txgnn_meta['rows_with_txgnn_score']} 个 pair。Stage 5 Top1000 包含 {top1000_summary['unique_drugs']} 个药物和 {top1000_summary['unique_proteins']} 个蛋白，全部具备直接和网络疾病证据。Top1000 DiffDock SMILES 运行完成 {stage6_meta['diffdock_completed']}/{stage6_meta['rows']} 个 pair，失败或缺失 {stage6_summary['diffdock_missing_or_failed']} 个 pair。Stage 6 共识排序的前列候选包括 Afatinib-EGFR、Dacomitinib-EGFR、Momelotinib-EGFR、Repotrectinib-JAK1、Pazopanib-KIT、Osimertinib-EGFR 等。

结论：当前数据足以进入 Step 7 机制解释和 Step 8 实验验证设计。全量 DiffDock 可作为结构证据增强继续后台运行，但不是继续 Step 7-8 的硬前置条件。论文和对外汇报中应严格表述为“Step 1-5 全量主筛选完成，Top1000 完成结构对接增强；全量 DiffDock 尚未完成”。

关键词：药物再定位；FDA 已批准药物；ConPLex；Open Targets；STRING；TxGNN；DiffDock；肿瘤；结构对接

## 1. 研究背景

药物再定位的核心优势在于利用已有批准药物的安全性、药代动力学和临床经验，缩短从计算假设到实验验证的周期。单一的亲和力预测或单一疾病数据库排序通常不足以支持候选优先级，因为高亲和力不等于疾病相关，高疾病相关也不等于存在合理结合构象。因此，本项目采用多阶段集成策略：先用全量深度学习 DTI 模型建立药物-蛋白相互作用底座，再用真实疾病数据库和蛋白相互作用网络约束疾病相关性，最后对高优先级候选进行结构对接增强。

本轮研究以肿瘤作为主要疾病场景。Open Targets 中使用 MONDO_0004992/cancer 作为主疾病条目，EFO_0000616/neoplasm 作为广义比较项；STRING 使用人类物种 9606 和高置信阈值；TxGNN 使用癌症疾病节点进行药物-疾病图模型评分。该设计目的不是一次性证明药物疗效，而是形成一份可审计、可复现、可进入实验设计的候选清单。

## 2. 五步走主筛选结果

{step_table}

需要特别说明：全量 pair 的亲和力预测和疾病相关性排序已经完成；DiffDock 属于结构对接增强层，不应被混同为 ConPLex 亲和力或真实结合自由能。全量 DiffDock 不是 Step 1-5 的完成条件。

## 3. 数据来源与真实接入状态

### 3.1 药物库

原始药物文件为 `/root/autodl-tmp/BioMaster/FDA_approved_small_molecules_2005_2026_with_structures.xls`。处理后的主文件为 `data/processed/drug_library_pubchem_chembl_mapped.csv`，包含 {drug_summary['rows']} 个小分子。当前结构来源为 ChEMBL API molfile，所有药物均有 SDF path、InChIKey 和分子式；PubChem CID 覆盖 913 个药物。DrugBank ID 当前仍为 0，因此后续若需要药物说明书、ATC、靶点药理和临床分期的强证据，应单独接入 DrugBank 或使用 Open Targets/ChEMBL/DrugCentral 等可替代来源。

### 3.2 蛋白和受体结构

蛋白库主文件为 `outputs/report_scale/protein_library_1000.csv`。AlphaFold 输入为 `/root/autodl-tmp/BioMaster/UP000005640_9606_HUMAN_v6.tar`，处理输出为 `data/processed/protein_library_1000_alphafold_paths.csv`。1000 个蛋白中 {receptor_meta['proteins_with_pdb_path']} 个具有 PDB/CIF 路径，缺失蛋白为 {', '.join(receptor_meta['missing_proteins'])}。结构就绪 pair 为 {receptor_meta['structure_ready_rows']}，即 913170/915000。

### 3.3 Open Targets

Open Targets 真实接入路径为 `data/processed/opentargets_target_disease_scores.csv`，元数据见 `data/processed/opentargets_target_disease_scores.metadata.json`。接口为 `{opentargets_meta['graphql_url']}`。主疾病为 MONDO_0004992/cancer，Open Targets 返回的 cancer 关联靶点中匹配本筛选蛋白库 {opentargets_meta['disease_stats'][0]['matched_screening_targets']} 个；neoplasm 比较项匹配 {opentargets_meta['disease_stats'][1]['matched_screening_targets']} 个。输出行为 {opentargets_meta['output_rows']}。

### 3.4 STRING

STRING 使用 v12.0 官方 API，不下载完整 human 全量文件。网络文件为 `data/processed/string_human_filtered_edges.csv`，元数据为 `data/processed/string_human_filtered_edges.metadata.json`。使用 species 9606、required_score {string_meta['required_score']}、partner_limit {string_meta['partner_limit']}，疾病种子基因为 {', '.join(string_meta['disease_seed_genes'])}。输出合并边数为 {string_meta['merged_edge_rows']}。这满足当前 Stage 5 的“疾病相关高置信子网”要求；完整 STRING 离线文件可以作为后续复现增强，不是当前阻塞项。

### 3.5 TxGNN

TxGNN 本地目录为 `data/raw/txgnn/TxGNNExplorer`，输出文件为 `data/processed/txgnn_drug_disease_scores.csv`。药物名称采用保守 exact normalization 与 TxGNN drug 节点匹配，不使用模糊匹配。915 个药物中 {txgnn_meta['crosswalk']['mapped_rows']} 个映射成功，{txgnn_meta['crosswalk']['unmapped_rows']} 个未映射。TxGNN 分数为药物-疾病层分数，被应用到同一药物的全部蛋白 pair；缺失 TxGNN 映射不作为负证据。

## 4. 计算流程

### 4.1 ConPLex 亲和力预测

ConPLex 输出为 `outputs/report_scale/conplex_affinity_scores_915k.csv`，共 {csv_rows('outputs/report_scale/conplex_affinity_scores_915k.csv')} 个 pair。该结果构成全量亲和力层。Stage 4 文件 `outputs/report_scale/stage4_affinity_candidates_915k.csv` 将 ConPLex 亲和力分数归一化为 `affinity_component` 和 `combined_ai_score`。当前 Stage 4 中 DiffDock 字段主要作为结构对接结果的预留字段，不能解释为全量 docking 已完成。

### 4.2 Open Targets + STRING 疾病优先级

Stage 5 第一层将 Stage 4 的 AI 分数、Open Targets 直接疾病关联和 STRING 网络传播证据合并。使用的公式为：

`OpenTargets_STRING_priority = 0.55 * normalized_combined_ai_score + 0.30 * OpenTargets_direct_score + 0.15 * STRING_network_score`

该层输出为 `outputs/report_scale/stage5_open_targets_string_ranked_candidates_915k.csv`。全量 {stage5_ot_string_meta['rows']} 个 pair 中，疾病证据状态为：{json.dumps(stage5_ot_string_meta['evidence_status_counts'], ensure_ascii=False)}。

### 4.3 TxGNN 融合

Stage 5 第二层加入 TxGNN 药物-癌症分数。映射药物使用：

`Stage5_final = 0.80 * OpenTargets_STRING_priority + 0.20 * TxGNN_indication_score`

未映射药物保留 OpenTargets/STRING 分数，不降权。最终输出为 `outputs/report_scale/stage5_txgnn_open_targets_string_ranked_candidates_915k.csv`，共 {stage5_txgnn_meta['rows']} 行；其中 {stage5_txgnn_meta['rows_with_txgnn_score']} 行有 TxGNN 分数，{stage5_txgnn_meta['rows_without_txgnn_score']} 行没有 TxGNN 分数。

### 4.4 Top-N 结构对接队列

从 Stage 5 全量排序中导出 Top1000、Top5000 和 Top10000 的 DiffDock-ready manifest：

{topn_table}

Top1000 使用 SMILES 作为配体输入进行 DiffDock，原因是早期 SDF 输入暴露出部分高优先级激酶抑制剂的构象/解析失败。SMILES 运行结果更稳定，完成率为 {stage6_meta['diffdock_completed']}/{stage6_meta['rows']}。

### 4.5 Stage 6 共识排序

Stage 6 对 Top1000 融合 Stage 5 疾病优先级和 DiffDock confidence：

`Stage6_consensus = 0.85 * Stage5_final_priority_score + 0.15 * normalized_DiffDock_confidence`

当 DiffDock 输出缺失时，当前实现保留 Stage 5 分数并将 `structural_status` 标记为 `missing_output`。这避免把工程失败误当成生物学负证据，但进入最终候选前必须重跑这些缺失项。

## 5. 结果

### 5.1 主筛选规模

本项目完成了 915 个 FDA 已批准小分子与 1000 个蛋白靶点的全量计算筛选。ConPLex 亲和力层和 Stage 5 疾病优先级层均覆盖 915000 个 pair。结构就绪和 DiffDock-ready pair 为 913170，少于全量 pair 的原因是 2 个蛋白缺失可用受体结构。

### 5.2 疾病证据覆盖

Open Targets cancer 直接证据覆盖 988 个本筛选蛋白，STRING 高置信子网传播得到 1053 个网络相关基因。Stage 5 Open Targets/STRING 层的全量证据分布为 direct_and_network 666120、direct 234240、network 915、none 13725。加入 TxGNN 后，738 个药物获得药物-癌症图模型分数，对应 738000 个 pair；177000 个 pair 因药物未映射到 TxGNN 而保留 Open Targets/STRING 分数。

### 5.3 Top1000 结构增强结果

Top1000 候选覆盖 {top1000_summary['unique_drugs']} 个药物、{top1000_summary['unique_proteins']} 个蛋白。疾病证据均为 direct_and_network。TxGNN 组件中 mapped 为 {top1000_summary['txgnn_component_status'].get('mapped', 0)}，unmapped 为 {top1000_summary['txgnn_component_status'].get('unmapped', 0)}。受体状态中 EGFR kinase-domain crop 为 {top1000_summary['diffdock_receptor_status'].get('curated_EGFR_kinase-domain_crop', 0)}，JAK1 kinase-domain crop 为 {top1000_summary['diffdock_receptor_status'].get('curated_JAK1_kinase-domain_crop', 0)}，full_length_ok 为 {top1000_summary['diffdock_receptor_status'].get('full_length_ok', 0)}，curated_full 为 {top1000_summary['diffdock_receptor_status'].get('curated_full', 0)}。

DiffDock Top1000 SMILES 运行产生 4 个 score 文件，完成 {stage6_meta['diffdock_completed']} 个 pair，缺失 {stage6_summary['diffdock_missing_or_failed']} 个 pair。DiffDock confidence 范围为 {stage6_meta['confidence_min']} 至 {stage6_meta['confidence_max']}。失败审计文件为 `outputs/report_scale/diffdock_top1000_smiles_failure_audit.csv`。

### 5.4 Stage 6 前 20 个候选

{top_table}

前列候选集中出现多个已知肿瘤相关靶点和药物类别，例如 EGFR 抑制剂、JAK/激酶相关候选和 KIT 相关候选。这说明流程能富集合理的阳性对照式信号，但也提示 Step 7 必须区分“已知适应症/已知机制确认”和“潜在再定位新假设”，不能仅凭排名声称新发现。

## 6. 讨论

本项目的主要价值在于把三个证据层统一到同一个 pair 级别排序表：第一层是药物-蛋白亲和力预测，第二层是肿瘤疾病相关性，第三层是高优先级候选的结构对接增强。相比只按 DTI 分数排序，本流程能够过滤掉缺乏疾病证据的高亲和力 pair；相比只按疾病数据库排序，本流程又保留了药物-靶点相互作用的计算证据；相比直接全量 docking，本流程先用疾病和 DTI 分数缩小结构对接规模，计算上更可执行。

当前 Top1000 结果已经足以进入机制分析和实验设计。原因有三点。第一，全量 ConPLex 和 Stage 5 排序已经完成，候选优先级底座不是抽样结果。第二，Top1000 全部具备直接和网络疾病证据，说明疾病关联层足够强。第三，Top1000 中 940 个 pair 已经获得 DiffDock confidence，可支持结构增强排序和结构可视化。但全量 DiffDock 尚未完成，因此全文不能表述为全量结构验证完成。

## 7. 局限性

1. DiffDock confidence 不是结合自由能，也不能替代实验 Kd、IC50 或 SPR/BLI 结果。
2. AlphaFold 结构是预测结构，可能不代表配体诱导构象、磷酸化状态、复合物状态或膜环境。
3. EGFR 和 JAK1 使用了人工策划的 kinase-domain crop，应在方法中透明报告，并在最终候选中优先检查 pocket 合理性。
4. Top1000 中有 60 个 pair 的 DiffDock 输出缺失；这些 pair 在最终候选前应优先重跑或用替代 docking 工具复核。
5. TxGNN 仅保守映射 738/915 个药物，未映射药物没有被视为负证据。这适合避免误杀，但会降低 TxGNN 层的覆盖完整性。
6. DrugBank ID 当前为 0/915，药物临床机制、安全性和说明书证据仍需在 Step 7 接入额外数据库。
7. 当前疾病重点为 cancer/MONDO_0004992，不代表对所有疾病均有效；如果扩展适应症，需要重新指定疾病节点和疾病种子基因。
8. 候选前列包含多个已知肿瘤靶点和已上市抗肿瘤药物，这有助于验证流程合理性，但新颖性需要单独做“已知/未知机制”标注。

## 8. Step 6-8 推进计划

### Step 6：结构共识和候选分层

当前已完成 Top1000 结构增强排序，主要输出为 `outputs/report_scale/stage6_top1000_consensus_candidates.csv`、`outputs/report_scale/stage6_top100_consensus_candidates.csv` 和失败审计表。建议将候选分为三层：Tier A 为 Stage6 排名前列且 DiffDock completed、疾病证据 direct_and_network、TxGNN mapped 的 pair；Tier B 为 Stage6 排名前列但 TxGNN unmapped 或 docking 缺失的 pair；Tier C 为 Top5000/Top10000 中需要后续扩展 docking 的候选。

全量 DiffDock 已恢复为后台 score-only 队列，当前 PID 为 `{full_pid}`，已有 score 文件数为 {full_score_files}。score-only 策略用于避免磁盘被 91 万级 pose 文件填满；需要结构可视化的候选应单独重跑并保留 SDF。

### Step 7：机制、安全性和可成药性分析

Step 7 应围绕 Top100 或 Tier A 候选完成以下分析：药物既有适应症和肿瘤相关适应症标注；靶点通路和上游/下游机制解释；靶点表达和癌种相关性；药物安全窗、给药途径、黑框警告和主要不良反应；已知靶点与预测靶点是否一致；是否存在明显的临床禁忌或药物相互作用风险。该步骤可以不等待全量 DiffDock，因为机制和安全性分析依赖的是候选列表、疾病证据和药物临床资料。

### Step 8：实验验证设计

Step 8 应将候选转化为可执行实验包。建议先选择 5-10 个 Tier A pair 进入体外验证，包括蛋白结合实验、酶活实验或细胞靶点占用实验；再选择 2-3 个机制清晰且安全性可控的候选进入细胞表型验证，例如增殖、凋亡、迁移、信号通路磷酸化或药物联用实验。对于已知 EGFR/KIT/JAK 相关候选，应重点寻找再定位癌种或新组合疗法，而不是重复证明已知适应症。

## 9. 数据和代码可用性

关键数据文件：

- 药物结构库：`data/processed/drug_library_pubchem_chembl_mapped.csv`
- 蛋白结构路径：`data/processed/protein_library_1000_alphafold_paths.csv`
- ConPLex 全量亲和力：`outputs/report_scale/conplex_affinity_scores_915k.csv`
- Stage4 AI 排序：`outputs/report_scale/stage4_affinity_candidates_915k.csv`
- Open Targets 证据：`data/processed/opentargets_target_disease_scores.csv`
- STRING 高置信子网：`data/processed/string_human_filtered_edges.csv`
- TxGNN 药物-疾病分数：`data/processed/txgnn_drug_disease_scores.csv`
- Stage5 全量排序：`outputs/report_scale/stage5_txgnn_open_targets_string_ranked_candidates_915k.csv`
- Top1000 DiffDock score：`outputs/report_scale/diffdock_top1000_smiles_scores.csv`
- Stage6 Top1000 共识排序：`outputs/report_scale/stage6_top1000_consensus_candidates.csv`
- Stage6 Top100 候选表：`outputs/report_scale/stage6_top100_consensus_candidates.csv`
- Top1000 DiffDock 失败审计：`outputs/report_scale/diffdock_top1000_smiles_failure_audit.csv`

关键脚本：

- `scripts/enrich_drug_structures_pubchem_chembl.py`
- `scripts/build_alphafold_receptor_manifest.py`
- `scripts/rerank_stage5_open_targets_string.py`
- `scripts/merge_stage5_with_txgnn.py`
- `scripts/build_diffdock_ready_manifest.py`
- `scripts/build_stage5_top_diffdock_manifest.py`
- `scripts/prepare_diffdock_full_run.py`
- `scripts/run_diffdock_full_queue.py`
- `scripts/run_diffdock_full_job.py`
- `scripts/merge_stage6_top_diffdock.py`
- `scripts/export_stage6_report_artifacts.py`

## 10. 可对外沟通的准确结论

可以说：

“我们已经完成 915 个 FDA 已批准小分子与 1000 个蛋白靶点的全量 AI-DTI 和疾病证据整合筛选，得到 915000 个 pair 的 Stage 5 排序。Top1000 候选已经完成结构对接增强，其中 940 个 pair 获得 DiffDock confidence。当前结果可以支持机制分析、候选分层和实验验证设计。”

不应说：

“全量 DiffDock 已经完成”或“所有候选都经过结构验证”。截至本文生成时，全量 DiffDock 仍在后台运行或可恢复，当前交付的结构增强结果严格限于 Top1000。

## 参考数据源和工具

本稿使用的主要公开数据源和工具包括 ChEMBL、PubChem、AlphaFold human proteome v6、Open Targets Platform GraphQL API、STRING v12.0 API、TxGNN、ConPLex 和 DiffDock。正式投稿前应按目标期刊格式补充标准参考文献、版本号、访问日期和 DOI/URL。
"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
