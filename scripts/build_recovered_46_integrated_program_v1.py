#!/usr/bin/env python3
"""Build the integrated 46-target recovery program and the 384-target audit.

Scope is frozen to targets whose *first* exclusion reason was lack of any
eligible experimental pocket.  The legacy five assay lanes are preserved for
traceability, while a mechanism-level branch controls computational and wet-lab
handling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs/recovered_target_program_integrated_v1"
RECOVERED = ROOT / "outputs/recovered_no_experimental_pocket_targets_ch37_v1/RECOVERED_NO_EXPERIMENTAL_POCKET_TARGETS_46_V1.csv"
DTA = ROOT / "outputs/recovered_dta_720x46_v1/RECOVERED_DTA_TARGET_SUMMARY_46_V1.csv"
GNINA = ROOT / "outputs/recovered_gnina_pocket_validation_23_v1/RECOVERED_GNINA_PREDICTED_POCKET_METRICS_23_V1.csv"
CANDIDATES = ROOT / "outputs/recovered_gnina_candidate_docking_v1/RECOVERED_GNINA_CANDIDATE_EVIDENCE_V1.csv"
BOLTZ = ROOT / "outputs/recovered_boltz2_loxl2_candidates_v1/RECOVERED_BOLTZ2_LOXL2_PAIR_SUMMARY_V1.csv"
EXCLUSION = ROOT / "outputs/final_target_package_ch37/FINAL_TARGET_EXCLUSION_AUDIT_888.csv"
MAINLINE = ROOT / "outputs/final_target_package_ch37/FINAL_FROZEN_CHEMBL_SM_TARGETS_WITH_DRUGLIKE_HOLO_338.csv"
UNIVERSE = ROOT / "outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv"


BRANCHES: dict[str, dict[str, str]] = {
    "B01_PROTEIN_KINASE": {
        "branch_zh": "蛋白激酶",
        "compute": "ATP位点与变构位点构象ensemble；ConPLEx+DrugCLIP交叉排序；历史正负对照校准GNINA；只有结构门控通过后才做局部MD",
        "orthogonal": "NanoBRET靶点占有或细胞磷酸化读数；近缘激酶选择性面板",
        "controls": "参考抑制剂、无活性类似物、DMSO、无酶、近缘激酶",
    },
    "B02_LIPID_METABOLIC_KINASE": {
        "branch_zh": "脂质/代谢物激酶",
        "compute": "保留ATP与脂质底物双位点、膜/脂质环境和构象ensemble；ConPLEx+DrugCLIP；对照校准口袋后再做候选对接",
        "orthogonal": "直接结合测定加细胞脂质组学或产物定量；同家族激酶反筛",
        "controls": "参考抑制剂、底物竞争、无酶、无ATP、DMSO、同家族激酶",
    },
    "B03_COFACTOR_RED_OX_ENZYME": {
        "branch_zh": "辅因子/金属依赖氧化还原酶",
        "compute": "按靶点保留血红素、FAD/FMN、金属、硒或LTQ环境；预测口袋ensemble和双DTA；缺辅因子的生成式结构不得升级为结合证据",
        "orthogonal": "直接结合SPR/BLI/MST/DSF与产物LC-MS或辅因子相关光谱；同工酶反筛",
        "controls": "参考抑制剂、无酶、无辅因子、底物空白、DMSO、同工酶",
    },
    "B04_MEMBRANE_NADPH_OXIDASE": {
        "branch_zh": "膜NADPH氧化酶",
        "compute": "跨膜螺旋、双血红素、FAD/NADPH及膜环境联合建模；构象/口袋ensemble；GNINA仅在靶点内对照校准后解释",
        "orthogonal": "膜或细胞ROS读数加直接结合/靶点占有；NOX同工酶及线粒体ROS反筛",
        "controls": "已知NOX抑制剂、敲除/未表达细胞、无NADPH、ROS清除剂、NOX同工酶",
    },
    "B05_NONREDOX_METABOLIC_ENZYME": {
        "branch_zh": "非氧化还原代谢酶",
        "compute": "底物、辅酶或膜界面感知的口袋ensemble；双DTA；历史正负对照校准GNINA；结构合格后才做候选复合物",
        "orthogonal": "直接结合与细胞/膜代谢产物定量；同工酶或相邻通路反筛",
        "controls": "参考抑制剂、底物竞争、无酶、DMSO、结构相近阴性、同工酶",
    },
    "B06_MULTISUBUNIT_RNA_POLYMERASE": {
        "branch_zh": "多亚基RNA聚合酶",
        "compute": "使用完整复合物/关键亚基界面而非孤立全长单链口袋；核酸与金属状态ensemble；单链预测仅作定位假设",
        "orthogonal": "重构转录体系加细胞新生RNA读数；POLR家族与DNA聚合酶反筛",
        "controls": "参考转录抑制剂、无模板、无NTP、无酶、POLR家族反筛",
    },
    "B07_UBIQUITIN_ADAPTOR": {
        "branch_zh": "泛素连接/衔接蛋白",
        "compute": "优先复合物界面与变构热点；单体fpocket位点仅为几何假设；需要界面结构或突变支持",
        "orthogonal": "体外泛素化/蛋白互作读数加细胞信号通路报告；界面突变复核",
        "controls": "无E1/E2、无ATP、催化/界面突变体、DMSO、非相关E3/衔接蛋白",
    },
    "B08_ION_CHANNEL_PORE": {
        "branch_zh": "成孔离子通道",
        "compute": "开放、关闭、失活或配体门控状态的膜内ensemble；孔道/侧窗位点；口袋对照通过前不做发现排序",
        "orthogonal": "自动膜片钳加离子通量或位点突变；多电压和多状态协议",
        "controls": "参考阻断剂/激动剂、未表达细胞、DMSO、seal/current质量控制、近缘通道",
    },
    "B09_CHANNEL_AUXILIARY_SUBUNIT": {
        "branch_zh": "离子通道辅助亚基",
        "compute": "必须在与成孔亚基复合物和膜环境中评估结合位点；孤立亚基口袋仅用于假设生成",
        "orthogonal": "配体结合加通道电流/膜定位调制；成孔亚基单独表达反筛",
        "controls": "已知α2δ配体、未表达细胞、成孔亚基单独表达、DMSO、辅助亚基突变体",
    },
    "B10_SOLUTE_TRANSPORTER": {
        "branch_zh": "溶质转运体",
        "compute": "内向、外向、闭塞状态及底物/离子耦联的膜内ensemble；通道口袋与双DTA；靶点内对照校准",
        "orthogonal": "细胞或膜泡摄取/外排加竞争结合或靶点占有；敲除细胞复核",
        "controls": "已知底物/抑制剂、敲除细胞、离子替换、DMSO、近缘转运体",
    },
    "B11_EPIGENETIC_ENZYME": {
        "branch_zh": "表观遗传酶",
        "compute": "金属/底物状态与催化口袋ensemble；双DTA和对照校准GNINA；考虑HDAC同工型选择性",
        "orthogonal": "去酰化酶活加NanoBRET/SPR/BLI；细胞底物修饰与HDAC家族反筛",
        "controls": "参考抑制剂、无Zn/无酶、底物空白、DMSO、HDAC家族面板",
    },
    "B12_TRANSCRIPTION_FACTOR_PPI": {
        "branch_zh": "转录因子/蛋白互作",
        "compute": "优先共调节因子复合物界面、降解调控界面和变构热点；孤立转录因子fpocket不作发现依据",
        "orthogonal": "蛋白互作TR-FRET/AlphaScreen或NanoBRET加细胞报告基因和靶基因表达",
        "controls": "界面突变体、无蛋白、DMSO、非相关转录因子、通路阳性/阴性对照",
    },
}


# gene -> (branch, mechanistic subtype, required structural context, primary assay)
TARGET_SPECS: dict[str, tuple[str, str, str, str]] = {
    "ABAT": ("B05_NONREDOX_METABOLIC_ENZYME", "PLP_DEPENDENT_AMINOTRANSFERASE", "PLP与底物/产物状态", "PLP依赖GABA转氨反应的底物消耗或产物定量"),
    "CPT2": ("B05_NONREDOX_METABOLIC_ENZYME", "MITOCHONDRIAL_ACYLTRANSFERASE", "膜界面与酰基辅酶A/肉碱底物", "线粒体CPT2酰基转移反应及酰基肉碱LC-MS"),
    "CYP26A1": ("B03_COFACTOR_RED_OX_ENZYME", "HEME_P450_MONOOXYGENASE", "血红素、NADPH-P450还原酶与视黄酸底物", "全反式视黄酸代谢LC-MS"),
    "DHCR24": ("B03_COFACTOR_RED_OX_ENZYME", "MEMBRANE_STEROL_REDUCTASE", "膜、FAD/NADPH与甾醇底物", "desmosterol到胆固醇的LC-MS定量"),
    "DIO1": ("B03_COFACTOR_RED_OX_ENZYME", "SELENOCYSTEINE_DEIODINASE", "含硒半胱氨酸催化中心、膜环境与还原伴侣", "甲状腺激素脱碘产物LC-MS；先获得可信全长/同源结构"),
    "DPYD": ("B03_COFACTOR_RED_OX_ENZYME", "MULTICOFACTOR_FLAVOENZYME", "FAD、FMN、Fe-S、NADPH与寡聚体", "尿嘧啶还原产物LC-MS或光谱动力学"),
    "EGLN3": ("B03_COFACTOR_RED_OX_ENZYME", "FE2_2OG_DIOXYGENASE", "Fe(II)、2-氧戊二酸和底物肽", "HIF肽羟化LC-MS或耦联读数"),
    "FAAH": ("B05_NONREDOX_METABOLIC_ENZYME", "MEMBRANE_SERINE_HYDROLASE", "膜界面与脂质底物通道", "AEA/FA类脂质底物水解LC-MS或荧光底物"),
    "GPD2": ("B03_COFACTOR_RED_OX_ENZYME", "MITOCHONDRIAL_FAD_DEHYDROGENASE", "线粒体膜、FAD与醌受体", "甘油-3-磷酸氧化及醌还原读数"),
    "HSD3B2": ("B03_COFACTOR_RED_OX_ENZYME", "MEMBRANE_STEROID_DEHYDROGENASE_ISOMERASE", "膜、NAD+与甾体底物", "甾体脱氢/异构化产物LC-MS"),
    "LOXL2": ("B03_COFACTOR_RED_OX_ENZYME", "COPPER_LTQ_AMINE_OXIDASE", "Cu与共价LTQ辅因子、成熟分泌蛋白", "Cu/LTQ重构LOXL2胺氧化酶活与H2O2/醛产物定量"),
    "NOX4": ("B04_MEMBRANE_NADPH_OXIDASE", "CONSTITUTIVE_NOX_MEMBRANE_OXIDASE", "膜、双血红素、FAD、NADPH与伴侣状态", "NOX4依赖ROS/H2O2生成的膜或细胞读数"),
    "PIK3CB": ("B02_LIPID_METABOLIC_KINASE", "CLASS_I_LIPID_KINASE", "膜、ATP、PIP2与调节亚基", "PIP2到PIP3的脂质激酶活性"),
    "PIKFYVE": ("B02_LIPID_METABOLIC_KINASE", "PHOSPHOINOSITIDE_LIPID_KINASE", "膜、ATP、PI3P与复合物状态", "PI3P到PI(3,5)P2的脂质激酶活性"),
    "POLR2A": ("B06_MULTISUBUNIT_RNA_POLYMERASE", "RNA_POLYMERASE_II_CATALYTIC_SUBUNIT", "完整Pol II复合物、DNA/RNA、Mg与NTP", "重构Pol II转录延伸或新生RNA读数"),
    "POLR3A": ("B06_MULTISUBUNIT_RNA_POLYMERASE", "RNA_POLYMERASE_III_CATALYTIC_SUBUNIT", "完整Pol III复合物、DNA/RNA、Mg与NTP", "重构Pol III转录或特异转录产物定量"),
    "PPP2CB": ("B05_NONREDOX_METABOLIC_ENZYME", "METAL_SER_THR_PHOSPHATASE", "Mn/Mg金属与PP2A全酶调节亚基", "PP2A底物肽去磷酸化及全酶复核"),
    "RPE65": ("B05_NONREDOX_METABOLIC_ENZYME", "MEMBRANE_RETINOID_ISOMERASE", "膜、Fe与视黄酯底物", "全反式视黄酯到11-顺式视黄醇的LC-MS"),
    "SPHK2": ("B02_LIPID_METABOLIC_KINASE", "SPHINGOLIPID_KINASE", "膜/脂质环境、ATP与鞘氨醇", "鞘氨醇到S1P的LC-MS或放射性激酶测定"),
    "TBXAS1": ("B03_COFACTOR_RED_OX_ENZYME", "HEME_P450_ISOMERASE", "膜、血红素与PGH2底物", "PGH2到TXA2稳定代谢物TXB2的LC-MS/免疫测定"),
    "TPO": ("B03_COFACTOR_RED_OX_ENZYME", "HEME_PEROXIDASE", "血红素、H2O2、碘离子与胞外催化域", "TPO过氧化/碘化活性及H2O2依赖性"),
    "TRAF3IP2": ("B07_UBIQUITIN_ADAPTOR", "E3_LIGASE_ADAPTOR_INTERFACE", "E1/E2/底物复合物与信号适配界面", "体外泛素化加TRAF3IP2依赖信号报告"),
    "TYR": ("B03_COFACTOR_RED_OX_ENZYME", "DINUCLEAR_COPPER_OXIDASE", "双铜中心、成熟糖基化膜蛋白与底物", "L-DOPA/酪氨酸氧化及黑色素生成"),
    "UGCG": ("B05_NONREDOX_METABOLIC_ENZYME", "MEMBRANE_GLYCOSYLTRANSFERASE", "膜、UDP-葡萄糖与神经酰胺", "神经酰胺葡糖基化产物LC-MS"),
    "AKT3": ("B01_PROTEIN_KINASE", "AGC_SER_THR_PROTEIN_KINASE", "ATP位点、PH域和磷酸化激活状态", "AKT3激酶活性与底物磷酸化"),
    "FRK": ("B01_PROTEIN_KINASE", "SRC_FAMILY_TYROSINE_KINASE", "ATP位点与激活环/自抑制状态", "FRK酪氨酸激酶活性"),
    "MAP3K10": ("B01_PROTEIN_KINASE", "MLK_FAMILY_PROTEIN_KINASE", "ATP位点与激活环状态", "MAP3K10激酶活性与下游磷酸化"),
    "MAP3K13": ("B01_PROTEIN_KINASE", "LZK_FAMILY_PROTEIN_KINASE", "ATP位点与激活环状态", "MAP3K13激酶活性与下游磷酸化"),
    "MAPKAPK5": ("B01_PROTEIN_KINASE", "CAMK_FAMILY_PROTEIN_KINASE", "ATP位点、核苷酸和激活状态", "MAPKAPK5激酶活性与底物磷酸化"),
    "YES1": ("B01_PROTEIN_KINASE", "SRC_FAMILY_TYROSINE_KINASE", "ATP位点与SH2/SH3自抑制状态", "YES1酪氨酸激酶活性"),
    "ANO1": ("B08_ION_CHANNEL_PORE", "CA_ACTIVATED_CHLORIDE_CHANNEL", "膜、Ca结合与开放/关闭状态", "ANO1电流的自动膜片钳多Ca状态剂量反应"),
    "KCNA5": ("B08_ION_CHANNEL_PORE", "VOLTAGE_GATED_POTASSIUM_CHANNEL", "膜、开放/关闭/失活状态", "Kv1.5多电压自动膜片钳"),
    "KCND3": ("B08_ION_CHANNEL_PORE", "A_TYPE_VOLTAGE_GATED_POTASSIUM_CHANNEL", "膜、开放/关闭/失活状态及辅助亚基", "Kv4.3多状态自动膜片钳"),
    "KCNJ2": ("B08_ION_CHANNEL_PORE", "INWARD_RECTIFIER_POTASSIUM_CHANNEL", "膜、PIP2与开放/关闭状态", "Kir2.1全细胞电流与PIP2依赖性"),
    "KCNJ8": ("B08_ION_CHANNEL_PORE", "KATP_PORE_SUBUNIT", "膜、PIP2、ATP及SUR复合物", "Kir6.1/SUR复合物KATP电流"),
    "KCNK18": ("B08_ION_CHANNEL_PORE", "TWO_PORE_POTASSIUM_CHANNEL", "膜、机械/脂质调控与门控状态", "TRESK全细胞电流及状态依赖性"),
    "RYR1": ("B08_ION_CHANNEL_PORE", "INTRACELLULAR_CA_RELEASE_CHANNEL", "四聚体、膜、Ca/ATP/调节蛋白；可信全长结构", "肌浆网Ca释放/单通道测定；先补可信全长结构"),
    "SCN11A": ("B08_ION_CHANNEL_PORE", "VOLTAGE_GATED_SODIUM_CHANNEL", "膜、开放/关闭/失活状态", "Nav1.9多电压与状态依赖自动膜片钳"),
    "HDAC11": ("B11_EPIGENETIC_ENZYME", "ZN_DEPENDENT_DEACYLASE", "Zn、底物肽与催化状态", "HDAC11去长链脂酰化/去酰化活性"),
    "NFE2L2": ("B12_TRANSCRIPTION_FACTOR_PPI", "STRESS_RESPONSE_TRANSCRIPTION_FACTOR", "KEAP1/共调节因子复合物与降解调控界面", "NFE2L2-KEAP1互作加ARE报告基因"),
    "CACNA2D2": ("B09_CHANNEL_AUXILIARY_SUBUNIT", "CALCIUM_CHANNEL_ALPHA2DELTA_AUXILIARY", "与CaV成孔亚基复合物、膜和成熟糖基化状态", "α2δ2配体结合加CaV电流/膜定位调制"),
    "CPT1B": ("B05_NONREDOX_METABOLIC_ENZYME", "MITOCHONDRIAL_MEMBRANE_ACYLTRANSFERASE", "线粒体外膜、酰基CoA、肉碱和丙二酰CoA", "CPT1B酰基转移与酰基肉碱LC-MS"),
    "NOX1": ("B04_MEMBRANE_NADPH_OXIDASE", "REGULATED_NOX_MEMBRANE_OXIDASE", "膜、双血红素、FAD/NADPH及NOXO1/NOXA1复合物", "NOX1复合物依赖ROS生成的膜或细胞读数"),
    "SLC10A2": ("B10_SOLUTE_TRANSPORTER", "SODIUM_BILE_ACID_COTRANSPORTER", "膜、Na与胆汁酸底物；内外向状态", "Na依赖胆汁酸摄取"),
    "SLC12A1": ("B10_SOLUTE_TRANSPORTER", "NA_K_CL_COTRANSPORTER", "膜、Na/K/Cl耦联及构象状态", "NKCC2离子通量/摄取及离子依赖性"),
    "SLC22A8": ("B10_SOLUTE_TRANSPORTER", "ORGANIC_ANION_TRANSPORTER", "膜、底物通道及内外向状态", "OAT3有机阴离子摄取与竞争抑制"),
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def join_unique(values: pd.Series) -> str:
    items = sorted({clean(v) for v in values if clean(v)})
    return ";".join(items)


def qualification_status(row: pd.Series) -> tuple[str, str, str]:
    gene = row["gene_symbol"]
    pocket = clean(row["computed_pocket_evidence"])
    qualification = clean(row.get("predicted_pocket_qualification"))
    if pocket.startswith("P3_"):
        return (
            "S0_LOW_CONFIDENCE_STRUCTURE_FIRST",
            "NO_PAIR_PROMOTION",
            "先获取实验/同源全长结构或高质量带MSA复合物模型，再重复口袋预测",
        )
    if qualification == "QUALIFIED_STRONG_RETROSPECTIVE_CONTROL_SEPARATION":
        if gene == "LOXL2":
            return (
                "S1_TWO_COMPUTATIONAL_CANDIDATES_REQUIRE_EXPERIMENT",
                "EXPERIMENTAL_CONFIRMATION_ONLY_NO_BINDING_CLAIM",
                "优先实验tecovirimat，其次tiotropium；不做MD；使用Cu/LTQ成熟LOXL2体系",
            )
        if gene == "SPHK2":
            return (
                "S2_POCKET_STRONG_DTA_MODEL_DISAGREEMENT",
                "DIAGNOSTIC_COMPOUND_PANEL_ONLY",
                "用5个GNINA救援化合物诊断模型，不将其列为正式发现候选",
            )
        if gene == "NOX1":
            return (
                "S2_POCKET_STRONG_NO_SUPPORTED_CANDIDATE",
                "SITE_VALIDATION_ONLY_NO_CANDIDATE_PROMOTION",
                "用历史正负对照验证预测位点；treprostinil不获候选资格",
            )
    if qualification == "MARGINAL_NOT_QUALIFIED":
        return (
            "S3_MARGINAL_POCKET_CONTROL_SEPARATION",
            "NO_DISCOVERY_PROMOTION",
            "增加构象/替代口袋并重复独立对照；通过预设统计门槛后再选候选",
        )
    if qualification == "FAILED_CONTROL_SEPARATION":
        return (
            "S4_POCKET_FAILED_RETROSPECTIVE_CONTROLS",
            "NO_DISCOVERY_PROMOTION",
            "回到构象、辅因子、膜环境或口袋选择；当前位点不得用于候选排名",
        )
    if pocket.startswith("P2_"):
        return (
            "S5_GEOMETRIC_RESCUE_WITHOUT_12X12_CONTROLS",
            "NO_FORMAL_DISCOVERY_UNTIL_CONTROLS_AND_REPLICATION",
            "补足12阳性+12阴性并在至少两个构象复现fpocket位点",
        )
    return (
        "S5_DUAL_PREDICTED_POCKET_WITHOUT_12X12_CONTROLS",
        "NO_FORMAL_DISCOVERY_UNTIL_CONTROLS",
        "补足12阳性+12阴性实验对照并完成靶点内GNINA校准",
    )


def wetlab_stage(row: pd.Series) -> tuple[str, str, str]:
    gene = row["gene_symbol"]
    status = row["current_program_status"]
    if gene == "LOXL2":
        return (
            "W1_CANDIDATE_CONFIRMATION",
            "tecovirimat;tiotropium",
            "10点半对数剂量、技术三复孔、至少3次独立实验；先tecovirimat后tiotropium",
        )
    if gene == "SPHK2":
        return (
            "W2_MODEL_DIAGNOSTIC_PANEL",
            "nilotinib;dabigatran etexilate;ponatinib;imatinib;cabozantinib",
            "8点剂量、技术三复孔、至少2次独立实验；仅用于解决DTA模型分歧",
        )
    if gene == "NOX1":
        return (
            "W2_PREDICTED_SITE_VALIDATION",
            "historical_positive_and_negative_controls;treprostinil_as_diagnostic_negative",
            "12阳性+12阴性对照两次独立复测；不把treprostinil作为发现候选",
        )
    if status.startswith("S3_"):
        return (
            "W3_MARGIN_REPLICATION",
            "historical_12_positive_and_12_negative_controls",
            "冻结同一对照面板，在替代口袋/构象重复；至少2次独立实验或计算复现",
        )
    if status.startswith("S4_"):
        return (
            "W4_STRUCTURE_AND_POCKET_REMODEL",
            "historical_controls_only_after_receptor_remodel",
            "先纠正构象/辅因子/膜环境，再以冻结对照面板复核；禁止发现候选",
        )
    if status.startswith("S0_"):
        return (
            "W6_STRUCTURE_ACQUISITION_FIRST",
            "reference_ligands_only",
            "先获取可信全长/同源/复合物结构和最低12+12实验对照，不进行候选剂量反应",
        )
    return (
        "W5_CONTROL_ACQUISITION",
        "acquire_minimum_12_positive_and_12_negative_controls",
        "优先收集同一测定体系的12阳性+12阴性；未满足前只做探索性口袋复核",
    )


def aggregate_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target_id, group in candidates.groupby("target_chembl_id", sort=False):
        formal = group[group["candidate_triage_status"].eq("COMPUTATIONAL_TRIAGE_PASS_REQUIRES_EXPERIMENT")]
        diagnostic = group[group["candidate_triage_status"].eq("ORTHOGONAL_RESCUE_SIGNAL_MODEL_DISAGREEMENT_REVIEW")]
        rows.append({
            "target_chembl_id": target_id,
            "candidate_pairs_docked": len(group),
            "formal_computational_triage_pair_count": len(formal),
            "formal_computational_triage_drugs": join_unique(formal["drug_names"]),
            "diagnostic_rescue_pair_count": len(diagnostic),
            "diagnostic_rescue_drugs": join_unique(diagnostic["drug_names"]),
            "candidate_statuses_observed": join_unique(group["candidate_triage_status"]),
        })
    return pd.DataFrame(rows)


def build_classification(recovered: pd.DataFrame) -> pd.DataFrame:
    observed = set(recovered["gene_symbol"])
    specified = set(TARGET_SPECS)
    if observed != specified:
        raise RuntimeError(f"Mechanistic classification mismatch missing={observed-specified}, extra={specified-observed}")
    rows: list[dict[str, Any]] = []
    for _, row in recovered.iterrows():
        branch, subtype, context, assay = TARGET_SPECS[row["gene_symbol"]]
        protocol = BRANCHES[branch]
        rows.append({
            "target_chembl_id": row["target_chembl_id"],
            "gene_symbol": row["gene_symbol"],
            "target_name": row["target_name"],
            "legacy_assay_lane": row["assay_lane"],
            "mechanistic_branch": branch,
            "mechanistic_branch_zh": protocol["branch_zh"],
            "mechanistic_subclass": subtype,
            "required_structural_biochemical_context_zh": context,
            "class_specific_compute_protocol_zh": protocol["compute"],
            "target_specific_primary_assay_zh": assay,
            "orthogonal_assay_and_counterscreen_zh": protocol["orthogonal"],
            "required_controls_zh": protocol["controls"],
        })
    return pd.DataFrame(rows)


def build_scope_audit(recovered: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    exclusion = pd.read_csv(EXCLUSION, low_memory=False)
    mainline = pd.read_csv(MAINLINE, low_memory=False)
    universe = pd.read_csv(UNIVERSE, low_memory=False)
    reason_counts = exclusion["first_exclusion_reason"].value_counts().to_dict()
    expected_reasons = {
        "INCLUDED": 338,
        "EXCLUDE_GPCR": 143,
        "EXCLUDE_NO_CHEMBL_SMALL_MOLECULE_MOA": 295,
        "EXCLUDE_UNSUPPORTED_TARGET_CLASS": 42,
        "EXCLUDE_NO_ELIGIBLE_EXPERIMENTAL_POCKET": 46,
        "EXCLUDE_NO_PREFERRED_EXPERIMENTAL_POCKET": 21,
        "EXCLUDE_NO_PREFERRED_DRUGLIKE_HOLO_POCKET": 3,
    }
    if len(universe) != 888 or reason_counts != expected_reasons:
        raise RuntimeError(f"Authoritative 888-target funnel changed: {reason_counts}")
    main_ids = set(mainline["target_chembl_id"])
    recovery_ids = set(recovered["target_chembl_id"])
    if len(main_ids) != 338 or len(recovery_ids) != 46 or main_ids & recovery_ids:
        raise RuntimeError("338/46 active branches are not unique and disjoint")
    if recovery_ids != set(exclusion.loc[
        exclusion["first_exclusion_reason"].eq("EXCLUDE_NO_ELIGIBLE_EXPERIMENTAL_POCKET"),
        "target_chembl_id",
    ]):
        raise RuntimeError("Recovery branch is not the exact first-exclusion 46")

    recovered_active = recovered[[
        "target_chembl_id", "gene_symbol", "uniprot_accession", "target_name", "assay_lane",
        "computed_pocket_evidence", "pocket_consensus_class",
    ]].copy()
    recovered_active["active_target_branch"] = "RECOVERED_NO_EXPERIMENTAL_POCKET_46"
    recovered_active["binding_site_evidence_source"] = "COMPUTATIONAL_POCKET_PREDICTION"
    recovered_active["first_exclusion_reason"] = "EXCLUDE_NO_ELIGIBLE_EXPERIMENTAL_POCKET"
    main_active = mainline[[
        "target_chembl_id", "gene_symbol", "uniprot_accession", "target_name", "assay_lane",
    ]].copy()
    main_active["computed_pocket_evidence"] = ""
    main_active["pocket_consensus_class"] = ""
    main_active["active_target_branch"] = "STRICT_EXPERIMENTAL_POCKET_MAINLINE_338"
    main_active["binding_site_evidence_source"] = "PREFERRED_DRUGLIKE_HOLO_EXPERIMENTAL_POCKET"
    main_active["first_exclusion_reason"] = "INCLUDED"
    active = pd.concat([main_active, recovered_active], ignore_index=True).sort_values(
        ["active_target_branch", "gene_symbol", "target_chembl_id"]
    )
    if len(active) != 384 or active["target_chembl_id"].nunique() != 384:
        raise RuntimeError("Active target union is not exactly 384")

    audit = exclusion.copy()
    hard_reasons = {
        "EXCLUDE_GPCR",
        "EXCLUDE_NO_CHEMBL_SMALL_MOLECULE_MOA",
        "EXCLUDE_UNSUPPORTED_TARGET_CLASS",
    }
    later_reasons = {
        "EXCLUDE_NO_PREFERRED_EXPERIMENTAL_POCKET",
        "EXCLUDE_NO_PREFERRED_DRUGLIKE_HOLO_POCKET",
    }
    audit["final_scope_branch"] = audit["first_exclusion_reason"].map(
        lambda reason: (
            "ACTIVE_STRICT_MAINLINE_338" if reason == "INCLUDED" else
            "ACTIVE_RECOVERED_NO_EXPERIMENTAL_POCKET_46" if reason == "EXCLUDE_NO_ELIGIBLE_EXPERIMENTAL_POCKET" else
            "HARD_GATE_EXCLUDED_480" if reason in hard_reasons else
            "LATER_POCKET_QUALITY_EXCLUDED_24" if reason in later_reasons else
            "UNCLASSIFIED"
        )
    )
    macro_counts = audit["final_scope_branch"].value_counts().to_dict()
    expected_macro = {
        "HARD_GATE_EXCLUDED_480": 480,
        "ACTIVE_STRICT_MAINLINE_338": 338,
        "ACTIVE_RECOVERED_NO_EXPERIMENTAL_POCKET_46": 46,
        "LATER_POCKET_QUALITY_EXCLUDED_24": 24,
    }
    if macro_counts != expected_macro:
        raise RuntimeError(f"Macro scope audit mismatch: {macro_counts}")
    summary = {
        "official_universe": 888,
        "hard_gate_excluded": 480,
        "hard_gate_reason_counts": {
            "EXCLUDE_GPCR": 143,
            "EXCLUDE_NO_CHEMBL_SMALL_MOLECULE_MOA": 295,
            "EXCLUDE_UNSUPPORTED_TARGET_CLASS": 42,
        },
        "after_hard_gates": 408,
        "strict_experimental_pocket_mainline": 338,
        "recovered_only_for_no_experimental_pocket": 46,
        "later_pocket_quality_excluded": 24,
        "later_pocket_quality_reason_counts": {
            "EXCLUDE_NO_PREFERRED_EXPERIMENTAL_POCKET": 21,
            "EXCLUDE_NO_PREFERRED_DRUGLIKE_HOLO_POCKET": 3,
        },
        "active_union": 384,
        "mainline_recovery_overlap": 0,
        "invariants": {
            "universe_partition": "480 + 338 + 46 + 24 = 888",
            "active_union": "338 + 46 = 384",
            "recovery_definition": "first_exclusion_reason == EXCLUDE_NO_ELIGIBLE_EXPERIMENTAL_POCKET",
            "raw_151_no_known_pocket_cache_is_not_formal_scope": True,
        },
    }
    return active, audit, summary


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    recovered = pd.read_csv(RECOVERED, low_memory=False)
    dta = pd.read_csv(DTA, low_memory=False)
    gnina = pd.read_csv(GNINA, low_memory=False)
    candidates = pd.read_csv(CANDIDATES, low_memory=False)
    boltz = pd.read_csv(BOLTZ, low_memory=False)
    if len(recovered) != 46 or recovered["target_chembl_id"].nunique() != 46:
        raise RuntimeError("Recovered input is not the frozen 46-target scope")
    if not recovered["first_exclusion_reason"].eq("EXCLUDE_NO_ELIGIBLE_EXPERIMENTAL_POCKET").all():
        raise RuntimeError("Recovered input contains an out-of-scope exclusion reason")

    classification = build_classification(recovered)
    candidate_agg = aggregate_candidates(candidates)
    boltz_agg = boltz.groupby("target_chembl_id").agg(
        boltz_pairs_completed=("pair_id", "size"),
        boltz_md_authorized_pairs=("md_authorized", "sum"),
        boltz_interpretations=("boltz_interpretation", join_unique),
    ).reset_index()
    outcomes = recovered.merge(classification.drop(columns=["gene_symbol", "target_name"]), on="target_chembl_id", validate="one_to_one")
    outcomes = outcomes.merge(
        dta[["target_chembl_id", "CONPLEX_TARGET_TOP5PCT_FRAGMENT_EXPLORATORY", "TWO_MODEL_TARGET_TOP10PCT_PREDICTED_POCKET"]],
        on="target_chembl_id", how="left", validate="one_to_one",
    )
    outcomes = outcomes.merge(gnina, on=["target_chembl_id", "gene_symbol", "computed_pocket_evidence"], how="left", validate="one_to_one", suffixes=("", "_gnina"))
    outcomes = outcomes.merge(candidate_agg, on="target_chembl_id", how="left", validate="one_to_one")
    outcomes = outcomes.merge(boltz_agg, on="target_chembl_id", how="left", validate="one_to_one")
    for column in [
        "candidate_pairs_docked", "formal_computational_triage_pair_count", "diagnostic_rescue_pair_count",
        "boltz_pairs_completed", "boltz_md_authorized_pairs",
    ]:
        outcomes[column] = outcomes[column].fillna(0).astype(int)
    for column in [
        "formal_computational_triage_drugs", "diagnostic_rescue_drugs", "candidate_statuses_observed",
        "boltz_interpretations",
    ]:
        outcomes[column] = outcomes[column].fillna("")
    status = outcomes.apply(qualification_status, axis=1)
    outcomes["current_program_status"] = status.map(lambda x: x[0])
    outcomes["discovery_authorization"] = status.map(lambda x: x[1])
    outcomes["next_action_zh"] = status.map(lambda x: x[2])
    outcomes["conplex_execution"] = "720_DRUGS_COMPLETE"
    outcomes["drugclip_execution"] = outcomes["gene_symbol"].map(
        lambda gene: "EXCLUDED_LOW_CONFIDENCE_FRAGMENT_STRUCTURE" if gene in {"DIO1", "RYR1"} else "720_DRUGS_EXACT_SEQUENCE_PREDICTED_POCKET_COMPLETE"
    )
    outcomes["gnina_control_execution"] = outcomes["predicted_pocket_qualification"].fillna("NOT_EVALUABLE_MINIMUM_12X12_NOT_MET")
    outcomes["formal_binding_evidence"] = False
    outcomes["scientific_boundary"] = "All candidate results are computational triage; binding/activity requires wet-lab measurement"

    wet = outcomes[[
        "target_chembl_id", "gene_symbol", "target_name", "legacy_assay_lane", "mechanistic_branch",
        "mechanistic_branch_zh", "mechanistic_subclass", "required_structural_biochemical_context_zh",
        "target_specific_primary_assay_zh", "orthogonal_assay_and_counterscreen_zh", "required_controls_zh",
        "computed_pocket_evidence", "historical_experimental_evidence", "gnina_control_execution",
        "current_program_status", "discovery_authorization", "next_action_zh",
    ]].copy()
    stage = wet.apply(wetlab_stage, axis=1)
    wet["wetlab_stage"] = stage.map(lambda x: x[0])
    wet["test_articles"] = stage.map(lambda x: x[1])
    wet["dose_and_replication_design_zh"] = stage.map(lambda x: x[2])
    wet["global_activity_gate_zh"] = (
        "主要测定中形成可重复完整剂量反应，至少两次独立实验IC50/EC50相差不超过3倍；"
        "正交结合或机制读数同方向；排除聚集、反应性、荧光/发光干扰"
    )
    wet["global_selectivity_gate_zh"] = "关键同工型/近缘靶点或机制反筛至少3倍窗口；高优先级候选目标为10倍窗口"
    wet["promotion_rule"] = wet["discovery_authorization"].map(
        lambda auth: "ONLY_AFTER_PRIMARY_AND_ORTHOGONAL_WETLAB_PASS" if "EXPERIMENTAL_CONFIRMATION" in auth else "NO_CANDIDATE_PROMOTION_AT_CURRENT_GATE"
    )

    branch_summary = outcomes.groupby(["mechanistic_branch", "mechanistic_branch_zh"], as_index=False).agg(
        target_count=("target_chembl_id", "size"),
        genes=("gene_symbol", join_unique),
        p1_dual_pocket_targets=("computed_pocket_evidence", lambda s: int(s.str.startswith("P1_").sum())),
        p2_geometric_rescue_targets=("computed_pocket_evidence", lambda s: int(s.str.startswith("P2_").sum())),
        p3_low_confidence_targets=("computed_pocket_evidence", lambda s: int(s.str.startswith("P3_").sum())),
        gnina_strong_targets=("gnina_control_execution", lambda s: int(s.eq("QUALIFIED_STRONG_RETROSPECTIVE_CONTROL_SEPARATION").sum())),
        gnina_marginal_targets=("gnina_control_execution", lambda s: int(s.eq("MARGINAL_NOT_QUALIFIED").sum())),
        gnina_failed_targets=("gnina_control_execution", lambda s: int(s.eq("FAILED_CONTROL_SEPARATION").sum())),
        no_12x12_targets=("gnina_control_execution", lambda s: int(s.eq("NOT_EVALUABLE_MINIMUM_12X12_NOT_MET").sum())),
        formal_computational_triage_pairs=("formal_computational_triage_pair_count", "sum"),
    ).sort_values("mechanistic_branch")
    if branch_summary["target_count"].sum() != 46 or len(branch_summary) != 12:
        raise RuntimeError("Mechanistic branch coverage is incomplete")

    active, full_audit, scope_summary = build_scope_audit(recovered)

    classification_path = OUTDIR / "RECOVERED_46_MECHANISTIC_CLASSIFICATION_V1.csv"
    outcomes_path = OUTDIR / "RECOVERED_46_INTEGRATED_TARGET_OUTCOMES_V1.csv"
    wet_path = OUTDIR / "RECOVERED_46_WETLAB_EXECUTION_PANEL_V1.csv"
    branch_path = OUTDIR / "RECOVERED_46_CLASS_BRANCH_SUMMARY_V1.csv"
    active_path = OUTDIR / "ACTIVE_TARGET_BRANCHES_384_V1.csv"
    audit_path = OUTDIR / "FULL_TARGET_SCOPE_AUDIT_888_V1.csv"
    classification.sort_values(["mechanistic_branch", "gene_symbol"]).to_csv(classification_path, index=False)
    outcomes.sort_values(["mechanistic_branch", "gene_symbol"]).to_csv(outcomes_path, index=False)
    wet.sort_values(["wetlab_stage", "mechanistic_branch", "gene_symbol"]).to_csv(wet_path, index=False)
    branch_summary.to_csv(branch_path, index=False)
    active.to_csv(active_path, index=False)
    full_audit.to_csv(audit_path, index=False)

    summary = {
        "scope": scope_summary,
        "recovered_target_count": 46,
        "mechanistic_branch_count": 12,
        "legacy_assay_lane_counts": outcomes["assay_lane"].value_counts().to_dict(),
        "mechanistic_branch_counts": outcomes["mechanistic_branch"].value_counts().sort_index().to_dict(),
        "pocket_evidence_counts": outcomes["computed_pocket_evidence"].value_counts().to_dict(),
        "gnina_control_status_counts": outcomes["gnina_control_execution"].value_counts().to_dict(),
        "program_status_counts": outcomes["current_program_status"].value_counts().to_dict(),
        "dta": {
            "conplex_targets_complete": int(outcomes["conplex_execution"].eq("720_DRUGS_COMPLETE").sum()),
            "drugclip_exact_sequence_targets_complete": int(outcomes["drugclip_execution"].str.startswith("720_DRUGS").sum()),
            "structure_first_targets_excluded_from_formal_3d": int(outcomes["drugclip_execution"].str.startswith("EXCLUDED").sum()),
        },
        "candidate_evidence": {
            "candidate_pairs_docked": int(outcomes["candidate_pairs_docked"].sum()),
            "formal_computational_triage_pairs_requiring_experiment": int(outcomes["formal_computational_triage_pair_count"].sum()),
            "diagnostic_rescue_pairs": int(outcomes["diagnostic_rescue_pair_count"].sum()),
            "formal_binding_evidence_pairs": 0,
        },
        "boltz_md": {
            "boltz_pairs_completed": int(outcomes["boltz_pairs_completed"].sum()),
            "md_authorized_pairs": int(outcomes["boltz_md_authorized_pairs"].sum()),
            "decision": "NO_MD_LOW_COMPLEX_CONFIDENCE_AND_MISSING_CU_LTQ_CONTEXT",
        },
        "wetlab": {
            "execution_rows": len(wet),
            "stage_counts": wet["wetlab_stage"].value_counts().to_dict(),
            "physical_execution_status": "EXTERNAL_LAB_EXECUTION_REQUIRED",
        },
        "scientific_boundary": "Predicted pockets and candidate scores are triage evidence, not measured binding or activity.",
        "outputs": {
            "classification": str(classification_path),
            "target_outcomes": str(outcomes_path),
            "wetlab_execution_panel": str(wet_path),
            "branch_summary": str(branch_path),
            "active_384": str(active_path),
            "full_888_audit": str(audit_path),
        },
    }
    summary_path = OUTDIR / "RECOVERED_46_INTEGRATED_PROGRAM_SUMMARY_V1.json"
    scope_path = OUTDIR / "ACTIVE_TARGET_SCOPE_AUDIT_384_V1.json"
    summary_path.write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    scope_path.write_text(json.dumps(json_safe(scope_summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    workbook = OUTDIR / "RECOVERED_46_INTEGRATED_PROGRAM_V1.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        branch_summary.to_excel(writer, sheet_name="12机制分支汇总", index=False)
        classification.sort_values(["mechanistic_branch", "gene_symbol"]).to_excel(writer, sheet_name="46靶点机制分类", index=False)
        outcomes.sort_values(["mechanistic_branch", "gene_symbol"]).to_excel(writer, sheet_name="46靶点计算结论", index=False)
        wet.sort_values(["wetlab_stage", "mechanistic_branch", "gene_symbol"]).to_excel(writer, sheet_name="46靶点实验执行", index=False)
        candidates.to_excel(writer, sheet_name="27候选对接证据", index=False)
        boltz.to_excel(writer, sheet_name="LOXL2_Boltz门控", index=False)
        active.to_excel(writer, sheet_name="384活跃靶点", index=False)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
