# SPR384 最终实验表 FDA 对照修订（2026-09-10）

已按用户要求，将明确可替换的31项拟用对照换入最终版；384候选、112靶点与候选优先顺序不变，84行候选对应的对照随之更新。31项使用19种药物，选用证据20项Kd、11项Ki。

这里的“明确可替换”指精确FDA药物身份、目标蛋白及选用记录已匹配，足以更新拟用对照方案；不意味着本批蛋白构建已经确认或SPR质控已经通过。实际构建信息尚未提供，因此每项均保留适用条件与实验室确认状态。

现112项对照中，52项FDA完整结构身份匹配，5项仅名称匹配，55项本轮未确认FDA身份。后两类不是全部被判定为未获批，也不是都不能做；本轮保留原拟用对照及其现有确认条件。

## 选择范围

没有机械采用初筛排名第一的药物：保留已有合适FDA药；不以细胞裂解液Kinobeads记录单独支持替换，不把JH2作用外推到JH1，不将功能IC50直接标为SPR Kd。HDAC1/2采用明确Ki记录而非可疑或二手Kd归类；PPARG采用配体竞争Ki，避免混用共激活肽结合。JAK1原cravacitinib精确来源未匹配的问题通过换用ruxolitinib的JH1构建837–1142的Ki记录解决。

## 本次替换

| 靶点 | 原拟用对照 | 本版拟用对照 | 选用端点/值（nM） | 新编号 |
| --- | --- | --- | --- | --- |
| PDE5A | 研究化合物 CHEMBL2180945（无通用名称） | TADALAFIL | Kd / 2.4 | CTRL-FDA-004 |
| FLT1 | STAUROSPORINE | SUNITINIB | Kd / 1.8 | CTRL-FDA-008 |
| XDH | TOPIROXOSTAT | FEBUXOSTAT | Ki / 0.1 | CTRL-FDA-012 |
| KIT | STAUROSPORINE | SUNITINIB | Kd / 0.37 | CTRL-FDA-014 |
| HDAC2 | 研究化合物 CHEMBL235842（无通用名称） | VORINOSTAT | Ki / 1.6 | CTRL-FDA-015 |
| FLT3 | STAUROSPORINE | SUNITINIB | Kd / 0.47 | CTRL-FDA-020 |
| PDGFRA | STAUROSPORINE | SUNITINIB | Kd / 0.79 | CTRL-FDA-026 |
| COMT | 研究化合物 CHEMBL5653589（无通用名称） | OPICAPONE | Ki / 1.0 | CTRL-FDA-029 |
| MAOB | LAZABEMIDE | SAFINAMIDE | Kd / 187.2 | CTRL-FDA-031 |
| RET | STAUROSPORINE | VANDETANIB | Kd / 34.0 | CTRL-FDA-032 |
| ESR1 | 研究化合物 CHEMBL377849（无通用名称） | RALOXIFENE | Ki / 2.0 | CTRL-FDA-034 |
| PGR | 研究化合物 CHEMBL2311103（无通用名称） | NORETHINDRONE | Ki / 1.87 | CTRL-FDA-037 |
| JAK3 | STAUROSPORINE | TOFACITINIB | Kd / 2.2 | CTRL-FDA-039 |
| ACHE | HUPERZINE A | DONEPEZIL | Kd / 8.0 | CTRL-FDA-043 |
| DYRK1A | HARMINE | ABEMACICLIB | Kd / 3.349 | CTRL-FDA-048 |
| PPARG | REVERSE TRIIODOTHYRONINE | PIOGLITAZONE | Ki / 420.0 | CTRL-FDA-050 |
| ESR2 | 研究化合物 CHEMBL475278（无通用名称） | RALOXIFENE | Ki / 23.0 | CTRL-FDA-052 |
| GSK3B | STAUROSPORINE | ABEMACICLIB | Kd / 7.673 | CTRL-FDA-056 |
| KDR | LINIFANIB | SUNITINIB | Kd / 1.5 | CTRL-FDA-060 |
| JAK1 | CRAVACITINIB | RUXOLITINIB | Ki / 0.2 | CTRL-FDA-064 |
| PIK3CB | PICTILISIB | GEDATOLISIB | Kd / 3.5 | CTRL-FDA-073 |
| HDAC1 | TRICHOSTATIN | VORINOSTAT | Ki / 1.3 | CTRL-FDA-076 |
| PIK3CG | PICTILISIB | GEDATOLISIB | Kd / 24.0 | CTRL-FDA-077 |
| TYK2 | STAUROSPORINE | TOFACITINIB | Ki / 4.39 | CTRL-FDA-082 |
| PIK3CA | WORTMANNIN | GEDATOLISIB | Kd / 1.0 | CTRL-FDA-089 |
| FGFR2 | FEXAGRATINIB | PAZOPANIB | Kd / 210.0 | CTRL-FDA-092 |
| ALK | STAUROSPORINE | CERITINIB | Kd / 1.3 | CTRL-FDA-095 |
| AXL | STAUROSPORINE | SUNITINIB | Kd / 9.0 | CTRL-FDA-104 |
| NTRK2 | STAUROSPORINE | MIDOSTAURIN | Kd / 310.0 | CTRL-FDA-105 |
| BRAF | SB590885 | SORAFENIB | Ki / 38.0 | CTRL-FDA-107 |
| ACVRL1 | AT-9283 | VANDETANIB | Kd / 470.0 | CTRL-FDA-109 |

## 文件与网站一致性

主表、四列CSV、详细表、Excel和112对照清单同步生成。FDA_CONTROL_REPLACEMENTS.csv保存每项原对照、审批来源、所选activity、原始方法描述和构建条件。旧版存于outputs/spr384_final_experiment_table_before_fda_v2_20260910。

网站SPR详情同时显示新药名、FDA审批链接、历史活性记录及构建适用条件。新CTRL-FDA编号和精确pair_id贯穿靶点页面、药物页面和结果模板；旧配对不会因改名被误接到新药物。历史结果记录按原身份独立归档。实验对照模板请重新下载。

验证：23项相关测试通过，前端构建通过；API包含384候选和112对照，31项替换的编号/结构/配对一致；3种CSV下载与本地逐字节一致，对照结果模板匹配当前112配对；桌面、移动端FDA证据可见且无页面错误。
