# 平台运行核查与近期 DTI／JEPA 可用模型

核查日期：2026-09-16 UTC。结论：平台曾离线，本轮已恢复原登录保护下的网站与固定域名隧道。近期确有可利用的公开权重，最贴合当前研究的是 Mol-JEPA、Nesso-1 和 ProbeMatchDTI；其中 Nesso 在本项目已有历史控制结果。模型可访问不等于已经通过本项目测试，本轮未替换生产评分、重选 SPR 或启动模型训练。

**平台实况与恢复。**

检查开始时本机18765端口没有监听，网站与Cloudflare进程均不在，公网返回530。按既有配置恢复 `run_explorer.py --no-infer --auth-file ...` 与 named tunnel；账号文件及隧道凭据未改动。公网 [登录页](https://palinova.xyz/login) 和本地登录页均恢复200，未登录的summary、SPR导出接口均返回401。

使用相同生产代码、前端和数据，在临时本地端口以临时账号完成登录、首页、SPR页、药物排名、移动端浏览器检查，无页面异常；API、SPR清单和CSV导出通过。该检查未使用生产账号登录，公网检查范围是可达性与登录保护。候选和对照导出逐字节等于交付文件，未上传任何实验结果。

| 项目 | 实际状态 |
|---|---|
| 默认主榜 | 720药×384靶点＝276,480对；ReTargetMap、历史DTIAM、ConPLex全覆盖，DrugCLIP覆盖382靶点 |
| 登记目录 | 888靶点；LYVE1、SLC8A1目前搜索均返回0项 |
| 完整研究矩阵 | 本地已生成720×890＝640,800对，尚未接入默认网站主榜 |
| 网站DTIAM | 仍读取历史 `public_retrained_v1/dtiam_720x384_deployment_v1`，不是九月新A/B |
| SPR | 384候选＋112对照，文件保持冻结；网站实测结果库当前0条，不代表湿实验没有开展 |
| 训练 | DTIAM六套已于9月15日完成；当前无训练进程，JEPA尚未启动 |
| 算力 | 驱动报告49,140 MiB显存且空闲；CPU配额20核，内存上限90 GiB，快照使用约25.8 GiB |
| 磁盘 | 系统盘约11 GiB、数据盘约65 GiB剩余；本轮未下载大型模型或数据包 |

因此，“服务恢复正常”和“研究成果全部上线”是两件事：新DTIAM、890矩阵及额外两靶点仍需单独完成展示接入。现有服务仍采用独立后台进程，未新增开机自启；本轮不能保证主机重启后自动恢复。

**本轮核实的模型，按实际用途而非论文榜单排序。**

| 模型／时间 | 可用资产核实 | 对我们的用途与限制 |
|---|---|---|
| **Mol-JEPA**，2026-08预印本、09-01修订 | 官方仓库链接HF权重；`model.safetensors`约181.7 MB，未设访问门槛，字节范围下载返回206 | SMILES生成512维分子表示和各模态预测表示。可比较是否改善分子输入，也可借鉴模态遮蔽、防塌缩；它本身不是任意蛋白—分子亲和预测器。[官方仓库](https://github.com/Boehringer-Ingelheim/mol-jepa)、[权重](https://huggingface.co/Flogrammer/Mol-JEPA)、[论文](https://arxiv.org/abs/2608.22642) |
| **Nesso-1**，2026-08预印本 | 代码、环境、本地历史结果已有；HF公开权重约165.4 MB，另有CCD与ESM依赖。官方可导出Pairformer交互张量 | 值得作为训练专用的昂贵组合视图，提取蛋白—配体块 `z_pl`。其输出是预测视图，不能当实验结构；亲和标量为log10(IC50/µM)，不可直接当Kd。[官方模型](https://huggingface.co/recursionpharma/nesso)、[特征提取代码](https://github.com/recursionpharma/nesso/blob/main/tutorial/extract_features.py) |
| **ProbeMatchDTI**，2026-09-02预印本 | 仓库5个无扩展名检查点为Git LFS；All_Model约78.1 MB，实际二进制范围下载返回206，而非只有空指针 | 原子／残基及多层表征交互，值得作为新直接DTI对照。需其ProtBERT、SMILES编码器及图输入，不能直接加载我们的ESM2＋BerMol缓存；README所述Agent目录未出现在本轮文件树，处理后数据亦需准备。[论文](https://arxiv.org/abs/2609.02549)、[代码与权重入口](https://github.com/developer-hq/ProbeMatchDTI) |
| **DTBind**，Research，2025-12-02 | 仓库列有occurrence/site/affinity三套真实模型文件，约4.3/3.6/10.8 MB；本轮未运行 | 适合补充位点、接触和结构教师比较。三任务输入要求不同，亲和分支需要复合物结构；不能包装为只需两个全局向量的替代模型。[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12669800/)、[官方代码](https://github.com/liqy09/DTBind) |
| **DrugCMF**，AAAI2026，5月代码发布 | 公开训练代码与预处理；README的“预训练模型”指四个基础编码器，本轮未发现训练完成的DTI头权重 | 模态置信度加权和交互值得借鉴，尤其适合避免劣质结构拖累序列分支。四套token特征准备较重，优先做方法消融。[论文](https://ojs.aaai.org/index.php/AAAI/article/view/39972)、[代码](https://github.com/deku-0621/DrugCMF) |
| **GRAM-DTI**，官方仓库标注ICLR2026 | 仓库以多模态预训练模块为主，本轮未发现最终任务权重；非即插即用 | 自适应模态丢弃值得参考；分子文本／功能描述须检查是否泄露已知靶点和适应证，不宜无审计加入筛选。[官方仓库](https://github.com/uta-smile/GRAM-DTI) |
| **ESP-DTI**，AAAI2026 | 有渐进采样与交叉注意力代码，本轮未发现任务权重 | 可作为课程采样参考；示例每轮查看TEST，复用时必须改为我们的验证选择／测试冻结流程。[论文](https://ojs.aaai.org/index.php/AAAI/article/download/37100/41062)、[代码](https://github.com/qianwindfeng/ESP-DTI) |

“未发现任务权重”仅指本轮检索的官方README、默认分支文件与公开入口，不证明作者没有权重。Mol-JEPA与DrugCMF标注CC BY-NC 4.0；Nesso代码和权重标注Apache-2.0。ProbeMatchDTI、DTBind本轮未找到明确仓库LICENSE。用途扩大时需核对对应许可，不能将论文开放获取等同于代码可任意商用。

本轮还查到ConfDTI论文提到公开复现资源，但指定的 `songsuan/ConfDTI` API返回404，故未列为当前可落地首选。[论文入口](https://doi.org/10.1021/acs.jcim.6c01036)。ProteinJEPA研究的是蛋白序列表征，其JEPA-only实验易塌缩，不能据名称直接当成组合亲和模型；保留性质监督和防塌缩检查仍有必要。[原论文](https://arxiv.org/abs/2605.07554)

**Nesso不是从零尝试：旧结果重新核算。**

本地 `nesso_control_benchmark_stratified20` 请求474条历史控制，成功449、缺失25。20个目标组中只有19个有可计算的Nesso AUROC：CACNA1B缺全部24条，MST1R缺1条。原SUMMARY把20组都写为可评价，本轮单独记录纠正，不改写历史文件。

逐目标复算的AUROC中位数：binder输出0.7639，负亲和标量0.9167。它说明这两个输出在历史控制上具有不同区分能力，不是当前A/B公共TEST成绩，也没有证明SPR384的未知配对命中率；官方训练重叠尚未排清，不能据它宣布Nesso超过DTIAM。旧任务约耗时5,979秒，不可直接外推更长蛋白或保存全部交互张量后的ETA。

这批旧输出没有找到 `predictions.safetensors`，所以尚无可直接拿来监督学生的 `z_pl` 缓存。若采用该教师，需要按冻结训练成员重跑小样本并立即池化保存，避免保存所有蛋白—蛋白大张量占满磁盘。

**对JEPA探索的具体建议。**

保留用户确定的部署接口：`ESM2蛋白嵌入＋分子嵌入 → 组合z → 分端点性质`。教师、接触和结构只在训练阶段使用。推荐先在同一A数据和划分上对照：

1. **监督基线**：相同组合网络和512维z，仅用实测分类及预先固定的性质目标。
2. **分子表示对照**：加入／替换Mol-JEPA分子表示，单独判断输入收益。这一组本身不算我们训练了组合JEPA。
3. **预测交互目标**：在基线之上预测冻结Nesso的 `z_pl` 汇聚表示，保留真实标签监督、缺失掩码及防塌缩诊断。这是JEPA式跨视图预测／蒸馏，Nesso不是实验真值。
4. **真实结构目标**：恢复本地PLINDER接触／距离教师，完成与新A/B留出的身份、同源、来源隔离后，比较真实结构目标的增益。9,903个旧训练系统不等于已全部通过新版排除，其中辅因子和大簇需要分开处理。

这些组需保持学生架构、监督成员与可比预算一致，并单列新增教师计算成本。先小批测吞吐、检查z方差／有效秩与打乱教师配对对照，再决定规模。成功标准是药物内Top-K、靶点内排序、相近召回下的失活误报及远邻／无阳性参照泛化；表示损失下降不能作为晋级条件。冻结SPR384不参与调参，既有TEST已反复查看，只能继续作回顾性诊断。

本轮没有给这些外部模型增加实跑成绩，没有安装全部新依赖或下载313 GB的Mol-JEPA训练包。当前优先级判断是本项目适配性判断，不是已完成同数据性能排名。

复查文件：[平台运行与接口检查](../outputs/platform_frontier_audit_20260916/PLATFORM_CHECK.json)、[检查点访问核实](../outputs/platform_frontier_audit_20260916/CHECKPOINT_ACCESS.json)、[Nesso旧结果纠正](../outputs/platform_frontier_audit_20260916/NESSO_HISTORICAL_RECHECK.json)、[逐目标复算](../outputs/platform_frontier_audit_20260916/NESSO_HISTORICAL_RECHECK.csv)。复查脚本为[平台检查入口](../scripts/check_platform_runtime_20260916.py)。官方仓库文件树、README和HF清单保存在本地同目录；不将全文README作为本项目文档重新发布。
