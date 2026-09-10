# BioMaster Discovery Atlas 可视化平台

平台直接读取本仓库的数据、选定模型包和结构文件，提供药物／靶点检索、双向排名、疾病与通路证据、三维口袋、完整实体档案和 SPR 实验设计追溯。没有用演示分数代替实际结果。

## 视觉与交互更新（2026-09-09）

新版采用深蓝导航、白色内容卡片、电蓝操作与青绿证据标识；正文16px，主要辅助文字至少12px。靶点概览将简洁功能说明、基本属性和真实三维结构并排展示，完整说明在档案页阅读。

排名表支持「排名优先／分数优先」、模型内排名条形提示和横向滚动；配对抽屉支持 Escape、键盘焦点循环和关闭后还焦点。全局搜索支持 Ctrl／Cmd+K、方向键选择与 Enter 打开；编号和页面链接可一键复制。档案字段可检索，关系图支持节点选择、缩放、平移和聚焦，三维视图的工具与状态已重新布局。

页面过渡、统计数值和交互反馈使用短动效，并遵守操作系统「减少动态效果」偏好。HTML 使用 `Cache-Control: no-store`，避免更新后仍显示旧界面。

独立浏览器验收覆盖1440、1920、390宽度，19组页面／状态及10项交互检查。可计算背景上的文字字号与对比度检查通过；渐变背景另行截图核对。结果见 `outputs/biomaster_explorer/readability/READABILITY_VALIDATION.json`，脚本为 `web/tests/readability.py`。

## 展示界面更新 r2（2026-09-09）

保留深蓝、白色和电蓝配色，统一导航、标题、卡片、表格的字号、间距与圆角。
SPR 页面修复异常的 212px 统计数字，使用 27–30px 紧凑摘要；新增候选/对照筛选、配对搜索与冻结排名，展开后按实验准备、排名证据、审核放行分区，并保留完整原始记录。

验证覆盖 9 类页面 × 1440/1920/390 三种宽度，以及 SPR 筛选、键盘展开、来源追溯与离线运行。
新版离线包目录：`outputs/offline_release_20260909_r2/`，包含 `START_BIOMASTER.bat`、中文说明、SHA-256 文件清单和验证报告。解压运行方式与首版一致。

## 启动

在项目根目录执行：

```bash
cd web
npm ci
npm run build
cd ..
python scripts/run_explorer.py
```

打开 **http://127.0.0.1:18765**。IDE 连接远程服务器时，在 Ports（端口）面板中将远程 **18765** 转发至本机 **18765**；如果 IDE 分配了其他本地端口，请使用该本地端口访问。默认仅监听本机，不需要另起前端进程。

当前项目环境已经具有运行依赖。新 Python 环境需要先安装 `web/requirements.txt`。完整展示还需要当前机器的 `outputs/`、`data/` 和 `downloads/chembl_37/` 数据；这些大型文件不进入 Git。前端依赖由 `web/package-lock.json` 锁定，字体与 3Dmol 在本地提供，页面运行不依赖外部 CDN。

首次启动从 2026-09-06 独立模型包校验文件并在 CPU 上进行全目录 FP32 推理，缓存于 `outputs/biomaster_explorer/cache/`。这是现有权重推理，不会训练或改变模型。后续启动复用通过包指纹校验的缓存。当前机器首次全目录推理约 16 秒，注释索引另需数秒。

```bash
# 指定端口与数据根目录
python scripts/run_explorer.py --port 18765 --root /root/autodl-tmp/BioMaster

# 只使用已有选定模型缓存；缺失时保持分数为空
python scripts/run_explorer.py --no-infer

# 开发：后端继续使用18765，另起Vite即可热更新
cd web
npm run dev
```

## 页面与交互

| 入口 | 已实现内容 |
|---|---|
| 项目全景 | 从本地索引计算覆盖统计，真实 EGFR 三维结构，药物／靶点快捷入口 |
| 药物图谱 | 720 个目录模型分子，英文名称、ChEMBL、DrugBank、InChIKey 等编号检索，分页浏览 |
| 靶点图谱 | 888 个官方登记靶点，包括 745 个非 GPCR 靶点；可筛选 384 个评分核心 |
| 综合概览 | 分子二维结构／靶点档案、证据数量、疾病／通路速览、模型排名 |
| 靶点／老药排名 | ReTargetMap、DrugCLIP、DTIAM、ConPLex 原始分数、各模型 rank 与分母；筛选、分页、配对比较侧栏、完整 CSV 导出 |
| 疾病与机制 | 原适应症与临床研究、已知靶点、TxGNN、Open Targets 疾病、Reactome 通路；全量搜索与分页 |
| 证据网络 | 实体与已知靶点／疾病／通路的交互关系图，预测边采用虚线，点击节点查看来源；每侧最多 7 节点，完整数据在列表中 |
| 完整档案 | 药物身份、理化性质、机制、蛋白功能、序列、GO、亚细胞定位、成药性、化学探针、组织表达等已有注释 |
| 口袋与三维结构 | 全部口袋搜索／分页，真实受体与共晶配体、AlphaFold 位点映射、P2Rank 口袋；旋转、复位、聚焦、表面、全屏与结构下载 |
| SPR 设计 | 64 靶点／512 配对设计，分组、对照、构建建议、冻结排名和未放行状态 |
| 数据来源 | 模型版本、文件来源、覆盖与缺失状态、研究口径 |

搜索框支持方向键选择、Enter 打开和 Escape 关闭。实体与详情标签使用 URL hash，可复制链接、刷新和通过浏览器前进／后退。小屏设备有折叠菜单；长排名表在表格区域横向滚动。

## 本次实际接入范围

以下为当前文件快照的索引结果，页面以运行时统计为准：

| 数据 | 数量与范围 |
|---|---|
| 目录模型分子 | 720 |
| 登记靶点 | 888，其中非 GPCR 745 |
| 统一评分核心 | 720 × 384 = 276,480 对 |
| 三维结构 | 878 个登记靶点可加载 |
| 实验／预测口袋记录 | 43,239；包含多个实验位点注释，并非同等独立验证的口袋 |
| 通路成员记录 | 5,366，覆盖 850 个登记靶点 |
| 靶点–疾病关联 | 258,386，覆盖 325 个登记靶点 |
| 药物适应症／临床研究 | 15,779，其中 ChEMBL phase 4 为 3,017，临床研究为 12,762 |
| TxGNN | 561 个可映射目录分子；本地快照只包含 cancer（MONDO:0004992） |
| GO 注释 | 78,555 条证据记录 |
| 组织表达 | 105,978 条；GTEx TPM 与 HPA nTPM 保留各自来源和单位 |
| SPR 设计 | 512 对：448 候选、64 对照，全部尚未放行 |

登记实体可以查看已有背景和结构；未进入评分核心的实体显示没有可用排名。当前系统检索已有项目实体，并不自动对任意新分子／新蛋白完成特征构建或推理。

## 评分和证据语义

1. 默认 ReTargetMap 使用 `outputs/biomaster_best_model_20260906/retargetmap_selected_v1/` 的 ≤2025 部署权重。药物查询读取正向输出，靶点查询读取反向输出。部署分数不是独立测试成绩、结合概率或实验亲和力。
2. 每个模型在每个查询的完整可评分候选集合中先计算 rank，再筛选和分页。原始分数降序，等分按实体 ID 确定顺序；缺失保持 `null`，不补零、不形成虚假排名。
3. 药物查询的核心上限为 384 个靶点，靶点查询上限为 720 个老药；888／745 登记集合不是排名分母。反向结果明确标注辅助证据。
4. DrugCLIP 实际覆盖 382 个核心靶点。跨实验／预测口袋来源的统一排名尚未校准，只作探索。原始分数不能跨模型直接比较；配对图比较的是模型内排名百分位。
5. 2026-09-01 冻结合同 Borda 分数单列保留在 API 与 CSV 的 `frozen_*` 字段，不替代默认选定模型。
6. 已知关系和未标注关系分别显示；未标注不等于阴性。ChEMBL phase 4 与低于 phase 4 的临床研究分别标注。活性代谢物的模型身份、原上市药物信息和理化性质明确保留归属。
7. TxGNN 只展示本地确实存在的 cancer 预测，不扩展成全疾病结果。Open Targets 关联与 Reactome 通路用于生物学解释，不替代直接结合与治疗效应验证。
8. 实验准备受体与同源共晶配体使用同一坐标体系，按配体 5 Å 邻域高亮受体残基。实验位点映射 AlphaFold 时使用 UniProt canonical 编号；共晶配体并不代表当前查询药物的预测姿势。
9. SPR 记录为 2026-09-09 设计快照，没有实测结果。42 个目录外对照分子只保留在对应靶点记录，不按同名错误映射到目录药物。

## 代码与数据入口

| 文件 | 职责 |
|---|---|
| `biomaster/explorer_data.py` | 真实矩阵读取、选定模型缓存、实体索引、双向排名、搜索、分页 |
| `biomaster/explorer_annotations.py` | FDA 身份映射、ChEMBL、TxGNN、Open Targets、Reactome、GTEx／HPA |
| `biomaster/explorer_structures.py` | 实验／预测口袋索引、结构白名单、坐标语义 |
| `biomaster/explorer_experiments.py` | SPR 设计与目录实体精确关联 |
| `biomaster/explorer_server.py` | 本地 HTTP API、CSV 导出、结构文件、静态前端 |
| `scripts/run_explorer.py` | 单命令启动入口 |
| `web/src/` | React／TypeScript 界面、3Dmol 视图、证据网络与详情 |

数据只读，生成物仅写入可视化缓存与验收目录，不重写现有训练结果、冻结合同或实验设计。

主要 API：

```text
GET /api/summary
GET /api/search?q=EGFR&kind=all&limit=20
GET /api/entity/drug/imatinib
GET /api/entity/target/P00533
GET /api/rankings?kind=target&id=CHEMBL203&model=biomaster&page=1&page_size=20
GET /api/rankings.csv?kind=drug&id=imatinib&model=drugclip
GET /api/evidence?kind=target&id=CHEMBL203&section=pathways&page=1&page_size=50
GET /api/structure/CHEMBL203?pocket=holo
GET /api/molecule/KTUFNOKKBVMGRW-UHFFFAOYSA-N.svg
```

`evidence` 支持 `known_targets`、`known_diseases`、`txgnn_diseases`、`target_diseases`、`pathways`、`pockets`、`experiments`。实体详情返回各列表前 100 条和 `*_total`，全量浏览必须使用分页接口。评分 CSV 导出包含所有筛选结果，不限于当前页。

## 验证与预览

```bash
python -m pytest -q tests/test_explorer.py tests/test_explorer_annotations.py tests/test_explorer_experiments.py
cd web
npm run build
cd ..

# 浏览器验收需要安装Playwright及Chromium
python -m pip install playwright
python -m playwright install chromium
python web/tests/smoke.py --url http://127.0.0.1:18765
```

浏览器验收覆盖真实数据首页、完整目录、搜索、二维分子、双方向排名、缺失分母、全量 CSV、证据分页、关系图、档案、SPR、未评分登记实体、404 和移动端。结构专项已验证 EGFR 的 585 个口袋可完整访问、实验受体 28 个邻近残基、AlphaFold P2Rank Pocket 1 的 24 个高亮残基，以及表面生成和视角控制。

验收摘要与截图输出到 `outputs/biomaster_explorer/`。三维浏览器接口参考 [3Dmol 官方文档](https://www.3dmol.org/doc/GLViewer.html)，前端构建兼容当前 Node 22.11 环境，采用 [Vite 6](https://vite.dev/blog/announcing-vite6)。

## Research Studio 工作台改版

首页以真实药物排名图、模型覆盖和研究入口组织；切换药物或模型会重新读取对应评分，图中节点可直接进入实体档案。四模型保持各自分母，首页没有模拟趋势、命中率或实验结果。

实体概览保留完整身份与来源上下文；进入排名、结构等分析页面时，标题区自动压缩。排名支持数据表、排名矩阵和平行坐标三种视图；矩阵颜色区分 Top 5、Top 20、其余前 25% 与其他位置，平行坐标使用 `(rank - 1) / (denominator - 1)`，并在缺失评分处断线。摘要仅统计当前显示候选，不代表全目录命中率。

新增 `relationship=all|known|unannotated` API／CSV 筛选参数。关系筛选与名称搜索一样，都发生在完整排名计算之后，不改变原始排名和评分分母；CSV 导出所有匹配项。目录默认使用紧凑列表，提供真实证据计数及覆盖筛选，也可切换卡片视图。

靶点疾病页提供当前已加载子集的关联分数图；明确区分子集排序和全量疾病排名。三维页采用口袋列表、受体画布、证据详情三栏，列表内滚动和分页，支持残基点击聚焦；实验共晶、实验位点映射与预测口袋分别标注。证据网络支持关系类型与名称筛选。

「研究清单」将收藏实体和最近访问保存在当前浏览器的 localStorage 中；刷新后保留，不跨浏览器同步。收藏仅保存实体标识与显示名称，后续打开时读取当前项目数据，不保存可能过期的评分副本。

新增工作台验收：

```bash
python web/tests/studio_acceptance.py --url http://127.0.0.1:18765
```

验证包含真实矩阵与轨迹交互、关系筛选后排名保留、筛选 CSV、目录视图与证据筛选、清单持久化、三维视口尺寸，以及 1440／1920／390 宽度的实际页面。工作台报告和截图位于 `outputs/biomaster_explorer/studio/`。

### 2026-09-09 公网展示加载优化

保持原深蓝、白色、电蓝配色。入口 HTML 在脚本下载期间显示启动提示，超过 12 秒提供刷新入口；页面与证据加载使用骨架占位、状态播报和慢连接提示。新增断网提示、页面组件异常恢复，以及目录、排名、证据与 SPR 的失败重试。

结构工作台、证据网络、机制与补充证据等模块按需下载；实体页切换模块时保留标题和导航。公共 API 客户端合并相同在途请求，消费者独立取消，避免快速切换后旧响应覆盖新页面；成功结果仅在当前页面会话缓存 60 秒，最多 80 项、估算 12 MiB。失败不缓存，网络异常和 502/503/504 最多自动重试一次，整体 30 秒超时。刷新页面会清除内存缓存。

服务端为文本、JSON、JS、CSS、SVG 与结构坐标按客户端能力提供 gzip，支持 `gzip;q=0`；带构建哈希的 JS/CSS 长期缓存，其他静态文件缓存一小时，HTML 重新验证，API 保持 HTTP no-store。静态 ETag 可返回 304；HEAD 不返回正文。保留前一发布版资源，避免已打开的旧页面资源丢失。

验证结果位于 `outputs/loading_optimization_20260909/`：

- 入口 JS/CSS 原始大小 522,700 → 435,401 字节（减少 16.7%），gzip 合计 121,818 字节；不含后续按需模块、字体和数据。
- 模拟延迟 180 ms、下行约 190 KB/s，首页主要内容约 1.6 秒可见；这是本机浏览器模拟结果，不代表各地区公网速度。
- `python -m pytest tests/test_explorer.py -q`：8 项通过，含压缩、缓存验证、HEAD、API 与导出。
- `python web/tests/request_behavior.py`：请求合并、独立取消、缓存过期、重试、失败恢复和超时。
- `python web/tests/loading_experience.py --url http://127.0.0.1:18765`：慢网、断网、缓存复用、移动端恢复。
- 原有真实数据 smoke、三尺寸 SPR 测试通过；公网 URL 的首页、数据和懒加载 SPR 页面另行验证。

公网服务继续使用原 Cloudflare 临时网址。隧道与网站均为独立后台进程，运行日志及 PID 在 `outputs/cloudflare_tunnel/`；未配置开机自动启动。离线包未重新生成。

实现参考：[React lazy](https://react.dev/reference/react/lazy)、[HTTP Cache-Control](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Cache-Control)。

### 2026-09-09 首页入口与跨实体证据浏览

首页四张证据卡分别进入 `#/browse/chembl`、`#/browse/pathways`、`#/browse/structures`、`#/browse/txgnn`，展示实际记录，支持搜索、分页、展开依据和进入所选实体。数据来源概览保留原入口。

- ChEMBL：15,779 条适应症，支持 Phase 4 / Phase 1–3 筛选；可切换作用机制。
- 通路：5,366 条注释，覆盖 850 个靶点，按通路或实体检索。
- 结构：878 个有结构的蛋白目录，分别计数实验位点与预测口袋；用户明确选择后进入三维工作台。口袋注释类型不等于底层坐标一定是实验结构，记录详情保留坐标来源及坐标体系。
- TxGNN：18,150 条已接入的各药物全疾病 Top30 快照，覆盖 605 个药物；不宣称此目录是全部药物疾病配对。
- 配对检索：`#/browse/pairs`，选择从药物或核心靶点进入各自完整排名，不跨查询混排分数。

新增只读 `/api/browse`，从已加载快照汇总，先检索/筛选再分页；不会触发推理。新增模块按需加载，沿用原配色及公共请求缓存。

验证：`tests/test_explorer.py` 9 项通过；`web/tests/research_browse.py` 验证四张首页卡片、两张统计卡、实际记录、搜索、Phase 筛选、翻页、实体显式选择与五类目录移动端布局。截图和公网验证记录见 `outputs/browse_refresh_20260909/`。离线包未更新。

侧边栏新增独立「SPR 设计」入口（`#/spr`），展示 64 个设计靶点 / 512 组配对，支持名称与编号搜索；选择靶点进入现有 SPR 明细。进入实体实验页时侧边栏保持 SPR 高亮，首页 SPR 卡片也改为进入完整目录。沿用原配色；已验证搜索、明细、导航高亮及移动端无横向溢出。

### 2026-09-10 展示网站账号登录

线上启用 txc、zqs、wxn、lyj、qjj 五个账号，权限相同。密码以独立随机盐 + PBKDF2-SHA256（260,000 次）哈希存放于服务器 `/root/.config/biomaster/users.json`（权限 0600），不进入前端或项目仓库。

登录入口 `/login`，右上角「退出」通过 POST `/logout` 注销服务器会话。会话有效期 12 小时，采用随机令牌和 HttpOnly / SameSite=Lax Cookie；HTTPS 下增加 Secure。服务器重启后需要重新登录。页面、API、结构文件、导出均经过服务端登录检查，未登录 API 返回 401，前端跳转登录。启用登录后响应采用 private/no-store，连续错误登录有 5 分钟限流。

启动命令：

```bash
python scripts/run_explorer.py --port 18765 --no-infer --auth-file /root/.config/biomaster/users.json
```

标准启动入口也会自动读取上述默认文件（如果存在），避免常规重启漏掉登录保护。公网及临时隧道均连接同一个受保护服务。验证包括 `tests/test_explorer_auth.py`，公网账号登录/注销结果位于 `outputs/login_validation_20260910/`。

账号管理（在服务器项目目录执行）：

```bash
python scripts/manage_explorer_users.py list
python scripts/manage_explorer_users.py add newuser
python scripts/manage_explorer_users.py password txc
python scripts/manage_explorer_users.py delete newuser
```

添加和改密会交互式要求两次密码，终端不回显；添加时已有账号不会被覆盖，不允许删除最后一个账号。账号文件使用锁和原子替换，保持权限 0600。当前登录服务在启动时读取配置，修改后必须重启网站服务才生效，重启会清除全部会话。不要仅重启 Cloudflare 隧道。

### SPR 展示精简（2026-09-10）

统一原生下拉框的高度、内边距、边框、箭头和键盘焦点；SPR 筛选区在桌面按列对齐，在窄屏分行。配对详情顶部保留原用途→目标假设，下方精简为机制与来源、疾病关联证据、实验准备，避免同一用途/靶点重复展示。原用途范围、审核过程、完整排名口径等保留在「完整记录与来源」；LLM 详细支持依据与来源、配比统计方法均默认折叠。实验准备待确认状态以中文展示，原始状态仍可追溯。设计尚未放行、无实测结果等状态保留。原有配色和数据不变。


### SPR 详情视觉排版（2026-09-10）

由独立视觉设计 agent 完成详情排版，主 agent 修复外层 summary 样式向内部折叠栏泄漏并完成公网验证。新增 spr-detail.css 独立作用域，正文14px、小节标题15px、标签12px、折叠行13px；首行四项15px半粗且以细线分列。实验安排和证据区域使用对齐网格，审查结论独立分区，证据数值等宽对齐。审查依据保持折叠，折叠行约42px。保留原配色、中文目标疾病、实验状态及完整来源。

web/tests/spr_typography.py 在1440、1920、390宽度验证字号一致、折叠尺寸、键盘开合、完整来源和无横向溢出。截图及结果位于 outputs/spr_visual_design_20260910/。
