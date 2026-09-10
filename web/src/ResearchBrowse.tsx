import { useEffect, useState } from "react";
import { ArrowLeft, ArrowUpRight, Search, Database, ChevronLeft, ChevronRight } from "lucide-react";
import { api, fmt, label, type Evidence, type EntityRef, type Summary } from "./types";
import { LoadingState } from "./LoadingState";
import { Value } from "./EvidenceBrowser";
import "./research-browse.css";

const sections = {
  chembl: { title: "适应症与作用机制", description: "跨药物检索 ChEMBL 适应症与机制注释，查看疾病名称、研究阶段与已登记作用靶点。", choices: [["known_diseases", "适应症记录"], ["known_targets", "作用机制"]] },
  pathways: { title: "通路与生物学背景", description: "按通路名称、Reactome 编号或基因检索全站注释，了解哪些靶点参与同一生物过程。", choices: [["pathways", "通路注释"]] },
  structures: { title: "结构与口袋目录", description: "先浏览全站蛋白和口袋覆盖，再选择感兴趣的靶点进入三维工作台。实验位点与预测口袋分别计数。", choices: [["structures", "蛋白结构"]] },
  txgnn: { title: "疾病重新定位线索", description: "浏览已接入的各药物全疾病 Top30 快照，按药物或疾病名称查找线索。模型分数不代表治疗有效概率，合并疾病节点也不代表子病种特异预测。", choices: [["txgnn_diseases", "Top30 疾病快照"]] },
  pairs: { title: "配对检索空间", description: "选择药物查看其完整靶点排名，或从靶点反向检索药物；随后可比较各模型、筛选已有关系并导出配对。不同查询的分数不作全站混排。", choices: [["pairs_drug", "从药物出发"], ["pairs_target", "从靶点出发"]] },
};
type Result = { total: number; all_total: number; entity_total: number; matched_entities: number; items: { entity: EntityRef; record: Evidence }[] };
const number = (v: number) => v.toLocaleString("en-US");
const recordLabels: Record<string, string> = { name: "记录名称", id: "记录编号", source: "证据来源", url: "来源链接", external_url: "外部来源", evidence: "证据说明", phase: "最高研究阶段", molecule_id: "ChEMBL 分子编号", action_type: "作用类型", action: "作用方向", organism: "物种", direct_interaction: "直接相互作用注释", score: "模型分数", logit: "模型原始输出", direction: "疾病方向", source_path: "快照来源", type: "结构类型", format: "坐标格式", coordinate_system: "坐标体系", accession: "UniProt 编号", mean_plddt: "平均 pLDDT", pocket_mean_plddt: "口袋平均 pLDDT", note: "说明", pocket_count: "口袋总数", experimental_pockets: "实验位点数", predicted_pockets: "预测口袋数", in_catalog: "已登记到项目目录", explorable: "可在项目中浏览", source_id: "原始来源编号", identifiers: "关联标识" };
const displayRecord = (record: Evidence) => Object.fromEntries(Object.entries(record).map(([key, value]) => [recordLabels[key] || key, value]));
const navigate = (path: string) => { location.hash = "/" + path; };

export default function ResearchBrowse({ section, summary }: { section: string; summary: Summary }) {
  const config = sections[section as keyof typeof sections];
  const [view, setView] = useState(config?.choices[0][0] || "known_diseases");
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [page, setPage] = useState(1);
  const [revision, setRevision] = useState(0);
  const [data, setData] = useState<Result | null>(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  useEffect(() => {
    if (!config) return;
    const controller = new AbortController();
    setBusy(true); setError("");
    const timer = setTimeout(() => {
      const params = new URLSearchParams({ section: view, search: query, category, page: String(page), page_size: "20" });
      api<Result>(`/api/browse?${params}`, controller.signal).then(setData)
        .catch(e => { if (e.name !== "AbortError") setError(e.message); })
        .finally(() => { if (!controller.signal.aborted) setBusy(false); });
    }, query ? 200 : 0);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [view, query, category, page, revision, config]);
  if (!config) return <section className="panel"><h1>未找到该浏览目录</h1><button className="button secondary" onClick={() => navigate("home")}>返回项目全景</button></section>;
  const open = (entity: EntityRef) => navigate(`${entity.kind}/${encodeURIComponent(entity.id)}/${section === "structures" ? "structure" : section === "pairs" ? "rankings" : "diseases"}`);
  const action = section === "structures" ? "打开三维工作台" : section === "pairs" ? "查看完整排名" : section === "pathways" ? "查看靶点生物学背景" : "查看药物疾病与机制";
  return <div className="research-browse">
    <button className="text-button" onClick={() => navigate("home")}><ArrowLeft size={16} />返回项目全景</button>
    <section className="panel browse-intro"><span className="eyebrow">RESEARCH EVIDENCE / EXPLORE</span><h1>{config.title}</h1><p>{config.description}</p>
      <nav className="browse-navigation" aria-label="研究内容目录">{Object.entries(sections).map(([key, item]) => <button key={key} aria-current={section === key ? "page" : undefined} onClick={() => navigate(`browse/${key}`)}>{item.title}</button>)}</nav>
      <div className="browse-stats"><span><strong>{data ? number(data.all_total) : "—"}</strong>{section === "structures" ? "个有结构的蛋白" : section === "pairs" ? "个可检索实体" : "条当前目录记录"}</span>{section !== "structures" && <span><strong>{section === "pairs" ? number(summary.counts.pairs) : data ? number(data.entity_total) : "—"}</strong>{section === "pairs" ? "对已计算配对" : "个关联实体"}</span>}{section === "structures" && <span><strong>{number(summary.counts.pockets)}</strong>个已接入口袋</span>}</div>
    </section>
    <section className="panel browse-results" aria-busy={busy}>
      <div className="browse-toolbar"><div className="browse-choices" role="group" aria-label="记录类型">{config.choices.map(([id, name]) => <button key={id} aria-pressed={view === id} onClick={() => { setView(id); setCategory("all"); setPage(1); setData(null); }}>{name}</button>)}</div>
        <label className="filter-search"><Search size={17} /><input aria-label="搜索浏览目录" placeholder={section === "structures" ? "搜索蛋白、基因或 UniProt 编号…" : "搜索名称、疾病、通路或编号…"} value={query} onChange={e => { setQuery(e.target.value); setPage(1); }} /></label>
        {(view === "known_diseases" || section === "structures") && <select aria-label="筛选证据类型" value={category} onChange={e => { setCategory(e.target.value); setPage(1); }}><option value="all">全部记录</option>{view === "known_diseases" ? <><option value="approved">已批准 · Phase 4</option><option value="clinical">临床研究 · Phase 1–3</option></> : <><option value="experimental">有实验位点注释</option><option value="predicted">有预测口袋</option></>}</select>}
      </div>
      {busy ? <LoadingState text="正在检索研究记录…" /> : error ? <div className="error-panel" role="alert">{error}<button className="button secondary" onClick={() => setRevision(v => v + 1)}>重新加载目录</button></div> : !data?.items.length ? <div className="empty"><Database size={26} /><strong>没有匹配的记录</strong><p>尝试更短的名称或其他编号。</p><button className="button secondary" onClick={() => { setQuery(""); setCategory("all"); setPage(1); }}>清除筛选</button></div> : <>
        <p className="browse-match" role="status">找到 {number(data.total)} 条记录 · 涉及 {number(data.matched_entities)} 个实体</p>
        <div className="browse-records">{data.items.map(({entity, record}, i) => <article className="browse-record" key={`${entity.id}:${page}:${i}`}>
          <div className="browse-record-main"><div className="browse-record-heading"><span className="badge neutral">{entity.kind === "drug" ? "药物" : "靶点"} · {entity.name}</span><small>{entity.id}</small></div>
            <h2>{label(record)}</h2>
            <div className="browse-facts">
              {record.phase != null && <span>{record.phase === 4 ? "已批准" : "临床研究"} · Phase {record.phase}</span>}
              {record.action_type && <span>作用类型：{record.action_type}</span>}
              {record.score != null && <span>模型分数：{fmt(record.score, 4)}</span>}
              {section === "structures" && <><span>口袋 {number(record.pocket_count)}</span><span>实验位点 {number(record.experimental_pockets)}</span><span>预测口袋 {number(record.predicted_pockets)}</span></>}
              {section === "pairs" && <span>{number(record.pair_count)} 对配对 · 可比较各模型完整排名</span>}
              <span>{record.source || record.id || "当前项目注释"}</span>
            </div>
            {record.evidence && <p>{record.evidence}</p>}
            {section !== "pairs" && <details><summary>查看记录详情与依据</summary><div className="browse-record-details"><Value value={displayRecord(record)} /></div></details>}
          </div><button className="button secondary" onClick={() => open(entity)}>{action}<ArrowUpRight size={16} /></button>
        </article>)}</div>
        <div className="browse-pagination"><span>第 {page} / {Math.max(1, Math.ceil(data.total / 20))} 页</span><button className="button secondary" aria-label="上一页浏览记录" disabled={page <= 1} onClick={() => setPage(v => v - 1)}><ChevronLeft size={16} />上一页</button><button className="button secondary" aria-label="下一页浏览记录" disabled={page * 20 >= data.total} onClick={() => setPage(v => v + 1)}>下一页<ChevronRight size={16} /></button></div>
      </>}
    </section>
  </div>;
}
