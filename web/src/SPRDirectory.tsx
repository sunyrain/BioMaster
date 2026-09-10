import { useEffect, useState } from "react";
import { ArrowUpRight, FlaskConical, Search } from "lucide-react";
import { api, type EntityRef } from "./types";
import { LoadingState } from "./LoadingState";
import "./research-browse.css";
import Experiments from "./Experiments";
import SPRComposition from "./SPRComposition";
import SPRResults from "./SPRResults";

export default function SPRDirectory({ initialScope }: { initialScope?: string }) {
  const [scope, setScope] = useState(initialScope === "results" ? "results" : "baseline");
  const [targets, setTargets] = useState<EntityRef[]>([]);
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setBusy(true); setError("");
    api<{items: EntityRef[]}>("/api/search?kind=target&limit=2000", controller.signal)
      .then(data => setTargets(data.items.filter(t => (t.annotations?.experiments || 0) > 0)))
      .catch(e => { if (e.name !== "AbortError") setError(e.message); })
      .finally(() => { if (!controller.signal.aborted) setBusy(false); });
    return () => controller.abort();
  }, [revision]);
  const visible = targets.filter(t => `${t.name} ${t.id} ${t.subtitle || ""}`.toLowerCase().includes(query.trim().toLowerCase()));
  const pairs = targets.reduce((n, t) => n + (t.annotations?.experiments || 0), 0);
  return <div className="research-browse">
    {scope !== "results" && <section className="panel browse-intro"><span className="eyebrow">EXPERIMENT DESIGN / SPR</span><h1>SPR 设计</h1><p>浏览当前 SPR 设计中的全部靶点，查看384个候选及112个另计参考对照。最终实验表按优先级排列，包含小分子药物名称、旧靶点名称、新靶点名称及拟用对照。新靶点的已知药物名称一列对应本靶点指定的参考化合物。</p>
      <div className="browse-stats"><span><strong>{busy ? "—" : targets.length}</strong>个设计靶点</span><span><strong>{busy ? "—" : pairs}</strong>组配对（384候选 + 112对照）</span><span><strong>设计快照</strong>候选2026.09.09 · 对照修订2026.09.10；实测结果另行入库</span></div>
    </section>}
    <div className="spr-filters sru-directory-tabs" role="group" aria-label="SPR 工作区"><button className={scope === "baseline" ? "active" : ""} onClick={() => setScope("baseline")}>基线384 · 细分实验建议</button><button className={scope === "expanded" ? "active" : ""} onClick={() => setScope("expanded")}>新增384 · 证据候选</button><button className={scope === "results" ? "active" : ""} onClick={() => setScope("results")}>实验结果 · 上传与记录</button></div>
    {scope === "results" && <SPRResults />}
    {scope === "expanded" && <section className="panel browse-intro"><p>在原384之外，完整评分空间得到991对通过本地规则的配对（尚未证明是全新关系），本轮审核其中384对，其余607对尚未进入本轮。192对来自结合与化学支持通道，192对来自放宽后的联合通道。</p><p>联合通道：结合排名前30/384、TxGNN疾病前100、OT≥0.3。结合通道：排名前10/384、阳性近邻相似度≥0.35，不强制图谱交集。两条通道均保留身份、既有关系及原始活动排除。</p></section>}
    {scope === "baseline" && <section className="panel browse-intro"><h2>最终实验表 · 384候选，对照另计</h2><p>FDA优先修订版：31个靶点已换入FDA药物对照，涉及84行候选。384候选及优先顺序沿用基线，对照另计。每项显示FDA来源、选用活性记录和构建条件；仍需本批蛋白测通。已更换的对照使用CTRL-FDA编号，请重新下载对照结果模板。</p><div className="experiment-links"><a className="button primary" href="/api/spr-final.csv" download>下载最终实验表 CSV</a><a className="button secondary" href="/api/spr-final.xlsx" download>下载 Excel（含对照分表）</a><a className="button secondary" href="/api/spr-final-four-columns.csv" download>仅四列 CSV</a><a className="button secondary" href="/api/spr-final-controls.csv" download>112对照清单</a></div></section>}
    {scope === "baseline" && <SPRComposition />}
    {scope !== "results" && <Experiments key={scope} expanded={scope === "expanded"} />}
    {scope !== "results" && <section className="panel browse-results">
      <label className="filter-search"><Search size={17} /><input aria-label="搜索 SPR 设计靶点" placeholder="搜索基因、蛋白名称或靶点编号…" value={query} onChange={e => setQuery(e.target.value)} /></label>
      {busy ? <LoadingState text="正在读取 SPR 设计目录…" /> : error ? <div className="error-panel" role="alert">{error}<button className="button secondary" onClick={() => setRevision(v => v + 1)}>重新加载目录</button></div> : !visible.length ? <div className="empty"><FlaskConical size={26} /><strong>没有匹配的设计靶点</strong>{query && <button className="button secondary" onClick={() => setQuery("")}>清除筛选</button>}</div> : <>
        <p className="browse-match" role="status">显示 {visible.length} / {targets.length} 个设计靶点</p>
        <div className="catalog-grid">{visible.map(target => <article className="panel" key={target.id}><div className="browse-record-heading"><FlaskConical size={20} /><h2>{target.name}</h2></div><p className="browse-match">{target.id}</p><p>{target.annotations?.experiments} 组实验配对</p><button className="button secondary" style={{marginTop:16}} onClick={() => { location.hash = `/target/${encodeURIComponent(target.id)}/experiments`; }}>查看 {target.name} 设计<ArrowUpRight size={16} /></button></article>)}</div>
      </>}
    </section>}
  </div>;
}
