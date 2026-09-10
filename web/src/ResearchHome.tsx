import { useEffect, useId, useRef, useState, type CSSProperties } from "react";
import {
  ArrowRight, ArrowUpRight, Atom, Box, Check,
  ChevronDown, Database, Dna, FlaskConical,
  Layers3, LoaderCircle, Network, Search, X,
} from "lucide-react";
import { api, EntityRef, MODEL_COLORS, MODEL_NAMES, Model, Rankings, Summary } from "./types";
import { AnimatedNumber } from "./motion";
import "./research-home.css";

type Props = {
  summary: Summary;
  onNavigate: (path: string) => void;
  onOpenEntity: (entity: EntityRef) => void;
};
const count = (value?: number) => (value ?? 0).toLocaleString("en-US");
const primaryModels: Model[] = ["biomaster", "drugclip", "dtiam", "conplex"];

function ResearchSearch({ featured, onOpenEntity }: {
  featured: EntityRef[]; onOpenEntity: Props["onOpenEntity"];
}) {
  const [query, setQuery] = useState("");
  const [items, setItems] = useState<EntityRef[]>([]);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState(-1);
  const wrap = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const listId = useId();
  const results = query.trim() ? items : featured.slice(0, 5);
  useEffect(() => {
    setSelected(-1);
    setItems([]);
    setError("");
    if (!query.trim()) { setBusy(false); return; }
    const controller = new AbortController();
    setBusy(true);
    const timer = window.setTimeout(() => {
      api<{ items: EntityRef[] }>(`/api/search?q=${encodeURIComponent(query)}&kind=all&limit=6`, controller.signal)
        .then(data => { if (!controller.signal.aborted) setItems(data.items); })
        .catch(err => { if (!controller.signal.aborted) setError(err.message); })
        .finally(() => { if (!controller.signal.aborted) setBusy(false); });
    }, 160);
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [query]);
  useEffect(() => {
    const close = (event: PointerEvent) => {
      if (!wrap.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, []);
  const choose = (entity: EntityRef) => { setOpen(false); onOpenEntity(entity); };
  return <div className="rh-search" ref={wrap} onBlur={event => {
    if (!event.currentTarget.contains(event.relatedTarget as Node)) setOpen(false);
  }}>
    <div className={`rh-search-field${open ? " is-open" : ""}`}>
      <Search size={22} aria-hidden="true" />
      <input ref={input} value={query} placeholder="搜索药物、基因、UniProt 或 InChIKey"
        aria-label="在研究工作台搜索药物或靶点" role="combobox" aria-expanded={open}
        aria-controls={listId} aria-autocomplete="list"
        aria-activedescendant={open && selected >= 0 && results[selected] ? `${listId}-${selected}` : undefined}
        onFocus={() => setOpen(true)} onChange={event => { setQuery(event.target.value); setOpen(true); }}
        onKeyDown={event => {
          if (event.key === "Escape") { setOpen(false); return; }
          if (event.key === "ArrowDown") { event.preventDefault(); setOpen(true); setSelected(i => Math.min(results.length - 1, i + 1)); }
          if (event.key === "ArrowUp") { event.preventDefault(); setSelected(i => Math.max(0, i - 1)); }
          if (event.key === "Enter" && !busy && !error && results.length) {
            event.preventDefault(); choose(results[Math.max(0, selected)]);
          }
        }} />
      {busy ? <LoaderCircle className="rh-spin" size={19} /> : query ?
        <button type="button" aria-label="清空搜索" onClick={() => { setQuery(""); input.current?.focus(); }}><X size={18} /></button> :
        <span className="rh-search-enter" aria-hidden="true">↵</span>}
    </div>
    {open && <div className="rh-search-popover">
      <div className="rh-search-caption">{query.trim() ? "匹配的项目实体" : "快速进入"}<span>↑ ↓ 选择 · Enter 打开</span></div>
      {busy ? <p className="rh-search-message" role="status">正在检索项目目录…</p> : error ?
        <p className="rh-search-message rh-error" role="alert">{error}</p> : !results.length ?
          <p className="rh-search-message" role="status">未找到匹配实体，请尝试药物英文名或数据库编号。</p> : null}
      <div id={listId} role="listbox" aria-label="实体搜索结果">
        {!busy && !error && results.map((entity, index) => <button key={`${entity.kind}-${entity.id}`}
          id={`${listId}-${index}`} role="option" type="button" aria-selected={selected === index}
          className={selected === index ? "is-selected" : ""} onMouseEnter={() => setSelected(index)}
          onClick={() => choose(entity)}>
          <span className={`rh-result-icon ${entity.kind}`}>{entity.kind === "drug" ? <Atom size={18} /> : <Dna size={18} />}</span>
          <span><strong>{entity.name}</strong><small>{entity.subtitle || entity.id}</small></span>
          <span className="rh-result-type">{entity.kind === "drug" ? "药物" : "靶点"}</span><ArrowUpRight size={17} />
        </button>)}
      </div>
    </div>}
  </div>;
}

function RankingMap({ drugs, models, onOpenEntity, onNavigate }: {
  drugs: EntityRef[]; models: Summary["models"]; onOpenEntity: Props["onOpenEntity"]; onNavigate: Props["onNavigate"];
}) {
  const [drugId, setDrugId] = useState(drugs[0]?.id || "");
  const [model, setModel] = useState<Model>("biomaster");
  const [data, setData] = useState<Rankings | null>(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  const [hovered, setHovered] = useState("");
  const graphId = useId().replace(/:/g, "");
  const drug = drugs.find(row => row.id === drugId) || drugs[0];
  const color = MODEL_COLORS[model];
  useEffect(() => {
    if (!drug) { setBusy(false); return; }
    const controller = new AbortController();
    setBusy(true); setError(""); setData(null); setHovered("");
    api<Rankings>(`/api/rankings?kind=drug&id=${encodeURIComponent(drug.id)}&model=${model}&page_size=6`, controller.signal)
      .then(result => { if (!controller.signal.aborted) setData(result); })
      .catch(err => { if (!controller.signal.aborted) setError(err.message); })
      .finally(() => { if (!controller.signal.aborted) setBusy(false); });
    return () => controller.abort();
  }, [drug?.id, model, retry]);
  const rows = data?.items.filter(row => row.rank != null) || [];
  return <section className="rh-map-panel" style={{ "--rh-model": color } as CSSProperties} aria-label="真实模型排名关系图">
    <div className="rh-panel-heading">
      <div><span className="rh-eyebrow">LIVE RANKING EXPLORER</span><h2>一个分子，探索更多靶点</h2></div>
      <label className="rh-model-select"><span className="rh-model-dot" />
        <select aria-label="切换关系图排名模型" value={model} onChange={event => setModel(event.target.value as Model)}>
          {primaryModels.map(id => <option key={id} value={id} disabled={models.find(row => row.id === id)?.available === false}>{MODEL_NAMES[id]}</option>)}
        </select><ChevronDown size={15} />
      </label>
    </div>
    <div className="rh-drug-tabs" aria-label="选择图谱中心药物">
      {drugs.map(item => <button type="button" key={item.id} aria-pressed={drug?.id === item.id}
        className={drug?.id === item.id ? "active" : ""} onClick={() => setDrugId(item.id)}>{item.name}</button>)}
    </div>
    <div className="rh-network-stage" aria-busy={busy}>
      {busy ? <div className="rh-map-state" role="status"><LoaderCircle size={29} className="rh-spin" /><strong>正在读取真实模型排名</strong><span>完整目录排序 · 保留原始分母</span></div> :
        error ? <div className="rh-map-state rh-error" role="alert"><Database size={28} /><strong>排名暂时无法加载</strong><span>{error}</span><button className="rh-text-link" type="button" onClick={() => setRetry(value => value + 1)}>重新加载 <ArrowRight size={16} /></button></div> :
          !drug || !rows.length ? <div className="rh-map-state"><Network size={28} /><strong>当前暂无可展示的模型排名</strong><span>可从药物图谱查看已有档案。</span><button type="button" className="rh-text-link" onClick={() => onNavigate("drugs")}>进入药物图谱 <ArrowRight size={16} /></button></div> :
            <>
              <svg className="rh-network-svg" viewBox="0 0 740 370" role="group" aria-labelledby={`${graphId}-title ${graphId}-desc`}>
                <title id={`${graphId}-title`}>{drug.name} 的 {MODEL_NAMES[model]} 前六名靶点</title>
                <desc id={`${graphId}-desc`}>连线表示模型预测排名，并非已证实相互作用。点击节点可进入实体档案。绿色已知标记来自已有关系记录。</desc>
                <defs>
                  <pattern id={`${graphId}-grid`} width="24" height="24" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r=".8" fill="#cad7e8" /></pattern>
                  <linearGradient id={`${graphId}-edge`}><stop offset="0%" stopColor={color} stopOpacity=".16" /><stop offset="100%" stopColor={color} stopOpacity=".7" /></linearGradient>
                  <linearGradient id={`${graphId}-drug`} x1="0" y1="0" x2="1" y2="1"><stop stopColor="#183c68" /><stop offset="1" stopColor="#0c1c34" /></linearGradient>
                </defs>
                <rect width="740" height="370" fill={`url(#${graphId}-grid)`} />
                <text x="32" y="29" className="rh-svg-caption">QUERY MOLECULE</text>
                <text x="455" y="29" className="rh-svg-caption">TOP {rows.length} / {data?.denominator} TARGETS</text>
                <line x1="396" y1="46" x2="396" y2="346" stroke="#d6e0ed" strokeDasharray="4 7" />
                {rows.map((row, index) => {
                  const y = 65 + index * 52;
                  const path = `M 248 194 C 345 194, 345 ${y}, 451 ${y}`;
                  return <g key={`edge-${row.id}`} className={`rh-graph-edge${hovered && hovered !== row.id ? " is-dimmed" : ""}${hovered === row.id ? " is-highlighted" : ""}`}>
                    <path d={path} fill="none" stroke={`url(#${graphId}-edge)`} strokeWidth={index === 0 ? 3 : 1.5} />
                    <path className="rh-edge-flow" d={path} fill="none" stroke={color} strokeWidth="2" strokeDasharray="3 160" style={{ animationDelay: `${index * -.65}s` }} />
                    <circle cx="451" cy={y} r="3.5" fill={color} />
                  </g>;
                })}
                <g className="rh-graph-node rh-query-node" tabIndex={0} role="button" aria-label={`打开 ${drug.name} 药物档案`}
                  onClick={() => onOpenEntity(drug)} onKeyDown={event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onOpenEntity(drug); } }}>
                  <rect x="27" y="126" width="221" height="136" rx="17" fill={`url(#${graphId}-drug)`} />
                  <g transform="translate(48 144)" stroke="#7bdbd1" fill="none" strokeWidth="1.4">
                    <circle cx="17" cy="17" r="3" fill="#7bdbd1" /><ellipse cx="17" cy="17" rx="17" ry="7" /><ellipse cx="17" cy="17" rx="17" ry="7" transform="rotate(60 17 17)" /><ellipse cx="17" cy="17" rx="17" ry="7" transform="rotate(120 17 17)" />
                  </g>
                  <text x="95" y="167" fill="#b5c9e5" fontSize="12" letterSpacing="1.3">DRUG PROFILE</text>
                  <text x="48" y="209" fill="#fff" fontSize={drug.name.length > 18 ? "18" : "23"} fontWeight="700">{drug.name.length > 23 ? `${drug.name.slice(0, 22)}…` : drug.name}</text>
                  <text x="48" y="240" fill="#b5c9e5" fontSize="13">分子档案 · 疾病 · 已知作用靶点</text>
                  <circle cx="246" cy="194" r="5" fill={color} stroke="#fff" strokeWidth="2" />
                </g>
                {rows.map((row, index) => {
                  const y = 65 + index * 52;
                  return <g key={row.id} className={`rh-graph-node rh-target-node${hovered === row.id ? " is-hovered" : ""}`} tabIndex={0} role="button"
                    aria-label={`${row.name}，第 ${row.rank} 名，${row.known_relation ? "已有关系记录，" : ""}打开靶点档案`}
                    onMouseEnter={() => setHovered(row.id)} onMouseLeave={() => setHovered("")} onFocus={() => setHovered(row.id)} onBlur={() => setHovered("")}
                    onClick={() => onOpenEntity(row)} onKeyDown={event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onOpenEntity(row); } }}>
                    <rect x="452" y={y - 20} width="264" height="40" rx="8" fill="#fff" stroke={hovered === row.id ? color : "#d8e2ef"} />
                    <text x="466" y={y + 5} fill={color} fontSize="14" fontWeight="700">#{row.rank}</text>
                    <text x="508" y={y + 5} fill="#162a45" fontSize="16" fontWeight="700">{row.name.length > 12 ? row.name.slice(0, 11) + "…" : row.name}</text>
                    {row.known_relation && <><rect x="635" y={y - 12} width="43" height="23" rx="5" fill="#dff5ee" /><text x="656.5" y={y + 4} textAnchor="middle" fill="#00634e" fontSize="12" fontWeight="600">已知</text></>}
                    <path d={`M 694 ${y - 4} l 4 4 l -4 4`} fill="none" stroke="#7287a1" strokeWidth="1.7" />
                  </g>;
                })}
                <text x="31" y="343" className="rh-svg-caption">预测排序连线 · 点击节点进入档案</text>
              </svg>
              <div className="rh-network-mobile">
                <button className="rh-mobile-query" type="button" onClick={() => onOpenEntity(drug)}><Atom size={21} /><span><small>中心药物</small><strong>{drug.name}</strong></span><ArrowUpRight size={18} /></button>
                <div className="rh-mobile-axis"><span>模型排名</span><span>{data?.denominator} 个可评分靶点</span></div>
                {rows.map(row => <button type="button" className="rh-mobile-target" key={row.id} onClick={() => onOpenEntity(row)}>
                  <span className="rh-mobile-rank">#{row.rank}</span><strong>{row.name}</strong>{row.known_relation && <span className="rh-known">已知</span>}<ArrowUpRight size={17} />
                </button>)}
              </div>
            </>}
    </div>
    <div className="rh-map-footer"><span><i className="rh-legend-line" /> 模型预测 <i className="rh-legend-known" /> 已有关系记录</span>
      {drug && <button className="rh-text-link" type="button" onClick={() => onNavigate(`drug/${encodeURIComponent(drug.id)}/rankings`)}>完整排名 <ArrowRight size={16} /></button>}
    </div>
    <p className="rh-map-note">显示完整目录中的前 6 名。已有关系是数据库注释；预测分数和排名不代表实验验证。</p>
  </section>;
}

export default function ResearchHome({ summary, onNavigate, onOpenEntity }: Props) {
  const c = summary.counts;
  const [entityKind, setEntityKind] = useState<"drugs" | "targets">("targets");
  const entities = summary.featured[entityKind];
  const sprPending = c.spr_not_released_pairs ?? 0;
  const metrics = [
    { label: "药物目录", value: c.drugs, unit: "种", detail: "模型分子与上市药物身份", icon: Atom, path: "drugs", tone: "teal" },
    { label: "靶点登记", value: c.targets, unit: "个", detail: `${count(c.scored_targets)} 个进入统一评分核心`, icon: Dna, path: "targets", tone: "blue" },
    { label: "配对检索空间", value: c.pairs, unit: "对", detail: `${count(c.drugs)} 药物 × ${count(c.scored_targets)} 靶点`, icon: Network, path: "browse/pairs", tone: "navy" },
    { label: "结构与口袋", value: c.pockets, unit: "个", detail: `${count(c.structures)} 个靶点已有结构`, icon: Box, path: "browse/structures", tone: "violet" },
  ];
  const sources = [
    { name: "ChEMBL 37", path: "browse/chembl", label: "适应症与作用机制", value: c.known_indications, unit: "条适应症记录", detail: `已批准 ${count(c.approved_indications)} · 临床研究 ${count(c.clinical_indications)}`, icon: FlaskConical, color: "teal" },
    { name: "Open Targets / Reactome", path: "browse/pathways", label: "通路与生物学背景", value: c.pathways, unit: "条通路注释", detail: `覆盖 ${count(c.pathway_targets)} 个靶点 · GO / 组织表达`, icon: Network, color: "blue" },
    { name: "AlphaFold / P2Rank", path: "browse/structures", label: "蛋白结构与结合口袋", value: c.structures, unit: "个靶点有结构", detail: `${count(c.pockets)} 个口袋 · 实验与预测分别标注`, icon: Box, color: "violet" },
    { name: "TxGNN", path: "browse/txgnn", label: "疾病重新定位线索", value: c.txgnn_drugs, unit: "种药物有预测", detail: "已有疾病关联快照 · 预测非临床结论", icon: Layers3, color: "amber" },
  ];
  return <div className="rh-root">
    <section className="rh-intro">
      <div className="rh-intro-top"><span className="rh-eyebrow">Palinova / RESEARCH WORKSPACE</span><span className="rh-data-status"><i /> 本地证据已连接</span></div>
      <div className="rh-intro-row"><div><h1>从分子出发，<br className="rh-mobile-break" /><span>连接新的发现。</span></h1><p>跨模型检索药物与靶点，贯通疾病、通路与三维结构。</p></div>
        <div className="rh-intro-links"><button type="button" onClick={() => onNavigate("drugs")}>药物图谱 <ArrowUpRight size={17} /></button><button type="button" onClick={() => onNavigate("targets")}>靶点图谱 <ArrowUpRight size={17} /></button></div>
      </div>
      <ResearchSearch featured={[...summary.featured.targets.slice(0, 3), ...summary.featured.drugs.slice(0, 2)]} onOpenEntity={onOpenEntity} />
      <div className="rh-search-hints"><span>快速探索</span>{summary.featured.targets.slice(0, 3).map(item => <button type="button" key={item.id} onClick={() => onOpenEntity(item)}>{item.name}<ArrowUpRight size={12} /></button>)}{summary.featured.drugs.slice(0, 2).map(item => <button type="button" key={item.id} onClick={() => onOpenEntity(item)}>{item.name}<ArrowUpRight size={12} /></button>)}</div>
    </section>

    <div className="rh-metric-grid">
      {metrics.map(metric => <button type="button" className={`rh-metric ${metric.tone}`} key={metric.label} onClick={() => onNavigate(metric.path)}>
        <div className="rh-metric-title"><span>{metric.label}</span><metric.icon size={20} /></div>
        <div className="rh-metric-value"><AnimatedNumber value={metric.value ?? 0} /><span>{metric.unit}</span></div>
        <div className="rh-metric-detail"><span>{metric.detail}</span><ArrowUpRight size={16} /></div>
      </button>)}
    </div>

    <div className="rh-main-grid">
      <RankingMap drugs={summary.featured.drugs} models={summary.models} onOpenEntity={onOpenEntity} onNavigate={onNavigate} />
      <aside className="rh-model-panel">
        <div className="rh-panel-heading"><div><span className="rh-eyebrow">MULTI-MODEL EVIDENCE</span><h2>四种视角，交叉比较</h2></div><Layers3 size={21} /></div>
        <p className="rh-model-intro">相同检索空间，保留各模型的独立排名与评分口径。</p>
        <div className="rh-model-list">{primaryModels.map((id, index) => {
          const model = summary.models.find(item => item.id === id);
          const coverage = model?.available ? Math.min(100, (Number(model.targets) || 0) / Math.max(1, c.scored_targets) * 100) : 0;
          return <div className="rh-model-row" key={id} style={{ "--rh-model": MODEL_COLORS[id] } as CSSProperties}>
            <div className="rh-model-name"><span className="rh-model-number">0{index + 1}</span><strong>{MODEL_NAMES[id]}</strong><span>{model?.available ? "已接入" : "未接入"}</span></div>
            <p>{id === "biomaster" ? "分子 + 蛋白表征 · 双向检索" : id === "drugclip" ? "结构表征 · 口袋比较" : id === "dtiam" ? "独立比较模型 · 药物靶点关联" : "序列表征 · 药物蛋白匹配"}</p>
            <div className="rh-coverage-line"><span style={{ width: `${coverage}%` }} /></div>
            <div className="rh-coverage-caption"><span>{count(Number(model?.coverage) || 0)} 配对</span><span>{count(Number(model?.targets) || 0)} / {count(c.scored_targets)} 靶点</span></div>
          </div>;
        })}</div>
        <button type="button" className="rh-model-source" onClick={() => onNavigate("sources")}>查看版本与评分口径 <ArrowRight size={17} /></button>
      </aside>
    </div>

    <section className="rh-evidence-section">
      <div className="rh-section-heading"><div><span className="rh-eyebrow">CONNECTED RESEARCH CONTEXT</span><h2>让每个候选，都有证据可追溯</h2></div><button type="button" className="rh-text-link" onClick={() => onNavigate("sources")}>全部数据来源 <ArrowRight size={16} /></button></div>
      <div className="rh-evidence-grid">{sources.map(source => <button type="button" className={`rh-evidence-card ${source.color}`} key={source.name} onClick={() => onNavigate(source.path)}>
        <div className="rh-source-top"><span className="rh-source-icon"><source.icon size={21} /></span><ArrowUpRight size={17} /></div>
        <span className="rh-source-name">{source.name}</span><h3>{source.label}</h3>
        <div className="rh-source-count"><strong>{count(source.value)}</strong><span>{source.unit}</span></div>
        <p>{source.detail}</p>
      </button>)}</div>
    </section>

    <div className="rh-bottom-grid">
      <section className="rh-entities-panel">
        <div className="rh-panel-heading"><div><span className="rh-eyebrow">QUICK ACCESS</span><h2>从关注的实体开始</h2></div>
          <div className="rh-segment" aria-label="切换精选实体类型"><button type="button" aria-pressed={entityKind === "targets"} className={entityKind === "targets" ? "active" : ""} onClick={() => setEntityKind("targets")}>靶点</button><button type="button" aria-pressed={entityKind === "drugs"} className={entityKind === "drugs" ? "active" : ""} onClick={() => setEntityKind("drugs")}>药物</button></div>
        </div>
        <div className="rh-entity-list">{entities.length ? entities.slice(0, 6).map((entity, index) => <button type="button" key={entity.id} onClick={() => onOpenEntity(entity)}>
          <span className="rh-entity-index">{String(index + 1).padStart(2, "0")}</span><span className={`rh-entity-kind ${entity.kind}`}>{entity.kind === "target" ? <Dna size={21} /> : <Atom size={21} />}</span>
          <span className="rh-entity-identity"><strong>{entity.name}</strong><small>{entity.subtitle || entity.id}</small></span><span className="rh-entity-go"><span>查看档案</span><ArrowUpRight size={16} /></span>
        </button>) : <p className="rh-empty-inline">当前暂无精选实体，可进入完整目录检索。</p>}</div>
        <button type="button" className="rh-entity-all" onClick={() => onNavigate(entityKind)}>浏览全部 {count(c[entityKind])} {entityKind === "drugs" ? "种药物" : "个靶点"}<ArrowRight size={17} /></button>
      </section>
      <section className="rh-spr-panel">
        <div className="rh-spr-overline"><span className="rh-eyebrow">FROM COMPUTATION TO EXPERIMENT</span><FlaskConical size={23} /></div>
        <h2>把候选，推进实验设计。</h2><p className="rh-spr-description">SPR 方案将计算候选、参考对照与蛋白构建关联到同一份可追溯设计中。</p>
        <div className="rh-spr-counts"><div><strong>{count(c.spr_design_targets)}</strong><span>设计靶点</span></div><span className="rh-spr-times">/</span><div><strong>{count(c.spr_design_pairs)}</strong><span>设计配对</span></div></div>
        <div className="rh-spr-composition" aria-label={`${count(c.spr_candidate_pairs)} 个候选配对，${count(c.spr_control_pairs)} 个对照配对`}><span style={{ flex: c.spr_candidate_pairs || 0 }} /><span style={{ flex: c.spr_control_pairs || 0 }} /></div>
        <div className="rh-spr-legend"><span><i /> 候选 {count(c.spr_candidate_pairs)}</span><span><i /> 参考对照 {count(c.spr_control_pairs)}</span></div>
        <div className="rh-spr-status"><span className="rh-spr-check"><Check size={16} /></span><div><strong>计算设计已完成</strong><p>{sprPending ? `${count(sprPending)} 配对待实验放行 · 暂无 SPR 实测结果` : "设计快照 · 实验状态以配对档案为准"}</p></div></div>
        <div className="rh-spr-actions"><button type="button" onClick={() => onNavigate("spr")}>浏览全部 SPR 设计 <ArrowUpRight size={17} /></button><button type="button" onClick={() => onNavigate("sources")} aria-label="查看 SPR 设计数据来源"><Database size={18} /></button></div>
      </section>
    </div>
    <div className="rh-workspace-footnote"><span><Database size={15} /> 数据快照 · {summary.version}</span><button type="button" onClick={() => onNavigate("browse/structures")}>浏览结构与口袋目录 <ArrowUpRight size={15} /></button></div>
  </div>;
}
