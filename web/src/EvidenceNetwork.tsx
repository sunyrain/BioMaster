import { useEffect, useId, useMemo, useRef, useState } from "react";
import { ArrowUpRight, ExternalLink, Focus, Move, Network, RotateCcw, Search, X, ZoomIn, ZoomOut } from "lucide-react";
import { Entity, Evidence, fmt, label } from "./types";
import "./motion.css";
import "./evidence-workbench.css";

type Node = {
  id: string; x: number; y: number; row: Evidence; category: string; color: string; side: "left" | "right";
};
const BASE_VIEW = { x: 0, y: 0, scale: 1 };
const shortLabel = (value: string) => value.length > 27 ? `${value.slice(0, 25)}…` : value;

export default function EvidenceNetwork({ entity }: { entity: Entity }) {
  const [selected, setSelected] = useState<Node | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [view, setView] = useState(BASE_VIEW);
  const [category, setCategory] = useState("all");
  const [query, setQuery] = useState("");
  const svgRef = useRef<SVGSVGElement>(null);
  const drag = useRef<{ x: number; y: number; viewX: number; viewY: number } | null>(null);
  const gridId = useId().replace(/:/g, "");
  const allNodes = useMemo(() => {
    const result: Node[] = [];
    const left = entity.kind === "drug" ? entity.known_targets : entity.pathways;
    const right = entity.kind === "drug"
      ? [...entity.known_diseases.slice(0, 5), ...entity.txgnn_diseases.slice(0, 2).map((r) => ({ ...r, prediction: true }))]
      : entity.target_diseases;
    for (const [side, rows] of [["left", left], ["right", right]] as const) {
      const subset = rows.slice(0, 7);
      subset.forEach((row, i) => result.push({
        id: `${side}-${i}`, x: side === "left" ? 280 : 620,
        y: subset.length === 1 ? 285 : 90 + i * 390 / Math.max(1, subset.length - 1), row, side,
        category: row.prediction ? "TxGNN 预测" : entity.kind === "drug"
          ? side === "left" ? "已知靶点" : "已知疾病关联"
          : side === "left" ? "通路成员" : "Open Targets 疾病",
        color: row.prediction ? "#7C3AED" : side === "left" ? "#00856F" : "#2459E8",
      }));
    }
    return result;
  }, [entity]);
  const categories = [...new Set(allNodes.map(node => node.category))];
  const nodes = allNodes.filter(node => (category === "all" || node.category === category) &&
    `${label(node.row)} ${node.row.id || ""} ${node.row.source || ""}`.toLowerCase().includes(query.trim().toLowerCase()));
  useEffect(() => { setSelected(null); setHovered(null); setView(BASE_VIEW); setCategory("all"); setQuery(""); }, [entity.id]);
  useEffect(() => { setSelected(null); setHovered(null); setView(BASE_VIEW); }, [category, query]);
  const activeId = hovered || selected?.id;
  const zoom = (factor: number) => setView((current) => {
    const scale = Math.min(2.6, Math.max(0.75, current.scale * factor));
    return { scale, x: 450 - (450 - current.x) * scale / current.scale, y: 285 - (285 - current.y) * scale / current.scale };
  });
  const focus = () => {
    if (!selected) return;
    const x = selected.side === "left" ? 160 : 740;
    setView({ scale: 1.5, x: 450 - x * 1.5, y: 285 - selected.y * 1.5 });
  };

  return <section className="panel evn-panel evn-studio">
    <div className="section-title">
      <div><span className="eyebrow">CONNECTED BIOLOGICAL EVIDENCE</span><h2>{entity.name} · 证据关系网络</h2></div>
      <span className="badge neutral"><Network size={15} />{nodes.length} / {allNodes.length} 个快照节点</span>
    </div>
    <p className="section-description">实线为数据库关联，虚线为 TxGNN 预测。当前快照每侧取前 7 条；筛选作用于这组节点。<a href={`#/${entity.kind}/${encodeURIComponent(entity.id)}/diseases`}>查看完整疾病与机制 <ArrowUpRight size={13} /></a></p>
    <div className="evn-filter-toolbar">
      <div className="evn-category-filters" role="group" aria-label="网络关系类型筛选">
        {["all", ...categories].map(type => <button type="button" key={type} aria-pressed={category === type} onClick={() => setCategory(type)}>{type === "all" ? "全部关联" : type}<span>{type === "all" ? allNodes.length : allNodes.filter(node => node.category === type).length}</span></button>)}
      </div>
      <label className="evn-node-search"><Search size={15} /><input aria-label="搜索当前网络节点" placeholder="搜索当前网络节点…" value={query} onChange={event => setQuery(event.target.value)} />{query && <button type="button" aria-label="清除网络搜索" onClick={() => setQuery("")}><X size={14} /></button>}</label>
    </div>
    <div className="evn-toolbar">
      <div className="evn-legend"><span><i className="evn-dot teal" />{entity.kind === "drug" ? "已知靶点" : "生物通路"}</span><span><i className="evn-dot blue" />疾病关联</span>{entity.kind === "drug" && <span><i className="evn-dot violet" />TxGNN 预测</span>}</div>
      <div className="evn-tools" role="group" aria-label="证据网络视图控制">
        <button type="button" title="缩小网络" aria-label="缩小网络" onClick={() => zoom(1 / 1.2)} disabled={view.scale <= 0.75}><ZoomOut size={17} /></button>
        <span className="evn-zoom">{Math.round(view.scale * 100)}%</span>
        <button type="button" title="放大网络" aria-label="放大网络" onClick={() => zoom(1.2)} disabled={view.scale >= 2.6}><ZoomIn size={17} /></button>
        <button type="button" title="聚焦已选节点" aria-label="聚焦已选节点" disabled={!selected} onClick={focus}><Focus size={17} /></button>
        <button type="button" title="复位网络视图" aria-label="复位网络视图" onClick={() => setView(BASE_VIEW)}><RotateCcw size={17} /></button>
      </div>
    </div>
    <div className="evn-layout">
      <div className="evn-canvas">
        <svg ref={svgRef} viewBox="0 0 900 570" role="group" aria-label={`${entity.name} 的关联网络；可用 Tab 选择节点并按 Enter 查看详情`}
          onPointerDown={(event) => {
            if ((event.target as Element).closest(".evn-node") || event.button !== 0) return;
            drag.current = { x: event.clientX, y: event.clientY, viewX: view.x, viewY: view.y };
            event.currentTarget.setPointerCapture(event.pointerId);
          }}
          onPointerMove={(event) => {
            if (!drag.current || !svgRef.current) return;
            const ratio = 900 / svgRef.current.getBoundingClientRect().width;
            setView((current) => ({ ...current, x: drag.current!.viewX + (event.clientX - drag.current!.x) * ratio, y: drag.current!.viewY + (event.clientY - drag.current!.y) * ratio }));
          }}
          onPointerUp={() => { drag.current = null; }} onPointerCancel={() => { drag.current = null; }}>
          <defs><pattern id={gridId} width="24" height="24" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1" fill="#CDDAE9" /></pattern></defs>
          <rect width="900" height="570" fill={`url(#${gridId})`} />
          <g transform={`translate(${view.x} ${view.y}) scale(${view.scale})`}>
            <circle cx="450" cy="285" r="135" fill="#EAF0FF" opacity=".65" />
            {[90, 135].map((radius) => <circle key={radius} cx="450" cy="285" r={radius} stroke="#BDCBEB" strokeWidth="1" strokeDasharray={radius === 135 ? "4 9" : undefined} fill="none" />)}
            <text x="34" y="36" className="evn-column-label">{entity.kind === "drug" ? "TARGET ASSOCIATIONS" : "BIOLOGICAL PATHWAYS"}</text>
            <text x="634" y="36" className="evn-column-label">DISEASE ASSOCIATIONS</text>
            {nodes.map((node) => <path className="evn-edge" key={`edge-${node.id}`} d={`M 450 285 C ${node.side === "left" ? 350 : 550} 285 ${node.side === "left" ? 370 : 530} ${node.y} ${node.x} ${node.y}`} stroke={node.color} strokeWidth={activeId === node.id ? 3.5 : 2} strokeOpacity={activeId && activeId !== node.id ? 0.18 : 0.78} strokeDasharray={node.row.prediction ? "7 6" : undefined} fill="none" />)}
            <circle cx="450" cy="285" r="58" fill="#09172D" stroke="#2459E8" strokeWidth="3" />
            <circle cx="450" cy="285" r="66" fill="none" stroke="#2459E8" strokeOpacity=".18" strokeWidth="7" />
            <text x="450" y="284" fill="#FFFFFF" textAnchor="middle" fontSize="18" fontWeight="750">{entity.name.length > 12 ? `${entity.name.slice(0, 10)}…` : entity.name}</text>
            <text x="450" y="306" fill="#97BAFF" textAnchor="middle" fontSize="12" letterSpacing="1.5">{entity.kind === "drug" ? "DRUG" : "TARGET"}</text>
            {nodes.map((node) => {
              const cardX = node.side === "left" ? 25 : 620;
              const active = activeId === node.id;
              return <g role="button" tabIndex={0} aria-pressed={selected?.id === node.id} aria-label={`查看 ${label(node.row)}，${node.category}`} key={node.id}
                className={`evn-node${active ? " is-active" : ""}`} onClick={() => setSelected(node)}
                onMouseEnter={() => setHovered(node.id)} onMouseLeave={() => setHovered(null)}
                onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setSelected(node); } if (event.key === "Escape") setSelected(null); }}>
                <title>{label(node.row)} · {node.category}</title>
                <rect className="evn-node-card" x={cardX} y={node.y - 26} width="255" height="52" rx="9" fill={active ? "#EDF3FF" : "#FFFFFF"} stroke={active ? node.color : "#CAD6E6"} strokeWidth={active ? 2 : 1} />
                <rect x={cardX} y={node.y - 16} width="4" height="32" rx="2" fill={node.color} />
                <text x={cardX + 15} y={node.y - 3} fill="#142338" fontSize="16" fontWeight="650">{shortLabel(label(node.row))}</text>
                <text x={cardX + 15} y={node.y + 16} fill="#526176" fontSize="14">{node.category}</text>
                <circle cx={node.x} cy={node.y} r={active ? 6 : 4.5} fill={node.color} stroke="#FFFFFF" strokeWidth="2" />
              </g>;
            })}
          </g>
        </svg>
        {!nodes.length && <div className="evn-empty">{allNodes.length ? "当前筛选没有匹配节点；可调整类型或搜索词。" : "该实体暂无已接入的关系节点。"}</div>}
        <div className="evn-hint"><Move size={14} />拖动平移 · 点击节点查看来源 · 使用工具栏缩放</div>
      </div>
      <aside className="evn-inspector" aria-live="polite">
        {selected ? <>
          <div className="evn-inspector-heading"><span className="eyebrow">ASSOCIATION DETAILS</span><button type="button" aria-label="取消选中节点" onClick={() => setSelected(null)}><X size={17} /></button></div>
          <span className="evn-category" style={{ color: selected.color }}>{selected.category}</span>
          <h3>{label(selected.row)}</h3>
          <dl><div><dt>数据库编号</dt><dd>{selected.row.id || "—"}</dd></div><div><dt>证据来源</dt><dd>{selected.row.source || "本地已知关系"}</dd></div>
            {selected.row.score != null && <div><dt>关联分数</dt><dd className="evn-score">{fmt(selected.row.score)}</dd></div>}
            {selected.row.phase != null && <div><dt>临床阶段</dt><dd>{selected.row.phase === 4 ? "阶段 4 · 已批准" : `阶段 ${selected.row.phase} · 研究记录`}</dd></div>}
          </dl>
          {selected.row.prediction && <p className="evn-prediction-note">模型预测关联，尚不代表已验证的适应症。</p>}
          {(selected.row.explorable ?? selected.row.in_catalog) && <a className="button secondary" href={`#/target/${encodeURIComponent(selected.row.id)}/overview`}>打开靶点档案<ArrowUpRight size={16} /></a>}
          {selected.row.url && <a className="evn-source-link" href={selected.row.url} target="_blank" rel="noreferrer">查看原始来源<ExternalLink size={15} /></a>}
        </> : <>
          <div className="evn-inspector-icon"><Network size={27} /></div><span className="eyebrow">EXPLORE THE CONNECTIONS</span>
          <h3>关系检查器</h3><p>选择图中节点，在此查看数据库编号、关联分数与原始证据。</p>
          <div className="evn-inspector-stats"><strong>{nodes.length}</strong><span>当前展示的关系节点</span></div>
          <div className="evn-inspector-note">关联代表相应数据源的记录，关系类型以节点标注为准。</div>
        </>}
      </aside>
    </div>
  </section>;
}
