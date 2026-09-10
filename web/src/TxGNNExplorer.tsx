import { Fragment, useEffect, useId, useMemo, useState } from "react";
import {
  ArrowDownToLine, ChevronDown, ChevronLeft, ChevronRight, Database,
  ExternalLink, Info, Layers3, LoaderCircle, Network, Search,
  ShieldAlert, ShieldCheck, X,
} from "lucide-react";
import { api, type Entity } from "./types";
import "./txgnn-explorer.css";

type Scope = "eligible" | "all";
type Direction = { id: string; label: string; count?: number };
type TxMetadata = {
  status?: string; source_status?: string; raw_available?: boolean;
  interpretation_eligible?: boolean; raw_count?: number; eligible_count?: number;
  excluded_count?: number; identity_hold?: boolean; mapping_rule?: string;
  mapping_status?: string; graph_drug_id?: string; graph_drug_name?: string;
  direction_count?: number; score_semantics?: string; source_path?: string;
};
type TxRow = {
  id: string; name?: string; score?: number | null; logit?: number | null;
  rank?: number | null; denominator?: number | null;
  raw_rank?: number | null; raw_denominator?: number | null;
  eligible_rank?: number | null; eligible_denominator?: number | null;
  directions?: string[]; direction_labels?: string[];
  exclusion_mask?: number | null; exclusion_reasons?: string[];
  interpretation_eligible?: boolean; mapping_scope?: string;
  member_disease_ids?: string[]; source?: string; source_path?: string;
  relation?: string;
};
type TxResult = {
  items: TxRow[]; total: number; all_total: number; page: number;
  page_size: number; denominator?: number;
};
type DiseaseSummary = {
  directions?: Direction[]; unclassified_count?: number;
  disease_evidence?: { directions?: Direction[]; unclassified_count?: number };
};
const PAGE_SIZE = 20;
const finite = (value: unknown): number | null => typeof value === "number" && Number.isFinite(value) ? value : null;
const count = (value: unknown) => finite(value)?.toLocaleString("en-US") ?? "—";
const rawLogit = (row: TxRow) => finite(row.logit) ?? finite(row.score);
const scoreText = (value: unknown) => finite(value)?.toFixed(4) ?? "—";
const rankText = (rank: unknown, denominator: unknown) => finite(rank) == null ? "未进入该范围排名" : `#${count(rank)} / ${count(denominator)}`;
const maskReasons = (row: TxRow) => {
  if (row.exclusion_reasons?.length) return row.exclusion_reasons;
  const mask = finite(row.exclusion_mask);
  if (mask == null) return ["未提供排除标记"];
  return [
    ...(mask & 1 ? ["bit 1 · 旧图已有 indication / off-label / contraindication 关系"] : []),
    ...(mask & 2 ? ["bit 2 · EC 治疗 / 试验 / 禁忌来源的保守排除"] : []),
    ...(mask & 4 ? ["bit 4 · 药物身份映射冲突，暂扣解释资格"] : []),
    ...(mask & ~7 ? [`其他排除位 · mask ${mask}`] : []),
  ];
};

/** Full, paginated all-disease inference. The former cancer-only list is never a fallback. */
export default function TxGNNExplorer({ entity }: { entity: Entity }) {
  const metadata = entity.txgnn as TxMetadata | undefined;
  const instanceId = useId().replace(/:/g, "");
  const [scope, setScope] = useState<Scope>("eligible");
  const [direction, setDirection] = useState("");
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(1);
  const [pageDraft, setPageDraft] = useState("1");
  const [expanded, setExpanded] = useState("");
  const [revision, setRevision] = useState(0);
  const [directions, setDirections] = useState<Direction[]>([]);
  const [directionsError, setDirectionsError] = useState("");
  const [catalogRevision, setCatalogRevision] = useState(0);
  const [response, setResponse] = useState<{ key: string; result: TxResult } | null>(null);
  const [failure, setFailure] = useState<{ key: string; message: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const rawAvailable = metadata?.raw_available === true;
  const identityHold = metadata?.identity_hold === true || metadata?.status === "identity_hold";
  const interpretationEligible = metadata?.interpretation_eligible === true && !identityHold;
  const canRequest = entity.kind === "drug" && rawAvailable && (scope === "all" || interpretationEligible);
  const params = new URLSearchParams({ kind: "drug", id: entity.id, section: "txgnn_diseases", scope, direction, search: query.trim(), page: String(page), page_size: String(PAGE_SIZE) });
  const requestKey = params.toString();
  const result = response?.key === requestKey ? response.result : null;
  const error = failure?.key === requestKey ? failure.message : "";
  const loading = canRequest && (busy || (!result && !error));
  const scopeCount = scope === "eligible" ? metadata?.eligible_count : metadata?.raw_count;
  const denominator = finite(result?.denominator) ?? finite(scopeCount);
  const pages = Math.max(1, Math.ceil((result?.total ?? 0) / PAGE_SIZE));
  const exportParams = new URLSearchParams(params);
  exportParams.delete("page"); exportParams.delete("page_size");
  const csvUrl = `/api/evidence.csv?${exportParams.toString()}`;

  useEffect(() => {
    setScope("eligible"); setDirection(""); setQuery(""); setPage(1);
    setExpanded(""); setResponse(null); setFailure(null);
  }, [entity.id]);
  useEffect(() => { setExpanded(""); setPageDraft(String(page)); }, [requestKey, page]);
  useEffect(() => {
    const abort = new AbortController();
    api<DiseaseSummary>("/api/disease-summary", abort.signal).then(summary => {
      const catalog = summary.disease_evidence || summary;
      const options = [...(catalog.directions || [])];
      if (!options.some(item => item.id === "unclassified") && finite(catalog.unclassified_count) != null) {
        options.push({ id: "unclassified", label: "未分类", count: catalog.unclassified_count });
      }
      setDirections(options); setDirectionsError("");
    }).catch(reason => { if (reason.name !== "AbortError") setDirectionsError(reason.message); });
    return () => abort.abort();
  }, [catalogRevision]);
  useEffect(() => {
    if (!canRequest) { setBusy(false); return; }
    const abort = new AbortController();
    setBusy(true); setFailure(null);
    const timer = setTimeout(() => {
      api<TxResult>(`/api/evidence?${requestKey}`, abort.signal).then(data => {
        if (!Array.isArray(data.items) || !Number.isFinite(data.total)) throw new Error("全疾病响应格式不完整，请重新加载。");
        if (scope === "eligible" && data.items.some(row => row.exclusion_mask !== 0 || row.interpretation_eligible === false)) {
          throw new Error("候选响应含排除标记，已停止展示。请重新加载或切换原始追溯核对。");
        }
        if (!abort.signal.aborted) setResponse({ key: requestKey, result: data });
      }).catch(reason => {
        if (reason.name !== "AbortError") setFailure({ key: requestKey, message: reason.message });
      }).finally(() => { if (!abort.signal.aborted) setBusy(false); });
    }, query.trim() ? 180 : 0);
    return () => { clearTimeout(timer); abort.abort(); };
  }, [requestKey, canRequest, scope, query, revision]);

  const scoredRows = useMemo(() => (result?.items || []).flatMap((row, index) => {
    const value = rawLogit(row);
    return value == null ? [] : [{ row, value, index }];
  }), [result]);
  const minimum = Math.min(0, ...scoredRows.map(item => item.value));
  const maximum = Math.max(0, ...scoredRows.map(item => item.value));
  const chartSpan = maximum - minimum || 1;
  const zeroPosition = (0 - minimum) / chartSpan * 100;
  const ranked = (row: TxRow) => scope === "all" ? finite(row.raw_rank) ?? finite(row.rank) : finite(row.rank);
  const rowDenominator = (row: TxRow) => scope === "all"
    ? finite(row.raw_denominator) ?? finite(row.denominator) ?? denominator
    : finite(row.denominator) ?? denominator;
  const directionLabel = (id: string) => directions.find(item => item.id === id)?.label || id;
  const labels = (row: TxRow) => row.direction_labels?.length ? row.direction_labels : row.directions?.map(directionLabel) || [];
  const rowKey = (row: TxRow, index: number) => `${row.id}-${index}`;
  const rowId = (index: number) => `${instanceId}-tx-row-${index}`;
  const openChartRow = (row: TxRow, index: number) => {
    setExpanded(rowKey(row, index));
    document.getElementById(rowId(index))?.scrollIntoView({ block: "center", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  };
  const resetFilters = () => { setQuery(""); setDirection(""); setPage(1); };
  const availabilityTitle = !metadata ? "尚未接入本轮全疾病映射状态"
    : identityHold && scope === "eligible" ? "身份冲突暂扣 · 默认候选为空"
    : !rawAvailable ? "当前药物没有本轮 TxGNN 原始分数"
    : "当前实体未获得候选解释资格";
  const availabilityText = !metadata ? "当前实体缺少全疾病结果的映射元数据，暂不展示疾病推理。"
    : identityHold && scope === "eligible" ? "该药物的规范节点存在身份冲突。原始分数保留用于追溯，暂不进入疾病线索审查。"
    : !rawAvailable ? metadata.status === "source_evidence_only"
      ? "已有来源关系线索可供核查，但没有本轮全疾病模型分数。图谱来源关系不会自动变成模型推理。"
      : "本轮未获得可用的全疾病图节点映射或模型分数。缺失不表示药物与这些疾病没有关系。"
    : "原始结果仅用于身份与来源核对，当前不进入默认候选列表。";

  return <section className="txgnn-explorer" aria-label={`${entity.name} 的 TxGNN 全疾病线索`}>
    <header className="tx-heading">
      <div><span className="tx-kicker"><Network size={15} />TXGNN · ALL-DISEASE INFERENCE</span><h2>TxGNN 疾病线索</h2><p>原权重全疾病推理 · indication 查询 · 原始 logit</p></div>
      <span className={`tx-mapping-status ${identityHold ? "hold" : interpretationEligible ? "eligible" : "unavailable"}`}>
        {interpretationEligible ? <ShieldCheck size={16} /> : <ShieldAlert size={16} />}
        {identityHold ? "身份冲突暂扣" : interpretationEligible ? "可进入线索审查" : "无可解释候选"}
      </span>
    </header>

    <div className="tx-metrics" aria-label="当前药物全疾病覆盖">
      <div><span>原始疾病节点</span><strong>{count(metadata?.raw_count)}</strong><small>{rawAvailable ? "完整推理行" : "没有原始分数"}</small></div>
      <div><span>可审查候选</span><strong>{count(metadata?.eligible_count)}</strong><small>仅 exclusion_mask = 0</small></div>
      <div><span>排除 / 暂扣节点</span><strong>{count(metadata?.excluded_count)}</strong><small>已有关系与身份标记</small></div>
      <div><span>候选覆盖方向</span><strong>{count(metadata?.direction_count)}</strong><small>多标签，不含未分类</small></div>
    </div>

    <div className="tx-mapping-strip">
      <span><Database size={14} />图药节点 <strong>{metadata?.graph_drug_name || "未提供"}</strong>{metadata?.graph_drug_id && <code>{metadata.graph_drug_id}</code>}</span>
      <details className="tx-mapping-details"><summary>映射与来源 <ChevronDown size={13} /></summary><dl>
        <div><dt>本地模型实体</dt><dd>{entity.name} · {entity.id}</dd></div>
        <div><dt>映射规则</dt><dd>{metadata?.mapping_rule || "未提供"}</dd></div>
        <div><dt>映射状态</dt><dd>{metadata?.mapping_status || metadata?.status || "未提供"}</dd></div>
        {metadata?.source_status && <div><dt>来源审查状态</dt><dd>{metadata.source_status}</dd></div>}
        <div><dt>分数定义</dt><dd>{metadata?.score_semantics || "TxGNN 原始 indication logit，未校准为临床或 SPR 概率。"}</dd></div>
        <div><dt>来源文件</dt><dd><code>{metadata?.source_path || "未提供"}</code></dd></div>
      </dl></details>
    </div>

    <div className="tx-scope-toolbar">
      <div className="tx-scope-switch" role="group" aria-label="TxGNN 结果范围">
        <button type="button" aria-pressed={scope === "eligible"} onClick={() => { setScope("eligible"); setPage(1); }}><ShieldCheck size={15} />候选审查<span>{count(metadata?.eligible_count)}</span></button>
        <button type="button" aria-pressed={scope === "all"} onClick={() => { setScope("all"); setPage(1); }}><Layers3 size={15} />原始追溯<span>{count(metadata?.raw_count)}</span></button>
      </div>
      {canRequest && result && result.total > 0 && !loading ? <a href={csvUrl} className="tx-export" download><ArrowDownToLine size={15} />导出筛选结果 CSV</a>
        : <button type="button" className="tx-export" disabled><ArrowDownToLine size={15} />导出筛选结果 CSV</button>}
    </div>
    <p className={`tx-scope-note ${scope === "all" ? "raw" : ""}`}><Info size={16} /><span>{scope === "eligible"
      ? "候选仅纳入排除标记为 0 的结果。原始 logit 越高表示该模型内排序越靠前；分数不是治疗概率，也不代表实验验证。"
      : "原始追溯包含已有关系、保守排除与身份暂扣记录。排除标记用于限制解释范围，不表示模型验证的阳性或阴性。"}</span></p>

    <div className="tx-filter-toolbar">
      <label className="tx-direction-filter"><span>疾病方向</span><select aria-label="筛选 TxGNN 疾病方向" value={direction} disabled={!canRequest} onChange={event => { setDirection(event.target.value); setPage(1); }}>
        <option value="">全部方向（含未分类）</option>
        {directions.map(item => <option key={item.id} value={item.id}>{item.label}{finite(item.count) != null ? ` · ${count(item.count)} 节点` : ""}</option>)}
      </select></label>
      <label className="tx-search"><Search size={16} /><input aria-label="搜索 TxGNN 疾病" placeholder="疾病名称、MONDO 或合并节点编号…" value={query} disabled={!canRequest} onChange={event => { setQuery(event.target.value); setPage(1); }} />{query && <button type="button" aria-label="清除 TxGNN 疾病搜索" onClick={() => { setQuery(""); setPage(1); }}><X size={15} /></button>}</label>
      {(direction || query) && <button type="button" className="tx-clear-filters" onClick={resetFilters}>重置筛选</button>}
    </div>
    <div className="tx-filter-context"><span>方向数为图中疾病节点数；同一节点可有多个方向。</span><span>{scope === "eligible" ? "候选" : "原始"}全局排名分母 <strong>{count(denominator)}</strong> · 筛选不重排</span></div>
    {directionsError && <div className="tx-catalog-error" role="status">方向目录暂时无法加载。{canRequest && "仍可检索全部疾病。"}<button type="button" onClick={() => setCatalogRevision(value => value + 1)}>重试方向目录</button></div>}

    {!canRequest ? <div className={`tx-empty ${identityHold ? "hold" : ""}`} role="status">
      {identityHold ? <ShieldAlert size={30} /> : <Database size={30} />}<h3>{availabilityTitle}</h3><p>{availabilityText}</p>
      {identityHold && rawAvailable && scope === "eligible" && <button type="button" onClick={() => { setScope("all"); setPage(1); }}>查看原始追溯 <ChevronRight size={15} /></button>}
      <span>此处仅展示本轮全疾病结果；未映射实体保留缺失状态。</span>
    </div> : error ? <div className="tx-error" role="alert"><ShieldAlert size={27} /><h3>全疾病结果暂时无法加载</h3><p>{error}</p><button type="button" onClick={() => setRevision(value => value + 1)}>重新加载 TxGNN 结果</button></div>
    : loading ? <div className="tx-loading" role="status"><LoaderCircle className="spin" size={27} /><strong>正在读取全疾病结果</strong><p>应用当前范围、方向与搜索条件…</p></div>
    : !result?.items.length ? <div className="tx-empty" role="status"><Search size={27} /><h3>{query || direction ? "当前筛选没有匹配疾病" : "当前范围没有可审查结果"}</h3><p>{query || direction ? "可调整疾病方向，或使用疾病名称、MONDO 编号检索。" : "此范围未提供疾病候选；可在原始追溯中核查现有结果与排除标记。"}</p>{(query || direction) && <button type="button" onClick={resetFilters}>清除筛选条件</button>}</div>
    : <>
      {scoredRows.length > 0 && <div className="tx-page-chart" aria-label="本页原始 logit 速览">
        <div className="tx-chart-heading"><span>本页 logit 速览</span><small>本页前 {Math.min(6, scoredRows.length)} 条有分数记录 · 点击查看证据</small><span>本页分数轴 {scoreText(minimum)} – {scoreText(maximum)}</span></div>
        <div className="tx-logit-series">{scoredRows.slice(0, 6).map(({ row, value, index }) => <button type="button" key={rowKey(row, index)} onClick={() => openChartRow(row, index)} aria-label={`查看 ${row.name || row.id} 的 TxGNN 证据，原始 logit ${scoreText(value)}`}>
          <span className="tx-chart-name"><small>#{count(ranked(row))}</small><span>{row.name || row.id}</span></span><span className="tx-logit-track" aria-hidden="true"><i className="tx-zero-axis" style={{ left: `${zeroPosition}%` }} /><i className="tx-logit-bar" style={{ left: `${(Math.min(0, value) - minimum) / chartSpan * 100}%`, width: `${Math.abs(value) / chartSpan * 100}%` }} /><i className="tx-logit-dot" style={{ left: `${(value - minimum) / chartSpan * 100}%` }} /></span><strong>{scoreText(value)}</strong>
        </button>)}</div>
      </div>}
      <div className="tx-results-heading"><strong>{count(result.total)} 条筛选结果</strong><span>每页 {PAGE_SIZE} 条 · 全量结果可分页浏览与导出</span></div>
      <div className="tx-table-scroll"><table className="tx-ranking-table"><thead><tr><th>{scope === "eligible" ? "候选全局排名" : "原始全局排名"}</th><th>疾病节点</th><th>原始 logit</th><th>方向标签</th><th>解释范围</th><th><span className="tx-sr-only">展开证据</span></th></tr></thead><tbody>
        {result.items.map((row, index) => {
          const key = rowKey(row, index), opened = expanded === key;
          const mask = finite(row.exclusion_mask), rowEligible = mask === 0 && row.interpretation_eligible !== false;
          const members = row.member_disease_ids || [], directionNames = labels(row);
          const statusTags = rowEligible ? ["候选可审查"] : mask == null ? ["资格未标注"] : [
            ...(mask & 1 ? ["bit 1 · 旧图关系"] : []),
            ...(mask & 2 ? ["bit 2 · EC 排除"] : []),
            ...(mask & 4 ? ["bit 4 · 身份暂扣"] : []),
            ...(mask === 0 || mask & ~7 ? ["解释资格暂扣"] : []),
          ];
          return <Fragment key={key}>
            <tr className={`tx-result-row${opened ? " is-open" : ""}${!rowEligible ? " is-excluded" : ""}`} id={rowId(index)} onClick={() => setExpanded(opened ? "" : key)}>
              <td className="tx-rank-cell"><strong>{finite(ranked(row)) == null ? "—" : `#${count(ranked(row))}`}</strong><small>/ {count(rowDenominator(row))}</small></td>
              <td className="tx-disease-cell"><button type="button" className="tx-disease-button" aria-expanded={opened} aria-controls={`${rowId(index)}-detail`} onClick={event => { event.stopPropagation(); setExpanded(opened ? "" : key); }}><strong>{row.name || row.id}</strong><code>{row.id}</code>{members.length > 1 && <span className="tx-merged-badge">合并疾病节点 · {members.length} 个成员</span>}</button></td>
              <td className="tx-score-cell"><strong title={rawLogit(row) == null ? "原始分数缺失" : `原始 logit: ${rawLogit(row)}`}>{scoreText(rawLogit(row))}</strong>{rawLogit(row) == null && <small>分数缺失</small>}</td>
              <td className="tx-directions-cell"><div>{directionNames.length ? directionNames.slice(0, 3).map((name, i) => <span key={`${name}-${i}`}>{name}</span>) : <span className="unclassified">未分类</span>}{directionNames.length > 3 && <span title={directionNames.slice(3).join(" · ")}>+{directionNames.length - 3}</span>}</div></td>
              <td className="tx-status-cell"><div className="tx-status-tags">{statusTags.map(tag => <span key={tag} className={`tx-row-status ${rowEligible ? "eligible" : "excluded"}`}>{tag}</span>)}</div><small>mask {mask == null ? "—" : mask}</small></td>
              <td className="tx-chevron-cell"><ChevronDown size={16} className={opened ? "open" : ""} aria-hidden="true" /></td>
            </tr>
            {opened && <tr className="tx-detail-row"><td colSpan={6}><div className="tx-row-detail" id={`${rowId(index)}-detail`} role="region" aria-label={`${row.name || row.id} 的原始疾病证据`}>
              <div className="tx-detail-heading"><strong>原始分数与解释边界</strong><span>{row.relation || "indication"} 查询</span></div>
              <dl className="tx-detail-metrics"><div><dt>原始 logit · 完整数值</dt><dd>{rawLogit(row) == null ? "—" : String(rawLogit(row))}</dd></div><div><dt>候选范围全局排名</dt><dd>{rankText(row.eligible_rank !== undefined ? row.eligible_rank : scope === "eligible" ? row.rank : null, finite(row.eligible_denominator) ?? (scope === "eligible" ? finite(row.denominator) : finite(metadata?.eligible_count)))}</dd></div><div><dt>原始全局排名</dt><dd>{rankText(row.raw_rank, row.raw_denominator)}</dd></div><div><dt>排除 mask</dt><dd>{mask == null ? "未提供" : mask}</dd></div></dl>
              <div className={`tx-exclusion-detail ${rowEligible ? "eligible" : "excluded"}`}><strong>{rowEligible ? "mask 0 · 未触发本轮排除条件" : "排除 / 暂扣原因"}</strong>{rowEligible ? <p>可作为待审查疾病线索；不代表已证明有效或已完成全部适应症查新。</p> : <ul>{maskReasons(row).map(reason => <li key={reason}>{reason}</li>)}</ul>}</div>
              <div className="tx-detail-columns"><div><span className="tx-detail-label">疾病节点与映射</span><code>{row.id}</code>{row.mapping_scope && <p>{row.mapping_scope}</p>}{members.length > 0 && <div className="tx-member-ids">{members.map((member, i) => <span key={`${member}-${i}`}>{member}</span>)}</div>}{members.length > 1 && <p className="tx-merged-note">分数属于这个合并节点，不能视为每个成员病种的特异性预测。</p>}<div className="tx-expanded-directions">{directionNames.length ? directionNames.map((name, i) => <span key={`${name}-${i}`}>{name}</span>) : <span>未分类节点仍保留原始分数</span>}</div></div>
                <div><span className="tx-detail-label">来源追溯</span><strong>{row.source || "TxGNN 原权重全疾病推理"}</strong><code className="tx-source-path">{row.source_path || metadata?.source_path || "未提供来源路径"}</code>{/^https?:\/\//.test(row.source || "") && <a href={row.source} target="_blank" rel="noreferrer">打开来源 <ExternalLink size={13} /></a>}<p>logit 为未校准模型分数，不是 sigmoid 概率、药效或治疗推荐。</p></div></div>
            </div></td></tr>}
          </Fragment>;
        })}
      </tbody></table></div>
      <div className="tx-pagination"><span>显示 {(page - 1) * PAGE_SIZE + 1}–{Math.min(page * PAGE_SIZE, result.total)} / {count(result.total)} 条</span><div className="tx-page-buttons"><button type="button" aria-label="上一页 TxGNN 疾病" disabled={page <= 1 || loading} onClick={() => setPage(value => value - 1)}><ChevronLeft size={16} /></button><span>{page} / {pages}</span><button type="button" aria-label="下一页 TxGNN 疾病" disabled={page >= pages || loading} onClick={() => setPage(value => value + 1)}><ChevronRight size={16} /></button></div><form onSubmit={event => { event.preventDefault(); const target = Number(pageDraft); if (Number.isFinite(target)) setPage(Math.max(1, Math.min(pages, Math.floor(target)))); }}><label>跳至<input type="number" min={1} max={pages} value={pageDraft} onChange={event => setPageDraft(event.target.value)} aria-label="跳转 TxGNN 页码" /></label><button type="submit">前往</button></form></div>
    </>}
    <details className="tx-mask-legend"><summary>如何理解排除标记与多标签方向 <ChevronDown size={14} /></summary><div><p><strong>bit 1</strong>：旧图已有适应症、off-label 或禁忌关系。<strong>bit 2</strong>：EC 治疗、试验或禁忌来源的保守排除，不等于已批准适应症清单。<strong>bit 4</strong>：药物身份映射冲突。多项标记可叠加；仅 mask 0 进入默认候选。</p><p>同一疾病节点可有多个方向标签，方向数量不能相加视为独立疾病数。未分类节点仍保留推理结果；合并 MONDO 节点保留成员 ID，不拆分复制分数。</p></div></details>
  </section>;
}
