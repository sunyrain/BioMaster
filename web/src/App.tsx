import "./pair-drawer.css";
import { lazy, Suspense, useEffect, useId, useRef, useState } from "react";
import { LoadingState, PageBoundary, ConnectionStatus } from "./LoadingState";
import { createPortal } from "react-dom";
import {
  Bookmark,
  ChartNoAxesCombined,
  Grid2X2,
  List,
  ArrowDownToLine,
  ArrowLeft,
  ArrowRight,
  ArrowUpRight,
  Atom,
  BookOpen,
  Box,
  Check,
  Copy,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CircleHelp,
  Database,
  Dna,
  ExternalLink,
  FlaskConical,
  Layers3,
  LayoutDashboard,
  Link2,
  LoaderCircle,
  Network,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Target,
  X,
} from "lucide-react";
import {
  api,
  Entity,
  EntityRef,
  Evidence,
  fmt,
  Kind,
  label,
  MODEL_COLORS,
  MODEL_NAMES,
  Model,
  MODELS,
  Ranking,
  Rankings,
  Summary,
} from "./types";
const StructureViewer = lazy(() => import("./StructureViewer"));
import EvidenceBrowser, { Metadata } from "./EvidenceBrowser";
const EvidenceNetwork = lazy(() => import("./EvidenceNetwork"));
const Experiments = lazy(() => import("./Experiments"));
const SupplementalEvidence = lazy(() => import("./SupplementalEvidence"));
import { AnimatedNumber, Reveal } from "./motion";
import ResearchHome from "./ResearchHome";
const BiologyInsights = lazy(() => import("./BiologyInsights"));
import RankVisuals, { type RankView } from "./RankVisuals";
import Workspace, { SaveEntityButton, RecentEntities, recordVisit } from "./Workspace";

const AffinityAtlas = lazy(() => import("./AffinityAtlas"));
const SPRDirectory = lazy(() => import("./SPRDirectory"));
const ResearchBrowse = lazy(() => import("./ResearchBrowse"));

type Route = {
  page: "home" | "drugs" | "targets" | "sources" | "workspace" | "entity" | "browse" | "spr" | "affinity";
  kind?: Kind;
  id?: string;
  tab?: string;
};
const parseRoute = (): Route => {
  const parts = location.hash.replace(/^#\/?/, "").split("/");
  return parts[0] === "drug" || parts[0] === "target"
    ? {
        page: "entity",
        kind: parts[0],
        id: decodeURIComponent(parts[1] || ""),
        tab: parts[2] || "overview",
      }
    : parts[0] === "spr" ? { page: "spr", tab: parts[1] || "baseline" } : parts[0] === "browse" ? { page: "browse", tab: parts[1] || "chembl" } : {
        page: (["drugs", "targets", "sources", "workspace", "spr", "affinity"].includes(parts[0])
          ? parts[0]
          : "home") as Route["page"],
      };
};
const go = (path: string) => {
  location.hash = "/" + path;
};
const openEntity = (e: EntityRef, tab = "overview") =>
  go(`${e.kind}/${encodeURIComponent(e.id)}/${tab}`);
const number = (n?: number) => (n ?? 0).toLocaleString("en-US");

function Empty({
  title = "当前数据中暂无记录",
  text = "此处只展示已接入的项目结果。",
}: {
  title?: string;
  text?: string;
}) {
  return (
    <div className="empty">
      <Database size={24} />
      <strong>{title}</strong>
      <span>{text}</span>
    </div>
  );
}
function Loading({ text = "正在读取项目数据…" }: { text?: string }) {
  return (
    <LoadingState text={text} />
  );
}
function ErrorPanel({ error, retry }: { error: string; retry?: () => void }) {
  return (
    <div className="error-panel">
      <CircleHelp size={22} />
      <div>
        <strong>数据暂时无法加载</strong>
        <p>{error}</p>
        {retry && (
          <button className="button secondary" onClick={retry}>
            重新加载
          </button>
        )}
      </div>
    </div>
  );
}
function Badge({
  children,
  tone = "green",
}: {
  children: React.ReactNode;
  tone?: string;
}) {
  return <span className={`badge ${tone}`}>{children}</span>;
}
function SectionTitle({
  eyebrow,
  title,
  children,
}: {
  eyebrow?: string;
  title: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="section-title">
      <div>
        {eyebrow && <span className="eyebrow">{eyebrow}</span>}
        <h2>{title}</h2>
      </div>
      {children}
    </div>
  );
}
function SourceLink({ row }: { row: Evidence }) {
  const url = row.url || row.link;
  return url && /^https?:\/\//.test(url) ? (
    <a
      className="source-link"
      href={url}
      target="_blank"
      rel="noreferrer"
      title="打开证据来源"
    >
      <ExternalLink size={14} />
    </a>
  ) : null;
}

function CopyButton({
  text,
  label = "复制链接",
  className = "button secondary",
  children,
}: {
  text: string;
  label?: string;
  className?: string;
  children?: React.ReactNode;
}) {
  const [status, setStatus] = useState("idle");
  useEffect(() => {
    if (status === "idle") return;
    const timer = setTimeout(() => setStatus("idle"), 2200);
    return () => clearTimeout(timer);
  }, [status]);
  return (
    <button
      type="button"
      className={className}
      title={status === "copied" ? "已复制" : label}
      aria-label={status === "copied" ? "已复制" : label}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setStatus("copied");
        } catch {
          setStatus("failed");
        }
      }}
    >
      {children}
      {status === "copied" ? <Check size={15} /> : <Copy size={15} />}
      <span className={children ? "sr-only" : ""} aria-live="polite">
        {status === "copied"
          ? "已复制"
          : status === "failed"
            ? "复制失败"
            : label}
      </span>
    </button>
  );
}

function SearchBox({
  large = false,
  kind = "all",
  placeholder,
}: {
  large?: boolean;
  kind?: Kind | "all";
  placeholder?: string;
}) {
  const resultId = useId();
  const [query, setQuery] = useState("");
  const [items, setItems] = useState<EntityRef[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [expanded, setExpanded] = useState(false);
  const [index, setIndex] = useState(0);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!query.trim()) {
      setItems([]);
      setBusy(false);
      return;
    }
    const control = new AbortController();
    setBusy(true);
    const timer = setTimeout(() => {
      api<{ items: EntityRef[] }>(
        `/api/search?q=${encodeURIComponent(query)}&kind=${kind}&limit=8`,
        control.signal,
      )
        .then((r) => {
          setItems(r.items);
          setError("");
          setIndex(0);
        })
        .catch((e) => {
          if (e.name !== "AbortError") setError(e.message);
        })
        .finally(() => {
          if (!control.signal.aborted) setBusy(false);
        });
    }, 180);
    return () => {
      clearTimeout(timer);
      control.abort();
    };
  }, [query, kind]);
  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) setExpanded(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);
  useEffect(() => {
    if (expanded && items[index])
      document
        .getElementById(`${resultId}-${index}`)
        ?.scrollIntoView({ block: "nearest" });
  }, [index, expanded, items, resultId]);
  const select = (e: EntityRef) => {
    openEntity(e);
    setExpanded(false);
    setQuery("");
  };
  return (
    <div className={`search-box ${large ? "large" : ""}`} ref={ref}>
      <Search size={large ? 22 : 17} />
      <input
        aria-label={placeholder || "搜索药物或靶点"}
        placeholder={
          placeholder ||
          (large ? "输入药物名、基因名或标准编号" : "药物、靶点或编号")
        }
        value={query}
        onFocus={() => setExpanded(true)}
        onChange={(e) => {
          setQuery(e.target.value);
          setExpanded(true);
        }}
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            e.stopPropagation();
            setExpanded(false);
          }
          if (e.key === "ArrowDown") {
            e.preventDefault();
            setExpanded(true);
            setIndex((i) => Math.min(i + 1, Math.max(0, items.length - 1)));
          }
          if (e.key === "ArrowUp") {
            e.preventDefault();
            setIndex((i) => Math.max(i - 1, 0));
          }
          if (e.key === "Enter" && expanded && items[index])
            select(items[index]);
        }}
        role="combobox"
        aria-expanded={expanded && !!query}
        aria-controls={resultId}
        aria-activedescendant={
          expanded && query && items[index] ? `${resultId}-${index}` : undefined
        }
        aria-autocomplete="list"
        aria-keyshortcuts={large ? undefined : "Control+K Meta+K"}
      />
      {query ? (
        <button
          className="clear-search icon-button"
          aria-label="清空搜索"
          onClick={() => {
            setQuery("");
            setItems([]);
            ref.current?.querySelector("input")?.focus();
          }}
        >
          <X size={16} />
        </button>
      ) : null}
      {busy ? (
        <LoaderCircle size={18} className="spin" />
      ) : (
        <span className="search-key">{large ? "Enter" : "⌘ K"}</span>
      )}
      {expanded && query && (
        <div className="search-results" id={resultId} role="listbox">
          {error ? (
            <p>{error}</p>
          ) : items.length ? (
            items.map((e, i) => (
              <button
                key={`${e.kind}-${e.id}`}
                id={`${resultId}-${i}`}
                tabIndex={-1}
                role="option"
                aria-selected={i === index}
                className={i === index ? "selected" : ""}
                onClick={() => select(e)}
              >
                <span className={`entity-icon small ${e.kind}`}>
                  {e.kind === "drug" ? (
                    <FlaskConical size={18} />
                  ) : (
                    <Dna size={18} />
                  )}
                </span>
                <span>
                  <strong>{e.name}</strong>
                  <small>{e.subtitle || e.id}</small>
                </span>
                <Badge tone={e.kind === "drug" ? "green" : "blue"}>
                  {e.kind === "drug" ? "药物" : "靶点"}
                </Badge>
              </button>
            ))
          ) : !busy ? (
            <p>未找到匹配的项目实体，试试英文名称或标准编号。</p>
          ) : (
            <p>正在搜索…</p>
          )}
        </div>
      )}
    </div>
  );
}

function Catalog({ kind, summary }: { kind: Kind; summary: Summary }) {
  const [query, setQuery] = useState("");
  const [items, setItems] = useState<EntityRef[]>([]);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const [page, setPage] = useState(1);
  const [onlyScored, setOnlyScored] = useState(false);
  const [evidenceFilter, setEvidenceFilter] = useState("all");
  const [catalogSort, setCatalogSort] = useState("name");
  const [catalogOrder, setCatalogOrder] = useState(1);
  const sortCatalog = (key: string) => { setCatalogSort(key); setCatalogOrder(key === catalogSort ? -catalogOrder : 1); setPage(1); };
  const catalogHeader = (key: string, label: string) => <th aria-sort={catalogSort === key ? (catalogOrder === 1 ? "ascending" : "descending") : "none"}><button className="rv-sort" onClick={() => sortCatalog(key)}>{label} {catalogSort === key ? (catalogOrder === 1 ? "↑" : "↓") : "↕"}</button></th>;
  const [display, setDisplay] = useState<"table" | "grid">("table");
  useEffect(() => {
    const c = new AbortController();
    setBusy(true);
    setError("");
    const timer = setTimeout(
      () =>
        api<{ items: EntityRef[] }>(
          `/api/search?kind=${kind}&q=${encodeURIComponent(query)}&limit=2000`,
          c.signal,
        )
          .then((r) => {
            setItems(r.items);
            setError("");
            setPage(1);
          })
          .catch((e) => {
            if (e.name !== "AbortError") setError(e.message);
          })
          .finally(() => {
            if (!c.signal.aborted) setBusy(false);
          }),
      120,
    );
    return () => {
      c.abort();
      clearTimeout(timer);
    };
  }, [kind, query, revision]);
  const evidenceOptions = kind === "drug" ? [["known_targets", "已知靶点"], ["known_diseases", "适应症 / 研究疾病"], ["txgnn_diseases", "TxGNN 线索"], ["experiments", "SPR 设计"]] : [["pockets", "口袋结构"], ["pathways", "生物通路"], ["target_diseases", "疾病关联"], ["experiments", "SPR 设计"]];
  const filtered = items.filter(x => (!onlyScored || x.scored) && (evidenceFilter === "all" || (x.annotations?.[evidenceFilter] ?? 0) > 0)).sort((a, b) => {
    const delta = catalogSort === "name" ? a.name.localeCompare(b.name, "en", { numeric: true }) : catalogSort === "scored" ? Number(a.scored) - Number(b.scored) : (a.annotations?.[evidenceOptions[0][0]] ?? 0) - (b.annotations?.[evidenceOptions[0][0]] ?? 0);
    return delta * catalogOrder || a.id.localeCompare(b.id);
  });
  const pages = Math.max(1, Math.ceil(filtered.length / 24));
  return (
    <>
      <div className="page-heading">
        <div>
          <span className="eyebrow">
            {kind === "drug" ? "DRUG LIBRARY" : "TARGET LANDSCAPE"}
          </span>
          <h1>
            {kind === "drug" ? "药物图谱" : "靶点图谱"}
            <span className="title-dot">.</span>
          </h1>
          <p>
            {kind === "drug"
              ? "探索目录中的老药：已知机制、潜在靶点与疾病关联。"
              : "从蛋白功能到分子口袋，完整了解每一个研究靶点。"}
          </p>
        </div>
        <Badge tone="neutral">
          {number(summary.counts[kind === "drug" ? "drugs" : "targets"])} 个实体
        </Badge>
      </div>
      <div className="catalog-scope-bar">
        <span><strong>{number(summary.counts[kind === "drug" ? "drugs" : "targets"])}</strong>登记实体</span>
        <span><strong>{number(summary.counts[kind === "drug" ? "drugs" : "scored_targets"])}</strong>可查询排名</span>
        <span><strong>4</strong>对比模型</span>
        <span><Database size={14} />本地项目目录</span>
      </div>
      <div className="catalog-toolbar">
        <div className="filter-search">
          <Search size={18} />
          <input
            aria-label="筛选目录"
            placeholder={
              kind === "drug"
                ? "按药物名、ChEMBL、InChIKey 搜索…"
                : "按基因名、蛋白名、UniProt 搜索…"
            }
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <select className="catalog-evidence-select" aria-label="按证据覆盖筛选目录" value={evidenceFilter} onChange={e => { setEvidenceFilter(e.target.value); setPage(1); }}>
          <option value="all">全部证据覆盖</option>{evidenceOptions.map(([key, text]) => <option key={key} value={key}>含{text}</option>)}
        </select>
        {kind === "target" && (
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={onlyScored}
              onChange={(e) => {
                setOnlyScored(e.target.checked);
                setPage(1);
              }}
            />
            仅核心评分靶点
          </label>
        )}
        <span>{busy ? "搜索中…" : `${filtered.length} 个匹配`}</span>
        <div className="ranking-view-switch catalog-display-switch" aria-label="目录显示方式">
          <button aria-pressed={display === "table"} className={display === "table" ? "active" : ""} onClick={() => setDisplay("table")}><List size={15} />列表</button>
          <button aria-pressed={display === "grid"} className={display === "grid" ? "active" : ""} onClick={() => setDisplay("grid")}><Grid2X2 size={15} />卡片</button>
        </div>
      </div>
      {error ? (
        <ErrorPanel error={error} retry={() => setRevision(v => v + 1)} />
      ) : busy ? (
        <Loading />
      ) : filtered.length ? display === "table" ? (
        <div className="catalog-table-panel"><table className="catalog-table">
          <thead><tr>{catalogHeader("name", kind === "drug" ? "药物 / 标准化分子" : "靶点 / 蛋白名称")}{catalogHeader("evidence", `关联证据 · 按${evidenceOptions[0][1]}数`)}{catalogHeader("scored", "数据范围")}<th>操作</th></tr></thead>
          <tbody>{filtered.slice((page - 1) * 24, page * 24).map(e => <tr key={e.id} className="catalog-clickable-row" tabIndex={0} aria-label={`打开 ${e.name} 详情`} onClick={() => openEntity(e)} onKeyDown={event => { if (event.target === event.currentTarget && (event.key === "Enter" || event.key === " ")) { event.preventDefault(); openEntity(e); } }}>
            <td><button className="catalog-table-entity" onClick={event => { event.stopPropagation(); openEntity(e); }}><span className={`entity-icon small ${kind}`}>{kind === "drug" ? <FlaskConical size={19} /> : <Dna size={19} />}</span><span><strong>{e.name}</strong><small>{kind === "target" ? e.subtitle : `${e.classification || "分子式未收录"} · ${e.id}`}</small></span></button></td>
            <td><div className="catalog-evidence-badges">{evidenceOptions.slice(0, 3).map(([key, text]) => <span key={key} className={(e.annotations?.[key] ?? 0) ? "has-data" : "no-data"}><b>{number(e.annotations?.[key])}</b>{text}</span>)}</div></td><td><Badge tone={e.scored ? "green" : "neutral"}>{e.scored ? "核心评分" : "登记实体"}</Badge></td>
            <td><button aria-label={`打开 ${e.name} 详情`} onClick={event => { event.stopPropagation(); openEntity(e); }}><ArrowUpRight size={18} /></button></td>
          </tr>)}</tbody>
        </table></div>
      ) : (
        <div className="catalog-grid">
          {filtered.slice((page - 1) * 24, page * 24).map((e) => (
            <button
              className="catalog-card"
              onClick={() => openEntity(e)}
              key={e.id}
            >
              <div className="catalog-card-top">
                <span className={`entity-icon ${kind}`}>
                  {kind === "drug" ? (
                    <FlaskConical size={22} />
                  ) : (
                    <Dna size={22} />
                  )}
                </span>
                <Badge tone={e.scored ? "green" : "neutral"}>
                  {e.scored ? "核心评分" : "登记实体"}
                </Badge>
              </div>
              <h3>{e.name}</h3>
              <p>{e.subtitle || e.id}</p>
              <div className="catalog-evidence-badges">{evidenceOptions.slice(0, 2).map(([key, text]) => <span key={key}><b>{number(e.annotations?.[key])}</b>{text}</span>)}</div>
              <div className="catalog-card-bottom">
                <span>{kind === "drug" ? "探索药物证据" : "探索靶点证据"}</span>
                <ArrowUpRight size={17} />
              </div>
            </button>
          ))}
        </div>
      ) : (
        <Empty title="未找到匹配实体" text="可尝试英文名称或完整标准编号。" />
      )}
      <Pagination
        page={page}
        pages={pages}
        setPage={setPage}
        total={filtered.length}
      />
    </>
  );
}

function Pagination({
  page,
  pages,
  setPage,
  total,
}: {
  page: number;
  pages: number;
  setPage: (p: number) => void;
  total: number;
}) {
  return (
    <div className="pagination">
      <span>共 {number(total)} 条记录</span>
      <div>
        <button
          disabled={page <= 1}
          onClick={() => setPage(page - 1)}
          aria-label="上一页"
        >
          <ChevronLeft size={17} />
        </button>
        <span>
          {page} <i>/</i> {pages}
        </span>
        <button
          disabled={page >= pages}
          onClick={() => setPage(page + 1)}
          aria-label="下一页"
        >
          <ChevronRight size={17} />
        </button>
      </div>
    </div>
  );
}

function EvidenceList({
  rows,
  type,
  limit = 8,
}: {
  rows: Evidence[];
  type: "known" | "txgnn" | "pathway" | "target";
  limit?: number;
}) {
  const [expanded, setExpanded] = useState(false);
  if (!rows?.length)
    return (
      <Empty
        text={
          type === "txgnn"
            ? "当前 TxGNN 结果中没有该实体的可映射预测。"
            : "本地已接入的数据源中没有匹配记录。"
        }
      />
    );
  return (
    <>
      <div className="evidence-list">
        {rows.slice(0, expanded ? undefined : limit).map((row, i) => {
          const score = row.score ?? row.association_score ?? row.txgnn_score;
          return (
            <div className="evidence-row" key={`${row.id || label(row)}-${i}`}>
              <span className={`evidence-symbol ${type}`}>
                <>
                  {type === "pathway" ? (
                    <Network size={17} />
                  ) : type === "known" ? (
                    <Link2 size={17} />
                  ) : (
                    <Atom size={17} />
                  )}
                </>
              </span>
              <div>
                <strong>{label(row)}</strong>
                <small>
                  {[
                    row.disease_id || row.pathway_id || row.id,
                    row.source,
                    row.phase != null
                      ? row.phase === 4
                        ? "阶段 4 · 已批准"
                        : `阶段 ${row.phase} · 临床研究`
                      : "",
                    row.relation || "",
                  ]
                    .filter(Boolean)
                    .join(" · ")}
                </small>
                {row.description && <p>{row.description}</p>}
              </div>
              {score != null && (
                <span className="evidence-score">
                  {fmt(score)}
                  <small>{type === "txgnn" ? "预测分数" : "关联分数"}</small>
                </span>
              )}
              <SourceLink row={row} />
            </div>
          );
        })}
      </div>
      {rows.length > limit && (
        <button className="show-more" onClick={() => setExpanded((v) => !v)}>
          {expanded ? "收起记录" : `查看全部 ${rows.length} 条`}
          <ChevronDown size={15} />
        </button>
      )}
    </>
  );
}

function RankingsPanel({
  entity,
  compact = false,
}: {
  entity: Entity;
  compact?: boolean;
}) {
  const [view, setView] = useState<RankView>("matrix");
  const [model, setModel] = useState<Model>("biomaster");
  const [order, setOrder] = useState<"asc" | "desc">("asc");
  const sortModel = (next: Model) => { setOrder(next === model && order === "asc" ? "desc" : "asc"); setModel(next); setPage(1); };
  const [query, setQuery] = useState("");
  const [relationship, setRelationship] = useState("all");
  const [page, setPage] = useState(1);
  const [data, setData] = useState<Rankings | null>(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const [selected, setSelected] = useState<Ranking | null>(null);
  const previousFocus = useRef<HTMLElement | null>(null);
  const selectPair = (row: Ranking) => {
    previousFocus.current = document.activeElement as HTMLElement;
    setSelected(row);
  };
  useEffect(() => {
    if (!selected) return;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = overflow;
      if (previousFocus.current?.isConnected)
        previousFocus.current.focus({ preventScroll: true });
    };
  }, [selected]);
  const params = new URLSearchParams({
    kind: entity.kind,
    id: entity.id,
    model,
    order,
    search: query,
    relationship,
    page: String(page),
    page_size: compact ? "8" : "20",
  }).toString();
  useEffect(() => {
    const c = new AbortController();
    setBusy(true);
    setError("");
    const timer = setTimeout(() => {
      api<Rankings>(`/api/rankings?${params}`, c.signal)
        .then((r) => {
          setData(r);
          setError("");
        })
        .catch((e) => {
          if (e.name !== "AbortError") setError(e.message);
        })
        .finally(() => {
          if (!c.signal.aborted) setBusy(false);
        });
    }, 120);
    return () => {
      c.abort();
      clearTimeout(timer);
    };
  }, [params, revision]);
  useEffect(() => {
    setPage(1);
    setSelected(null);
  }, [entity.id, model, query, relationship]);
  const target = entity.kind === "drug" ? "靶点" : "药物";
  return (
    <section className="panel ranking-panel">
      <SectionTitle
        eyebrow={compact ? "RANKING EXPLORER" : undefined}
        title={entity.kind === "drug" ? "逐靶点排名" : "老药范围内逐药物排名"}
      >
        {compact ? (
          <button
            className="text-button"
            onClick={() => openEntity(entity, "rankings")}
          >
            完整排名 <ArrowRight size={15} />
          </button>
        ) : (
          <a
            className="button secondary small-button"
            href={`/api/rankings.csv?${params}`}
            download
          >
            <ArrowDownToLine size={15} />
            导出 CSV
          </a>
        )}
      </SectionTitle>
      <div className="ranking-toolbar">
        <div className="ranking-view-switch" aria-label="排名视图">
          {([["matrix", "排名矩阵", Grid2X2], ["parallel", "模型分歧", ChartNoAxesCombined]] as const).map(([id, text, Icon]) => <button key={id} className={view === id ? "active" : ""} aria-pressed={view === id} onClick={() => setView(id)}><Icon size={15} />{text}</button>)}
        </div>
        <div className="model-tabs">
          {MODELS.map((m) => (
            <button
              className={model === m ? "active" : ""}
              aria-pressed={model === m}
              key={m}
              onClick={() => { setModel(m); setOrder("asc"); setPage(1); }}
            >
              <i style={{ background: MODEL_COLORS[m] }} />
              {MODEL_NAMES[m]}
            </button>
          ))}
        </div>
        {!compact && (
          <div className="filter-search small-filter">
            <Search size={15} />
            <input
              aria-label="筛选排名"
              placeholder={`筛选${target}名称 / ID`}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
        )}
      </div>
      {!compact && <div className="ranking-filter-row">
        <div className="relationship-filters" aria-label="关系证据筛选">
          <span><SlidersHorizontal size={14} />关系证据</span>
          {[["all", "全部候选"], ["known", "已知关系"], ["unannotated", "关系未标注"]].map(([id, text]) => <button key={id} className={relationship === id ? "active" : ""} aria-pressed={relationship === id} onClick={() => setRelationship(id)}>{text}</button>)}
        </div><span>{data?.total ?? "—"} 个匹配 · 排名保持全目录位置</span>
      </div>}
      <div className="ranking-context">
        <span>
          {entity.kind === "target" ? (
            <Badge tone="amber">反向辅助证据</Badge>
          ) : (
            <Badge>药物 → 靶点</Badge>
          )}
          <span>
            排名分母 <strong>{data?.denominator ?? "—"}</strong> · 筛选不重排
          </span>
        </span>
        <span>{view === "matrix" ? "点击矩阵单元格查看配对证据" : "悬停轨迹查看模型分歧"}</span>
      </div>
      {error ? (
        <ErrorPanel error={error} retry={() => setRevision(v => v + 1)} />
      ) : busy ? (
        <Loading />
      ) : data?.items.length ? (
        <RankVisuals rows={data.items} model={model} view={view} order={order} onSort={sortModel} onSelect={selectPair} />
      ) : (
        <Empty
          title="当前模型没有可用排名"
          text="该实体可能尚未进入评分核心，或筛选条件没有匹配结果。"
        />
      )}
      {!compact && data && (
        <Pagination
          page={page}
          pages={Math.max(1, Math.ceil(data.total / data.page_size))}
          setPage={setPage}
          total={data.total}
        />
      )}
      <p className="panel-footnote">
        {model === "drugclip"
          ? "DrugCLIP 跨实验 / 预测口袋的统一排名属于探索性结果。 "
          : "原始分数用于同一模型内排序。 "}
        “未标注”不代表阴性；已知关系不等于已验证该预测。
      </p>
      {selected &&
        createPortal(
          <div className="drawer-backdrop" onClick={() => setSelected(null)}>
            <aside
              className="pair-drawer"
              onClick={(e) => e.stopPropagation()}
              role="dialog"
              aria-modal="true"
              aria-label="配对模型证据"
              onKeyDown={(e) => {
                if (e.key === "Escape") setSelected(null);
                if (e.key === "Tab") {
                  const nodes = Array.from(
                    e.currentTarget.querySelectorAll<HTMLElement>(
                      'button:not(:disabled), a[href], summary, [tabindex="0"]',
                    ),
                  ).filter((el) => el.getClientRects().length > 0);
                  const first = nodes[0],
                    last = nodes[nodes.length - 1];
                  if (e.shiftKey && document.activeElement === first) {
                    e.preventDefault();
                    last?.focus();
                  } else if (!e.shiftKey && document.activeElement === last) {
                    e.preventDefault();
                    first?.focus();
                  }
                }
              }}
            >
              <header className="pair-heading">
                <div className="pair-heading-top"><span className="eyebrow">配对证据</span><button autoFocus className="icon-button" onClick={() => setSelected(null)} aria-label="关闭配对面板"><X size={20} /></button></div>
                <div className="pair-identities">
                  <div><span>{entity.kind === "drug" ? "药物" : "靶点"}</span><h2>{entity.name}</h2></div>
                  <ArrowRight size={20} aria-hidden="true" />
                  <div><span>{selected.kind === "drug" ? "药物" : "靶点"}</span><h2>{selected.name}</h2></div>
                </div>
                <p>{entity.kind === "drug" ? "固定药物 · 比较候选靶点" : "固定靶点 · 比较目录老药"}</p>
              </header>
              <div className="pair-body">
              <div className="pair-relation-context"><Link2 size={19} /><div><strong>{selected.known_relation ? "已有数据库关系注释" : "当前关系未标注"}</strong><p>{selected.mechanism || (selected.known_relation ? "已知关系的详细机制以原始数据记录为准。" : "可作为待研究候选；未标注不代表实验阴性。")}</p>{selected.action && <span>作用类型：{({ INHIBITOR: "抑制剂", SUBSTRATE: "底物", ANTAGONIST: "拮抗剂", AGONIST: "激动剂", BLOCKER: "阻断剂", ACTIVATOR: "激活剂" } as Record<string, string>)[selected.action] || selected.action}</span>}</div></div>
              <section className="pair-comparison" aria-label="四模型对比">
                <div className="pair-section-heading"><h3>模型对比</h3><span>排名越小，位置越靠前</span></div>
                <table><thead><tr><th>模型</th><th>排名 / 总数</th><th>原始分数</th></tr></thead><tbody>
                {MODELS.map(m => <tr key={m}>
                  <th scope="row"><i style={{background: MODEL_COLORS[m]}} />{MODEL_NAMES[m]}</th>
                  <td><strong>{selected.ranks[m] == null ? "—" : `#${selected.ranks[m]}`}</strong><span> / {selected.denominators[m] ?? "—"}</span><div className="pair-rank-track"><i style={{background: MODEL_COLORS[m], width: selected.ranks[m] == null ? "0%" : `${Math.max(1, 100 * (1 - (selected.ranks[m]! - 1) / Math.max(1, selected.denominators[m] - 1)))}%`}} /></div></td>
                  <td>{fmt(selected.scores[m], 5)}</td>
                </tr>)}
                </tbody></table>
                <p className="pair-comparison-note">排名基于各模型已评分集合；原始分数不可跨模型比较。</p>
              </section>
              <details className="pair-extra">
                <summary>已知机制与冻结版本</summary>
                <dl>
                  <div>
                    <dt>已知作用方向</dt>
                    <dd>{selected.action || "未收录"}</dd>
                  </div>
                  <div>
                    <dt>已知机制</dt>
                    <dd>{selected.mechanism || "未收录"}</dd>
                  </div>
                  <div>
                    <dt>DrugCLIP 口袋来源</dt>
                    <dd>{selected.pocket_source || "未收录"}</dd>
                  </div>
                  <div>
                    <dt>2026-09-01 冻结 Borda</dt>
                    <dd>
                      {fmt(selected.frozen_score)} · #
                      {fmt(selected.frozen_rank)} /{" "}
                      {fmt(selected.frozen_denominator)}
                    </dd>
                  </div>
                </dl>
                <p>冻结版本与当前独立交付模型分别保留，不共用分数尺度。</p>
              </details>
              </div>
              <div className="pair-action-grid"><button className="button primary" onClick={() => { setSelected(null); openEntity(selected); }}>打开{target}详情 <ArrowUpRight size={17} /></button><button className="button secondary" onClick={() => { setSelected(null); openEntity(entity.kind === "target" ? entity : selected, "structure"); }}><Box size={16} />查看靶点口袋</button><SaveEntityButton entity={selected} /></div>
            </aside>
          </div>,
          document.body,
        )}
    </section>
  );
}

function Diseases({ entity }: { entity: Entity }) {
  return (
    <div className="disease-layout">
      {entity.kind === "target" && <div className="full-span"><BiologyInsights entity={entity} /></div>}
      {entity.kind === "drug" ? (
        <>
          <section className="panel">
            <SectionTitle
              eyebrow="KNOWN INDICATIONS"
              title="已知适应症与临床研究"
            >
              <Badge tone="blue">数据库关联</Badge>
            </SectionTitle>
            <p className="section-description">
              阶段 4 为 ChEMBL 已批准记录，低于阶段 4
              的关联按临床研究展示；适应症的司法辖区与时间以源记录为准。
            </p>
            <EvidenceBrowser entity={entity} section="known_diseases" />
          </section>
          <section className="panel">
            <SectionTitle eyebrow="TXGNN PREDICTIONS" title="TxGNN 疾病关联">
              <Badge tone="purple">模型预测</Badge>
            </SectionTitle>
            <p className="section-description">
              当前快照展示已计算的疾病关联。知识图谱预测分数不等于治疗有效概率。
            </p>
            <EvidenceBrowser entity={entity} section="txgnn_diseases" />
          </section>
          <section className="panel full-span">
            <SectionTitle eyebrow="ESTABLISHED BIOLOGY" title="已知作用靶点" />
            <EvidenceBrowser entity={entity} section="known_targets" />
          </section>
        </>
      ) : (
        <>
          <section className="panel">
            <SectionTitle eyebrow="OPEN TARGETS" title="靶点–疾病关联">
              <Badge tone="blue">聚合证据</Badge>
            </SectionTitle>
            <p className="section-description">
              Open Targets 数据库中的关联证据，不等同于药物直接结合或治疗效应。
            </p>
            <EvidenceBrowser entity={entity} section="target_diseases" />
          </section>
          <section className="panel">
            <SectionTitle eyebrow="PATHWAY CONTEXT" title="生物通路">
              <Badge tone="green">Open Targets / Reactome</Badge>
            </SectionTitle>
            <EvidenceBrowser entity={entity} section="pathways" />
          </section>
        </>
      )}
    </div>
  );
}
function KnownTargets({ rows }: { rows: Evidence[] }) {
  return !rows?.length ? (
    <Empty />
  ) : (
    <div className="known-targets">
      {rows.map((r, i) => (
        <div className="known-target" key={i}>
          <Dna size={19} />
          <div>
            {r.id && r.explorable !== false ? (
              <button
                className="text-button"
                onClick={() =>
                  openEntity({ id: r.id, kind: "target", name: label(r) })
                }
              >
                {label(r)}
                <ArrowUpRight size={13} />
              </button>
            ) : (
              <strong>{label(r)}</strong>
            )}
            <small>
              {[r.action_type, r.mechanism_of_action || r.description, r.source]
                .filter(Boolean)
                .join(" · ") || r.id}
            </small>
          </div>
          <SourceLink row={r} />
        </div>
      ))}
    </div>
  );
}

function Sources({ summary, entity }: { summary: Summary; entity?: Entity }) {
  const sources = entity?.sources || summary.sources;
  return (
    <>
      <div className={entity ? "source-intro" : "page-heading"}>
        <div>
          <span className="eyebrow">PROVENANCE & COVERAGE</span>
          <h1>
            {entity ? "证据从哪里来" : "数据与研究口径"}
            <span className="title-dot">.</span>
          </h1>
          <p>每一项结果都保留来源。缺失、预测与已知证据，分别呈现。</p>
        </div>
      </div>
      <section className="panel">
        <SectionTitle title="模型版本与覆盖范围" />
        <div className="source-models">
          {summary.models.map((m) => (
            <div key={m.id}>
              <span
                className="model-dot"
                style={{ background: MODEL_COLORS[m.id as Model] || "#aaa" }}
              />
              <div>
                <h3>{m.name}</h3>
                <p>{m.version}</p>
                <small>{m.role}</small>
              </div>
              <Badge tone="neutral">
                {typeof m.coverage === "number"
                  ? `${number(m.coverage)} 对`
                  : fmt(m.coverage)}
              </Badge>
            </div>
          ))}
        </div>
      </section>
      <section className="panel source-table">
        <SectionTitle title="已接入的数据文件">
          <Badge tone="neutral">{sources.length} 个来源</Badge>
        </SectionTitle>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>来源</th>
                <th>文件与版本</th>
                <th>状态</th>
              </tr>
            </thead>
            <tbody>
              {sources.map((s, i) => (
                <tr key={i}>
                  <td>
                    <strong>{s.name || s.type || "项目结果"}</strong>
                    {s.note && <small>{s.note}</small>}
                  </td>
                  <td>
                    <code>{s.path || s.source || s.version || "本地索引"}</code>
                  </td>
                  <td>
                    <Badge tone={s.available === false ? "amber" : "green"}>
                      {s.available === false ? "未找到" : "已接入"}
                    </Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section className="panel boundaries">
        <SectionTitle title="如何理解这些结果" />
        {summary.boundaries.map((b, i) => (
          <div key={i}>
            <ShieldCheck size={18} />
            <p>{b}</p>
          </div>
        ))}
        {entity?.missing?.length ? (
          <div>
            <Database size={18} />
            <p>当前实体缺失：{entity.missing.join("；")}</p>
          </div>
        ) : null}
      </section>
    </>
  );
}

function EntityPage({ route, summary }: { route: Route; summary: Summary }) {
  const [entity, setEntity] = useState<Entity | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(true);
  const [nonce, setNonce] = useState(0);
  const tab = route.tab || "overview";
  useEffect(() => {
    document
      .querySelector<HTMLElement>(".detail-tabs .active")
      ?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [tab]);
  useEffect(() => {
    const c = new AbortController();
    setBusy(true);
    setError("");
    setEntity(null);
    api<Entity>(
      `/api/entity/${route.kind}/${encodeURIComponent(route.id || "")}`,
      c.signal,
    )
      .then(setEntity)
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      })
      .finally(() => {
        if (!c.signal.aborted) setBusy(false);
      });
    return () => c.abort();
  }, [route.kind, route.id, nonce]);
  useEffect(() => { if (entity) recordVisit(entity); }, [entity]);
  if (busy) return <Loading text="正在关联排名、疾病与结构证据…" />;
  if (error)
    return <ErrorPanel error={error} retry={() => setNonce((v) => v + 1)} />;
  if (!entity) return null;
  const tabs = [
    ["overview", "综合概览"],
    ["rankings", entity.kind === "drug" ? "靶点排名" : "老药排名"],
    ["diseases", "疾病与机制"],
    ["network", "证据网络"],
    ["metadata", "完整档案"],
    ["experiments", "SPR 设计"],
    ...(entity.kind === "target" ? [["structure", "口袋与三维结构"]] : []),
    ["sources", "数据来源"],
  ];
  const identifiers = (
    entity.kind === "drug"
      ? ["inchikey", "chembl_id", "drugbank_id"]
      : ["chembl_id", "uniprot_id", "ensembl_id"]
  )
    .map((key) => [key, entity.identifiers?.[key]])
    .filter(([, value]) => value != null && value !== "");
  return (
    <div className={`entity-workbench ${tab === "overview" ? "is-overview" : "is-analysis"}`}>
      <button
        className="back-link"
        onClick={() => go(entity.kind === "drug" ? "drugs" : "targets")}
      >
        <ArrowLeft size={15} />
        {entity.kind === "drug" ? "药物图谱" : "靶点图谱"}
      </button>
      <section className="entity-header">
        <div className={`entity-icon big ${entity.kind}`}>
          {entity.kind === "drug" ? (
            <FlaskConical size={31} />
          ) : (
            <Dna size={31} />
          )}
        </div>
        <div className="entity-heading">
          <div className="entity-overline">
            <span className="eyebrow">
              {entity.kind === "drug" ? "DRUG PROFILE" : "TARGET PROFILE"}
            </span>
            <Badge tone={entity.scored ? "green" : "neutral"}>
              {entity.scored ? "核心评分实体" : "登记实体"}
            </Badge>
          </div>
          <h1>{entity.name}</h1>
          <p>{entity.subtitle}</p>
          <div className="identifiers">
            {identifiers.slice(0, 4).map(([k, v]) => (
              <CopyButton
                key={k}
                text={fmt(v)}
                label={`复制 ${k.replace(/_/g, " ")}`}
                className="identifier-chip"
              >
                <small>
                  {(
                    {
                      inchikey: "InChIKey",
                      chembl_id: "ChEMBL",
                      uniprot_id: "UniProt",
                      ensembl_id: "Ensembl",
                      drugbank_id: "DrugBank",
                    } as Record<string, string>
                  )[k] || k}
                </small>
                <code>{fmt(v)}</code>
              </CopyButton>
            ))}
          </div>
        </div>
        <div className="entity-header-aside"><span>EVIDENCE CONNECTED</span><strong>{entity.models.filter(m => MODELS.includes(m.id) && m.available).length}<small> / 4 模型</small></strong><span>{entity.kind === "drug" ? "药物 → 靶点检索" : "靶点 → 老药检索"}</span></div>
        <div className="entity-actions">
          <SaveEntityButton entity={entity} />
          <CopyButton text={location.href} />
          <a
            className="button secondary small-button"
            href={`/api/rankings.csv?kind=${entity.kind}&id=${encodeURIComponent(entity.id)}&model=biomaster`}
            download
          >
            <ArrowDownToLine size={16} />
            导出排名
          </a>
        </div>
      </section>
      <div className="entity-context-bar">
        <div><span>评分空间</span><strong>{entity.scored ? `${entity.kind === "drug" ? summary.counts.scored_targets : summary.counts.drugs} ${entity.kind === "drug" ? "个核心靶点" : "种目录老药"}` : "未进入评分核心"}</strong></div>
        <div><span>{entity.kind === "drug" ? "分子式" : "靶点分类"}</span><strong>{entity.kind === "drug" ? entity.properties.molecular_formula || "未收录" : entity.properties.family || "未收录"}</strong></div>
        <div><span>模型版本</span><strong>Selected · 2026.09.06</strong></div>
        <button onClick={() => openEntity(entity, "sources")}><ShieldCheck size={14} />数据来源与适用范围 <ArrowUpRight size={14} /></button>
      </div>
      <nav className="detail-tabs" aria-label="实体详情导航">
        {tabs.map(([id, name]) => (
          <button
            key={id}
            className={tab === id ? "active" : ""}
            aria-current={tab === id ? "page" : undefined}
            onClick={() => openEntity(entity, id)}
            onKeyDown={(e) => {
              const delta =
                e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
              if (delta) {
                e.preventDefault();
                const next =
                  tabs[
                    (tabs.findIndex((t) => t[0] === id) + delta + tabs.length) %
                      tabs.length
                  ];
                openEntity(entity, next[0]);
                requestAnimationFrame(() =>
                  document
                    .querySelector<HTMLButtonElement>(".detail-tabs .active")
                    ?.focus(),
                );
              }
            }}
          >
            {(() => { const Icon = ({ overview: LayoutDashboard, rankings: ChartNoAxesCombined, diseases: Target, network: Network, metadata: BookOpen, experiments: FlaskConical, structure: Box, sources: Database } as Record<string, typeof Dna>)[id]; return Icon ? <Icon size={15} /> : null; })()}
            {name}
            {id === "structure" && (
              <span>{entity.pockets_total ?? entity.pockets?.length ?? 0}</span>
            )}
          </button>
        ))}
      </nav>
      <Reveal key={tab} className="entity-tab-content">
        <Suspense fallback={<Loading text="正在准备研究内容…" />}>
        {tab === "overview" && (
          <>
            <div className="profile-stats">
              {[
                {
                  label: entity.kind === "drug" ? "已知作用靶点" : "疾病关联",
                  count:
                    entity.kind === "drug"
                      ? entity.known_targets_total
                      : entity.target_diseases_total,
                  icon: Target,
                  tab: "diseases",
                },
                {
                  label:
                    entity.kind === "drug" ? "已知疾病 / 适应症" : "生物通路",
                  count:
                    entity.kind === "drug"
                      ? entity.known_diseases_total
                      : entity.pathways_total,
                  icon: Network,
                  tab: "diseases",
                },
                {
                  label:
                    entity.kind === "drug" ? "TxGNN 疾病线索" : "可查看口袋",
                  count:
                    entity.kind === "drug"
                      ? entity.txgnn_diseases_total
                      : entity.pockets_total,
                  icon: entity.kind === "drug" ? Sparkles : Box,
                  tab: entity.kind === "drug" ? "diseases" : "structure",
                },
                {
                  label: "对比模型",
                  count: entity.models.filter(
                    (m) => MODELS.includes(m.id) && m.available,
                  ).length,
                  icon: Layers3,
                  tab: "rankings",
                },
              ].map((s) => (
                <button key={s.label} onClick={() => openEntity(entity, s.tab)}>
                  <s.icon size={19} />
                  <span>{s.label}</span>
                  <strong>
                    <AnimatedNumber value={s.count ?? 0} />
                  </strong>
                  <ArrowUpRight size={16} />
                </button>
              ))}
            </div>
            <div className="profile-columns">
              <section className="panel identity-panel">
                <SectionTitle
                  eyebrow={
                    entity.kind === "drug"
                      ? "MOLECULAR IDENTITY"
                      : "BIOLOGICAL CONTEXT"
                  }
                  title={entity.kind === "drug" ? "分子档案" : "靶点档案"}
                />
                {entity.kind === "drug" ? (
                  <div className="molecule-canvas">
                    <img
                      src={
                        entity.molecule_url ||
                        `/api/molecule/${encodeURIComponent(entity.id)}.svg`
                      }
                      alt={`${entity.name} 二维分子结构`}
                      onError={(e) => {
                        e.currentTarget.style.display = "none";
                        e.currentTarget.parentElement?.classList.add(
                          "unavailable",
                        );
                      }}
                    />
                    <span>二维分子结构暂不可用</span>
                  </div>
                ) : (
                  <div className="target-description">
                    <Dna size={35} />
                    <div>
                      <p>
                        {(() => {
                          const description =
                            entity.description ||
                            entity.subtitle ||
                            "本地靶点登记信息";
                          return description.length > 255
                            ? description.slice(0, 255).replace(/\s+\S*$/, "") +
                                "…"
                            : description;
                        })()}
                      </p>
                      <button
                        className="text-button description-expand"
                        onClick={() => openEntity(entity, "metadata")}
                      >
                        阅读完整功能注释 <ArrowUpRight size={16} />
                      </button>
                    </div>
                  </div>
                )}
                <dl className="property-grid">
                  {Object.entries(entity.properties || {})
                    .filter(
                      ([k, v]) =>
                        [
                          "molecular_weight",
                          "molecular_formula",
                          "approval_year",
                          "logp",
                          "tpsa",
                          "species_role",
                          "family",
                          "subfamily",
                          "organism",
                          "sequence_length",
                          "af_mean_plddt",
                          "assay_lane",
                        ].includes(k) &&
                        v != null &&
                        typeof v !== "object",
                    )
                    .slice(0, 6)
                    .map(([k, v]) => (
                      <div key={k}>
                        <dt>
                          {(
                            {
                              molecular_weight: "分子量",
                              molecular_formula: "分子式",
                              species_role: "模型物种",
                              family: "靶点家族",
                              subfamily: "靶点亚类",
                              af_mean_plddt: "平均 pLDDT",
                              assay_lane: "实验分路",
                              tpsa: "极性表面积",
                              formula: "分子式",
                              approval_year: "获批年份",
                              target_class: "靶点分类",
                              protein_length: "蛋白长度",
                              organism: "物种",
                              logp: "LogP",
                              gene_symbol: "基因",
                              route: "研究分路",
                              sequence_length: "蛋白长度",
                            } as Record<string, string>
                          )[k] || k.replace(/_/g, " ")}
                        </dt>
                        <dd>{fmt(v)}</dd>
                      </div>
                    ))}
                </dl>
              </section>
              {entity.kind === "target" ? (
                <section className="panel structure-overview">
                  <SectionTitle
                    eyebrow="STRUCTURE EXPLORER"
                    title="蛋白与结合口袋"
                  >
                    <button
                      className="text-button"
                      onClick={() => openEntity(entity, "structure")}
                    >
                      结构工作台 <ArrowUpRight size={17} />
                    </button>
                  </SectionTitle>
                  <div className="profile-structure-view">
                    <StructureViewer entity={entity} compact height={340} />
                  </div>
                  <div className="structure-preview-caption">
                    <Badge tone="blue">真实项目结构</Badge>
                    <span>拖动旋转 · 滚轮缩放</span>
                  </div>
                </section>
              ) : (
                <section className="panel preview-evidence">
                  <SectionTitle
                    eyebrow={
                      entity.kind === "drug"
                        ? "DISEASE CONTEXT"
                        : "PATHWAY CONTEXT"
                    }
                    title={
                      entity.kind === "drug" ? "疾病证据速览" : "生物通路速览"
                    }
                  >
                    <button
                      className="text-button"
                      onClick={() => openEntity(entity, "diseases")}
                    >
                      全部证据 <ArrowRight size={15} />
                    </button>
                  </SectionTitle>
                  <EvidenceList
                    rows={
                      entity.kind === "drug"
                        ? entity.known_diseases.length
                          ? entity.known_diseases
                          : entity.txgnn_diseases
                        : entity.pathways
                    }
                    type={
                      entity.kind === "drug"
                        ? entity.known_diseases.length
                          ? "known"
                          : "txgnn"
                        : "pathway"
                    }
                    limit={4}
                  />
                  <div className="info-note">
                    <ShieldCheck size={17} />
                    <p>
                      {entity.kind === "drug"
                        ? "已知适应症和 TxGNN 预测在「疾病与机制」中分别展示。"
                        : "通路和疾病关联用于解释靶点背景，不能替代配对实验。"}
                    </p>
                  </div>
                </section>
              )}
            </div>
            {entity.kind === "target" && (
              <section className="panel pathway-strip">
                <SectionTitle eyebrow="PATHWAY CONTEXT" title="相关生物通路">
                  <button
                    className="text-button"
                    onClick={() => openEntity(entity, "diseases")}
                  >
                    全部 {entity.pathways_total ?? entity.pathways.length} 条{" "}
                    <ArrowRight size={17} />
                  </button>
                </SectionTitle>
                {entity.pathways.length ? (
                  <div className="pathway-chips">
                    {entity.pathways.slice(0, 4).map((row, i) => (
                      <button
                        key={row.id || i}
                        className="pathway-chip"
                        onClick={() => openEntity(entity, "diseases")}
                      >
                        <Network size={18} />
                        <span>{label(row)}</span>
                        <ArrowUpRight size={16} />
                      </button>
                    ))}
                  </div>
                ) : (
                  <Empty title="当前暂无已接入的通路记录" />
                )}
              </section>
            )}
            <RankingsPanel entity={entity} compact />
          </>
        )}
        {tab === "rankings" && <RankingsPanel entity={entity} />}
        {tab === "diseases" && <Diseases entity={entity} />}
        {tab === "metadata" && <Metadata entity={entity} />}
        {tab === "network" && <EvidenceNetwork entity={entity} />}
        {tab === "experiments" && <Experiments entity={entity} />}
        {tab === "structure" && entity.kind === "target" && (
          <section className="structure-page">
            <SectionTitle
              eyebrow="POCKET & STRUCTURE EXPLORER"
              title="三维结构工作台"
            >
              <Badge tone="blue">拖动旋转 · 滚轮缩放</Badge>
            </SectionTitle>
            <StructureViewer entity={entity} />
            <div className="info-note">
              <Box size={19} />
              <p>
                展示项目已有的受体结构与口袋注释。预测结构及口袋不等同于实验确认的结合位点；未提供配体姿势时，不推断配体接触关系。
              </p>
            </div>
          </section>
        )}
        {tab === "sources" && <Sources summary={summary} entity={entity} />}
        </Suspense>
      </Reveal>
    </div>
  );
}

export default function App() {
  const [route, setRoute] = useState<Route>(parseRoute);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState("");
  const [mobile, setMobile] = useState(false);
  useEffect(() => {
    const shortcut = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        document
          .querySelector<HTMLInputElement>(".topbar .search-box input")
          ?.focus();
      }
      if (event.key === "Escape") setMobile(false);
    };
    window.addEventListener("keydown", shortcut);
    return () => window.removeEventListener("keydown", shortcut);
  }, []);
  const [nonce, setNonce] = useState(0);
  useEffect(() => {
    const listener = () => {
      setRoute(parseRoute());
      setMobile(false);
      window.scrollTo({ top: 0 });
    };
    window.addEventListener("hashchange", listener);
    return () => window.removeEventListener("hashchange", listener);
  }, []);
  useEffect(() => {
    const c = new AbortController();
    api<Summary>("/api/summary", c.signal)
      .then((r) => {
        setSummary(r);
        setError("");
      })
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => c.abort();
  }, [nonce]);
  const nav = [
    { id: "home", label: "项目全景", icon: LayoutDashboard },
    { id: "drugs", label: "药物图谱", icon: FlaskConical },
    { id: "targets", label: "靶点图谱", icon: Dna },
    { id: "spr", label: "SPR 设计", icon: FlaskConical },
    { id: "affinity", label: "亲和证据矩阵", icon: Grid2X2 },
    { id: "sources", label: "证据与来源", icon: Database },
    { id: "workspace", label: "研究清单", icon: Bookmark },
  ];
  const active =
    route.page === "entity" && route.tab === "experiments" ? "spr" : route.page === "entity"
      ? route.kind === "drug"
        ? "drugs"
        : "targets"
      : route.page === "browse" ? "sources" : route.page;
  return (
    <div className="app-shell">
      {mobile && (
        <div className="sidebar-backdrop" onClick={() => setMobile(false)} />
      )}
      <aside className={`sidebar ${mobile ? "open" : ""}`}>
        <button className="brand" onClick={() => go("home")}>
          <span className="brand-mark">
            <Dna size={27} />
          </span>
          <span>
            Palinova<small>DISCOVERY ATLAS</small>
          </span>
        </button>
        <div className="workspace-label">
          研究工作区 <span>01</span>
        </div>
        <nav>
          {nav.map((n) => (
            <button
              key={n.id}
              className={active === n.id ? "active" : ""}
              onClick={() => go(n.id)}
            >
              <n.icon size={19} />
              <span>{n.label}</span>
              {n.id === "drugs" && summary && (
                <small>{summary.counts.drugs}</small>
              )}
              {n.id === "targets" && summary && (
                <small>{summary.counts.targets}</small>
              )}
            </button>
          ))}
        </nav>
        <RecentEntities onOpen={openEntity} />
        <div className="sidebar-collection">
          <span className="eyebrow">CONNECTED EVIDENCE</span>
          <div>
            <span className="mini-dot green" />
            分子与蛋白
          </div>
          <div>
            <span className="mini-dot blue" />
            疾病与通路
          </div>
          <div>
            <span className="mini-dot purple" />
            口袋与结构
          </div>
        </div>
        <div className="sidebar-bottom">
          <div className="project-note">
            <BookOpen size={20} />
            <strong>让研究证据彼此连接</strong>
            <p>
              老药新靶点
              <br />
              多模型 · 多尺度 · 可追溯
            </p>
            <button onClick={() => go("sources")}>
              了解数据口径 <ArrowUpRight size={14} />
            </button>
          </div>
          <div className="workspace-status">
            <span className={`live-dot ${error ? "offline" : ""}`} />
            <span>
              本地数据工作区
              <small>
                {error
                  ? "连接异常"
                  : summary
                    ? "数据索引已连接"
                    : "正在连接数据索引"}
              </small>
            </span>
            <ShieldCheck size={17} />
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            <button
              className="mobile-toggle icon-button"
              onClick={() => setMobile((v) => !v)}
              aria-label="展开导航"
            >
              <SlidersHorizontal size={20} />
            </button>
            <span>工作区</span>
            <ChevronRight size={13} />
            <strong>{nav.find((n) => n.id === active)?.label}</strong>
            {route.page === "entity" && (
              <>
                <ChevronRight size={13} />
                <span>实体详情</span>
              </>
            )}
          </div>
          <div className="topbar-actions">
            <SearchBox />
            <span className="release-tag">
              <span className="live-dot" />
              RESEARCH EDITION
            </span>
            <form method="post" action="/logout"><button className="text-button" type="submit" aria-label="退出登录">退出</button></form>
          </div>
        </header>
        <main id="main-content">
          {summary?.warnings?.length ? (
            <div className="info-note data-notices">
              <Database size={17} />
              <div>
                <strong>数据接入提示</strong>
                {summary.warnings.map((w, i) => (
                  <p key={i}>{w}</p>
                ))}
              </div>
            </div>
          ) : null}
          <Reveal
            key={`${route.page}:${route.kind || ""}:${route.id || ""}`}
            className="page-content"
          >
            <PageBoundary key={`${route.page}:${route.kind || ""}:${route.id || ""}}`}><Suspense fallback={<Loading text="正在准备页面内容…" />} >
            {error ? (
              <ErrorPanel error={error} retry={() => setNonce((v) => v + 1)} />
            ) : !summary ? (
              <Loading text="正在载入研究概览…" />
            ) : route.page === "home" ? (
              <ResearchHome summary={summary} onNavigate={go} onOpenEntity={openEntity} />
            ) : route.page === "drugs" || route.page === "targets" ? (
              <Catalog
                key={route.page}
                kind={route.page === "drugs" ? "drug" : "target"}
                summary={summary}
              />
            ) : route.page === "workspace" ? (
              <Workspace onOpen={openEntity} onNavigate={go} />
            ) : route.page === "affinity" ? (
              <AffinityAtlas />
            ) : route.page === "spr" ? (
              <SPRDirectory key={route.tab} initialScope={route.tab} />
            ) : route.page === "browse" ? (
              <ResearchBrowse key={route.tab} section={route.tab || "chembl"} summary={summary} />
            ) : route.page === "sources" ? (
              <Sources summary={summary} />
            ) : (
              <EntityPage route={route} summary={summary} />
            )}
            </Suspense></PageBoundary>
          </Reveal>
        </main>
        <ConnectionStatus />
        <footer>
          <span>
            <Dna size={14} />
            Palinova <i>/</i> ReTargetMap
          </span>
          <span>Research evidence, connected.</span>
          <span>LOCAL DATA · {summary?.version || "2026"}</span>
        </footer>
      </div>
    </div>
  );
}
