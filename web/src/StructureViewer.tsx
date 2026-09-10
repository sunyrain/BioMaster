import { useEffect, useRef, useState } from "react";
import {
  Expand,
  Focus,
  LoaderCircle,
  Pause,
  Play,
  RotateCcw,
  Box,
  Layers3,
  Download,
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  Search,
  ZoomIn,
  ZoomOut,
  CheckCircle2,
  ExternalLink,
  ListFilter,
  Dna,
  X,
} from "lucide-react";
import type { Entity, Evidence } from "./types";

import { useReducedMotion } from "./motion";
import "./structure-workbench.css";

type RenderedResidue = { id: string; chain: string; resi: number; resn: string; icode: string };
const pocketKind = (pocket?: Evidence) => pocket?.coordinate_system === "experimental_structure"
  ? "实验共晶" : pocket?.type === "experimental" ? "实验位点 · AF 映射"
  : pocket?.type === "predicted" ? "预测口袋" : "本地位点";
const metric = (value: unknown, digits = 2) => typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
const annotationNames: Record<string, string> = {
  name: "口袋名称", source: "证据来源", type: "口袋类型", grade: "证据分级",
  residues: "显示使用的残基编号", canonical_residues: "UniProt 标准残基编号",
  center: "口袋中心坐标 / Å", pdb_id: "PDB 编号", chain: "链", ligand: "共晶配体",
  ligand_name: "配体名称", coordinate_system: "坐标体系", structure_source: "显示结构来源",
  resolution: "原始实验分辨率 / Å", redocking_rmsd: "重对接 RMSD / Å",
  score: "口袋分数", probability: "口袋预测概率", note: "说明",
};

export default function StructureViewer({
  entity,
  compact = false,
  height,
}: {
  entity: Entity;
  compact?: boolean;
  height?: number;
}) {
  const reducedMotion = useReducedMotion();
  const initialView = useRef<number[] | null>(null);
  const [loadRevision, setLoadRevision] = useState(0);
  const [surfaceLoading, setSurfaceLoading] = useState(false);
  const stage = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLDivElement>(null);
  const viewer = useRef<any>(null);
  const library = useRef<any>(null);
  const selection = useRef<Evidence>({ model: 0 });
  const [selected, setSelected] = useState("");
  const [chosen, setChosen] = useState<Evidence | undefined>();
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(1);
  const [pocketPage, setPocketPage] = useState<{
    items: Evidence[];
    total: number;
  } | null>(null);
  const [pocketsLoading, setPocketsLoading] = useState(false);
  const [pocketsError, setPocketsError] = useState("");
  const [rotating, setRotating] = useState(false);
  const [surface, setSurface] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [atomCount, setAtomCount] = useState(0);
  const [residueCount, setResidueCount] = useState(0);
  const [renderedResidues, setRenderedResidues] = useState<RenderedResidue[]>([]);
  const [activeResidue, setActiveResidue] = useState("");
  const [ready, setReady] = useState(0);
  const firstPockets = entity.pockets || [];
  const allPocketCount =
    (entity as Entity & { pockets_total?: number }).pockets_total ??
    firstPockets.length;
  const pockets = pocketPage?.items || firstPockets;
  const matchingPocketCount = pocketPage?.total ?? allPocketCount;
  const pocket: Evidence | undefined =
    chosen ||
    pockets.find((item) => item.id === selected) ||
    firstPockets.find((item) => item.structure_url) ||
    firstPockets[0];
  const structure = (entity as Entity & { structure?: Evidence }).structure;
  const sourceUrl: string | undefined =
    pocket?.structure_url || entity.structure_url || structure?.url;

  useEffect(() => {
    setSelected("");
    setChosen(undefined);
    setQuery("");
    setPage(1);
    setPocketPage(null);
    setRotating(false);
    setSurface(false);
  }, [entity.id]);

  useEffect(() => {
    if (compact || entity.kind !== "target") return;
    if (page === 1 && !query.trim()) {
      setPocketPage(null);
      setPocketsLoading(false);
      setPocketsError("");
      return;
    }
    const abort = new AbortController();
    setPocketsLoading(true);
    setPocketsError("");
    const timer = setTimeout(
      async () => {
        try {
          const response = await fetch(
            `/api/evidence?kind=target&id=${encodeURIComponent(entity.id)}&section=pockets&page=${page}&page_size=100&search=${encodeURIComponent(query)}`,
            { signal: abort.signal },
          );
          if (!response.ok)
            throw new Error(`口袋检索失败 (${response.status})`);
          const data = await response.json();
          if (!abort.signal.aborted)
            setPocketPage({ items: data.items || [], total: data.total ?? 0 });
        } catch (failure) {
          if (!abort.signal.aborted)
            setPocketsError((failure as Error).message);
        } finally {
          if (!abort.signal.aborted) setPocketsLoading(false);
        }
      },
      query ? 180 : 0,
    );
    return () => {
      clearTimeout(timer);
      abort.abort();
    };
  }, [entity.id, entity.kind, page, query, compact]);

  useEffect(() => {
    if (!canvas.current) return;
    const host = canvas.current;
    const controller = new AbortController();
    let disposed = false;
    let localViewer: any;
    let resizeObserver: ResizeObserver | undefined;
    setLoading(Boolean(sourceUrl));
    setSurfaceLoading(false);
    initialView.current = null;
    setError("");
    setAtomCount(0);
    setResidueCount(0);
    setRenderedResidues([]);
    setActiveResidue("");
    if (!sourceUrl) {
      setLoading(false);
      return;
    }

    async function load() {
      const timeout = setTimeout(() => controller.abort(), 45_000);
      try {
        const [module, response] = await Promise.all([
          import("3dmol"),
          fetch(sourceUrl!, { signal: controller.signal }),
        ]);
        if (!response.ok)
          throw new Error(`结构文件读取失败 (${response.status})`);
        const text = await response.text();
        if (disposed) return;
        const mol: any = (module as any).createViewer
          ? module
          : (module as any).default;
        if (!mol?.createViewer) throw new Error("三维渲染组件加载失败");
        library.current = mol;
        host.replaceChildren();
        localViewer = mol.createViewer(host, {
          backgroundColor: "#09172D",
          antialias: true,
          defaultcolors: mol.elementColors?.rasmol,
          id: `structure-${entity.id}`,
        });
        viewer.current = localViewer;
        const format = pocket?.structure_format || structure?.format || "pdb";
        localViewer.addModel(text, format);
        const atoms: Evidence[] = localViewer.selectedAtoms({ model: 0 });
        if (!atoms.length) throw new Error("该文件没有可显示的原子坐标");
        setAtomCount(atoms.length);
        let focus: Evidence = { model: 0 };
        let matchedResidues = 0;
        if (
          pocket?.coordinate_system === "uniprot_canonical" &&
          pocket?.residues?.length
        ) {
          focus = { model: 0, resi: pocket.residues };
          matchedResidues = new Set(
            localViewer
              .selectedAtoms(focus)
              .map((atom: Evidence) => `${atom.chain}:${atom.resi}`),
          ).size;
        }
        if (pocket?.ligand_url) {
          const ligandResponse = await fetch(pocket.ligand_url, {
            signal: controller.signal,
          });
          if (!ligandResponse.ok)
            throw new Error(`共晶配体读取失败 (${ligandResponse.status})`);
          const ligandText = await ligandResponse.text();
          if (disposed) return;
          localViewer.addModel(ligandText, pocket.ligand_format || "sdf");
          const ligandAtoms: Evidence[] = localViewer.selectedAtoms({
            model: 1,
          });
          if (!ligandAtoms.length) throw new Error("共晶配体文件没有原子坐标");
          // Both files come from the same frozen experimental protocol. Find
          // the neighborhood geometrically; canonical numbering is not used.
          const near = new Set<string>();
          for (const atom of atoms) {
            if (atom.elem === "H") continue;
            if (
              ligandAtoms.some(
                (other) =>
                  (atom.x - other.x) ** 2 +
                    (atom.y - other.y) ** 2 +
                    (atom.z - other.z) ** 2 <=
                  25,
              )
            ) {
              near.add(`${atom.chain}:${atom.resi}:${atom.icode || ""}`);
            }
          }
          const serials = atoms
            .filter((atom) =>
              near.has(`${atom.chain}:${atom.resi}:${atom.icode || ""}`),
            )
            .map((atom) => atom.serial);
          if (serials.length) focus = { model: 0, serial: serials };
          matchedResidues = near.size;
          localViewer.setStyle(
            { model: 1 },
            {
              stick: { radius: 0.22, colorscheme: "orangeCarbon" },
              sphere: { scale: 0.23, colorscheme: "orangeCarbon" },
            },
          );
        }
        selection.current = focus;
        setResidueCount(matchedResidues);
        const uniqueResidues = new Map<string, RenderedResidue>();
        if (matchedResidues) for (const atom of localViewer.selectedAtoms(focus)) {
          const id = `${atom.chain || ""}:${atom.resi}:${atom.icode || ""}`;
          if (!uniqueResidues.has(id)) uniqueResidues.set(id, {
            id, chain: atom.chain || "", resi: atom.resi, resn: atom.resn || "", icode: atom.icode || "",
          });
        }
        setRenderedResidues([...uniqueResidues.values()]);
        localViewer.setStyle(
          { model: 0 },
          { cartoon: { color: "#59D9DC", opacity: 1 } },
        );
        if (matchedResidues)
          localViewer.addStyle(focus, {
            stick: { radius: 0.19, color: "#FFC857" },
          });
        localViewer.zoomTo();
        localViewer.rotate(18, "x");
        localViewer.rotate(-18, "y");
        localViewer.zoom(1.12);
        localViewer.render();
        initialView.current = localViewer.getView();
        resizeObserver = new ResizeObserver(() => {
          if (!disposed) {
            localViewer.resize();
            localViewer.render();
          }
        });
        resizeObserver.observe(host);
        const webglCanvas = host.querySelector("canvas");
        webglCanvas?.addEventListener(
          "webglcontextlost",
          () => {
            if (!disposed) {
              setError("WebGL 上下文已中断，请刷新页面后重试。");
              setLoading(false);
            }
          },
          { signal: controller.signal },
        );
        if (!disposed) {
          setLoading(false);
          setReady((value) => value + 1);
        }
      } catch (failure) {
        if (disposed) return;
        setLoading(false);
        const message =
          controller.signal.aborted ? "结构加载超时，请重新加载。" : failure instanceof Error ? failure.message : "三维结构加载失败";
        setError(
          /webgl|context|GLViewer/i.test(message)
            ? "当前浏览器无法创建 WebGL 视图。请开启硬件加速或下载结构文件查看。"
            : message,
        );
      } finally {
        clearTimeout(timeout);
      }
    }
    load();
    return () => {
      disposed = true;
      controller.abort();
      resizeObserver?.disconnect();
      if (localViewer) {
        localViewer.spin(false);
        localViewer.clear();
      }
      if (viewer.current === localViewer) viewer.current = null;
      host.replaceChildren();
    };
  }, [entity.id, sourceUrl, pocket?.id, loadRevision]);

  useEffect(() => {
    const current = viewer.current;
    if (!current || loading || error) return;
    current.spin(rotating ? "y" : false, 0.35, true);
    return () => {
      current.spin(false);
    };
  }, [rotating, ready, loading, error]);

  useEffect(() => {
    const current = viewer.current;
    if (!current || !library.current || loading || error) return;
    let cancelled = false;
    current.removeAllSurfaces();
    setSurfaceLoading(surface);
    if (surface) {
      const area = residueCount ? selection.current : { model: 0 };
      Promise.resolve(
        current.addSurface(
          library.current.SurfaceType.VDW,
          { opacity: 0.5, color: "#57D6DC" },
          area,
          { model: 0 },
        ),
      )
        .then(() => {
          if (!cancelled) { current.render(); setSurfaceLoading(false); }
        })
        .catch(() => {
          if (!cancelled) { setSurfaceLoading(false); setError("分子表面生成失败；可切回骨架视图。"); }
        });
    }
    current.render();
    return () => {
      cancelled = true;
    };
  }, [surface, ready, loading, error, residueCount]);

  const reset = () => {
    if (initialView.current) viewer.current?.setView(initialView.current);
    else { viewer.current?.zoomTo(); viewer.current?.zoom(1.12); }
    viewer.current?.render();
  };
  const focus = () => {
    viewer.current?.zoomTo(selection.current, reducedMotion ? 0 : 450);
    viewer.current?.zoom(0.75, reducedMotion ? 0 : 450);
    viewer.current?.render();
  };
  const fullscreen = () => {
    if (document.fullscreenElement) document.exitFullscreen?.();
    else stage.current?.requestFullscreen?.().catch(() => {});
  };
  const disabled = loading || Boolean(error) || !sourceUrl;
  const currentSource =
    pocket?.structure_source || structure?.source || "本地结构";

  const status = loading ? "加载结构中" : error ? "结构加载异常" : sourceUrl ? "三维结构就绪" : "暂无三维结构";
  const zoom = (factor: number) => { viewer.current?.zoom(factor, reducedMotion ? 0 : 200); viewer.current?.render(); };

  const choosePocket = (item: Evidence) => {
    setSelected(item.id); setChosen(item); setSurface(false); setRotating(false);
    if (window.matchMedia("(max-width: 900px)").matches) stage.current?.closest(".swb-viewport")?.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "start" });
  };
  const focusResidue = (residue: RenderedResidue) => {
    setActiveResidue(residue.id);
    const atoms = { model: 0, chain: residue.chain, resi: residue.resi, ...(residue.icode ? { icode: residue.icode } : {}) };
    viewer.current?.zoomTo(atoms, reducedMotion ? 0 : 350);
    viewer.current?.zoom(0.65, reducedMotion ? 0 : 350);
    viewer.current?.render();
  };
  const description = !pocket ? "当前实体未接入口袋注释；画布展示可用的本地结构。" : pocket?.coordinate_system === "experimental_structure"
    ? "显示经项目流程准备的实验受体与共晶配体；高亮配体 5 Å 邻域。共晶配体不代表当前查询药物的预测结合姿态。"
    : pocket?.type === "experimental"
      ? "实验口袋按 UniProt 标准残基编号映射到 AlphaFold 预测模型；画布呈现预测骨架。"
      : "预测口袋与 AlphaFold 模型使用相同残基编号；口袋分数描述位点预测结果。";
  const renderStage = (
    <div ref={stage} className="mol-stage" style={{ height: height ?? (compact ? 360 : 470) }}>
      <div ref={canvas} aria-label={`${entity.name} 分子三维结构`} role="img" className="mol-canvas" />
      <div className="mol-stage-top">
        <div className="mol-stage-label"><i /><span>{compact ? `${entity.name} · 3D STRUCTURE` : (pocket?.name || pocket?.id || entity.name)}</span></div>
        <button type="button" onClick={fullscreen} aria-label="全屏查看结构" title="全屏查看结构" className="mol-icon-button"><Expand size={17} /></button>
      </div>
      <div className="mol-stage-tools" role="group" aria-label="三维结构视角控制">
        <button className="mol-icon-button" type="button" title={rotating ? "停止旋转" : "自动旋转"} aria-label={rotating ? "停止旋转" : "自动旋转"} aria-pressed={rotating} disabled={disabled} onClick={() => setRotating(!rotating)}>{rotating ? <Pause size={17} /> : <Play size={17} />}</button>
        <button className="mol-icon-button" type="button" title="放大结构" aria-label="放大结构" disabled={disabled} onClick={() => zoom(1.2)}><ZoomIn size={17} /></button>
        <button className="mol-icon-button" type="button" title="缩小结构" aria-label="缩小结构" disabled={disabled} onClick={() => zoom(1 / 1.2)}><ZoomOut size={17} /></button>
        <button className="mol-icon-button" type="button" title="复位视角" aria-label="复位视角" disabled={disabled} onClick={reset}><RotateCcw size={17} /></button>
        <button className="mol-icon-button" type="button" title={residueCount ? "聚焦口袋" : "当前结构无匹配口袋残基"} aria-label="聚焦口袋" disabled={disabled || !residueCount} onClick={focus}><Focus size={17} /></button>
      </div>
      {(loading || error || !sourceUrl) && <div className="mol-load-state" role={error ? "alert" : "status"}>
        <div className="mol-load-icon">{loading ? <LoaderCircle size={30} className="spin" /> : error ? <AlertCircle size={30} /> : <Box size={32} />}</div>
        <strong>{loading ? "正在载入分子坐标" : error || "暂无可加载的本地三维结构"}</strong>
        <span>{loading ? "准备蛋白骨架与口袋残基" : "口袋注释与结构来源仍可查看"}</span>
        {error && <button type="button" className="mol-button" onClick={() => { setSurface(false); setLoadRevision((value) => value + 1); }}><RotateCcw size={15} />重新加载</button>}
      </div>}
      <div className="mol-stage-bottom">
        <div className="mol-legend"><span><i className="protein" />蛋白</span><span><i className="pocket" />口袋{pocket?.ligand_url ? " · 5 Å 邻域" : ""}</span>{pocket?.ligand_url && <span><i className="ligand" />共晶配体</span>}</div>
        <span className="mol-gesture-hint">拖动旋转 · 滚轮缩放</span>
      </div>
    </div>
  );
  if (compact) return <div className="structure-viewer mol-viewer mol-compact">{renderStage}</div>;

  return <div className="structure-viewer mol-viewer structure-workbench">
    <div className="swb-workspace-bar">
      <div><Dna size={18} /><strong>{entity.name}</strong><span>结构工作区</span></div>
      <button type="button" className="swb-mobile-library" onClick={() => stage.current?.closest(".structure-workbench")?.querySelector(".swb-library")?.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "start" })}><ListFilter size={14} />口袋 {allPocketCount.toLocaleString()}</button>
      <span className={`mol-status${error ? " is-error" : ""}`} role="status">
        {loading ? <LoaderCircle size={14} className="spin" /> : error ? <AlertCircle size={14} /> : sourceUrl ? <CheckCircle2 size={14} /> : <Box size={14} />}{status}
      </span>
    </div>
    <div className="swb-layout">
      <aside className="swb-library" aria-label="口袋目录">
        <div className="swb-section-heading"><span><ListFilter size={16} />口袋目录</span><strong>{allPocketCount.toLocaleString()}</strong></div>
        <label className="swb-search"><Search size={15} /><input aria-label="检索全部口袋" placeholder="PDB、配体、来源…" value={query} onChange={(event) => { setQuery(event.target.value); setPage(1); }} />
          {pocketsLoading ? <LoaderCircle size={15} className="spin" /> : query ? <button type="button" aria-label="清除口袋搜索" onClick={() => { setQuery(""); setPage(1); }}><X size={14} /></button> : null}
        </label>
        <div className="swb-query-shortcuts"><span>快捷检索</span>{[...(firstPockets.some(item => item.pdb_id) ? [String(firstPockets.find(item => item.pdb_id)?.pdb_id).toUpperCase()] : []), "P2Rank"].map(term => <button type="button" key={term} onClick={() => { setQuery(term); setPage(1); }}>{term}</button>)}</div>
        <div className="swb-pocket-list" role="listbox" aria-label="选择结构口袋" aria-busy={pocketsLoading}>
          {pocketsError ? <div className="swb-list-empty" role="alert"><AlertCircle size={22} /><strong>口袋检索失败</strong><span>{pocketsError}</span></div>
          : !pockets.length ? <div className="swb-list-empty"><Box size={24} /><strong>{pocketsLoading ? "正在检索口袋" : "没有匹配口袋"}</strong><span>{query ? "可尝试 PDB 编号、配体名称或来源。" : "当前实体没有已接入口袋记录。"}</span></div>
          : pockets.map((item, index) => {
            const count = item.residue_count ?? item.canonical_residues?.length ?? item.residues?.length;
            const current = pocket?.id === item.id;
            return <button type="button" role="option" aria-selected={current} className={`swb-pocket-option${current ? " is-selected" : ""}`} key={`${item.id}-${index}`}
              disabled={pocketsLoading} tabIndex={current || (!pockets.some(p => p.id === pocket?.id) && index === 0) ? 0 : -1}
              onClick={() => choosePocket(item)} onKeyDown={event => {
                const options = Array.from(event.currentTarget.parentElement!.querySelectorAll<HTMLButtonElement>('[role="option"]'));
                const next = event.key === "ArrowDown" ? Math.min(index + 1, options.length - 1) : event.key === "ArrowUp" ? Math.max(0, index - 1) : event.key === "Home" ? 0 : event.key === "End" ? options.length - 1 : -1;
                if (next >= 0) { event.preventDefault(); options[next]?.focus(); }
              }}>
              <span className="swb-option-title"><strong>{item.name || item.id}</strong>{current && <CheckCircle2 size={14} />}</span>
              <span className={`swb-kind ${item.coordinate_system === "experimental_structure" ? "holo" : item.type === "predicted" ? "predicted" : "mapped"}`}>{pocketKind(item)}</span>
              <span className="swb-option-metrics">{count != null && <span>{count} 个注释残基</span>}{item.resolution != null ? <span>{metric(item.resolution)} Å</span> : item.probability != null ? <span>概率 {metric(item.probability, 3)}</span> : item.chain ? <span>链 {item.chain}</span> : null}</span>
            </button>;
          })}
        </div>
        <div className="swb-pagination"><span role="status">{pocketsLoading ? "检索中…" : matchingPocketCount ? `${(page - 1) * 100 + 1}–${Math.min(page * 100, matchingPocketCount)} / ${matchingPocketCount.toLocaleString()}` : "0 个匹配"}</span>
          <div><button type="button" aria-label="上一页口袋" disabled={page <= 1 || pocketsLoading} onClick={() => setPage(value => value - 1)}><ChevronLeft size={16} /></button><button type="button" aria-label="下一页口袋" disabled={page * 100 >= matchingPocketCount || pocketsLoading} onClick={() => setPage(value => value + 1)}><ChevronRight size={16} /></button></div>
        </div>
        {query && <p className="swb-result-scope">搜索匹配 {matchingPocketCount.toLocaleString()} / 全部 {allPocketCount.toLocaleString()} 个口袋</p>}
      </aside>
      <section className="swb-viewport" aria-label="当前三维画布">
        <div className="swb-view-toolbar">
          <span className="swb-coordinate-badge">{pocket?.coordinate_system === "experimental_structure" ? "实验结构坐标" : pocket?.coordinate_system === "uniprot_canonical" ? "UniProt / AlphaFold" : "本地坐标"}</span>
          <div className="mol-mode-controls" role="group" aria-label="分子显示模式">
            <button type="button" className="mol-button" disabled={loading || !sourceUrl} aria-pressed={!surface} onClick={() => { setSurface(false); setError(""); }}><Box size={15} />骨架</button>
            <button type="button" className="mol-button" disabled={disabled || surfaceLoading} aria-pressed={surface} onClick={() => setSurface(!surface)}>{surfaceLoading ? <LoaderCircle size={15} className="spin" /> : <Layers3 size={15} />}{surfaceLoading ? "生成表面" : "口袋表面"}</button>
          </div>
        </div>
        {renderStage}
        <div className="swb-canvas-footer"><div className="mol-atom-stats"><strong>{atomCount.toLocaleString()}</strong><span>原子</span><i /><strong>{residueCount}</strong><span>高亮残基</span></div><span>{(pocket?.structure_format || structure?.format || "pdb").toUpperCase()}</span></div>
        <div className="swb-residues">
          <div className="swb-section-heading"><span>画布中的口袋残基</span>{renderedResidues.length > 0 && <button type="button" onClick={() => { setActiveResidue(""); focus(); }} disabled={disabled}>聚焦全部 <Focus size={13} /></button>}</div>
          {renderedResidues.length ? <><div className="swb-residue-strip" aria-label="口袋残基，点击聚焦">
            {renderedResidues.map(residue => <button type="button" key={residue.id} aria-pressed={activeResidue === residue.id} disabled={disabled} title={`聚焦 ${residue.resn} ${residue.chain || "无链"}:${residue.resi}${residue.icode}`} onClick={() => focusResidue(residue)}>{residue.resn}<strong>{residue.resi}{residue.icode}</strong>{residue.chain && <small>{residue.chain}</small>}</button>)}
          </div><p>来自当前加载坐标；点击残基定位。</p></> : <p>{loading ? "读取残基映射…" : "当前画布没有可映射的口袋残基。"}</p>}
        </div>
      </section>
      <aside className="swb-inspector" aria-label="当前口袋证据">
        <div className="swb-section-heading"><span>当前口袋证据</span><span className="swb-live-indicator" /></div>
        <div className="swb-inspector-content">
          <span className={`swb-kind ${pocket?.coordinate_system === "experimental_structure" ? "holo" : pocket?.type === "predicted" ? "predicted" : "mapped"}`}>{pocketKind(pocket)}</span>
          <h3>{pocket?.name || entity.name}</h3>
          <code className="swb-pocket-id">{pocket?.id || entity.id}</code>
          <dl className="swb-facts">
            <div><dt>画布结构</dt><dd>{currentSource}</dd></div>
            {pocket?.pdb_id && <div><dt>关联 PDB</dt><dd>{String(pocket.pdb_id).toUpperCase()}{pocket.chain ? ` · 链 ${pocket.chain}` : ""}</dd></div>}
            {pocket?.ligand && <div><dt>共晶配体注释</dt><dd>{pocket.ligand}</dd></div>}
            {pocket?.resolution != null && <div><dt>原始实验分辨率</dt><dd>{metric(pocket.resolution)} Å</dd></div>}
            {pocket?.redocking_rmsd != null && <div><dt>重对接 RMSD</dt><dd>{metric(pocket.redocking_rmsd, 3)} Å</dd></div>}
            {pocket?.probability != null && <div><dt>口袋预测概率</dt><dd>{metric(pocket.probability, 3)}</dd></div>}
            {pocket?.score != null && <div><dt>口袋预测分数</dt><dd>{metric(pocket.score, 3)}</dd></div>}
            <div><dt>证据来源</dt><dd>{pocket?.source || currentSource}</dd></div>
          </dl>
          <p className="swb-evidence-note">{description}</p>
          {pocket?.ligand_name && <details className="swb-secondary-detail"><summary>配体名称</summary><p>{pocket.ligand_name}</p></details>}
          {pocket?.canonical_residues?.length > 0 && <details className="swb-secondary-detail"><summary>UniProt 标准残基 · {pocket.canonical_residues.length}</summary><p>{pocket.canonical_residues.join(", ")}</p></details>}
          {pocket?.external_url && /^https?:\/\//.test(pocket.external_url) && <a className="swb-link" href={pocket.external_url} target="_blank" rel="noreferrer">查看原始结构记录 <ExternalLink size={14} /></a>}
          {pocket && <details className="swb-secondary-detail swb-all-metadata"><summary>完整口袋注释</summary><dl>{Object.entries(pocket).map(([key, value]) => <div key={key}><dt>{annotationNames[key] || key.replace(/_/g, " ")}</dt><dd>{typeof value === "string" && /^https?:\/\//.test(value) ? <a href={value} target="_blank" rel="noreferrer">打开原始记录 ↗</a> : value == null ? "—" : Array.isArray(value) ? value.join(", ") : typeof value === "object" ? JSON.stringify(value) : String(value)}</dd></div>)}</dl></details>}
        </div>
        <div className="swb-downloads">{sourceUrl && <a href={sourceUrl} download><Download size={15} />下载当前结构</a>}{pocket?.ligand_url && <a href={pocket.ligand_url} download><Download size={15} />共晶配体 SDF</a>}</div>
      </aside>
    </div>
  </div>;
}
