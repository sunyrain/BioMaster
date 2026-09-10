import { useEffect, useState, type ReactNode } from 'react';
import { Download, Plus, Pencil, Trash2, Search, ClipboardPen } from 'lucide-react';

type Design = {experiment_id:string;pair_id:string;drug_name:string;target_name:string;target_id:string;group:string;priority_rank:number|null;priority:string|null;template_order:number};
export type Entry = Record<string,string>;
const groups: Record<string,string> = {baseline:'基线候选',control:'参考对照',expanded:'扩展候选',all:'全部分组'};
const initial = ():Entry => ({experiment_id:'',pair_id:'',sample_id:'',experiment_date:new Date().toISOString().slice(0,10),result:'',qc:'',KD:'',KD_unit:'',ka_M_inv_s_inv:'',kd_s_inv:'',Rmax_RU:'',chi2:'',construct:'',max_concentration_uM:'',notes:''});
export default function SPRInput({children,batch,onBatch,busy,onChange,onValidate,saved}: {children:ReactNode;batch:string;onBatch:(v:string)=>void;busy:boolean;onChange:()=>void;onValidate:(rows:Entry[])=>Promise<void>;saved:string}) {
  const [mode,setMode]=useState('online');
  const [scope,setScope]=useState('baseline');
  const [order,setOrder]=useState('priority');
  const [target,setTarget]=useState('');
  const [search,setSearch]=useState('');
  const [catalog,setCatalog]=useState<Design[]>([]);
  const [loading,setLoading]=useState(true);
  const [error,setError]=useState('');
  const [reload,setReload]=useState(0);
  const [draft,setDraft]=useState<Entry>(initial);
  const [queue,setQueue]=useState<Entry[]>([]);
  const [editing,setEditing]=useState<number|null>(null);
  const [formError,setFormError]=useState('');
  const [notice,setNotice]=useState('');
  useEffect(()=>{
    const c=new AbortController();setLoading(true);setError('');
    fetch('/api/spr-results/catalog?scope=all&order=priority',{signal:c.signal}).then(async r=>{const d=await r.json();if(!r.ok)throw new Error(d.error||'实验目录加载失败');return d;}).then(d=>setCatalog(d.items)).catch(e=>{if(e.name!=='AbortError')setError(e.message);}).finally(()=>{if(!c.signal.aborted)setLoading(false);});return()=>c.abort();
  },[reload]);
  useEffect(()=>{if(saved&&mode==='online'){setQueue([]);setDraft(initial());setEditing(null);setNotice('');}},[saved]);
  function change(key:string,v:string){setDraft(d=>({...d,[key]:v,...(key==='result'&&v!=='detected'?{KD:'',KD_unit:''}:{})}));setFormError('');onChange();}
  const pool=catalog.filter(r=>scope==='all'||r.group===scope);
  const targets=[...new Map(pool.map(r=>[r.target_id,r.target_name])).entries()].sort((a,b)=>a[1].localeCompare(b[1]));
  const visible=pool.filter(r=>(!target||r.target_id===target)&&`${r.experiment_id} ${r.drug_name} ${r.target_name} ${r.target_id}`.toLowerCase().includes(search.trim().toLowerCase())).sort((a,b)=>{
    const groupOrder=['baseline','control','expanded'];const g=groupOrder.indexOf(a.group)-groupOrder.indexOf(b.group);if(g)return g;
    if(order==='target'){const t=a.target_name.localeCompare(b.target_name);if(t)return t;}
    return order==='id'?a.experiment_id.localeCompare(b.experiment_id):(a.priority_rank??1e9)-(b.priority_rank??1e9)||a.experiment_id.localeCompare(b.experiment_id);
  });
  const selected=catalog.find(r=>r.experiment_id===draft.experiment_id);
  const params=new URLSearchParams({scope,order,target,search:search.trim()});
  const dirty=Object.entries(draft).some(([k,v])=>!['experiment_id','pair_id','experiment_date'].includes(k)&&!!v);
  function add(){
    if(!selected){setFormError('请先从左侧选择实验配对。');return;}
    if(!draft.sample_id.trim()||!draft.experiment_date||!draft.result||!draft.qc){setFormError('请填写样本编号、实验日期，并选择响应结果与质控结果。');return;}
    if(queue.some((r,i)=>i!==editing&&r.sample_id.trim()===draft.sample_id.trim())){setFormError('待提交列表中已有相同样本编号；重复实验请使用不同编号。');return;}
    if(draft.KD&&!draft.KD_unit){setFormError('填写 KD 后请选择单位。');return;}
    const entry={...draft,sample_id:draft.sample_id.trim()};
    setQueue(q=>editing===null?[...q,entry]:q.map((r,i)=>i===editing?entry:r));setEditing(null);
    setDraft({...initial(),experiment_id:draft.experiment_id,pair_id:draft.pair_id,experiment_date:draft.experiment_date});
    setFormError('');setNotice('已加入待提交列表，可继续填写下一次实验。');onChange();
  }
  return <>
    <div className="sri-mode" role="group" aria-label="结果录入方式">{[['online','在线填写','选择配对，中文表单录入'],['csv','CSV 批量上传','下载有序模板，批量导入']].map(([id,title,description])=><button key={id} disabled={busy} aria-pressed={mode===id} className={mode===id?'active':''} onClick={()=>{setMode(id);onChange();}}><strong>{title}</strong><span>{description}</span></button>)}</div>
    <section className="sru-card sri-design"><div className="sru-section-heading"><div><span className="sru-kicker">DESIGN ORDER</span><h3>选择实验范围与顺序</h3><p>优先顺序与原候选编号是两列；原编号保持不变。对照和扩展候选没有基线优先级。</p></div><a className="sru-template" aria-disabled={loading||!!error} href={loading||error?undefined:`/api/spr-results/template.csv?${params}`}><Download size={17}/>下载当前范围模板（{visible.length}）</a></div>
    <div className="sri-filters"><label>实验分组<select value={scope} disabled={busy} onChange={e=>{setScope(e.target.value);setTarget('');}}>{Object.entries(groups).map(([k,v])=><option key={k} value={k}>{v}</option>)}</select></label><label>排列方式<select value={order} disabled={busy} onChange={e=>setOrder(e.target.value)}><option value="priority">优先顺序</option><option value="id">原实验编号</option><option value="target">靶点 → 优先顺序</option></select></label><label>靶点筛选<select value={target} disabled={busy} onChange={e=>setTarget(e.target.value)}><option value="">全部靶点</option>{targets.map(([k,v])=><option key={k} value={k}>{v} · {k}</option>)}</select></label><label>搜索配对<div className="sri-search"><Search size={16}/><input value={search} disabled={busy} onChange={e=>setSearch(e.target.value)} placeholder="药物、靶点或候选编号"/></div></label></div>
    {error&&<p className="sru-error" role="alert">{error}<button onClick={()=>setReload(n=>n+1)}>重新加载</button></p>}
    <details className="sri-template-help"><summary>模板如何填写？</summary><p>模板版本后的六列是识别信息：模板序号、实验分组、优先顺序、优先级、药物名称、靶点名称。保留实验编号与配对编号，填写后面的样本编号、实验日期、响应结果、质控结果及可选测量值。</p><p>CSV 支持中文选项：检出响应 / 未检出响应 / 无法判定；通过 / 未通过 / 待确认。旧模板已退役；请使用 SPR_RESULTS_V2 中文模板并保留模板版本列。未完成实验的空白行自动跳过。</p></details>
    </section>
    {mode==='csv'?children:<>
      <div className="sri-entry-grid"><section className="sru-card sri-pair-card"><h3>1. 选择实验配对 <small>{visible.length} 组</small></h3><p className="sri-muted">按上方范围与顺序显示，点击配对开始填写。</p><div className="sri-pair-list" aria-label="可选实验配对">{loading?<p role="status">正在读取实验目录…</p>:visible.length===0?<p>没有匹配的配对，请调整筛选。</p>:visible.map((r,i)=><button key={r.experiment_id} disabled={busy||dirty||editing!==null} aria-pressed={selected?.experiment_id===r.experiment_id} className={selected?.experiment_id===r.experiment_id?'selected':''} onClick={()=>{setDraft({...draft,experiment_id:r.experiment_id,pair_id:r.pair_id});setNotice('');onChange();}}><span className="sri-seq">{i+1}</span><span><strong>{r.drug_name} <span>→ {r.target_name}</span></strong><small>{r.experiment_id} · {groups[r.group]}{r.priority_rank?` · 优先顺序 ${r.priority_rank}`:''}</small></span></button>)}</div></section>
      <section className="sru-card sri-form"><h3><ClipboardPen size={20}/>2. {editing===null?'填写实验结果':'编辑待提交结果'}</h3><div className="sri-batch"><label className="sru-label" htmlFor="spr-online-batch">实验批次 <span>本次待提交记录共用一个批次</span></label><input id="spr-online-batch" className="sru-input" value={batch} maxLength={120} disabled={busy} onChange={e=>onBatch(e.target.value)} placeholder="例如 SPR-20260910-AR-01"/></div>{!selected?<div className="sru-empty"><strong>先选择药物与靶点</strong><p>身份信息自动带入，无须复制实验编号。</p></div>:<><div className="sri-selected"><strong>{selected.drug_name} → {selected.target_name}</strong><span>{selected.experiment_id} · {groups[selected.group]} · {selected.priority||'此组不设基线优先级'}</span><small>{selected.pair_id}</small></div>
      <fieldset disabled={busy}><div className="sri-fields"><label>样本编号 *<input value={draft.sample_id} maxLength={120} onChange={e=>change('sample_id',e.target.value)} placeholder="例如 AR-001-R1"/></label><label>实验日期 *<input type="date" value={draft.experiment_date} max={new Date().toISOString().slice(0,10)} onChange={e=>change('experiment_date',e.target.value)}/></label><label>响应结果 *<select aria-label="响应结果 *" value={draft.result} onChange={e=>change('result',e.target.value)}><option value="">请选择实测响应</option><option value="detected">检出响应</option><option value="not_detected">未检出响应</option><option value="inconclusive">无法判定</option></select></label><label>质控结果 *<select aria-label="质控结果 *" value={draft.qc} onChange={e=>change('qc',e.target.value)}><option value="">请选择质控结论</option><option value="pass">通过</option><option value="fail">未通过</option><option value="review">待确认</option></select></label>
      {draft.result==='detected'&&<><label>KD（可选）<input type="number" min="0" step="any" value={draft.KD} onChange={e=>change('KD',e.target.value)} placeholder="填写拟合值"/></label><label>KD 单位<select aria-label="KD 单位" value={draft.KD_unit} onChange={e=>change('KD_unit',e.target.value)}><option value="">请选择单位</option>{['M','mM','uM','nM','pM'].map(u=><option key={u}>{u}</option>)}</select></label></>}
      </div><details className="sri-advanced"><summary>更多测量信息 · 动力学、构建与浓度</summary><div className="sri-fields">{[['ka_M_inv_s_inv','ka / M⁻¹s⁻¹'],['kd_s_inv','kd / s⁻¹'],['Rmax_RU','Rmax / RU'],['chi2','χ²'],['max_concentration_uM','最高浓度 / µM'],['construct','蛋白构建']].map(([k,label])=><label key={k}>{label}<input type={k==='construct'?'text':'number'} min="0" step="any" value={draft[k]} onChange={e=>change(k,e.target.value)}/></label>)}</div></details><label className="sri-notes">备注 / 原始曲线档案编号<textarea value={draft.notes} maxLength={4000} onChange={e=>change('notes',e.target.value)} placeholder="实验条件、异常响应或原始资料的档案编号" rows={2}/></label></fieldset>
      {formError&&<p className="sru-error" role="alert">{formError}</p>}{notice&&<p className="sri-notice" role="status">{notice}</p>}<div className="sri-actions"><button className="sru-primary" disabled={busy} onClick={add}><Plus size={17}/>{editing===null?'加入待提交列表':'保存本条修改'}</button><button disabled={busy} onClick={()=>{setDraft(initial());setEditing(null);setFormError('');setNotice('');onChange();}}>清空本条 / 重选配对</button></div><p className="sri-muted">填写后先加入列表，再校验提交。切换配对前请保存或清空本条。</p></>}
      </section></div>
      <section className="sru-card"><div className="sru-section-heading"><div><h3>3. 待提交列表 · {queue.length} 条</h3><p>可以编辑、移除或继续添加重复实验；此处尚未入库。</p></div><button className="sru-primary" disabled={busy||!batch.trim()||!queue.length||editing!==null||dirty} onClick={()=>onValidate(queue)}>校验在线记录并预览</button></div>{!queue.length?<p className="sri-muted">填写右侧表单，将实验记录加入此列表。</p>:<div className="sru-table-scroll"><table><thead><tr><th>样本</th><th>实验配对</th><th>响应 / 质控</th><th>KD</th><th>操作</th></tr></thead><tbody>{queue.map((r,i)=>{const d=catalog.find(d=>d.experiment_id===r.experiment_id);return <tr key={i}><td>{r.sample_id}<small>{r.experiment_date}</small></td><td>{d?.drug_name} → {d?.target_name}<small>{r.experiment_id}</small></td><td>{{detected:'检出响应',not_detected:'未检出响应',inconclusive:'无法判定'}[r.result]} / {{pass:'通过',fail:'未通过',review:'待确认'}[r.qc]}</td><td>{r.KD||'—'} {r.KD_unit}</td><td><div className="sri-actions"><button aria-label={`编辑 ${r.sample_id}`} disabled={busy||dirty||editing!==null} onClick={()=>{setDraft({...r});setEditing(i);setNotice('');onChange();}}><Pencil size={15}/></button><button aria-label={`移除 ${r.sample_id}`} disabled={busy||editing!==null} onClick={()=>{setQueue(q=>q.filter((_,n)=>n!==i));onChange();}}><Trash2 size={15}/></button></div></td></tr>;})}</tbody></table></div>}<p className="sri-muted">待提交列表仅保留在当前页面，请完成提交后再离开。检出响应不等于特异性结合；全部结果入库后待审核。</p></section>
    </>}
  </>;
}
