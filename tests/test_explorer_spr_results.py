import csv
import io
import json
import http.client
import threading
from types import SimpleNamespace

import pytest
from biomaster.explorer_spr_results import SPRResults, FIELDS, csv_bytes, LABELS, TEMPLATE_VERSION
from biomaster.explorer_server import ExplorerHTTPServer

@pytest.fixture
def store(tmp_path):
    return SPRResults(tmp_path)

@pytest.fixture
def registry():
    return {'C384-001': {'experiment_id':'C384-001','pair_id':'DRUG__TARGET','drug_id':'DRUG','target_id':'TARGET','drug_name':'drug','gene_symbol':'GENE','is_control':False}}

def payload(rows=None, batch='B1'):
    row = {'experiment_id':'C384-001','pair_id':'DRUG__TARGET','sample_id':'S1','experiment_date':'2026-09-09','result':'detected','qc':'pass','KD':'0.25','KD_unit':'uM','chi2':'0'}
    out = io.StringIO()
    writer=csv.DictWriter(out,fieldnames=list(LABELS.values()))
    writer.writeheader()
    for entry in (rows if rows is not None else [row]):
        entry={**entry,'template_version':TEMPLATE_VERSION}
        if entry.get('result'): entry['result']={'detected':'检出响应','not_detected':'未检出响应','inconclusive':'无法判定'}.get(entry['result'],entry['result'])
        if entry.get('qc'): entry['qc']={'pass':'通过','fail':'未通过','review':'待确认'}.get(entry['qc'],entry['qc'])
        writer.writerow({LABELS[k]:v for k,v in entry.items()})
    return {'batch':batch,'filename':'result.csv','content':out.getvalue()}

def base_row():
    return {'experiment_id':'C384-001','pair_id':'DRUG__TARGET','sample_id':'S1','experiment_date':'2026-09-09','result':'detected','qc':'pass','KD':'0.25','KD_unit':'uM','chi2':'0'}


def test_preview_commit_persistence_and_original_units(store,registry):
    result=store.preview(payload(),registry,'alice')
    assert result['items'][0]['KD_nM']==250
    assert result['items'][0]['chi2']==0
    assert store.results()['total']==0
    with pytest.raises(ValueError,match='不属于'):
        store.commit(result['token'],'bob')
    saved=store.commit(result['token'],'alice')
    assert saved['count']==1
    assert store.commit(result['token'],'alice')['already_saved']
    records=SPRResults(store.path.parents[3]).results()['items']
    assert records[0]['KD_unit']=='uM' and records[0]['review_status']=='pending'
    assert records[0]['uploaded_by']=='alice'
    assert store.results(pair='other')['total']==0
    assert store.results(search='GENE')['total']==1
    assert store.preview(payload(),registry,'alice')['error_count']==1

@pytest.mark.parametrize('field,value',[('pair_id','WRONG'),('experiment_id','UNKNOWN'),('KD','nan'),('KD','-1'),('KD','inf'),('KD_unit','mg/mL'),('result','not_detected'),('experiment_date','2099-01-01'),('qc','unknown')])
def test_invalid_rows_never_get_commit_token(store,registry,field,value):
    row=base_row()
    row[field]=value
    checked=store.preview(payload([row]),registry,'alice')
    assert checked['error_count']==1 and checked['token'] is None
    assert store.results()['total']==0

def test_blank_template_rows_skip_and_replicates_remain_distinct(store,registry):
    rows=[base_row()]
    rows += [{**rows[0],'sample_id':'S2','result':'not_detected','KD':'','KD_unit':''}, {'experiment_id':'C384-001','pair_id':'DRUG__TARGET'}]
    result=store.preview(payload(rows),registry,'alice')
    assert result['skipped_count']==1 and result['valid_count']==2
    store.commit(result['token'],'alice')
    assert store.results()['total']==2
    assert store.results()['items'][1].get('KD_nM') is None

def test_concurrent_previews_atomic_duplicate_rejection(store,registry):
    first=store.preview(payload(),registry,'alice')
    row=base_row()
    second=store.preview(payload([{**row,'sample_id':'S2'},row]),registry,'alice')
    store.commit(first['token'],'alice')
    with pytest.raises(ValueError,match='未导入任何行'):
        store.commit(second['token'],'alice')
    assert store.results()['total']==1

def test_expiry_format_and_formula_escape(store,registry):
    result=store.preview(payload(),registry,'alice')
    with store.connect() as db: db.execute('UPDATE uploads SET created=0')
    with pytest.raises(ValueError,match='过期'): store.commit(result['token'],'alice')
    with pytest.raises(ValueError): store.preview({**payload(),'content':'x,x\n1,2'},registry,'alice')
    assert "'=HYPERLINK" in csv_bytes([{'notes':'=HYPERLINK("x")'}],['notes']).decode('utf-8-sig')

@pytest.fixture
def http_server(tmp_path,monkeypatch,registry):
    data=SimpleNamespace(root=tmp_path)
    server=ExplorerHTTPServer(('127.0.0.1',0),data,tmp_path)
    monkeypatch.setattr(server.spr_results,'registry',lambda data: registry)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield server
    server.shutdown();server.server_close();thread.join()

def call(server,path,body=None,headers=None):
    conn=http.client.HTTPConnection(*server.server_address,timeout=5)
    conn.request('POST' if body is not None else 'GET',path,json.dumps(body) if body is not None else None,headers or {'Content-Type':'application/json'})
    response=conn.getresponse(); result=(response.status,response.read());conn.close();return result

def test_http_upload_auth_origin_and_export(http_server):
    assert call(http_server,'/api/spr-results/preview',payload(),{'Content-Type':'application/json','Origin':'https://evil.example'})[0]==403
    http_server.auth=SimpleNamespace(username=lambda h:None)
    assert call(http_server,'/api/spr-results/preview',payload())[0]==401
    http_server.auth=None
    status,body=call(http_server,'/api/spr-results/preview',payload())
    assert status==200
    token=json.loads(body)['token']
    assert call(http_server,'/api/spr-results/commit',{'token':token})[0]==200
    status,body=call(http_server,'/api/spr-results.csv')
    assert status==200
    assert len(list(csv.DictReader(io.StringIO(body.decode('utf-8-sig')))))==1
    assert call(http_server,'/api/spr-results/template.csv')[0]==200
    assert call(http_server,'/api/spr-results?page=0')[0]==400

def test_template_order_context_and_chinese_roundtrip(store, registry):
    first={**registry['C384-001'],'group':'baseline','priority_rank':2,'priority':'P1','target_name':'GENE'}
    second={**first,'experiment_id':'C384-040','priority_rank':1,'pair_id':'D2__T2','drug_id':'D2','target_id':'T2','drug_name':'Drug 2','target_name':'AAA'}
    control={**first,'experiment_id':'CTRL-001','group':'control','priority_rank':None,'priority':None}
    reg={r['experiment_id']:r for r in [first,control,second]}
    template=store.template(reg).decode('utf-8-sig')
    rows=list(csv.DictReader(io.StringIO(template)))
    assert [r['实验编号'] for r in rows]==['C384-040','C384-001']
    assert [r['模板序号'] for r in rows]==['1','2']
    assert rows[0]['药物名称']=='Drug 2' and rows[0]['优先顺序']=='1'
    assert [r['experiment_id'] for r in store.catalog(reg,order='id')]==['C384-001','C384-040']
    assert len(store.catalog(reg,scope='all'))==3
    assert len(store.catalog(reg,target='T2'))==1
    rows[0].update(样本编号='S1',实验日期='2026-09-09',响应结果='检出响应',质控结果='通过',KD='2',KD单位='uM')
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    preview=store.preview({'batch':'CN','filename':'cn.csv','content':out.getvalue()},reg,'alice')
    assert preview['valid_count']==1 and preview['skipped_count']==1 and preview['error_count']==0
    assert preview['items'][0]['KD_nM']==2000
    store.commit(preview['token'],'alice')
    assert store.results()['items'][0]['drug_name']=='Drug 2'

def test_online_uses_same_validation_and_tracks_source(store,registry):
    row=base_row()
    checked=store.preview({'batch':'ONLINE','rows':[row]},registry,'alice')
    assert checked['items'][0]['KD_nM']==250
    store.commit(checked['token'],'alice')
    record=store.results()['items'][0]
    assert record['entry_method']=='online' and record['filename']=='online-entry.csv'
    assert store.preview({'batch':'ONLINE','rows':[row]},registry,'alice')['token'] is None
    assert store.preview({'batch':'ONLINE2','rows':[{**row,'pair_id':'WRONG'}]},registry,'alice')['error_count']==1
    with pytest.raises(ValueError): store.preview({'batch':'ONLINE','rows':[]},registry,'alice')


def test_retired_template_and_version_rejected(store,registry):
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=FIELDS);writer.writeheader();writer.writerow(base_row())
    with pytest.raises(ValueError,match='退役'): store.preview({'batch':'OLD','filename':'old.csv','content':out.getvalue()},registry,'alice')
    current=payload();current['content']=current['content'].replace(TEMPLATE_VERSION,'RETIRED')
    assert store.preview(current,registry,'alice')['token'] is None
