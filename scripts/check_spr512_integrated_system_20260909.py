"""Smoke-test the real project catalog, selected scoring path, and local HTTP routes."""
import sys,json,threading,http.client
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from biomaster.explorer_data import ExplorerData
from biomaster.explorer_server import ExplorerHTTPServer

def main():
 data=ExplorerData(ROOT);summary=data.summary();assert summary['counts']['disease_targets']==886;assert summary['counts']['txgnn_drugs']==605
 server=ExplorerHTTPServer(('127.0.0.1',0),data,ROOT/'web/dist');thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 def get(path):
  conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=60);conn.request('GET',path);r=conn.getresponse();body=r.read();assert r.status==200,(r.status,body[:200]);conn.close();return json.loads(body)
 try:
  target=get('/api/entity/target/CHEMBL2326');rank=get('/api/rankings?kind=drug&id=USZAGAREISWJDP-UHFFFAOYSA-N&page_size=3');assert len(target['disease_review']['spr_records'])==8 and rank['items']
  result={'counts':summary['counts'],'http_entity_status':200,'http_ranking_status':200,'ranking_denominator':rank['denominator'],'spr_target_records':8,'infer_enabled':True,'training':False,'source_paths':target['disease_review']['spr_records'][0]['source_paths']}
  (ROOT/'outputs/spr512_integrated_disease_20260909/SYSTEM_SMOKE_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(result)
 finally:server.shutdown();server.server_close()
if __name__=='__main__':main()
