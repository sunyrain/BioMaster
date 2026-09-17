"""Behavior tests: rank denominators, absence, full exports, and file isolation."""
import csv
import hashlib
import http.client
import io
import json
from pathlib import Path
import threading

import numpy as np
import pandas as pd
import pytest

from biomaster.explorer_data import BUNDLE, FROZEN, PAIRS, ExplorerData, clean
from biomaster.explorer_server import ExplorerHTTPServer
from biomaster.dtiam_release import SCORES as DTIAM_SCORES, MANIFEST as DTIAM_MANIFEST, RELEASE_ID as DTIAM_RELEASE


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    rows = []
    for drug in ("D1", "D2"):
        for i, target in enumerate(("T1", "T2", "T3")):
            rows.append({"ligand_inchikey": drug, "drug_names": "Drug " + drug,
                         "ligand_smiles": "CCO", "target_chembl_id": target, "gene_symbol": "Gene " + target,
                         "conplex_score": [0.9, 0.6, 0.2][i], "drugclip_cosine_mean": [0.2, 0.8, None][i],
                         "is_any_frozen_known_relationship": drug == "D1" and target == "T1"})
    frame = pd.DataFrame(rows)
    path = tmp_path / PAIRS
    path.parent.mkdir(parents=True)
    frame.to_csv(path, index=False)
    frozen = frame[["ligand_inchikey", "drug_names", "target_chembl_id", "gene_symbol"]].copy()
    frozen["biomaster_independent_borda_score"] = [0.4, 0.2, 0.1, 0.5, 0.3, 0.2]
    frozen["dtiam_probability"] = [0.5, 0.4, 0.3, 0.9, 0.1, 0.2]
    path = tmp_path / FROZEN
    path.parent.mkdir(parents=True)
    frozen.to_csv(path, index=False)
    upgraded = frozen[['ligand_inchikey', 'target_chembl_id']].copy()
    upgraded['dtiam_probability'] = [0.2, 0.8, 0.3, 0.9, 0.1, 0.5]
    upgraded['dtiam_model_version'] = DTIAM_RELEASE
    path = tmp_path / DTIAM_SCORES
    path.parent.mkdir(parents=True)
    upgraded.to_csv(path, index=False)
    (tmp_path / DTIAM_MANIFEST).write_text(json.dumps(dict(status='COMPLETE', release_id=DTIAM_RELEASE,
        rows=len(upgraded), scores_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    bundle = tmp_path / BUNDLE
    bundle.mkdir(parents=True)
    (bundle / "MANIFEST.json").write_text('{"files": {}}')
    monkeypatch.setattr(ExplorerData, "_selected_scores", lambda self, pairs: np.array([
        [3, 0.1], [3, 0.2], [1, 0.3], [2, 0.9], [1, 0.8], [0, 0.7]], dtype=np.float32))
    data = ExplorerData(tmp_path)
    data.ensure_loaded()
    return data


def test_rank_computed_before_search_pagination_and_missing_is_null(catalog):
    result = catalog.rankings("drug", "D1", search="T2", page_size=1)
    assert result["denominator"] == 3
    assert result["total"] == 1
    assert result["items"][0]["rank"] == 2  # tie is resolved by entity identifier
    assert result["items"][0]["id"] == "T2"
    all_rows = catalog.rankings("drug", "D1", "drugclip")
    assert all_rows["denominator"] == 2
    assert [row["id"] for row in all_rows["items"]] == ["T2", "T1", "T3"]
    missing = all_rows["items"][-1]
    assert missing["score"] is None and missing["rank"] is None
    assert missing["ranks"]["conplex"] == 3
    assert missing["known_relation"] is False
    # Unknown relationship means no recorded evidence, never a negative label.
    assert "negative" not in missing
    json.dumps(all_rows, allow_nan=False)


def test_reverse_rank_uses_reverse_head(catalog):
    result = catalog.rankings("target", "T1")
    assert result["auxiliary"] is True
    assert result["denominator"] == 2
    assert result["items"][0]["id"] == "D2"
    assert result["items"][0]["score"] == pytest.approx(0.9)
    assert catalog.rankings("drug", "D1")["items"][0]["score"] == 3


def test_dtiam_upgrade_replaces_historical_scores_in_both_directions_and_export(catalog, http_server):
    result = catalog.rankings('drug', 'D1', 'dtiam')
    assert result['dtiam_model_version'] == DTIAM_RELEASE
    assert result['source']['version'] == DTIAM_RELEASE
    assert [r['id'] for r in result['items']] == ['T2', 'T3', 'T1']
    assert result['items'][0]['score'] == pytest.approx(.8)
    target = catalog.rankings('target', 'T2', 'dtiam')
    assert target['items'][0]['id'] == 'D1' and target['denominator'] == 2
    status, _, body = get(http_server, '/api/rankings.csv?kind=drug&id=D1&model=dtiam')
    rows = list(csv.DictReader(io.StringIO(body.decode('utf-8-sig'))))
    assert status == 200 and len(rows) == 3
    assert rows[0]['dtiam_model_version'] == DTIAM_RELEASE
    assert rows[0]['entity_id'] == 'T2' and float(rows[0]['dtiam_score']) == .8
    assert catalog.rankings('drug', 'D1', 'biomaster')['items'][0]['score'] == 3


def test_missing_dtiam_release_does_not_relabel_frozen_historical_scores(catalog):
    (catalog.root / DTIAM_SCORES).unlink()
    reloaded = ExplorerData(catalog.root)
    reloaded.ensure_loaded()
    result = reloaded.rankings('drug', 'D1', 'dtiam')
    assert result['denominator'] == 0
    assert all(row['score'] is None and row['rank'] is None for row in result['items'])
    assert reloaded.pairs.frozen.notna().all()


def test_dtiam_upgrade_checksum_rejects_changed_artifact(catalog):
    from biomaster.dtiam_release import load_scores
    with (catalog.root / DTIAM_SCORES).open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        load_scores(catalog.root)


def test_dtiam_review_overlay_preserves_other_models_and_matches_exact_identity(catalog):
    from biomaster.dtiam_release import overlay_baseline
    before = pd.DataFrame({'pair_id': ['D1__T2', 'D1__T1', 'D0__T0'],
        'biomaster': [3., 3., 1.], 'drugclip': [.8, .2, .1], 'conplex': [.6, .9, .1], 'dtiam': [.4, .5, .9]})
    after = overlay_baseline(catalog.root, before)
    pd.testing.assert_frame_equal(before[['pair_id', 'biomaster', 'drugclip', 'conplex']],
                                  after[['pair_id', 'biomaster', 'drugclip', 'conplex']])
    assert after.dtiam.iloc[:2].tolist() == [.8, .2]
    assert pd.isna(after.dtiam.iloc[2])
    assert after.dtiam_model_version.eq(DTIAM_RELEASE).all()


def test_relationship_filter_preserves_global_rank_and_export(http_server, catalog):
    known = catalog.rankings("drug", "D1", "drugclip", relationship="known")
    assert known["total"] == 1 and known["denominator"] == 2
    assert known["items"][0]["id"] == "T1" and known["items"][0]["rank"] == 2
    status, _, body = get(http_server, "/api/rankings.csv?kind=drug&id=D1&relationship=unannotated&page_size=1")
    rows = list(csv.DictReader(io.StringIO(body.decode("utf-8-sig"))))
    assert status == 200 and len(rows) == 2
    assert [row["rank"] for row in rows] == ["2.0", "3.0"]
    assert all(row["known_relation"] == "False" and row["denominator"] == "3" for row in rows)
    assert get(http_server, "/api/rankings?kind=drug&id=D1&relationship=invalid")[0] == 400


def test_evidence_full_pagination_and_catalog_limits(catalog):
    catalog.targets["T1"]["target_diseases"] = [{"id": str(i), "name": "disease"} for i in range(205)]
    entity = catalog.entity("target", "T1")
    assert len(entity["target_diseases"]) == 100
    assert entity["target_diseases_total"] == 205
    page = catalog.evidence("target", "T1", "target_diseases", page=3, page_size=100)
    assert page["total"] == 205 and len(page["items"]) == 5
    assert catalog.search("", "all", 2000)["total"] == 5
    with pytest.raises(ValueError):
        catalog.rankings("drug", "D1", page=0)
    with pytest.raises(KeyError):
        catalog.entity("target", "not-real")


def test_structure_allowlist_blocks_paths_and_symlinks(catalog, tmp_path):
    directory = tmp_path / "data/processed/alphafold_receptors_v6"
    directory.mkdir(parents=True)
    allowed = directory / "test.pdb"
    allowed.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n")
    catalog.structure_files[("T1", "")] = allowed
    assert catalog.structure_path("T1") == allowed
    with pytest.raises(KeyError):
        catalog.structure_path("T1", "../../.git/config")
    outside = tmp_path / "secret.pdb"
    outside.write_text("private")
    link = directory / "escape.pdb"
    link.symlink_to(outside)
    catalog.structure_files[("T1", "escape")] = link
    with pytest.raises(KeyError):
        catalog.structure_path("T1", "escape")


@pytest.fixture
def http_server(catalog, tmp_path):
    static = tmp_path / "web/dist"
    static.mkdir(parents=True)
    (static / "index.html").write_text("<h1>BioMaster</h1>")
    (tmp_path / "secret.txt").write_text("SECRET")
    server = ExplorerHTTPServer(("127.0.0.1", 0), catalog, static)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def get(server, path):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    connection.request("GET", path)
    response = connection.getresponse()
    result = response.status, dict(response.getheaders()), response.read()
    connection.close()
    return result


def test_http_endpoints_export_all_rows_and_reject_traversal(http_server):
    status, headers, body = get(http_server, "/api/rankings?kind=drug&id=D1&model=drugclip&page_size=1")
    assert status == 200 and len(json.loads(body)["items"]) == 1
    status, headers, body = get(http_server, "/api/rankings.csv?kind=drug&id=D1&model=drugclip&page_size=1")
    rows = list(csv.DictReader(io.StringIO(body.decode("utf-8-sig"))))
    assert status == 200 and len(rows) == 3
    assert rows[-1]["rank"] == "" and rows[-1]["denominator"] == "2"
    assert "attachment" in headers["Content-Disposition"]
    assert get(http_server, "/api/entity/target/unknown")[0] == 404
    assert get(http_server, "/api/rankings?kind=drug&id=D1&page_size=999")[0] == 400
    status, _, body = get(http_server, "/%2e%2e/%2e%2e/secret.txt")
    assert status == 404 and b"SECRET" not in body
    assert get(http_server, "/api/structure/T1?pocket=../../secret.txt")[0] == 404
    assert get(http_server, "/targets/T1")[0] == 200


def test_empty_checkout_works_without_fabricated_scores(tmp_path):
    data = ExplorerData(tmp_path, infer=False)
    summary = data.summary()
    assert summary["counts"]["pairs"] == 0
    assert all(model["coverage"] == 0 for model in summary["models"])
    assert summary["warnings"]
    assert clean({"missing": np.nan}) == {"missing": None}


def test_compression_cache_validation_and_head(http_server):
    import gzip
    asset = http_server.static_dir / "assets" / "index-abcdefgh.js"
    asset.parent.mkdir()
    original = b"const message = 'BioMaster';\n" * 200
    asset.write_bytes(original)

    def request(method, path, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", http_server.server_port)
        conn.request(method, path, headers=headers or {})
        response = conn.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        conn.close()
        return result

    status, headers, body = request("GET", "/assets/index-abcdefgh.js", {"Accept-Encoding": "gzip"})
    assert status == 200 and gzip.decompress(body) == original
    assert len(body) < len(original) / 2
    assert "immutable" in headers["Cache-Control"]
    assert headers["Vary"] == "Accept-Encoding"
    status, _, body = request("GET", "/assets/index-abcdefgh.js", {"Accept-Encoding": "gzip", "If-None-Match": headers["ETag"]})
    assert status == 304 and not body
    status, head, body = request("HEAD", "/assets/index-abcdefgh.js", {"Accept-Encoding": "gzip"})
    assert status == 200 and not body and head["Content-Length"] == headers["Content-Length"]
    _, identity, body = request("GET", "/assets/index-abcdefgh.js", {"Accept-Encoding": "gzip;q=0, *;q=1"})
    assert "Content-Encoding" not in identity and body == original
    assert request("GET", "/")[1]["Cache-Control"] == "no-cache"
    assert request("GET", "/api/summary")[1]["Cache-Control"] == "no-store"
    assert request("GET", "/assets/missing-abcdefgh.js")[0] == 404


def test_browse_cross_entity_search_filter_and_pagination(catalog):
    from biomaster.explorer_browse import browse
    catalog.drugs['D1']['known_diseases'] = [{'name': 'Shared disease', 'phase': 4}]
    catalog.drugs['D2']['known_diseases'] = [{'name': 'Shared disease', 'phase': 2}, {'name': 'Other', 'phase': 1}]
    result = browse(catalog, 'known_diseases', search='Shared', page_size=1)
    assert result['total'] == 2 and result['all_total'] == 3 and result['matched_entities'] == 2
    second = browse(catalog, 'known_diseases', search='Shared', page=2, page_size=1)
    assert result['items'][0]['entity']['id'] != second['items'][0]['entity']['id']
    assert browse(catalog, 'known_diseases', category='approved')['total'] == 1
    assert browse(catalog, 'known_diseases', category='clinical')['total'] == 2
    assert browse(catalog, 'known_diseases', search='absent')['total'] == 0
    catalog.targets['T1']['structure'] = {'source':'AlphaFold DB'}
    catalog.targets['T1']['pockets'] = [{'type':'experimental'}, {'type':'predicted'}]
    structures = browse(catalog, 'structures', category='experimental')
    row = next(x['record'] for x in structures['items'] if x['entity']['id'] == 'T1')
    assert row['pocket_count'] == 2 and row['predicted_pockets'] == 1
    pairs = browse(catalog, 'pairs_drug')
    assert sum(x['record']['pair_count'] for x in pairs['items']) == 6
    with pytest.raises(ValueError): browse(catalog, 'unknown')
    with pytest.raises(ValueError): browse(catalog, 'pathways', category='approved')
    with pytest.raises(ValueError): browse(catalog, 'structures', page=0)


def test_seven_models_rank_over_full_entity_catalog_and_refresh_live(catalog):
    from biomaster.catalog_models import connect, save
    db=connect(catalog.root,write=True)
    save(db,[dict(model='nesso',drug_id='D1',target_id='T1',score=.1,status='completed'),
             dict(model='nesso',drug_id='D2',target_id='T1',score=.9,status='completed'),
             dict(model='probematch',drug_id='D1',target_id='T1',score=.8,status='completed')])
    result=catalog.rankings('target','T1','nesso')
    assert result['catalog_count']==2 and result['total']==2
    assert result['denominator']==2
    assert [r['id'] for r in result['items']]==['D2','D1']
    assert result['items'][0]['ranks']['nesso']==1
    assert result['items'][0]['coverage_complete']['nesso'] is True
    assert result['items'][0]['scores']['probematch'] is None
    assert len(result['items'][0]['scores'])==7
    # Search/relationship filters keep the original per-target ranks, never SPR ranks.
    filtered=catalog.rankings('target','T1','nesso',search='D1')
    assert filtered['denominator']==2 and filtered['items'][0]['rank']==2
    assert filtered['items'][0]['coverage_complete']['probematch'] is False
    # New scores appear without process restart and cannot change the old model's head.
    before=catalog.rankings('target','T1','biomaster')
    save(db,[dict(model='probematch',drug_id='D2',target_id='T1',score=.95,status='completed')])
    updated=catalog.rankings('target','T1','probematch')
    assert updated['items'][0]['id']=='D2' and updated['denominator']==2
    after=catalog.rankings('target','T1','biomaster')
    assert [(r['id'],r['score'],r['rank']) for r in before['items']]==[(r['id'],r['score'],r['rank']) for r in after['items']]
    assert after['items'][0]['score']==pytest.approx(.9)
    db.execute("INSERT INTO target_status VALUES ('dtbind','T1','unavailable','publisher_sequence_mismatch')");db.commit()
    missing=catalog.rankings('target','T1','dtbind')
    assert missing['total']==2 and missing['denominator']==0
    assert all(r['rank'] is None and r['score'] is None for r in missing['items'])
    assert missing['items'][0]['model_reasons']['dtbind']=='publisher_sequence_mismatch'
    db.close()


def test_seven_model_csv_exports_full_directory_with_missing_rows(http_server,catalog):
    from biomaster.catalog_models import connect,save
    db=connect(catalog.root,write=True)
    save(db,[dict(model='dtbind',drug_id='D2',target_id='T1',score=.8,status='completed')]);db.close()
    status,_,body=get(http_server,'/api/rankings.csv?kind=target&id=T1&model=dtbind&page_size=1')
    rows=list(csv.DictReader(io.StringIO(body.decode('utf-8-sig'))))
    assert status==200 and len(rows)==2
    assert rows[0]['entity_id']=='D2' and rows[1]['rank']==''
    assert 'nesso_rank' in rows[0] and 'probematch_score' in rows[0]
