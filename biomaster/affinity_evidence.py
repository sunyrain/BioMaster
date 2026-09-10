"""Read-only access to versioned experimental affinity evidence.

KD/Ki, potency, apparent KD and contextual signals remain separate. A record
is evidence of a measurement, not automatically evidence of positive binding.
"""
from pathlib import Path
import json
import sqlite3
from contextlib import closing

DEFAULT_DB = Path(__file__).resolve().parents[1] / 'data/processed/biomaster_affinity_evidence_20260910.sqlite'

def get_pair_evidence(ligand_inchikey: str, target_chembl_id: str, *, database=DEFAULT_DB, limit=200):
    """Return full-key, single-accession records and their original quality flags.

    Construct-unverified, mutant and lysate records are retained with scope;
    multi-accession complex membership is deliberately not inferred as binding.
    """
    limit = max(1, min(int(limit), 2000))
    db = Path(database).resolve()
    with closing(sqlite3.connect(db.as_uri() + '?mode=ro', uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute('''select m.*, t.target_chembl_id, t.gene_symbol
            from measurements m join project_targets t
            on t.uniprot_accession=m.target_accessions
            where m.ligand_key=? and t.target_chembl_id=?
            order by case m.endpoint when 'Kd' then 0 when 'Ki' then 1 else 2 end,
            m.source,m.id limit ?''', (ligand_inchikey, target_chembl_id, limit)).fetchall()
        result = []
        for row in rows:
            entry = dict(row)
            entry['metadata'] = json.loads(entry.pop('metadata_json'))
            entry['assays'] = []
            entry['quality_review'] = []
            if conn.execute("select 1 from sqlite_master where name='evidence_quality_reviews'").fetchone():
                entry['quality_review'] = [dict(x) for x in conn.execute(
                    'select * from evidence_quality_reviews where source=? and record_id=?',
                    (entry['source'], entry['record_id']))]
            if entry['source'] == 'BindingDB_202609':
                assays = conn.execute('''select a.* from bindingdb_assay_links l
                    join bindingdb_assays a using(entry_assay_id)
                    where l.reactant_set_id=?''', (entry['record_id'].split(':')[0],)).fetchall()
                entry['assays'] = [dict(x) for x in assays]
            result.append(entry)
        return result

def evidence_inventory(*, database=DEFAULT_DB):
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)) as conn:
        return [dict(source=s, endpoint=e, source_rows=n)
                for s,e,n in conn.execute('select source,endpoint,count(*) from measurements group by source,endpoint')]
