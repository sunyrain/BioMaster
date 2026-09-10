"""Check cross-cutoff labels against read-only raw ChEMBL37 activities.

Run after reproduce_audit.py. Only in-memory SQLite TEMP tables are created.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from extract_chembl37_target_calibration_v5 import classify_pairs  # noqa: E402


def main():
    scope = pd.read_csv(OUT / "temporal_cross_cutoff_relations.csv")
    scope = scope[scope.source_kind.eq("chembl37_comprehensive")].drop_duplicates(
        ["target_chembl_id", "parent_molregno"]
    )
    db = ROOT / "downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db"
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    connection.execute("CREATE TEMP TABLE requested (target_id TEXT, parent INTEGER, PRIMARY KEY(target_id,parent))")
    connection.executemany("INSERT INTO requested VALUES (?,?)", [
        (r.target_chembl_id, int(r.parent_molregno)) for r in scope.itertuples()
    ])
    connection.execute("""
        CREATE TEMP TABLE molecule_targets AS
        SELECT r.target_id, r.parent, mh.molregno, td.tid
        FROM requested r JOIN molecule_hierarchy mh ON mh.parent_molregno=r.parent
        JOIN target_dictionary td ON td.chembl_id=r.target_id
        UNION
        SELECT r.target_id, r.parent, r.parent AS molregno, td.tid
        FROM requested r JOIN target_dictionary td ON td.chembl_id=r.target_id
    """)
    config = yaml.safe_load((ROOT / "configs/calibrated_pipeline_v5.yaml").read_text())
    text = "LOWER(COALESCE(a.activity_comment,'') || ' ' || COALESCE(a.standard_text_value,'') || ' ' || COALESCE(a.text_value,''))"
    terms = " OR ".join(f"{text} LIKE '%{term}%'" for term in config["chembl_calibration"]["explicit_inactive_patterns"])
    query = f"""
        SELECT r.target_id AS target_chembl_id, r.parent AS parent_molregno,
               a.activity_id, a.pchembl_value, d.year AS document_year,
               CASE WHEN ({terms}) THEN 1 ELSE 0 END AS explicit_inactive
        FROM molecule_targets r
        CROSS JOIN activities a INDEXED BY fk_act_molregno ON a.molregno=r.molregno
        JOIN assays ass ON ass.assay_id=a.assay_id AND ass.tid=r.tid
        LEFT JOIN molecule_hierarchy mh ON mh.molregno=a.molregno
        LEFT JOIN docs d ON d.doc_id=COALESCE(a.doc_id,ass.doc_id)
        WHERE COALESCE(mh.parent_molregno,a.molregno)=r.parent
          AND ass.assay_type='B' AND ass.confidence_score>=9
          AND COALESCE(a.potential_duplicate,0)=0
          AND COALESCE(a.data_validity_comment,'') IN ('','Manually validated')
          AND ((a.pchembl_value IS NOT NULL AND a.standard_type IN ('Ki','Kd','IC50')
                AND a.standard_relation='=') OR ({terms}))
    """
    raw = pd.read_sql_query(query, connection)
    raw.pchembl_value = pd.to_numeric(raw.pchembl_value, errors="coerce")
    raw.document_year = pd.to_numeric(raw.document_year, errors="coerce")
    def aggregate(frame):
        agg = frame.groupby(["target_chembl_id", "parent_molregno"]).agg(
            mean_pchembl=("pchembl_value", "mean"), min_pchembl=("pchembl_value", "min"),
            max_pchembl=("pchembl_value", "max"), any_explicit_inactive=("explicit_inactive", "max"),
            rows=("activity_id", "size"),
        ).reset_index()
        return classify_pairs(agg, config)
    all_time = aggregate(raw)
    before = aggregate(raw[raw.document_year.le(2022)])
    aligned = scope.merge(all_time, on=["target_chembl_id", "parent_molregno"], suffixes=("_stored", "_reaggregated"), validate="one_to_one")
    assert len(aligned) == len(scope)
    assert aligned.calibration_label_stored.eq(aligned.calibration_label_reaggregated).all()
    assert np.allclose(aligned.mean_pchembl_stored, aligned.mean_pchembl_reaggregated, equal_nan=True)
    comparison = aligned.merge(before, on=["target_chembl_id", "parent_molregno"], validate="one_to_one")
    assert len(comparison) == len(scope)
    comparison["label_changed_by_future_evidence"] = comparison.calibration_label.ne(comparison.calibration_label_stored)
    comparison["mean_changed_by_future_evidence"] = ~np.isclose(comparison.mean_pchembl, comparison.mean_pchembl_stored, equal_nan=True)
    comparison.to_csv(OUT / "temporal_raw_reaggregation.csv", index=False)
    summary = {
        "database": str(db.relative_to(ROOT)), "eligible_raw_activity_rows": len(raw),
        "actual_training_source_relations_audited": len(comparison),
        "all_time_reaggregation_matches_stored_labels_and_means": True,
        "label_changed_by_future_evidence": int(comparison.label_changed_by_future_evidence.sum()),
        "mean_changed_by_future_evidence": int(comparison.mean_changed_by_future_evidence.sum()),
        "label_transitions_pre2023_to_stored": [
            {"pre2023": str(a), "stored": str(b), "rows": int(n)}
            for (a,b),n in comparison.groupby(["calibration_label", "calibration_label_stored"]).size().items()
        ],
        "scope_limit": "Only cross-cutoff relations retained in actual training are audited. Future-driven exclusions and target context are additional unaudited paths.",
    }
    (OUT / "TEMPORAL_RAW_AUDIT.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
