SELECT a.activity_id,a.standard_type,a.standard_relation,a.standard_value,a.standard_units,a.activity_comment,a.standard_text_value,a.text_value,a.potential_duplicate,a.data_validity_comment,
        ass.assay_type,ass.confidence_score,ass.relationship_type,ass.variant_id,ass.chembl_id assay_chembl_id,ass.description,
        t.chembl_id target_chembl_id,t.pref_name target_name,cs.standard_inchi_key,cs.canonical_smiles,
        d.year,d.doi,d.pubmed_id,d.patent_id
        FROM target_dictionary t JOIN assays ass ON ass.tid=t.tid JOIN activities a ON a.assay_id=ass.assay_id
        LEFT JOIN molecule_hierarchy mh ON mh.molregno=a.molregno
        LEFT JOIN compound_structures cs ON cs.molregno=COALESCE(mh.parent_molregno,a.molregno)
        LEFT JOIN docs d ON d.doc_id=COALESCE(a.doc_id,ass.doc_id)
        WHERE t.organism='Homo sapiens' AND t.target_type='SINGLE PROTEIN'
        AND (a.standard_type IN ('Kd','Ki','IC50','EC50') OR LOWER(COALESCE(a.activity_comment,'')) IN ('inactive','not active','no activity'))