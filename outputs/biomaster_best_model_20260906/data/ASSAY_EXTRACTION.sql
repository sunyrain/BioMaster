SELECT t.target_index AS target_feature_index,
          COALESCE(mh.parent_molregno,a.molregno) AS parent_molregno,
          ass.assay_id, a.standard_type, a.standard_units, d.year AS document_year,
          COUNT(*) AS numeric_rows, SUM(a.pchembl_value) AS pchembl_sum,
          MIN(a.pchembl_value) AS min_pchembl, MAX(a.pchembl_value) AS max_pchembl
            FROM target_scope t JOIN assays ass ON ass.tid=t.tid JOIN activities a ON a.assay_id=ass.assay_id
    LEFT JOIN molecule_hierarchy mh ON mh.molregno=a.molregno
    LEFT JOIN docs d ON d.doc_id=COALESCE(a.doc_id,ass.doc_id)
    WHERE ass.assay_type='B' AND ass.confidence_score>=9 AND ass.relationship_type='D' AND ass.variant_id IS NULL
      AND COALESCE(a.potential_duplicate,0)=0 AND COALESCE(a.data_validity_comment,'') IN ('','Manually validated')
      AND ((a.pchembl_value IS NOT NULL AND a.standard_type IN ('Ki','Kd','IC50') AND a.standard_relation='=')
           OR LOWER(COALESCE(a.activity_comment,'')||' '||COALESCE(a.standard_text_value,'')||' '||COALESCE(a.text_value,'')) LIKE '%inactive%'
           OR LOWER(COALESCE(a.activity_comment,'')||' '||COALESCE(a.standard_text_value,'')||' '||COALESCE(a.text_value,'')) LIKE '%not active%'
           OR LOWER(COALESCE(a.activity_comment,'')||' '||COALESCE(a.standard_text_value,'')||' '||COALESCE(a.text_value,'')) LIKE '%no activity%')

          AND d.year IS NOT NULL AND d.year<=2022
          AND a.standard_type IN ('Ki','Kd','IC50') AND a.standard_relation='='
          AND a.pchembl_value IS NOT NULL AND a.standard_units IS NOT NULL
        GROUP BY t.target_index, COALESCE(mh.parent_molregno,a.molregno),
          ass.assay_id,a.standard_type,a.standard_units,d.year