
    SELECT t.target_index, COALESCE(mh.parent_molregno,a.molregno) AS parent_molregno,
           d.year AS document_year, COUNT(*) AS activity_rows,
           COUNT(DISTINCT COALESCE(a.doc_id,ass.doc_id)) AS document_count,
           COUNT(CASE WHEN a.standard_type IN ('Ki','Kd','IC50') AND a.standard_relation='=' THEN a.pchembl_value END) AS numeric_rows,
           SUM(CASE WHEN a.standard_type IN ('Ki','Kd','IC50') AND a.standard_relation='=' THEN a.pchembl_value END) AS pchembl_sum,
           MIN(CASE WHEN a.standard_type IN ('Ki','Kd','IC50') AND a.standard_relation='=' THEN a.pchembl_value END) AS min_pchembl,
           MAX(CASE WHEN a.standard_type IN ('Ki','Kd','IC50') AND a.standard_relation='=' THEN a.pchembl_value END) AS max_pchembl,
           MAX(CASE WHEN LOWER(COALESCE(a.activity_comment,'')||' '||COALESCE(a.standard_text_value,'')||' '||COALESCE(a.text_value,'')) LIKE '%inactive%'
                         OR LOWER(COALESCE(a.activity_comment,'')||' '||COALESCE(a.standard_text_value,'')||' '||COALESCE(a.text_value,'')) LIKE '%not active%'
                         OR LOWER(COALESCE(a.activity_comment,'')||' '||COALESCE(a.standard_text_value,'')||' '||COALESCE(a.text_value,'')) LIKE '%no activity%'
                    THEN 1 ELSE 0 END) AS any_explicit_inactive
    FROM target_scope t JOIN assays ass ON ass.tid=t.tid JOIN activities a ON a.assay_id=ass.assay_id
    LEFT JOIN molecule_hierarchy mh ON mh.molregno=a.molregno
    LEFT JOIN docs d ON d.doc_id=COALESCE(a.doc_id,ass.doc_id)
    WHERE ass.assay_type='B' AND ass.confidence_score>=9 AND ass.relationship_type='D' AND ass.variant_id IS NULL
      AND COALESCE(a.potential_duplicate,0)=0 AND COALESCE(a.data_validity_comment,'') IN ('','Manually validated')
      AND ((a.pchembl_value IS NOT NULL AND a.standard_type IN ('Ki','Kd','IC50') AND a.standard_relation='=')
           OR LOWER(COALESCE(a.activity_comment,'')||' '||COALESCE(a.standard_text_value,'')||' '||COALESCE(a.text_value,'')) LIKE '%inactive%'
           OR LOWER(COALESCE(a.activity_comment,'')||' '||COALESCE(a.standard_text_value,'')||' '||COALESCE(a.text_value,'')) LIKE '%not active%'
           OR LOWER(COALESCE(a.activity_comment,'')||' '||COALESCE(a.standard_text_value,'')||' '||COALESCE(a.text_value,'')) LIKE '%no activity%')
    GROUP BY t.target_index,COALESCE(mh.parent_molregno,a.molregno),d.year
    