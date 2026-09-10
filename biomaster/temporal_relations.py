"""Date-before-label aggregation for retrospective ChEMBL relation prediction."""
import numpy as np
import pandas as pd


def aggregate_window(annual, start=None, end=None):
    """Only records inside the requested period can affect its binary label."""
    valid=annual.document_year.notna()
    if start is not None:valid &= annual.document_year.ge(start)
    if end is not None:valid &= annual.document_year.le(end)
    part=annual.loc[valid]
    result=part.groupby(['drug_feature_index','target_feature_index'],sort=True).agg(
        activity_rows=('activity_rows','sum'),numeric_rows=('numeric_rows','sum'),
        pchembl_sum=('pchembl_sum','sum'),min_pchembl=('min_pchembl','min'),max_pchembl=('max_pchembl','max'),
        any_explicit_inactive=('any_explicit_inactive','max'),
        min_document_year=('document_year','min'),max_document_year=('document_year','max')).reset_index()
    result['mean_pchembl']=result.pchembl_sum/result.numeric_rows.replace(0,np.nan)
    conflict=(result.min_pchembl.le(5)&result.max_pchembl.ge(6))|(result.any_explicit_inactive.eq(1)&result.max_pchembl.ge(6))
    result['binary_label']=np.nan
    result.loc[result.mean_pchembl.ge(6)&~conflict,'binary_label']=1
    result.loc[(result.mean_pchembl.le(5)|result.any_explicit_inactive.eq(1))&~conflict,'binary_label']=0
    result['conflicting']=conflict
    return result


def first_observations(annual):
    frame=annual.assign(undated=annual.document_year.isna())
    return frame.groupby(['drug_feature_index','target_feature_index'],sort=True).agg(
        first_document_year=('document_year','min'),has_undated=('undated','max')).reset_index()


def new_relations(period, first, cutoff):
    result=period.merge(first,on=['drug_feature_index','target_feature_index'],validate='one_to_one')
    return result.loc[result.first_document_year.gt(cutoff)&~result.has_undated].copy()
