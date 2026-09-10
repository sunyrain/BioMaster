"""Future measurements must not alter earlier labels or first-seen eligibility."""
import numpy as np
import pandas as pd

from biomaster.temporal_relations import aggregate_window, first_observations, new_relations


def annual(records):
    rows=[]
    for drug,year,value in records:
        rows.append(dict(drug_feature_index=drug,target_feature_index=0,document_year=year,
                         activity_rows=1,numeric_rows=1,pchembl_sum=value,min_pchembl=value,
                         max_pchembl=value,any_explicit_inactive=0))
    return pd.DataFrame(rows)


def test_later_positive_cannot_relabel_earlier_negative():
    data=annual([(0,2020,4.),(0,2024,8.)])
    assert aggregate_window(data,end=2022).binary_label.tolist()==[0]
    assert aggregate_window(data,start=2023,end=2025).binary_label.tolist()==[1]
    assert aggregate_window(data).conflicting.tolist()==[True]
    assert new_relations(aggregate_window(data,start=2023),first_observations(data),2022).empty


def test_grey_earlier_measurement_blocks_novel_relation():
    data=annual([(0,2020,5.5),(0,2023,7.),(1,2023,7.)])
    assert aggregate_window(data,end=2022).binary_label.isna().all()
    result=new_relations(aggregate_window(data,start=2023),first_observations(data),2022)
    assert result.drug_feature_index.tolist()==[1]


def test_undated_record_excludes_first_seen_claim_but_not_period_label():
    data=annual([(0,np.nan,4.),(0,2024,7.),(1,2025,8.)])
    period=aggregate_window(data,start=2023,end=2025)
    assert period.binary_label.tolist()==[1,1]
    assert new_relations(period,first_observations(data),2022).drug_feature_index.tolist()==[1]


def test_multiple_measurement_counts_are_weighted_and_cutoff_inclusive():
    data=annual([(0,2022,5.8),(0,2023,7.)])
    data.loc[0,['numeric_rows','pchembl_sum']]=[9,52.2]
    assert np.isclose(aggregate_window(data).mean_pchembl.iloc[0],5.92)
    assert aggregate_window(data,end=2022).max_document_year.iloc[0]==2022
    assert aggregate_window(data,start=2023,end=2025).min_document_year.iloc[0]==2023


def test_aliases_collapse_before_first_observation_and_labeling():
    data=annual([(0,2020,7.),(1,2024,8.)])
    data.drug_feature_index=data.drug_feature_index.map({0:0,1:0})
    assert new_relations(aggregate_window(data,start=2023),first_observations(data),2022).empty
