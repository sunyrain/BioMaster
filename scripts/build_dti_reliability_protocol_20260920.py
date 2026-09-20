#!/usr/bin/env python3
"""Create the research run registry and prospective analysis contracts, without fitting."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'outputs/dti_research_preparation_20260920'
CONFIG=ROOT/'configs/dti_reliability_20260920'
DATA='data/research/dti_reliability_20260920_v1'
SEEDS=[101,202,303]


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def build():
    CONFIG.mkdir(parents=True,exist_ok=True)
    data_summary=json.loads((REPORT/'DATA_PREPARATION_SUMMARY.json').read_text())
    asset_summary=json.loads((REPORT/'ASSET_SUMMARY.json').read_text())
    assert data_summary['status']=='PREPARED_AND_VALIDATED'
    assert asset_summary['missing_asset_entries']==0
    architectures={
        'drug_only':dict(inputs=['bermol'],formula='g(d)',control='No target information; constant within a drug-to-target query.'),
        'protein_only':dict(inputs=['esm2_dtiam'],formula='h(p)',control='No drug information; constant within a target-to-drug query.'),
        'additive':dict(inputs=['bermol','esm2_dtiam'],formula='g(d)+h(p)+b',control='No pair-dependent interaction term.'),
        'dual_cosine':dict(inputs=['bermol','esm2_dtiam'],formula='a*cos(phi(d),psi(p))+b',activation='GELU',learnable_positive_scale=True),
        'bilinear':dict(inputs=['bermol','esm2_dtiam'],formula='sum(u(d)*v(p))+g(d)+h(p)',rank=64),
        'global_mlp':dict(inputs=['bermol','esm2_dtiam'],formula='MLP([u,v,u*v,abs(u-v)])',pair_width=256,pair_layers=2),
        'lightgbm':dict(inputs=['bermol','esm2_dtiam'],formula='GBDT(concat(d,p))',num_leaves=63,learning_rate=.05,
                        max_estimators=3000,min_data_in_leaf=50,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,
                        early_stopping_rounds=150,early_stopping_metric='validation_binary_logloss',
                        reporting_checkpoint='Best bidirectional query-macro AP among saved 25-tree checkpoints; test never selects.')}
    protocol=dict(
        protocol_id='dti_reliability_20260920_v1',status='PREPARED_FROZEN_CONFIGURATION_NOT_EXTERNALLY_PREREGISTERED',
        created_utc=datetime.now(timezone.utc).isoformat(),objective='Separate shared signal, correlated error, endpoint scope and pair selectivity in DTI decisions.',
        evidence_status=dict(existing_computational_tests='REPEATEDLY_VIEWED_EXPLORATORY',
            new_repartitions='RETROSPECTIVE_FRESH_SUPERVISED_FITS_REQUIRED',
            public_pretraining_overlap='UNKNOWN_UNLESS_EXPLICITLY_AUDITED',wetlab_384='UNKNOWN_PENDING_USER',
            new_external_confirmatory_panel='NOT_YET_QUALIFIED'),
        frozen_artifacts={str(p.relative_to(ROOT)):sha(p) for p in [REPORT/'DATA_MANIFEST.json',REPORT/'FEATURE_REGISTRY.json',REPORT/'WEIGHT_REGISTRY.csv']},
        seeds=SEEDS,split_seed=20260920,architectures=architectures,
        inputs=dict(primary=['bermol','esm2_dtiam'],normalize='Per-feature training-only mean/std; persist and reuse on validation/test.',
            esm2_window='Original DTIAM-compatible first 1022 residues with its documented pooling, unchanged across controls.',
            optional_drugclip='14,533 required drug embeddings unavailable; mask explicitly or use a separately declared common-coverage comparison.',
            structure_increment='Token/atom/structure models require an additional input-control experiment; pooled features are not local geometry.'),
        labels=dict(positive_nM_max=1000,negative_nM_min=10000,grey_interval='1000 < value_nM < 10000 excluded from binary labels',
            inequalities='Inherit audited bounds; a censored interval that crosses a boundary does not establish that label.',
            explicit_inactive='Audited experimental inactivity retained as binary negative; never fabricated numeric Kd.',
            unmeasured='UNKNOWN, never negative',arm_A='Kd/Ki binary evidence plus explicit inactivity',
            arm_B='Kd/Ki/IC50/EC50 binary evidence plus explicit inactivity; endpoint pooling is the tested hypothesis.',
            regression='Separate exact Kd, Ki, IC50, EC50 heads and metrics; no pooled physical affinity claim.'),
        neural_training=dict(initialization='Fresh trainable layers, frozen common encoders',projection_width=256,
            projection='Linear -> LayerNorm -> GELU, identical corresponding towers',dropout=.1,
            optimizer='AdamW',learning_rate=.0003,weight_decay=.0001,batch_size=1024,gradient_clip_norm=5,
            sampling='Uniform observed pairs with no permanent per-target cap and no invented decoys.',
            validation_every='One complete epoch',minimum_epochs=10,maximum_epochs=100,patience=12,min_delta=.0001,
            checkpoint='Max validation bidirectional query-macro AP; missing direction is not imputed.',
            time_cap_hours_per_run=8,time_cap_status='BUDGET_CENSORED_NOT_CONVERGED',
            convergence='No validation improvement for patience after minimum epochs; never assert mathematical convergence.',
            mixed_precision='Disabled for initial parity pilot; optional later amendment must be logged.',
            train_repeats='Three prespecified seeds; retain all outcomes including failure.',
            hyperparameters='Initial controlled settings fixed here; further tuning is a separately logged sensitivity study, never test-selected.'),
        losses=dict(binary='BCEWithLogits on eligible observed binary labels, unweighted in primary architecture study.',
            ranking='Equal query weighting, half target-to-drug and half drug-to-target where measured positive/negative comparisons exist; softplus(-(s_pos-s_neg)).',
            ranking_context='Binary label-order regularization across queried measurements; not claimed to be within-assay continuous affinity ordering.',
            ranking_weight=.1,regression='Huber loss per exact endpoint; average within endpoint then over available endpoints.',
            regression_scaling='p-activity centered/scaled using training rows for that endpoint only; save scales.',regression_weight=.1,
            total='L_binary + 0.1*L_rank (if enabled) + 0.1*L_reg (if enabled)',
            unlabelled_auxiliary='Unavailable endpoint/loss masked, not zero-filled as a label.',
            assay_sensitivity='Optional within-assay ranking only for same target/endpoint/source, >=5 distinct training members after filtering, delta p-activity >=0.3.',
            assay_scope='Existing 639,511 auxiliary rows are original-training-only; overlaps and unavailable held-out measurements reported explicitly.'),
        metrics=dict(primary=['target_query_macro_AP_on_mixed_queries','drug_query_macro_AP_on_mixed_queries'],
            checkpoint_summary='Arithmetic mean of the two directions; both direction scores and query counts always reported.',
            secondary=['pooled_AP','AUROC','explicit_inactive_FPR','exact_endpoint_Spearman','exact_endpoint_RMSE',
                'risk_coverage','coverage_failure_rate','Spearman_query','Kendall_tau_b','TopK_overlap','shared_errors'],
            topk=dict(primary_k=10,sensitivity_k=[5,20,50,100],minimum_query_size=10,
                labels='Measured-panel P@10 only when all compared predictions in that panel have usable measurements. Never call this full-catalog P@10.',
                ties='Exact expected boundary membership for overlap and P@K; average ranks for correlations.',
                constant='Undefined correlation reported with count; constant-score ranking AP equals that query prevalence.'),
            random_baseline='AUROC=0.5; independent TopK overlap=K^2/N; uninformative constant-score AP=prevalence. Finite random-order AP reference computed by permutations.',
            inactive_fpr='Choose score threshold on validation positives at >=90% recall, then freeze; report achieved test recall and FPR on explicit-inactive subset.',
            regression_comparison='Same endpoint and comparable assay; report label spread and censoring separately.',
            single_class='Report counts and applicable false-positive or recall summaries; exclude from mixed-query AP with explicit denominator.',
            classification_vs_affinity='Do not interpret similarity/logit as Kd or calibrated probability.'),
        inference_track=dict(model_families=['ReTargetMap','DTIAM_A','DrugCLIP','ConPLex','ProbeMatchDTI','DTBind_occurrence','Nesso_binder'],
            source_snapshot='outputs/model_interpretation_20260920/latest_results',
            scope='Frozen complete 50 x 719 rectangle for exploration; growing full 720 x 384 matrix needs a new immutable snapshot.',
            heads='Nesso binder and pIC50=6-log10(IC50/uM) analyzed separately; DTBind occurrence/affinity/site are separate tasks.',
            nesso_gates='0.5 and per-query Top50 are already viewed exploratory diagnostics; no retrospectively invented primary threshold.',
            missing='Retain missing/unsupported/error distinctions and report coverage; no zero imputation.',
            extra_models='EviDTI and Scope only after native example replay, feature parity and applicable target-universe checks.'),
        statistical_plan=dict(unit='Query-level paired contrasts; targets grouped by homology, drugs by scaffold; documents audited separately.',
            bootstrap_replicates=2000,confidence_level=.95,
            dependence='For pooled or cross-query metrics use crossed scaffold x target-cluster resampling; add document-cluster sensitivity and disclose independent units.',
            seeds='Three seed-level values plus mean/range; pair bootstrap does not convert three fits into thousands of independent fits.',
            primary_comparisons=['global_mlp vs additive','global_mlp vs dual_cosine','global_mlp vs lightgbm'],
            primary_family='3 comparisons x 2 directions x 2 primary splits = 12 tests; Holm correction within this family.',
            secondary='Benjamini-Hochberg FDR for prespecified secondary families; post-hoc diagnostics labeled exploratory.',
            effect_size='Report paired absolute AP difference and uncertainty, not significance alone.',
            causal_scope='Only controlled fresh training supports attribution to the changed factor; public deployed architecture comparisons are observational.'),
        selection_protocol=dict(keep_tests_sealed_for_new_fits='No early stopping, weights, gates or feature normalization from test.',
            interpretation='Historical label access is disclosed despite prospective run rules.',
            equal_compute='Save common update-step checkpoints and unique-pair exposure; compare both convergence-selected and equal-update budgets.',
            data_contrasts='Matched target x class x explicit-inactive totals plus nested common-pair/additional-B members; chemical diversity and source mix still differ.',
            no_150_deletion='Reuse existing rotating-150/balanced results as exploratory. Any new sampler comparison rotates examples and logs exposure, never silently discards all later records.',
            fusion='Weights selected on validation. Compare best single, training-only chemical nearest neighbor, rank-mean fusion, and disagreement risk at equal retained budget.'),
        execution=dict(new_training_started=False,training_runner_status='NEW_CONTROLLED_TRAINER_NOT_IMPLEMENTED; data loader and preflight ready.',
            stages=['P0 data/assets/protocol','P1 scaffold controls and existing-result reuse','P2 cold controls and targeted interventions','P3 source/double-cold/time sensitivity','P4 independent measured panel and wetlab analyses'],
            resources='Existing Nesso/DTBind production jobs retained; no new GPU allocation by this preparation.',
            scheduling='One GPU job at a time after resource availability; CPU-heavy tasks <=4 threads during current inference.',
            disk_floor_GiB=15,new_training_eta='Not measured; profile 200 train steps plus one full validation per architecture before scheduling.',
            amendment='Append versioned deviations with reason and whether validation/test outcomes were known; never rewrite a completed protocol version silently.'),
        release=dict(weights='Publish retrieval manifests and hashes; redistribute bytes only when the specific weight terms allow it.',
            raw_data='Retain original source/assay/identity references and license metadata; publish derived members or rebuild scripts as permitted.',
            heldout='No test labels or SPR results exposed to training loader by default.',external_registration='Not performed.'))
    save(CONFIG/'PROTOCOL.json',protocol)
    rows=[]
    def add(stage,arm,view,arch,loss,seed,train_members=None,eval_prefix=None,variant='default'):
        key=f'{stage}_{arm}_{view}_{arch}_{loss}_{variant}_s{seed}'
        rows.append(dict(run_id=key,stage=stage,track='controlled_fresh_training',arm=arm,view=view,architecture=arch,
            loss=loss,variant=variant,seed=seed,feature_set='bermol+esm2_dtiam',
            train_members=train_members or f'{DATA}/{arm}_{view}_train.parquet',
            validation_members=f'{DATA}/{eval_prefix}_VALIDATION.parquet' if eval_prefix else f'{DATA}/{arm}_{view}_validation.parquet',
            test_members=f'{DATA}/{eval_prefix}_TEST.parquet' if eval_prefix else f'{DATA}/{arm}_{view}_test.parquet',
            status='PLANNED_NOT_STARTED',requires='Controlled trainer + data preflight + resource profile',
            output_dir=f'outputs/dti_reliability_runs_20260920/{key}',test_use='Final report only, no model selection'))
    for view in ['scaffold_replay','cold_target']:
        for arch in architectures:
            for seed in SEEDS:add('E1','A',view,arch,'binary',seed)
    for arm in ['A','B']:
        for loss in (['binary_rank','binary_regression','binary_both'] if arm=='A' else ['binary','binary_rank','binary_regression','binary_both']):
            for seed in SEEDS:add('E2',arm,'scaffold_replay','global_mlp',loss,seed)
    for name,members in [('matched_A','A_MATCHED_TRAIN'),('matched_B','B_MATCHED_TRAIN'),
                         ('common','A_B_COMMON_TRAIN'),('common_plus_B','A_B_COMMON_TRAIN+B_EXTRA_SHARED_TARGETS_TRAIN')]:
        paths=';'.join(f'{DATA}/{p}.parquet' for p in members.split('+'))
        for seed in SEEDS:add('E3','AB','matched_composition','global_mlp','binary',seed,paths,'A_B_COMMON',name)
    for view in ['source_purged','double_cold','temporal_conservative']:
        for seed in SEEDS:add('E4','A',view,'global_mlp','binary',seed)
    for seed in SEEDS:add('E5','A','scaffold_replay','dual_cosine','binary',seed,variant='relu_projection')
    runs=pd.DataFrame(rows);assert runs.run_id.is_unique and len(runs)==87
    runs.to_csv(REPORT/'EXPERIMENT_MATRIX.csv',index=False)
    legacy=pd.read_csv(ROOT/'outputs/biomaster_model_consolidation_20260911/MODEL_LEDGER.csv')
    dtiam=pd.read_csv(ROOT/'outputs/biomaster_dtiam_ab_20260912/MODEL_LEDGER.csv')
    existing=pd.concat([legacy.assign(registry_source='Palinova consolidation'),dtiam.assign(registry_source='DTIAM A/B')],ignore_index=True)
    existing['evidence_status']='EXPLORATORY_TEST_RESULTS_PREVIOUSLY_VIEWED'
    existing.to_csv(REPORT/'EXISTING_EXPERIMENTS.csv',index=False)
    pd.DataFrame([
        dict(dataset='Historical endpoint/scaffold tests',status='VIEWED_REPEATEDLY',allowed_use='Exploration and fresh-fit retrospective comparison'),
        dict(dataset='Existing 378-pair / 141-pair panels',status='VIEWED_REPEATEDLY',allowed_use='Exploratory diagnostics only'),
        dict(dataset='New cold/source/time members',status='RETROSPECTIVE_REPARTITION',allowed_use='Fresh supervised fits; public pretraining overlap unknown'),
        dict(dataset='SPR384',status='UNKNOWN_PENDING_USER',allowed_use='Preserve all measurements; no claim of unblinded independence'),
        dict(dataset='New external measured panel',status='NOT_YET_QUALIFIED',allowed_use='Needs provenance, coverage, label and exposure audit')
    ]).to_csv(REPORT/'DATA_ACCESS_LOG.csv',index=False)
    prediction_schema={'$schema':'https://json-schema.org/draft/2020-12/schema','title':'DTI research prediction row','type':'object',
        'required':['run_id','model_version','pair_id','query_direction','score','higher_is_better','score_semantics','status','input_manifest_sha256'],
        'properties':{'run_id':{'type':'string'},'model_version':{'type':'string'},'pair_id':{'type':'string'},
            'query_direction':{'enum':['target_to_drug','drug_to_target']},'score':{'type':['number','null']},
            'higher_is_better':{'type':'boolean'},'score_semantics':{'enum':['classification_logit','classifier_probability','retrieval_similarity','directional_logit','pKd','pKi','pIC50','pEC50','log10_IC50_uM']},
            'status':{'enum':['OK','UNSUPPORTED_INPUT','MISSING_FEATURE','INFERENCE_FAILED','UNRELIABLE_OUTPUT']},
            'input_manifest_sha256':{'type':'string','pattern':'^[0-9a-f]{64}$'},'reason':{'type':'string'}},
        'allOf':[{'if':{'properties':{'status':{'const':'OK'}}},'then':{'properties':{'score':{'type':'number'}}}},
                 {'if':{'properties':{'status':{'enum':['UNSUPPORTED_INPUT','MISSING_FEATURE','INFERENCE_FAILED']}}},'then':{'properties':{'score':{'type':'null'}},'required':['reason']}}]}
    save(CONFIG/'PREDICTION_ROW.schema.json',prediction_schema)
    columns=['candidate_id','drug_name','drug_parent_inchikey','drug_lot','target_name','target_sequence_sha256','construct_id',
        'assay_run_id','plate_position','reference_compound','reference_qc_status','protein_activity_qc','instrument',
        'concentration_nM','replicate_id','solubility_flag','aggregation_flag','nonspecific_binding_flag','curve_quality',
        'measurement_type','measurement_relation','measurement_value','measurement_unit','detection_limit_nM',
        'result_status','raw_sensorgram_path','analysis_version','measured_at_utc','first_result_access_utc','notes']
    pd.DataFrame(columns=columns).to_csv(REPORT/'WETLAB_RESULTS_TEMPLATE.csv',index=False)
    save(CONFIG/'WETLAB_CONTRACT.json',dict(result_status=['QUANTIFIABLE','MEASURED_NO_BINDING_WITHIN_LIMIT','INDETERMINATE','TECHNICAL_FAILURE','NOT_MEASURED'],
        numeric_binding_endpoint='Kd in nM only after acceptable reference/blank/construct/curve checks; preserve relation and censoring.',
        failure_policy='TECHNICAL_FAILURE, INDETERMINATE and NOT_MEASURED never become binary negatives.',
        threshold_policy='Lab detection and binding-decision thresholds must be specified from assay conditions before outcome-based selection.',
        selection_scope='The frozen SPR384 selection pool; controls separate, raw failures retained.',
        result_access='UNKNOWN_PENDING_USER',candidate_changes=False))
    save(CONFIG/'AMENDMENTS.json',[])
    save(REPORT/'PREPARATION_STATUS.json',dict(status='PREPARATION_COMPLETE_TRAINING_NOT_STARTED',
        planned_new_controlled_fits=len(runs),planned_by_stage=runs.groupby('stage').size().to_dict(),
        registered_existing_fit_directories=existing.checkpoint.nunique(),existing_result_rows=len(existing),
        materialized_data_views=data_summary['materialized_member_files'],asset_entries=asset_summary['registered_asset_entries'],
        blockers_for_confirmatory_paper=['Unseen external panel not yet qualified','SPR result-access status unknown',
            'Additional public adapters need parity validation','Controlled training implementation and runs remain'],
        current_inference_workers_unchanged=True))
    print(json.dumps(json.loads((REPORT/'PREPARATION_STATUS.json').read_text()),indent=2))


if __name__=='__main__':build()
