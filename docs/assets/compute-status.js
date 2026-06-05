window.BIOMASTER_COMPUTE_STATUS = {
  "updatedUtc": "2026-06-05T10:38:12.912090Z",
  "sourceUpdatedUtc": "2026-06-05T10:34:40Z",
  "summary": {
    "completedModuleCount": 58,
    "activeComputationCount": 6,
    "experimentClosureStatus": "not_complete",
    "experimentClosurePassedRequiredChecks": 1,
    "experimentClosureTotalRequiredChecks": 8,
    "artifactManifestCount": 15377,
    "artifactManifestTotalSizeBytes": 1767731956,
    "sourceScriptCount": 57,
    "candidateRows": 3921,
    "sotaReadyA": 112,
    "sotaReadyB": 783,
    "experimentReadyRows": 885,
    "reviewReadyRows": 763,
    "balancedPanelRows": 300,
    "wave1Rows": 96
  },
  "active": {
    "fullDiffdock": {
      "status": "running",
      "completedJobs": 2495,
      "totalJobs": 3653,
      "completedJobPct": 68.3,
      "scoredRows": 623750,
      "totalRows": 913170,
      "scoredRowPct": 68.306,
      "completedOutputs": 484295,
      "missingOutputsInScoredJobs": 139455,
      "activeLocks": [
        2495,
        2496,
        2497,
        2498
      ],
      "activeDetails": [
        {
          "jobId": 2495,
          "expectedRows": 250,
          "scoreRows": 0,
          "completedOutputs": 0,
          "inFlightRank1SdfCount": 135,
          "scoreExists": false,
          "lockPath": "outputs/report_scale/diffdock_full_run/scores/diffdock_full_chunk_02495.scores.csv.lock"
        },
        {
          "jobId": 2496,
          "expectedRows": 250,
          "scoreRows": 0,
          "completedOutputs": 0,
          "inFlightRank1SdfCount": 119,
          "scoreExists": false,
          "lockPath": "outputs/report_scale/diffdock_full_run/scores/diffdock_full_chunk_02496.scores.csv.lock"
        },
        {
          "jobId": 2497,
          "expectedRows": 250,
          "scoreRows": 0,
          "completedOutputs": 0,
          "inFlightRank1SdfCount": 4,
          "scoreExists": false,
          "lockPath": "outputs/report_scale/diffdock_full_run/scores/diffdock_full_chunk_02497.scores.csv.lock"
        },
        {
          "jobId": 2498,
          "expectedRows": 250,
          "scoreRows": 0,
          "completedOutputs": 0,
          "inFlightRank1SdfCount": 0,
          "scoreExists": false,
          "lockPath": "outputs/report_scale/diffdock_full_run/scores/diffdock_full_chunk_02498.scores.csv.lock"
        }
      ],
      "etaHours": 63.9908,
      "etaDays": 2.6663,
      "estimatedFinishUtc": "2026-06-08T02:37:12Z",
      "busyGpuCount": 4,
      "gpuCount": 4,
      "gpus": [
        {
          "index": 0,
          "busId": "00000000:1C:00.0",
          "memoryUsedMb": 11899.0,
          "memoryTotalMb": 32760.0,
          "utilizationPct": 35.0,
          "temperatureC": 41.0,
          "powerW": 148.73
        },
        {
          "index": 1,
          "busId": "00000000:1D:00.0",
          "memoryUsedMb": 4203.0,
          "memoryTotalMb": 32760.0,
          "utilizationPct": 100.0,
          "temperatureC": 53.0,
          "powerW": 272.83
        },
        {
          "index": 2,
          "busId": "00000000:1E:00.0",
          "memoryUsedMb": 11829.0,
          "memoryTotalMb": 32760.0,
          "utilizationPct": 36.0,
          "temperatureC": 43.0,
          "powerW": 127.97
        },
        {
          "index": 3,
          "busId": "00000000:DC:00.0",
          "memoryUsedMb": 10659.0,
          "memoryTotalMb": 32760.0,
          "utilizationPct": 30.0,
          "temperatureC": 41.0,
          "powerW": 168.42
        }
      ],
      "createdUtc": "2026-06-05T10:37:46Z",
      "interpretationZh": "全量 DiffDock 正在覆盖 druggable-proteome 候选队列。已完成 chunk 中的 missing output 是技术性缺失，不应解释为生物学阴性。"
    },
    "standardPoseFull3921": {
      "status": "completed",
      "summary": {
        "created_utc": "2026-06-04T08:20:24Z",
        "scope": "Standard PoseBusters/ProLIF structural validation over Top3921 of 3921 final candidates.",
        "source": "outputs/sota_validation/final_prioritization/final_priority_pose_quality_matrix.csv",
        "sourceRows": 3921,
        "candidateRows": 3921,
        "structureInputReadyRows": 3776,
        "structureInputReadyPct": 96.302,
        "uniqueProteinsConverted": 270,
        "posebustersConfig": "dock",
        "posebustersPassRows": 573,
        "posebustersPassPct": 14.6136,
        "prolifOkRows": 3776,
        "prolifOkPct": 96.302,
        "prolifInteractionRows": 23445,
        "standardSupportedRows": 502,
        "standardSupportedPct": 12.8029,
        "knownRows": 136,
        "novelRows": 1711,
        "standardPoseValidationTierCounts": {
          "D_standard_pose_fail": 3181,
          "A_standard_structure_supported": 46,
          "C_standard_interaction_review": 71,
          "B_standard_structure_acceptable": 456,
          "C_standard_pose_review": 22,
          "D_standard_pose_not_ready": 145
        },
        "medianProlifInteractionCount": 5.0,
        "medianStandardPoseValidationScore": 60.0,
        "top100": {
          "rows": 100,
          "standardSupportedRows": 21,
          "posebustersPassRows": 23,
          "prolifOkRows": 100,
          "knownRows": 9,
          "novelRows": 47,
          "tierCounts": {
            "D_standard_pose_fail": 76,
            "A_standard_structure_supported": 5,
            "C_standard_interaction_review": 2,
            "B_standard_structure_acceptable": 16,
            "C_standard_pose_review": 1
          }
        },
        "directionRows": [
          {
            "direction": "cardiovascular",
            "rows": 547,
            "structureInputReadyRows": 532,
            "standardSupportedRows": 58,
            "standardSupportedPct": 10.6033,
            "knownRows": 23,
            "novelRows": 166,
            "tierCounts": "{\"A_standard_structure_supported\": 10, \"B_standard_structure_acceptable\": 48, \"C_standard_interaction_review\": 8, \"C_standard_pose_review\": 9, \"D_standard_pose_fail\": 457, \"D_standard_pose_not_ready\": 15}",
            "posebustersPassRows": 66,
            "prolifOkRows": 532,
            "medianProlifInteractionCount": 5.0,
            "medianStandardPoseValidationScore": 60.0
          },
          {
            "direction": "immunology_inflammation",
            "rows": 844,
            "structureInputReadyRows": 817,
            "standardSupportedRows": 97,
            "standardSupportedPct": 11.4929,
            "knownRows": 28,
            "novelRows": 227,
            "tierCounts": "{\"A_standard_structure_supported\": 7, \"B_standard_structure_acceptable\": 90, \"C_standard_interaction_review\": 8, \"C_standard_pose_review\": 3, \"D_standard_pose_fail\": 709, \"D_standard_pose_not_ready\": 27}",
            "posebustersPassRows": 105,
            "prolifOkRows": 817,
            "medianProlifInteractionCount": 5.0,
            "medianStandardPoseValidationScore": 60.0
          },
          {
            "direction": "infectious_disease",
            "rows": 940,
            "structureInputReadyRows": 895,
            "standardSupportedRows": 115,
            "standardSupportedPct": 12.234,
            "knownRows": 31,
            "novelRows": 406,
            "tierCounts": "{\"A_standard_structure_supported\": 10, \"B_standard_structure_acceptable\": 105, \"C_standard_interaction_review\": 15, \"C_standard_pose_review\": 5, \"D_standard_pose_fail\": 760, \"D_standard_pose_not_ready\": 45}",
            "posebustersPassRows": 130,
            "prolifOkRows": 895,
            "medianProlifInteractionCount": 5.0,
            "medianStandardPoseValidationScore": 60.0
          },
          {
            "direction": "neurology_psychiatry",
            "rows": 793,
            "structureInputReadyRows": 744,
            "standardSupportedRows": 125,
            "standardSupportedPct": 15.7629,
            "knownRows": 35,
            "novelRows": 299,
            "tierCounts": "{\"A_standard_structure_supported\": 9, \"B_standard_structure_acceptable\": 116, \"C_standard_interaction_review\": 22, \"C_standard_pose_review\": 1, \"D_standard_pose_fail\": 596, \"D_standard_pose_not_ready\": 49}",
            "posebustersPassRows": 147,
            "prolifOkRows": 744,
            "medianProlifInteractionCount": 4.0,
            "medianStandardPoseValidationScore": 60.0
          },
          {
            "direction": "oncology",
            "rows": 797,
            "structureInputReadyRows": 788,
            "standardSupportedRows": 107,
            "standardSupportedPct": 13.4253,
            "knownRows": 19,
            "novelRows": 613,
            "tierCounts": "{\"A_standard_structure_supported\": 10, \"B_standard_structure_acceptable\": 97, \"C_standard_interaction_review\": 18, \"C_standard_pose_review\": 4, \"D_standard_pose_fail\": 659, \"D_standard_pose_not_ready\": 9}",
            "posebustersPassRows": 125,
            "prolifOkRows": 788,
            "medianProlifInteractionCount": 5.0,
            "medianStandardPoseValidationScore": 60.0
          }
        ],
        "methodNote": "PoseBusters dock configuration checks ligand chemistry and ligand-protein plausibility; ProLIF generates protein-ligand interaction fingerprints from the same docked SDF/PDB files. This layer is a standard-tool validation extension over high-priority candidates, not a new affinity model."
      },
      "summaryFile": {
        "exists": true,
        "path": "/root/autodl-tmp/BioMaster/outputs/sota_validation/standard_pose_validation_full3921/standard_pose_validation_summary.json",
        "sizeBytes": 4883,
        "mtimeUtc": "2026-06-04T08:20:24.250034Z"
      },
      "logFile": {
        "exists": true,
        "path": "/root/autodl-tmp/BioMaster/logs/standard_pose_validation_full3921.log",
        "sizeBytes": 13403,
        "mtimeUtc": "2026-06-04T08:20:28.374057Z"
      },
      "interpretationZh": "full3921 标准 PoseBusters/ProLIF 结构验证已完成。"
    },
    "experimentClosure": {
      "overallStatus": "not_complete",
      "passedRequiredChecks": 1,
      "totalRequiredChecks": 8,
      "failedRequiredChecks": [
        "main_diffdock_complete",
        "single_ligand_rescue_complete_or_documented",
        "multi_ligand_rescue_complete",
        "final_merge_outputs_exist",
        "final_merged_rows_sufficient",
        "finalizer_completed",
        "post_finalization_completed"
      ],
      "createdUtc": "2026-06-05T10:37:43.574938Z",
      "auditJson": "outputs/sota_validation/experiment_closure_audit/experiment_closure_audit.json",
      "auditMarkdown": "outputs/sota_validation/experiment_closure_audit/EXPERIMENT_CLOSURE_AUDIT.md",
      "interpretationZh": "全部 required checks 通过前，实验计算闭环不能标记完成。"
    },
    "ligandRescue": {
      "status": "queued_for_auto_start_after_main_queue",
      "ligandId": "CHEMBL3039504",
      "queuedRows": 998,
      "jobs": 16,
      "watcherReady": false,
      "watcherLog": "logs/diffdock_ligand_rescue_after_full.log",
      "interpretationZh": "CHEMBL3039504 triggered clustered DiffDock technical failures with the original multi-fragment/salt SDF. A largest-organic-parent rescue queue has been prepared and should be run only after a real GPU slot is free. A background watcher is active and will start this rescue queue after the main full DiffDock queue completes and GPUs are confirmed idle."
    },
    "multiLigandRescue": {
      "status": "queued_for_auto_start_after_single_ligand_rescue",
      "candidateLigands": 48,
      "queuedLigands": 48,
      "queuedRows": 47871,
      "jobs": 768,
      "watcherReady": false,
      "watcherLog": "logs/diffdock_multi_ligand_rescue_after_single.log",
      "latestAuditScoredLigands": 594,
      "latestAuditScoredRows": 601500,
      "latestAuditMissingRows": 135843,
      "latestAuditMissingPct": 22.584,
      "latestAuditRescueRecommendedLigands": 64,
      "latestAuditRescueRecommendedRows": 64837,
      "latestAuditZeroCompletedChunks": 205,
      "latestAuditMaskRotateZeroCompletedChunks": 205,
      "interpretationZh": "预构建 multi-ligand rescue 队列是当前可运行快照；最终 watcher 会在主队列和单 ligand rescue 完成后重跑 failure audit，再按最新技术缺失结果重建 rescue 队列。"
    }
  },
  "completedModules": [
    {
      "module": "ConPLex affinity screen and disease-direction ranking",
      "status": "completed",
      "evidence": "Existing ranked candidate matrices were used as the base layer for all downstream validation.",
      "candidateRows": null,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "DiffDock disease-direction structural docking",
      "status": "completed_with_stable_residual_failures",
      "evidence": "SDF and SMILES rescue rounds were completed; the remaining missing outputs are stable graph/sampling boundary cases.",
      "candidateRows": 23744,
      "completedRows": 23707,
      "completionPct": 99.84417115902964
    },
    {
      "module": "Open Targets multi-disease target evidence",
      "status": "completed",
      "evidence": "Expanded disease-direction evidence was integrated into the P0 validation coverage audit.",
      "candidateRows": null,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "TxGNN multi-direction drug-disease evidence",
      "status": "completed",
      "evidence": "Representative disease directions were inferred for oncology, cardiovascular, infectious, neurology/psychiatry, and immunology/inflammation.",
      "candidateRows": null,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Known-target benchmark and enrichment",
      "status": "completed",
      "evidence": "Pair-level AUROC/AUPRC, Recall@K, random enrichment, and positive-pair ranks were generated.",
      "candidateRows": null,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Known-target stratified validation",
      "status": "completed",
      "evidence": "Record-level known-target recall was stratified by approval year, therapeutic area, and route.",
      "candidateRows": null,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "ADMET and repurposability filtering",
      "status": "completed_rule_based",
      "evidence": "Transparent RDKit descriptors, PAINS/Brenk alerts, route feasibility, and label-text flags were computed.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "ML ADMET/toxicity endpoint audit",
      "status": "completed_local_tdc_qsar",
      "evidence": "Public TDC ADME/Tox endpoint data were used to train local molecular-fingerprint QSAR models for hERG, DILI, Ames, CYP inhibition, P-gp, BBB, and HIA, then score the FDA drug library and final candidates.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "TxGNN KG explainability paths",
      "status": "completed",
      "evidence": "Direct drug-target, drug-disease, target-disease, PPI bridge, and known-target disease bridge paths were generated.",
      "candidateRows": null,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "DiffDock pose sanity audit",
      "status": "completed_lightweight_geometry",
      "evidence": "RDKit/PDB heavy-atom contact checks were run over the disease-direction structural candidate set.",
      "candidateRows": 23744,
      "completedRows": 23060,
      "completionPct": null
    },
    {
      "module": "AlphaFold pocket-confidence structural audit",
      "status": "completed_plddt_pocket_qc",
      "evidence": "Docked ligand pockets were audited against AlphaFold pLDDT values from receptor PDB B-factor fields, producing structure-adjusted ranking and low-confidence review tables.",
      "candidateRows": 23744,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Stage6 Top10000 oncology structural audit",
      "status": "completed_pose_and_pocket_qc",
      "evidence": "The completed Stage6 Top10000 oncology consensus table was audited with pose-sanity and AlphaFold pocket-confidence checks; missing DiffDock rows are retained as technical missing outputs, not biological negatives.",
      "candidateRows": 10000,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Residue-contact pose interpretability audit",
      "status": "completed_final_candidate_contact_qc",
      "evidence": "Final-candidate docked poses were converted into residue-contact, contact-class, pocket pLDDT, and expert discussion tables.",
      "candidateRows": 3921,
      "completedRows": 3776,
      "completionPct": null
    },
    {
      "module": "RDKit pose-quality gate",
      "status": "completed_local_posebusters_like_qc",
      "evidence": "Final-candidate docked poses were audited for ligand conformer readability, bond geometry, intramolecular clashes, ligand-receptor severe overlaps, contact coverage, and pocket pLDDT as a local PoseBusters-like quality gate.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Standard PoseBusters/ProLIF Top100 structural validation",
      "status": "completed_standard_tool_expert_check",
      "evidence": "High-priority Top100 docked poses were checked with standard PoseBusters and ProLIF tooling, producing pose-validity calls plus residue-level ligand-protein interaction fingerprints for expert review.",
      "candidateRows": 100,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Standard PoseBusters/ProLIF Top300 structural validation extension",
      "status": "completed_standard_tool_extension_top300",
      "evidence": "The completed Top300 extension broadens the standard PoseBusters/ProLIF stress test beyond the original Top100 while preserving the Top100 result as the formal high-priority expert-check layer.",
      "candidateRows": 300,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Standard PoseBusters/ProLIF full final-candidate structural validation",
      "status": "completed_standard_tool_extension_full3921",
      "evidence": "The full final-priority candidate set is being checked with the same standard PoseBusters/ProLIF tooling used for the Top100 and Top300 expert-check layers.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "AutoDock Vina Top100 structural consensus rescoring",
      "status": "completed_vina_python_score_only_top100",
      "evidence": "Top100 high-priority DiffDock poses were converted to PDBQT and independently stress-tested with AutoDock Vina score-only plus local optimization. The result is a conservative structural consensus layer, not a new affinity model.",
      "candidateRows": 100,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "smina Top100 structural rescoring",
      "status": "completed_smina_score_only_top100",
      "evidence": "The prepared Top100 structural queue was independently rescored with smina score-only mode, including receptor PDBQT conversion for missing receptor caches. This is a second classical docking/scoring stress test and not GNINA CNN scoring.",
      "candidateRows": 100,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "GNINA CNN Top100 rescoring execution audit",
      "status": "completed_gnina_cnn_top100",
      "evidence": "GNINA CNN rescoring has been executed over the prepared Top100 structural queue; completed rows now carry parsed CNNscore/CNNaffinity and consensus tiers.",
      "candidateRows": 100,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Boltz-2 Top50 second-model complex validation",
      "status": "completed_boltz2_fast_spotcheck_top50",
      "evidence": "Boltz-2 was executed locally on the selected Top50 complex spot-check set as an orthogonal structure-generation and affinity-probability layer. Salt/solvate/counter-ion failures were rerun as largest organic parent fragments and are explicitly flagged.",
      "candidateRows": 50,
      "completedRows": 50,
      "completionPct": null
    },
    {
      "module": "Boltz-2 high-sampling finalist validation",
      "status": "completed_boltz2_high_sampling_finalists",
      "evidence": "Selected original-ligand Boltz-2 finalists were rerun with higher recycling, structure sampling, and affinity sampling. This is a stronger second-model structural corroboration layer for finalists, not a DiffDock rerun.",
      "candidateRows": 12,
      "completedRows": 12,
      "completionPct": null
    },
    {
      "module": "ChEMBL target druggability and clinical-stage audit",
      "status": "completed_uniprot_exact_match",
      "evidence": "Final candidates were mapped by exact UniProt accession to the ChEMBL druggable-proteome table, adding clinical phase, druggable modality, target class, and disease-direction context.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Drug chemotype scaffold and similarity-diversity audit",
      "status": "completed_rdkit_scaffold_cluster_qc",
      "evidence": "FDA small-molecule SMILES were mapped to candidate drugs, then audited by Murcko scaffolds and Morgan-fingerprint Butina clusters to quantify chemotype concentration and generate a scaffold-capped shortlist.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "SOTA-ready multi-evidence decision matrix",
      "status": "completed_integrated_expert_triage",
      "evidence": "Final candidates were re-integrated across model score, KG/disease evidence, structure confidence, target druggability, chemotype diversity, direction specificity, and risk flags into expert action queues.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Network medicine / PPI proximity audit",
      "status": "completed_expanded_string_huri_interactome",
      "evidence": "Open Targets disease-module seeds and the current STRING/HuRI PPI graph were used to add coverage-aware target proximity evidence; uncovered targets are labeled as data gaps rather than negative biology.",
      "candidateRows": 23744,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "SOTA-network candidate reprioritization",
      "status": "completed_network_medicine_integration",
      "evidence": "The network-medicine layer was merged into the existing SOTA-ready matrix without overwriting the original ranking, producing a network-adjusted matrix, shortlists, TopK metrics, and rank-shift review.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "HPA tissue-expression context audit",
      "status": "completed_hpa_tissue_plausibility_qc",
      "evidence": "Human Protein Atlas consensus RNA expression was mapped to disease-direction tissue panels, adding tissue-plausibility support and mismatch review queues without replacing disease causality evidence.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "GTEx tissue-expression context audit",
      "status": "completed_gtex_tissue_plausibility_qc",
      "evidence": "GTEx V10 median-TPM tissue expression was mapped to disease-direction tissue panels as an independent bulk-tissue corroboration layer for target-context plausibility.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "DepMap oncology dependency scoring",
      "status": "completed_depmap_crispr_dependency_qc",
      "evidence": "DepMap Public CRISPR dependency probabilities were mapped to oncology candidate targets to add a cancer-cell vulnerability layer; this supports target context, not direct drug binding or clinical efficacy.",
      "candidateRows": null,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "LINCS/CMap disease-signature reversal scoring",
      "status": "completed_drug_direction_signature_reversal",
      "evidence": "LINCS/CMap Level 5 compound perturbation signatures were scored against direction-level disease signatures. The score is a drug-disease transcriptomic reversal layer, not direct target-binding proof.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Integrated expert review panel",
      "status": "completed_multi_evidence_expert_shortlist",
      "evidence": "Affinity, disease evidence, pathway/CREEDS, LINCS/CMap, GTEx/HPA, DepMap, ADMET, contraindication flags, structure audits, and known-target/novelty labels were merged into a diversity-constrained Top50 and Wave1 panel.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "External SOTA model input package",
      "status": "completed_input_package_with_local_dti_execution",
      "evidence": "Executable input queues were generated for GNINA CNN rescoring, Boltz/Chai-style complex spot-checks, and independent DTI ensemble inference. The local supervised DTI branch has now been executed; formal DrugBAN/DeepDTA pretrained inference still requires validated weights and dependencies.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Local supervised independent DTI corroboration",
      "status": "completed_local_supervised_dti_top1000",
      "evidence": "A reproducible local ExtraTrees DTI model was trained from FDA label drug-UniProt positives and sampled unlabeled negatives, excluding exact FDA-positive candidate pairs before scoring the Top1000 SMILES+sequence queue. This provides an orthogonal DTI corroboration layer relative to ConPLex, while not being claimed as DrugBAN/DeepDTA pretrained inference.",
      "candidateRows": 1000,
      "completedRows": 1000,
      "completionPct": null
    },
    {
      "module": "Local supervised independent DTI full final-candidate extension",
      "status": "completed_local_supervised_dti_full3921",
      "evidence": "The same local supervised DTI protocol was expanded from the prepared Top1000 queue to the full 3921 final-priority candidate matrix, preserving exact FDA-positive candidate-pair exclusion from training.",
      "candidateRows": 3921,
      "completedRows": 3921,
      "completionPct": null
    },
    {
      "module": "SOTA model feasibility audit",
      "status": "completed_engineering_readiness_audit",
      "evidence": "Local availability of SOTA-adjacent validation layers was audited across transcriptomics, tissue context, structural rescoring, independent DTI, complex prediction, and ML ADMET prerequisites.",
      "candidateRows": null,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Rank stability and consensus-priority audit",
      "status": "completed_multi_ranking_stability_qc",
      "evidence": "Final-priority, structure-adjusted, target-adjusted, and SOTA-ready rankings were compared by TopK overlap, rank correlation, consensus candidates, and large rank-delta review.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Experimental validation planning panel",
      "status": "completed_assay_planning_triage",
      "evidence": "Final candidates were converted into assay-planning queues with transparent validation score, disease-balanced shortlist, novelty shortlist, positive controls, risk-review queue, and assay modality labels.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Wet-lab pre-experiment focusing package",
      "status": "completed_decision_grade_wetlab_focus",
      "evidence": "The expert Top50 and Wave1 panels were converted into a stricter wet-lab execution package with core 6, focused 12, backup 24, disease-direction Top10, purchase/assay queue, primary assay, orthogonal assay, counterscreen, execution protocol, vendor/platform checklist, pre-experiment go/hold decision matrix, and a final pre-purchase manual-gate checklist.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Validation panel benchmark and calibration",
      "status": "completed_known_target_calibration_qc",
      "evidence": "Experimental validation queues were audited against known drug-target positives, TopK enrichment, group calibration, and queue-level control/novel/interpretability composition.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Validation panel diversity and wave-1 coverage audit",
      "status": "completed_panel_design_qc",
      "evidence": "The assay-planning outputs were audited for concentration across disease direction, drug, target, scaffold, mechanism gate, and assay modality; a capped wave-1 validation panel was generated for practical follow-up design.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "FDA label mechanism and action-type audit",
      "status": "completed_label_mechanism_qc",
      "evidence": "FDA Target ChEMBL IDs were expanded through the local target-component cache to UniProt accessions, then matched to candidate drug-protein pairs to separate label-target recalls, same-drug new-target extensions, and clinically labeled targets paired with different approved drugs.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "FDA label temporal generalization audit",
      "status": "completed_retrospective_time_sliced_qc",
      "evidence": "FDA label-target rows were stratified by approval-year era to stress-test whether the validation ranking and assay queues recover modern and recent label mechanisms; this is a retrospective time-sliced audit, not a true prospective deployment test.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Final multi-evidence candidate prioritization",
      "status": "completed",
      "evidence": "Model score, disease evidence, KG path, ADMET, known-target recall, and pose sanity were integrated into final expert triage tables.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Final-priority known-target validation",
      "status": "completed",
      "evidence": "Known drug-target positives inside the final triage table were used to test whether the integrated priority score still enriches recoverable biology.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Final-priority ablation and robustness audit",
      "status": "completed",
      "evidence": "Single-layer and leave-one-component-out rankings were compared against known-target positives, with TopK enrichment and bootstrap confidence intervals.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Novelty and known-mechanism leakage audit",
      "status": "completed",
      "evidence": "Known benchmark pairs, direct KG drug-target edges, known disease-use hypotheses, safety context, sparse evidence, and strict novel pairs were separated for interpretation.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Cross-disease direction specificity audit",
      "status": "completed",
      "evidence": "The same drug-target pair was compared across disease directions to distinguish disease-focused candidates from broad multi-direction generalists.",
      "candidateRows": null,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "TopK significance and stratified random baseline audit",
      "status": "completed",
      "evidence": "Observed known-target hits in final TopK were compared with global random and disease-direction-stratified random baselines.",
      "candidateRows": null,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Multi-evidence concordance audit",
      "status": "completed",
      "evidence": "Model, disease, KG, ADMET, structure, and direction-label evidence layers were counted to separate mature multi-evidence candidates from sparse-evidence priorities.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Candidate diversity and concentration audit",
      "status": "completed",
      "evidence": "Top candidate concentration was quantified by drug, target, and target family, and a diversity-capped expert shortlist was generated.",
      "candidateRows": null,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "Computation artifact manifest and reproducibility inventory",
      "status": "completed",
      "evidence": "Core SOTA computation outputs were indexed with size, timestamp, row/column counts where available, SHA256 hashes, and source-script attribution.",
      "candidateRows": null,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "AutoDock Vina Top300 structural consensus rescoring extension",
      "status": "completed_vina_python_score_only_top300",
      "evidence": "The AutoDock Vina score-only/local-optimization stress test has been extended from Top100 to Top300 final-priority candidates.",
      "candidateRows": 300,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "AutoDock Vina final3921 structural consensus rescoring",
      "status": "completed_vina_python_score_only_final3921",
      "evidence": "Final-priority rows with available structure inputs are independently stress-tested with AutoDock Vina score-only plus local optimization; merged shard output is used when available.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "smina Top300 structural rescoring extension",
      "status": "completed_smina_score_only_top300",
      "evidence": "The smina score-only stress test has been extended from Top100 to Top300 final-priority candidates.",
      "candidateRows": 300,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "smina final3921 structural rescoring",
      "status": "completed_smina_score_only_final3921",
      "evidence": "All final-priority rows with available structure inputs were independently rescored with smina score-only mode.",
      "candidateRows": 3921,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "GNINA CNN Top300 structural rescoring extension",
      "status": "completed_gnina_cnn_top300_cpu",
      "evidence": "GNINA CNN rescoring has been extended from Top100 to Top300 in CPU-only mode, preserving GPU capacity for the full DiffDock expansion.",
      "candidateRows": 300,
      "completedRows": null,
      "completionPct": null
    },
    {
      "module": "GNINA CNN full final-candidate structural rescoring",
      "status": "completed_gnina_cnn_full3921_cpu",
      "evidence": "The GNINA CNN structural consensus stress test has been expanded to the full final-candidate structural queue in CPU-only mode.",
      "candidateRows": 2664,
      "completedRows": null,
      "completionPct": null
    }
  ],
  "moduleStatusCounts": [
    {
      "label": "completed",
      "value": 15
    },
    {
      "label": "completed_with_stable_residual_failures",
      "value": 1
    },
    {
      "label": "completed_rule_based",
      "value": 1
    },
    {
      "label": "completed_local_tdc_qsar",
      "value": 1
    },
    {
      "label": "completed_lightweight_geometry",
      "value": 1
    },
    {
      "label": "completed_plddt_pocket_qc",
      "value": 1
    },
    {
      "label": "completed_pose_and_pocket_qc",
      "value": 1
    },
    {
      "label": "completed_final_candidate_contact_qc",
      "value": 1
    },
    {
      "label": "completed_local_posebusters_like_qc",
      "value": 1
    },
    {
      "label": "completed_standard_tool_expert_check",
      "value": 1
    },
    {
      "label": "completed_standard_tool_extension_top300",
      "value": 1
    },
    {
      "label": "completed_standard_tool_extension_full3921",
      "value": 1
    },
    {
      "label": "completed_vina_python_score_only_top100",
      "value": 1
    },
    {
      "label": "completed_smina_score_only_top100",
      "value": 1
    },
    {
      "label": "completed_gnina_cnn_top100",
      "value": 1
    },
    {
      "label": "completed_boltz2_fast_spotcheck_top50",
      "value": 1
    },
    {
      "label": "completed_boltz2_high_sampling_finalists",
      "value": 1
    },
    {
      "label": "completed_uniprot_exact_match",
      "value": 1
    },
    {
      "label": "completed_rdkit_scaffold_cluster_qc",
      "value": 1
    },
    {
      "label": "completed_integrated_expert_triage",
      "value": 1
    },
    {
      "label": "completed_expanded_string_huri_interactome",
      "value": 1
    },
    {
      "label": "completed_network_medicine_integration",
      "value": 1
    },
    {
      "label": "completed_hpa_tissue_plausibility_qc",
      "value": 1
    },
    {
      "label": "completed_gtex_tissue_plausibility_qc",
      "value": 1
    },
    {
      "label": "completed_depmap_crispr_dependency_qc",
      "value": 1
    },
    {
      "label": "completed_drug_direction_signature_reversal",
      "value": 1
    },
    {
      "label": "completed_multi_evidence_expert_shortlist",
      "value": 1
    },
    {
      "label": "completed_input_package_with_local_dti_execution",
      "value": 1
    },
    {
      "label": "completed_local_supervised_dti_top1000",
      "value": 1
    },
    {
      "label": "completed_local_supervised_dti_full3921",
      "value": 1
    },
    {
      "label": "completed_engineering_readiness_audit",
      "value": 1
    },
    {
      "label": "completed_multi_ranking_stability_qc",
      "value": 1
    },
    {
      "label": "completed_assay_planning_triage",
      "value": 1
    },
    {
      "label": "completed_decision_grade_wetlab_focus",
      "value": 1
    },
    {
      "label": "completed_known_target_calibration_qc",
      "value": 1
    },
    {
      "label": "completed_panel_design_qc",
      "value": 1
    },
    {
      "label": "completed_label_mechanism_qc",
      "value": 1
    },
    {
      "label": "completed_retrospective_time_sliced_qc",
      "value": 1
    },
    {
      "label": "completed_vina_python_score_only_top300",
      "value": 1
    },
    {
      "label": "completed_vina_python_score_only_final3921",
      "value": 1
    },
    {
      "label": "completed_smina_score_only_top300",
      "value": 1
    },
    {
      "label": "completed_smina_score_only_final3921",
      "value": 1
    },
    {
      "label": "completed_gnina_cnn_top300_cpu",
      "value": 1
    },
    {
      "label": "completed_gnina_cnn_full3921_cpu",
      "value": 1
    }
  ],
  "evidenceLayers": [
    {
      "label": "KG explainability",
      "value": 97.16908951798011,
      "suffix": "%",
      "note": "28774 path rows"
    },
    {
      "label": "Structure A/B pocket",
      "value": 87.1502,
      "suffix": "%",
      "note": "23051 audited poses"
    },
    {
      "label": "Pose interpretability",
      "value": 48.7631,
      "suffix": "%",
      "note": "52091 residue contacts"
    },
    {
      "label": "Local DTI support",
      "value": 29.992348890589135,
      "suffix": "%",
      "note": "3921 scored rows"
    },
    {
      "label": "ML ADMET low/manageable",
      "value": 54.7054,
      "suffix": "%",
      "note": "11 endpoints"
    },
    {
      "label": "Tissue context HPA",
      "value": 76.154,
      "suffix": "%",
      "note": "relevant tissue expression"
    },
    {
      "label": "GTEx context",
      "value": 73.2211,
      "suffix": "%",
      "note": "GTEx Analysis V10"
    },
    {
      "label": "Network medicine",
      "value": 85.23335883703137,
      "suffix": "%",
      "note": "STRING subnet support"
    }
  ],
  "structuralModels": [
    {
      "label": "DiffDock disease-direction reps",
      "completed": 23707,
      "total": 23744,
      "pct": 99.84417115902964,
      "note": "SDF/SMILES rescue complete"
    },
    {
      "label": "Vina final3921",
      "completed": 3763,
      "total": 3921,
      "pct": 95.9704,
      "note": "classical score-only stress test"
    },
    {
      "label": "smina final3921",
      "completed": 3776,
      "total": 3921,
      "pct": 96.302,
      "note": "independent classical docking score"
    },
    {
      "label": "GNINA CNN full3921",
      "completed": 2583,
      "total": 2664,
      "pct": 96.9595,
      "note": "CPU execution, GPU preserved for DiffDock"
    },
    {
      "label": "Boltz-2 Top50",
      "completed": 50,
      "total": 50,
      "pct": 100.0,
      "note": "orthogonal complex model"
    },
    {
      "label": "Boltz-2 high-sampling finalists",
      "completed": 12,
      "total": 12,
      "pct": 100.0,
      "note": "higher confidence finalist rerun"
    }
  ],
  "sotaReady": {
    "candidateRows": 3921,
    "uniqueDrugs": 117,
    "uniqueTargets": 271,
    "tierCounts": [
      {
        "label": "D_blocked_or_requires_resolution",
        "value": 1336
      },
      {
        "label": "C_context_or_secondary_review",
        "value": 863
      },
      {
        "label": "D_low_priority_or_sparse_support",
        "value": 827
      },
      {
        "label": "B_review_ready_priority",
        "value": 783
      },
      {
        "label": "A_sota_ready_expert_priority",
        "value": 112
      }
    ],
    "actionCounts": [
      {
        "label": "novel_pair_expert_review",
        "value": 1463
      },
      {
        "label": "safety_or_contraindication_review",
        "value": 1162
      },
      {
        "label": "mechanism_extension_repurposing",
        "value": 564
      },
      {
        "label": "target_context_review",
        "value": 366
      },
      {
        "label": "positive_control_or_known_mechanism",
        "value": 192
      },
      {
        "label": "structure_low_confidence_review",
        "value": 174
      }
    ],
    "top100KnownRows": 10,
    "top100NovelRows": 48,
    "top100StructureABRows": 100,
    "top100TargetABRows": 100,
    "expertShortlistRows": 300,
    "novelShortlistRows": 300,
    "diverseShortlistRows": 243
  },
  "validationPanel": {
    "candidateRows": 3921,
    "experimentReadyRows": 885,
    "reviewReadyRows": 763,
    "novelReadyRows": 1108,
    "positiveControlReadyRows": 137,
    "balancedPanelRows": 300,
    "balancedPanelDirections": 5,
    "balancedUniqueDrugs": 61,
    "balancedUniqueTargets": 120,
    "balancedUniqueScaffolds": 57,
    "tierCounts": [
      {
        "label": "D_risk_hold",
        "value": 1162
      },
      {
        "label": "A_experiment_ready",
        "value": 885
      },
      {
        "label": "B_expert_review_ready",
        "value": 763
      },
      {
        "label": "D_structure_hold",
        "value": 552
      },
      {
        "label": "D_low_priority",
        "value": 444
      },
      {
        "label": "C_secondary_or_context_review",
        "value": 115
      }
    ],
    "gateCounts": [
      {
        "label": "novel_candidate",
        "value": 1822
      },
      {
        "label": "risk_review",
        "value": 1076
      },
      {
        "label": "mechanism_extension",
        "value": 716
      },
      {
        "label": "positive_control",
        "value": 307
      }
    ],
    "assayCounts": [
      {
        "label": "cell_based_receptor_function_assay",
        "value": 3416
      },
      {
        "label": "biochemical_kinase_or_enzyme_assay",
        "value": 163
      },
      {
        "label": "electrophysiology_or_channel_function_assay",
        "value": 124
      },
      {
        "label": "transporter_activity_assay",
        "value": 117
      },
      {
        "label": "target_engagement_and_cell_phenotype_assay",
        "value": 96
      },
      {
        "label": "cellular_reporter_or_target_engagement_assay",
        "value": 5
      }
    ]
  },
  "wetlabFocus": {
    "candidateRows": 3921,
    "experimentReadyRows": 885,
    "reviewReadyRows": 763,
    "expertTop50Rows": 50,
    "wave1Rows": 24,
    "purchaseAndAssayQueueRows": 94,
    "firstExperimentPanelRows": 12,
    "firstExperimentRoleCounts": [
      {
        "label": "novel_repurposing_candidate",
        "value": 5
      },
      {
        "label": "mechanism_extension",
        "value": 4
      },
      {
        "label": "positive_control",
        "value": 3
      }
    ],
    "firstExperimentDirectionCounts": [
      {
        "label": "oncology",
        "value": 6
      },
      {
        "label": "neurology_psychiatry",
        "value": 4
      },
      {
        "label": "cardiovascular",
        "value": 2
      }
    ],
    "prePurchaseFocusRows": 94,
    "prePurchaseTierCounts": [
      {
        "label": "hold",
        "value": 66
      },
      {
        "label": "backup_24",
        "value": 11
      },
      {
        "label": "batch_1_core_6",
        "value": 6
      },
      {
        "label": "batch_1_extension_12",
        "value": 6
      },
      {
        "label": "expert_review_hold",
        "value": 5
      }
    ],
    "prePurchaseActionCounts": [
      {
        "label": "hold_before_purchase",
        "value": 51
      },
      {
        "label": "backup_after_purchase_feasibility_check",
        "value": 21
      },
      {
        "label": "ready_for_final_manual_purchase_check",
        "value": 17
      },
      {
        "label": "expert_review_before_purchase",
        "value": 5
      }
    ],
    "experimentExecutionProtocolRows": 12,
    "experimentExecutionProtocolCoreRows": 6,
    "experimentExecutionProtocolExtensionRows": 6,
    "experimentExecutionProtocolAssayCounts": [
      {
        "label": "cell_based_receptor_function_assay",
        "value": 9
      },
      {
        "label": "biochemical_kinase_or_enzyme_assay",
        "value": 3
      }
    ],
    "procurementPlatformChecklistRows": 12,
    "procurementPlatformChecklistCoreRows": 6,
    "procurementPlatformChecklistExtensionRows": 6,
    "procurementPlatformChecklistStatusCounts": [
      {
        "label": "pending",
        "value": 12
      }
    ],
    "finalPreExperimentGateRows": 94,
    "finalPreExperimentGateTierCounts": [
      {
        "label": "hold_before_purchase",
        "value": 66
      },
      {
        "label": "backup_replacement_after_manual_confirmation",
        "value": 11
      },
      {
        "label": "go_core_6_after_manual_confirmation",
        "value": 6
      },
      {
        "label": "go_extension_12_after_manual_confirmation",
        "value": 6
      },
      {
        "label": "expert_review_before_purchase",
        "value": 5
      }
    ],
    "finalPreExperimentGateGoCandidateRows": 12,
    "finalPreExperimentGateBackupRows": 11,
    "finalPreExperimentGateExpertReviewRows": 5,
    "finalPreExperimentGateHoldRows": 66,
    "finalPreExperimentManualGateCount": 8,
    "finalPreExperimentOutputs": {
      "finalGateCsv": "outputs/sota_validation/wetlab_validation_package/wetlab_final_pre_experiment_gate.csv",
      "top12Csv": "outputs/sota_validation/wetlab_validation_package/wetlab_final_pre_experiment_gate_top12.csv",
      "summaryJson": "outputs/sota_validation/wetlab_validation_package/wetlab_final_pre_experiment_gate_summary.json",
      "markdown": "outputs/sota_validation/wetlab_validation_package/WETLAB_FINAL_PRE_EXPERIMENT_GATE.md"
    },
    "detailedCandidateReviewRows": 12,
    "detailedCandidateReviewLiteratureAuditRows": 12,
    "detailedCandidateReviewOutputs": {
      "tex": "outputs/sota_validation/wetlab_candidate_detailed_review/WETLAB_CANDIDATE_DETAILED_REVIEW.tex",
      "markdown": "outputs/sota_validation/wetlab_candidate_detailed_review/WETLAB_CANDIDATE_DETAILED_REVIEW.md",
      "literatureAuditCsv": "outputs/sota_validation/wetlab_candidate_detailed_review/wetlab_candidate_detailed_review_literature_audit.csv",
      "pdf": "outputs/sota_validation/wetlab_candidate_detailed_review/WETLAB_CANDIDATE_DETAILED_REVIEW.pdf",
      "publishedPdf": "docs/assets/wetlab-candidate-detailed-review.pdf"
    },
    "outputs": {
      "wave1": "outputs/sota_validation/wetlab_validation_package/wetlab_wave1_24.csv",
      "top50": "outputs/sota_validation/wetlab_validation_package/wetlab_expert_top50.csv",
      "directionTop10": "outputs/sota_validation/wetlab_validation_package/wetlab_direction_top10.csv",
      "focusedTop12": "outputs/sota_validation/wetlab_validation_package/wetlab_focused_top12.csv",
      "purchaseAndAssayQueue": "outputs/sota_validation/wetlab_validation_package/wetlab_purchase_and_assay_queue.csv",
      "preExperimentDecisionMatrix": "outputs/sota_validation/wetlab_validation_package/wetlab_pre_experiment_decision_matrix.csv",
      "firstExperimentPanel": "outputs/sota_validation/wetlab_validation_package/wetlab_first_experiment_panel_12.csv",
      "prePurchaseFocusPackage": "outputs/sota_validation/wetlab_validation_package/wetlab_pre_purchase_focus_package.csv",
      "experimentExecutionProtocol": "outputs/sota_validation/wetlab_validation_package/wetlab_experiment_execution_protocol_12.csv",
      "procurementPlatformChecklist": "outputs/sota_validation/wetlab_validation_package/wetlab_platform_procurement_checklist_12.csv",
      "markdown": "outputs/sota_validation/wetlab_validation_package/WETLAB_VALIDATION_RECOMMENDATION.md",
      "focusMarkdown": "outputs/sota_validation/wetlab_validation_package/WETLAB_PRE_EXPERIMENT_FOCUSING_STRATEGY.md",
      "decisionMarkdown": "outputs/sota_validation/wetlab_validation_package/WETLAB_PRE_EXPERIMENT_DECISION_MATRIX.md",
      "firstExperimentMarkdown": "outputs/sota_validation/wetlab_validation_package/WETLAB_FIRST_EXPERIMENT_PANEL.md",
      "prePurchaseFocusMarkdown": "outputs/sota_validation/wetlab_validation_package/WETLAB_PRE_PURCHASE_FOCUS_PACKAGE.md",
      "experimentExecutionMarkdown": "outputs/sota_validation/wetlab_validation_package/WETLAB_EXPERIMENT_EXECUTION_PROTOCOL.md",
      "procurementPlatformChecklistMarkdown": "outputs/sota_validation/wetlab_validation_package/WETLAB_PLATFORM_PROCUREMENT_CHECKLIST.md",
      "summaryJson": "outputs/sota_validation/wetlab_validation_package/wetlab_validation_summary.json"
    },
    "interpretationZh": "湿实验前收敛口径为：从全候选进入采购/实验队列，再压缩为第一轮 12 个候选，首批优先核心 6 个；最终实验前门控会继续要求临床可达浓度、机制方向、模型表达、精确文献 novelty、采购质量和 assay interference 等人工核查后才能下单。"
  },
  "benchmarks": {
    "knownPairRecallAt100000Pct": 16.47982062780269,
    "knownPairEnrichmentAt100000": 8.000936434977579,
    "knownRecordRecallAt100000Pct": 19.524617996604416,
    "finalPriorityValidationRecallAt100Pct": 11.0294,
    "finalPriorityValidationPrecisionAt100Pct": 15.0,
    "finalPriorityValidationEnrichmentAt100": 4.3246,
    "significanceTop100ObservedHits": 15,
    "significanceTop100GlobalExpectedHits": 3.468503,
    "significanceTop100GlobalP": "1.1926398e-06",
    "concordanceMultiEvidencePct": 82.2239,
    "concordanceTop100MultiEvidencePct": 100.0
  },
  "dependencies": {
    "moduleImports": {
      "rdkit": {
        "available": true,
        "detail": "/root/miniconda3/lib/python3.12/site-packages/rdkit/__init__.py"
      },
      "torch": {
        "available": true,
        "detail": "/root/miniconda3/lib/python3.12/site-packages/torch/__init__.py"
      },
      "networkx": {
        "available": true,
        "detail": "/root/miniconda3/lib/python3.12/site-packages/networkx/__init__.py"
      },
      "sklearn": {
        "available": true,
        "detail": "/root/miniconda3/lib/python3.12/site-packages/sklearn/__init__.py"
      },
      "tdc": {
        "available": true,
        "detail": "/root/miniconda3/lib/python3.12/site-packages/tdc/__init__.py"
      },
      "dgl": {
        "available": false,
        "detail": "module_not_found"
      },
      "dgllife": {
        "available": false,
        "detail": "module_not_found"
      },
      "deepchem": {
        "available": false,
        "detail": "module_not_found"
      },
      "prolif": {
        "available": true,
        "detail": "/root/miniconda3/lib/python3.12/site-packages/prolif/__init__.py"
      },
      "openbabel": {
        "available": true,
        "detail": "/root/miniconda3/lib/python3.12/site-packages/openbabel/__init__.py"
      },
      "posebusters": {
        "available": true,
        "detail": "/root/miniconda3/lib/python3.12/site-packages/posebusters/__init__.py"
      },
      "vina": {
        "available": true,
        "detail": "/root/miniconda3/lib/python3.12/site-packages/vina/__init__.py"
      },
      "boltz": {
        "available": false,
        "detail": "module_not_found"
      },
      "chai_lab": {
        "available": false,
        "detail": "module_not_found"
      }
    },
    "binaries": {
      "gnina": {
        "available": true,
        "runtimeReady": true,
        "detail": "/root/autodl-tmp/tools/gnina/gnina.1.3.2"
      },
      "vina": {
        "available": false,
        "runtimeReady": null,
        "detail": ""
      },
      "smina": {
        "available": true,
        "runtimeReady": null,
        "detail": "/root/autodl-tmp/conda_envs/smina/bin/smina"
      },
      "obabel": {
        "available": true,
        "runtimeReady": null,
        "detail": "/root/miniconda3/bin/obabel"
      },
      "babel": {
        "available": false,
        "runtimeReady": null,
        "detail": ""
      }
    },
    "priorityQueue": [
      "P1 LINCS/CMap expression reversal: completed and integrated; extend only with disease-subtype signatures if required.",
      "P2 GNINA CNN structural rescoring: Vina, smina, and GNINA Top100 rescoring are complete; expand only if more structural stress-test coverage is required.",
      "P2 smina structural rescoring: Top100 score-only audit is complete and should be interpreted as a classical docking stress test, not a CNN model.",
      "P2 external SOTA input package: GNINA Top100, Boltz/Chai Top50, and independent DTI Top1000 queues are input-ready; Boltz-2 and local supervised DTI have already been executed.",
      "P2 Boltz-2 second-model validation: Top50 fast spot-check and high-sampling finalist validation are complete; only Chai-1/AF3-style external corroboration remains optional.",
      "P2 optional standard ProLIF/PoseBusters expansion: Top100 expert-check is complete; expand to Top300 only if more structural examples are needed.",
      "P2 optional Vina expansion: Top100 score-only audit is complete; expand to Top300/Top500 if more stress-test coverage is needed.",
      "P2 independent DTI ensemble: local supervised DTI Top1000 is complete; formal DrugBAN/DeepDTA pretrained extensions still need dependencies/checkpoints."
    ],
    "blockers": [
      {
        "label": "LINCS/CMap perturbation signatures",
        "status": "external_data_missing",
        "detailZh": "候选药物和疾病方向已完成 readiness audit，但本地没有扰动签名和疾病 DEG/signature 文件。"
      },
      {
        "label": "DrugBAN formal pretrained inference",
        "status": "dependency_missing",
        "detailZh": "local supervised DTI 已完成；正式 DrugBAN 分支仍缺 dgl 和 dgllife 环境。"
      },
      {
        "label": "DeepDTA/GraphDTA validated checkpoints",
        "status": "checkpoint_missing",
        "detailZh": "需要可信 pretrained weights 或重新训练配置，当前不把本地模型冒充外部 pretrained SOTA。"
      },
      {
        "label": "Chai-1 / AF3-style optional corroboration",
        "status": "optional_external_model",
        "detailZh": "Boltz-2 Top50 和 high-sampling finalists 已完成；Chai-1 或 AF3-style 可作为最终候选外部结构佐证。"
      }
    ]
  },
  "finalPriority": {
    "candidateRows": 3921,
    "tierCounts": [
      {
        "label": "D",
        "value": 1763
      },
      {
        "label": "C",
        "value": 1516
      },
      {
        "label": "B",
        "value": 622
      },
      {
        "label": "A",
        "value": 20
      }
    ],
    "reviewTrackCounts": [
      {
        "label": "B_novel_pair_disease_context_review",
        "value": 1296
      },
      {
        "label": "C_safety_or_contraindication_review",
        "value": 974
      },
      {
        "label": "D_deprioritize_until_issue_resolved",
        "value": 903
      },
      {
        "label": "A_repurposing_mechanism_review",
        "value": 497
      },
      {
        "label": "C_secondary_model_priority",
        "value": 103
      },
      {
        "label": "A_positive_control_or_known_mechanism",
        "value": 98
      },
      {
        "label": "B_mechanism_review",
        "value": 50
      }
    ]
  },
  "notableModules": {
    "DiffDock disease-direction structural docking": {
      "module": "DiffDock disease-direction structural docking",
      "status": "completed_with_stable_residual_failures",
      "candidateRows": 23744,
      "completedRows": 23707,
      "missingRows": 37,
      "completionPct": 99.84417115902964,
      "evidence": "SDF and SMILES rescue rounds were completed; the remaining missing outputs are stable graph/sampling boundary cases."
    },
    "Standard PoseBusters/ProLIF Top300 structural validation extension": {
      "module": "Standard PoseBusters/ProLIF Top300 structural validation extension",
      "status": "completed_standard_tool_extension_top300",
      "candidateRows": 300,
      "structureInputReadyRows": 300,
      "posebustersPassRows": 61,
      "posebustersPassPct": 20.3333,
      "prolifOkRows": 300,
      "prolifInteractionRows": 1539,
      "standardSupportedRows": 56,
      "standardSupportedPct": 18.6667,
      "tierCounts": {
        "D_standard_pose_fail": 238,
        "A_standard_structure_supported": 8,
        "C_standard_interaction_review": 5,
        "B_standard_structure_acceptable": 48,
        "C_standard_pose_review": 1
      },
      "evidence": "The completed Top300 extension broadens the standard PoseBusters/ProLIF stress test beyond the original Top100 while preserving the Top100 result as the formal high-priority expert-check layer."
    },
    "GNINA CNN full final-candidate structural rescoring": {
      "module": "GNINA CNN full final-candidate structural rescoring",
      "status": "completed_gnina_cnn_full3921_cpu",
      "candidateRows": 2664,
      "inputReadyRows": 2583,
      "gninaScoredRows": 2583,
      "gninaScoredPct": 96.9595,
      "gninaSupportedRows": 74,
      "gninaSupportedPct": 2.7778,
      "medianGninaCnnScore": 0.1292,
      "medianGninaCnnAffinity": 4.8315,
      "statusCounts": {
        "ok": 2583,
        "not_ready": 81
      },
      "tierCounts": {
        "C_gnina_cnn_signal_pose_review": 794,
        "B_gnina_cnn_structural_support": 69,
        "A_gnina_cnn_structural_support": 5,
        "C_gnina_pose_supported_weak_cnn": 984,
        "D_gnina_low_support_review": 731,
        "D_gnina_not_scored": 81
      },
      "evidence": "The GNINA CNN structural consensus stress test has been expanded to the full final-candidate structural queue in CPU-only mode."
    },
    "Boltz-2 high-sampling finalist validation": {
      "module": "Boltz-2 high-sampling finalist validation",
      "status": "completed_boltz2_high_sampling_finalists",
      "candidateRows": 12,
      "completedRows": 12,
      "completedPct": 100.0,
      "knownRows": 4,
      "novelOrExtensionRows": 8,
      "abSupportedRows": 11,
      "abSupportedPct": 91.66666666666667,
      "medianHighConfidenceScore": 0.857646644115448,
      "medianHighLigandIptm": 0.979526549577713,
      "medianHighAffinityProbabilityBinary": 0.9136049151420593,
      "medianCompositeDeltaVsFast": 32.9983,
      "tierCounts": {
        "A_boltz_high_sampling_supported": 8,
        "B_boltz_high_sampling_supported": 3,
        "C_boltz_high_sampling_review": 1
      },
      "evidence": "Selected original-ligand Boltz-2 finalists were rerun with higher recycling, structure sampling, and affinity sampling. This is a stronger second-model structural corroboration layer for finalists, not a DiffDock rerun."
    },
    "Local supervised independent DTI full final-candidate extension": {
      "module": "Local supervised independent DTI full final-candidate extension",
      "status": "completed_local_supervised_dti_full3921",
      "candidateRows": 3921,
      "scoredRows": 3921,
      "uniqueCandidateDrugs": 117,
      "uniqueCandidateProteins": 271,
      "knownRows": 136,
      "novelRows": 3785,
      "abSupportedRows": 1176,
      "abSupportedPct": 29.992348890589135,
      "tierCounts": {
        "D_independent_dti_low_model_support": 1567,
        "C_independent_dti_model_review": 1178,
        "B_independent_dti_model_supported": 784,
        "A_independent_dti_model_supported": 392
      },
      "candidateKnownBenchmark": {
        "knownRows": 136,
        "auroc": 0.9277818789338722,
        "averagePrecision": 0.4792794820151002,
        "topK": [
          {
            "cutoff": 50,
            "rows": 50,
            "knownHits": 31,
            "precisionPct": 62.0,
            "recallPct": 22.7941,
            "enrichmentVsQueueBaseline": 17.875147
          },
          {
            "cutoff": 100,
            "rows": 100,
            "knownHits": 58,
            "precisionPct": 58.0,
            "recallPct": 42.6471,
            "enrichmentVsQueueBaseline": 16.721912
          },
          {
            "cutoff": 300,
            "rows": 300,
            "knownHits": 85,
            "precisionPct": 28.3333,
            "recallPct": 62.5,
            "enrichmentVsQueueBaseline": 8.16875
          },
          {
            "cutoff": 500,
            "rows": 500,
            "knownHits": 98,
            "precisionPct": 19.6,
            "recallPct": 72.0588,
            "enrichmentVsQueueBaseline": 5.650853
          },
          {
            "cutoff": 1000,
            "rows": 1000,
            "knownHits": 130,
            "precisionPct": 13.0,
            "recallPct": 95.5882,
            "enrichmentVsQueueBaseline": 3.748015
          }
        ]
      },
      "validationMetrics": [
        {
          "split": "pair_stratified_holdout",
          "status": "ok",
          "trainRows": 5595,
          "testRows": 1866,
          "testPositives": 207,
          "testNegatives": 1659,
          "auroc": 0.7779932617577087,
          "averagePrecision": 0.6542721573896474,
          "positiveScoreMedian": 0.4109449939592624,
          "negativeScoreMedian": 0.17291740913116782
        },
        {
          "split": "drug_group_holdout",
          "status": "ok",
          "trainRows": 5503,
          "testRows": 1958,
          "testPositives": 229,
          "testNegatives": 1729,
          "auroc": 0.6961289687099846,
          "averagePrecision": 0.4974029826322436,
          "positiveScoreMedian": 0.25736816694359155,
          "negativeScoreMedian": 0.16616528293094054
        },
        {
          "split": "target_group_holdout",
          "status": "ok",
          "trainRows": 5477,
          "testRows": 1984,
          "testPositives": 235,
          "testNegatives": 1749,
          "auroc": 0.4895758062357821,
          "averagePrecision": 0.29096630015348923,
          "positiveScoreMedian": 0.16619598447096406,
          "negativeScoreMedian": 0.1778239691348913
        }
      ],
      "evidence": "The same local supervised DTI protocol was expanded from the prepared Top1000 queue to the full 3921 final-priority candidate matrix, preserving exact FDA-positive candidate-pair exclusion from training."
    },
    "Experimental validation planning panel": {
      "module": "Experimental validation planning panel",
      "status": "completed_assay_planning_triage",
      "candidateRows": 3921,
      "experimentReadyRows": 885,
      "reviewReadyRows": 763,
      "novelExperimentOrReviewReadyRows": 1108,
      "balancedPanelRows": 300,
      "balancedPanelUniqueDrugs": 61,
      "balancedPanelUniqueTargets": 120,
      "balancedPanelUniqueScaffolds": 57,
      "evidence": "Final candidates were converted into assay-planning queues with transparent validation score, disease-balanced shortlist, novelty shortlist, positive controls, risk-review queue, and assay modality labels."
    },
    "FDA label temporal generalization audit": {
      "module": "FDA label temporal generalization audit",
      "status": "completed_retrospective_time_sliced_qc",
      "candidateRows": 3921,
      "exactLabelRows": 137,
      "exactLabel2016PlusRows": 34,
      "exactLabel2021PlusRows": 14,
      "targetContext2021PlusRows": 1332,
      "top100Exact2016PlusRows": 1,
      "top100Exact2021PlusRows": 0,
      "top300Exact2016PlusRows": 4,
      "top300Exact2021PlusRows": 0,
      "balancedExact2021PlusRows": 2,
      "wave1Exact2021PlusRows": 1,
      "evidence": "FDA label-target rows were stratified by approval-year era to stress-test whether the validation ranking and assay queues recover modern and recent label mechanisms; this is a retrospective time-sliced audit, not a true prospective deployment test."
    }
  }
};
