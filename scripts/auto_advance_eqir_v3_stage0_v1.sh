#!/usr/bin/env bash
set -euo pipefail

task_root="/root/autodl-tmp/BioMaster"
jobs_rel="outputs/old_drug_target_sota_v1/v3_label_blind_structure_audit_v1/small_molecule_execution_inputs_v1/V3_LABEL_BLIND_GNINA_ALL_READY_JOBS_V1.csv"
queue_dir_rel="outputs/old_drug_target_sota_v1/v3_label_blind_structure_audit_v1/small_molecule_execution_inputs_v1"
run_summary_rel="${queue_dir_rel}/GNINA_CALIBRATION_RUN_SUMMARY_V3_STAGE0_SMALL_MOLECULE_FULL8_V1.json"
manifest_rel="outputs/old_drug_target_sota_v1/v3_label_blind_structure_audit_v1/small_molecule_applicability_v1/V3_LABEL_BLIND_GNINA_EFFECTIVE_MANIFEST_V1.csv.gz"
ledger_rel="outputs/old_drug_target_sota_v1/v3_label_blind_structure_audit_v1/small_molecule_applicability_v1/V3_LABEL_BLIND_EFFECTIVE_EVALUATION_LEDGER_V1.csv.gz"
evaluation_dir_rel="outputs/old_drug_target_sota_v1/v3_label_blind_structure_audit_v1/small_molecule_applicability_v1/evaluation_v1"
stage0_dir_rel="outputs/old_drug_target_sota_v1/biomaster_eqir_v3_stage0_small_molecule_v1"
runner_pattern="run_gnina_calibration_338_v1.py.*v3_stage0_small_molecule_full8"

cd "${task_root}"

while [[ ! -f "${run_summary_rel}" ]]; do
  if ! pgrep -f "${runner_pattern}" >/dev/null; then
    sleep 2
    if [[ ! -f "${run_summary_rel}" ]]; then
      echo "GNINA runner stopped without a formal run summary; Stage 0 remains locked." >&2
      exit 2
    fi
  fi
  sleep 30
done

python - "${run_summary_rel}" <<'PY'
import json
import sys

path = sys.argv[1]
summary = json.load(open(path, encoding="utf-8"))
required = {
    "requested_targets": 289,
    "completed_targets": 289,
    "failed_targets": 0,
    "requested_controls": 17241,
    "completed_controls": 17241,
}
observed = {key: summary.get(key) for key in required}
if observed != required:
    raise SystemExit(
        "Formal GNINA queue is not complete; Stage 0 remains locked: "
        + json.dumps({"required": required, "observed": observed}, sort_keys=True)
    )
print(json.dumps({"queue_gate": "PASS", **observed}, sort_keys=True))
PY

python scripts/extract_v3_label_blind_gnina_scores.py \
  --manifest "${manifest_rel}" \
  --jobs "${jobs_rel}" \
  --output-dir "${evaluation_dir_rel}"

python - "${evaluation_dir_rel}/V3_LABEL_BLIND_GNINA_EXTRACTION_SUMMARY_V1.json" <<'PY'
import json
import sys

path = sys.argv[1]
summary = json.load(open(path, encoding="utf-8"))
required = {
    "status": "PASS",
    "requested_targets": 289,
    "completed_targets": 289,
    "requested_pairs": 17241,
    "completed_pairs": 17241,
    "partial": False,
    "labels_used": False,
    "cross_target_raw_score_comparison_performed": False,
    "target_local_ecdf_only": True,
}
observed = {key: summary.get(key) for key in required}
if observed != required:
    raise SystemExit(
        "Formal label-blind extraction failed its completeness gate: "
        + json.dumps({"required": required, "observed": observed}, sort_keys=True)
    )
print(json.dumps({"extraction_gate": "PASS", **observed}, sort_keys=True))
PY

python scripts/evaluate_biomaster_eqir_v3_stage0.py \
  --iterations 2000 \
  --permutations 200 \
  --scores "${evaluation_dir_rel}/V3_LABEL_BLIND_GNINA_PAIR_SCORES_V1.csv.gz" \
  --extraction-summary "${evaluation_dir_rel}/V3_LABEL_BLIND_GNINA_EXTRACTION_SUMMARY_V1.json" \
  --ledger "${ledger_rel}" \
  --prediction-root outputs/old_drug_target_sota_v1/biomaster_odti_routed_ranker_v1 \
  --output-dir "${stage0_dir_rel}"

python - "${stage0_dir_rel}/EQIR_V3_STAGE0_SUMMARY_V1.json" <<'PY'
import json
import sys

path = sys.argv[1]
summary = json.load(open(path, encoding="utf-8"))
print(
    json.dumps(
        {
            "stage0_finished": True,
            "decision": summary["decision"],
            "algorithm_innovation_claim": summary["algorithm_innovation_claim"],
        },
        sort_keys=True,
    )
)
PY
