#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="outputs/gnina_calibration_338_v1"
STATE="$OUT/GNINA_AUTOMATIC_PIPELINE_STATE_V1.json"
RUN_SUMMARY="$OUT/execution_inputs_prepared_v1/GNINA_CALIBRATION_RUN_SUMMARY_FULL_V1.json"

write_state() {
  local stage="$1"
  local status="$2"
  local message="$3"
  python - "$STATE" "$stage" "$status" "$message" <<'PY'
import datetime
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "updated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "stage": sys.argv[2],
    "status": sys.argv[3],
    "message": sys.argv[4],
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

on_error() {
  local exit_code=$?
  write_state "FAILED" "BLOCKED" "Automatic pipeline stopped with exit code ${exit_code}; inspect run_logs/auto_advance.log."
  exit "$exit_code"
}
trap on_error ERR

write_state "CALIBRATION_DOCKING" "RUNNING" "Waiting for the 269-target GNINA calibration runner."
while pgrep -f '[r]un_gnina_calibration_338_v1.py.*--run-label full' >/dev/null; do
  sleep 120
done

if [[ ! -s "$RUN_SUMMARY" ]]; then
  write_state "CALIBRATION_RECOVERY" "RUNNING" "Full-run summary is absent; starting the resumable recovery run."
  python scripts/run_gnina_calibration_338_v1.py \
    --gpus 0,1,2,3 \
    --jobs-per-gpu 2 \
    --cpus-per-job 8 \
    --run-label full
fi

write_state "POSTPROCESS" "RUNNING" "Auditing outputs, evaluating Gate-A/Gate-B, and mapping qualification to the frozen pair space."
bash scripts/run_gnina_calibration_postprocess_v1.sh

python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/gnina_calibration_338_v1/evaluation")
required = [
    root / "GNINA_CALIBRATION_RUN_INTEGRITY_SUMMARY_FULL_V1.json",
    root / "GNINA_CALIBRATION_EVALUATION_SUMMARY_FULL_V1.json",
    root / "GNINA_TARGET_QUALIFICATION_338_FULL_V1.csv",
    root / "GNINA_QUALIFICATION_MAPPED_FDA243360_FULL_V1.csv.gz",
    root / "GNINA_POSTCALIBRATION_DTA_REVIEW_QUEUE_FULL_V1.csv.gz",
    root / "GNINA_POSTCALIBRATION_SCOPE_SUMMARY_FULL_V1.json",
]
missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
if missing:
    raise RuntimeError(f"Missing post-calibration outputs: {missing}")
integrity = json.loads(required[0].read_text(encoding="utf-8"))
evaluation = json.loads(required[1].read_text(encoding="utf-8"))
if integrity.get("status") != "PASS":
    raise RuntimeError(f"Integrity audit did not pass: {integrity}")
if evaluation.get("partial_evaluation"):
    raise RuntimeError(f"Evaluation is unexpectedly partial: {evaluation}")
PY

write_state "QUALIFIED_REVIEW_QUEUE_READY" "COMPLETED" "Full calibration, target qualification, and the post-calibration DTA review queue are complete. Candidate novelty thresholds must be frozen before unknown-pair GNINA docking."
trap - ERR
echo "Automatic GNINA calibration pipeline completed."
