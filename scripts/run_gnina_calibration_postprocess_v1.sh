#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

run_summary="outputs/gnina_calibration_338_v1/execution_inputs_prepared_v1/GNINA_CALIBRATION_RUN_SUMMARY_FULL_V1.json"
if ! python -c 'import json,sys; s=json.load(open(sys.argv[1])); raise SystemExit(0 if s["failed_targets"] == 0 else 1)' "$run_summary"; then
  python scripts/run_gnina_calibration_338_v1.py \
    --gpus 0,1,2,3 \
    --jobs-per-gpu 2 \
    --cpus-per-job 8 \
    --run-label retry1
fi

python scripts/audit_gnina_calibration_run_v1.py
python scripts/evaluate_gnina_calibration_338_v1.py \
  --bootstrap 2000 \
  --permutations 5000 \
  --label full
python scripts/build_gnina_target_qualification_and_scope_v1.py

echo "GNINA calibration postprocessing completed."
