#!/usr/bin/env python3
"""Build a blank, provenance-preserving W1→V3 semantic input template.

The generated CSV is intentionally not V3 training-ready.  It contains the
frozen identity/bridge fields and blank semantic fields that must be filled by
the assay team after controlled unblinding and QC.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biomaster.odti_w1_v3 import W1_V3_FIELDS  # noqa: E402


BRIDGE = ROOT / (
    "outputs/biomaster_odti_w1_v3_bridge_v1/"
    "W1_V17_TO_ODTI_V3_BRIDGE_MANIFEST_V1.csv"
)
OUT_DIR = ROOT / "outputs/biomaster_odti_w1_v3_semantic_input_v1"
OUT_CSV = OUT_DIR / "W1_V3_SEMANTIC_RESULT_INPUT_16_V1.csv"
OUT_JSON = OUT_DIR / "W1_V3_SEMANTIC_RESULT_INPUT_TEMPLATE_V1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not BRIDGE.is_file():
        raise FileNotFoundError(BRIDGE)
    with BRIDGE.open(newline="", encoding="utf-8") as handle:
        bridge_rows = list(csv.DictReader(handle))
    if len(bridge_rows) != 16:
        raise ValueError(f"expected 16 W1 bridge rows, found {len(bridge_rows)}")

    rows: list[dict[str, str]] = []
    for source in bridge_rows:
        row = {field: "" for field in W1_V3_FIELDS}
        for field in (
            "w1_candidate_id",
            "prospective_pair_id",
            "calibration_pair_id",
            "training_store_overlap_status",
            "drug_entity_key",
            "target_entity_key",
            "assay_lane",
            "plate_id",
            "blinded_sample_code",
            "independent_run",
        ):
            row[field] = source.get(field, "")
        # These values document the current pre-assay state.  They are not
        # valid semantic labels and are deliberately rejected by the adapter.
        row["activity_class"] = "PENDING"
        row["assay_status"] = "PENDING"
        row["unblinding_status"] = source.get("unblinding_status", "LOCKED_NOT_UNBLINDED")
        rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(W1_V3_FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "status": "PASS_TEMPLATE_ONLY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "BIOMASTER_ODTI_W1_V3_SEMANTIC_INPUT_TEMPLATE_V1",
        "claim_status": "TEMPLATE_ONLY; NO_W1_READOUT_AND_NO_V3_TRAINING_ROWS",
        "source_bridge": str(BRIDGE.relative_to(ROOT)),
        "source_bridge_sha256": sha256(BRIDGE),
        "output_csv": str(OUT_CSV.relative_to(ROOT)),
        "output_csv_sha256": sha256(OUT_CSV),
        "rows": len(rows),
        "fields": list(W1_V3_FIELDS),
        "semantic_fields_left_blank_until_real_readout": [
            "assay_id",
            "replicate_id",
            "readout_value",
            "readout_unit",
            "censor_lower",
            "censor_upper",
            "replicate_variance",
            "assay_metadata_json",
            "raw_data_filename",
            "raw_data_file_sha256",
        ],
        "explicit_rejections": [
            "PENDING activity_class",
            "PENDING assay_status",
            "LOCKED_NOT_UNBLINDED unblinding_status",
            "missing raw-data and replicate provenance",
        ],
        "unknown_pair_is_negative": False,
        "do_not_use_as_training_input": True,
    }
    OUT_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
