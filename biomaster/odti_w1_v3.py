"""Safe semantic adapter for prospective W1 observations into ODTI V3 rows.

The operator-facing W1 templates deliberately contain assay/QC fields and
remain blinded.  This module accepts only a post-assay semantic table whose
labels and provenance have been explicitly entered.  It never infers a
negative label from a missing or pending observation.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Any, Iterable, Mapping


W1_V3_FIELDS: tuple[str, ...] = (
    "w1_candidate_id",
    "prospective_pair_id",
    "calibration_pair_id",
    "training_store_overlap_status",
    "drug_entity_key",
    "target_entity_key",
    "assay_id",
    "assay_lane",
    "plate_id",
    "blinded_sample_code",
    "independent_run",
    "replicate_id",
    "readout_value",
    "readout_unit",
    "activity_class",
    "censor_lower",
    "censor_upper",
    "assay_status",
    "replicate_variance",
    "assay_metadata_json",
    "raw_data_filename",
    "raw_data_file_sha256",
    "unblinding_status",
)

ALLOWED_ACTIVITY_CLASSES = {
    "active",
    "inactive",
    "borderline",
    "failed",
    "not_interpretable",
}
ALLOWED_ASSAY_STATUSES = {
    "PASS",
    "FAILED",
    "FAILED_QC",
    "INCONCLUSIVE",
    "NOT_INTERPRETABLE",
}
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


class W1V3AdapterError(ValueError):
    """Raised when a W1 semantic table is not safe for V3 ingestion."""

    def __init__(self, errors: list[dict[str, Any]]) -> None:
        self.errors = errors
        message = "; ".join(
            f"row {item.get('row')}: {item.get('code')} - {item.get('message')}"
            for item in errors[:8]
        )
        if len(errors) > 8:
            message += f"; ... and {len(errors) - 8} more"
        super().__init__(message or "W1 V3 adapter rejected input")


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _finite_float(value: object) -> float | None:
    text = _text(value)
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_pending(value: object) -> bool:
    return _text(value).upper() in {"", "PENDING", "TEMPLATE_PENDING", "LOCKED_NOT_UNBLINDED"}


def _normalize_row(row: Mapping[str, object]) -> dict[str, str]:
    return {field: _text(row.get(field)) for field in W1_V3_FIELDS}


def validate_w1_v3_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    require_two_independent_runs: bool = True,
    require_controlled_unblinding: bool = True,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Validate and normalize explicit W1 semantic rows.

    The function is intentionally strict.  It expects a separate semantic
    table, not the frozen V17 operator template.  A valid return value is safe
    to write as a V3 observation append-only table; it is not yet a training
    split or a model result.
    """

    normalized = [_normalize_row(row) for row in rows]
    errors: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    by_pair: dict[str, list[dict[str, str]]] = defaultdict(list)

    for index, row in enumerate(normalized, start=1):
        def error(code: str, message: str) -> None:
            errors.append({"row": index, "code": code, "message": message})

        missing_identity = [
            field
            for field in (
                "w1_candidate_id",
                "prospective_pair_id",
                "training_store_overlap_status",
                "drug_entity_key",
                "target_entity_key",
                "assay_id",
                "assay_lane",
                "plate_id",
                "blinded_sample_code",
                "independent_run",
                "replicate_id",
                "readout_unit",
                "assay_status",
                "assay_metadata_json",
                "raw_data_filename",
                "raw_data_file_sha256",
            )
            if not row[field]
        ]
        if missing_identity:
            error("MISSING_REQUIRED_FIELD", ", ".join(missing_identity))
        if row["activity_class"].lower() not in ALLOWED_ACTIVITY_CLASSES:
            error(
                "INVALID_ACTIVITY_CLASS",
                f"expected one of {sorted(ALLOWED_ACTIVITY_CLASSES)}, got {row['activity_class']!r}",
            )
        if row["assay_status"].upper() not in ALLOWED_ASSAY_STATUSES:
            error(
                "INVALID_ASSAY_STATUS",
                f"expected one of {sorted(ALLOWED_ASSAY_STATUSES)}, got {row['assay_status']!r}",
            )
        if _is_pending(row["activity_class"]) or _is_pending(row["assay_status"]):
            error("PENDING_OR_LOCKED_SEMANTICS", "semantic labels are still pending or blinded")
        if require_controlled_unblinding and row["unblinding_status"].upper() not in {
            "CONTROLLED_UNBLINDED",
            "CONTROLLED_KEY_RELEASED",
        }:
            error("NOT_CONTROLLED_UNBLINDED", "V3 ingestion requires a controlled post-assay unblinding status")
        if not HEX64.fullmatch(row["raw_data_file_sha256"]):
            error("INVALID_RAW_DATA_HASH", "raw_data_file_sha256 must be a 64-character SHA-256")
        try:
            metadata = json.loads(row["assay_metadata_json"])
            if not isinstance(metadata, dict):
                raise ValueError("metadata is not an object")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            error("INVALID_ASSAY_METADATA_JSON", str(exc))

        readout = _finite_float(row["readout_value"])
        lower = _finite_float(row["censor_lower"])
        upper = _finite_float(row["censor_upper"])
        variance = _finite_float(row["replicate_variance"])
        if row["readout_value"] and readout is None:
            error("INVALID_READOUT_VALUE", "readout_value must be finite numeric")
        if row["censor_lower"] and lower is None:
            error("INVALID_CENSOR_LOWER", "censor_lower must be finite numeric")
        if row["censor_upper"] and upper is None:
            error("INVALID_CENSOR_UPPER", "censor_upper must be finite numeric")
        if row["replicate_variance"] and (variance is None or variance < 0):
            error("INVALID_REPLICATE_VARIANCE", "replicate_variance must be finite and non-negative")
        if lower is not None and upper is not None and lower > upper:
            error("INVALID_CENSOR_INTERVAL", "censor_lower cannot exceed censor_upper")

        activity = row["activity_class"].lower()
        if activity in {"active", "inactive", "borderline"} and readout is None and lower is None and upper is None:
            error("MISSING_OBSERVATION_VALUE", "active/inactive/borderline requires readout or censor bound")
        if activity in {"failed", "not_interpretable"} and row["assay_status"].upper() == "PASS":
            error("FAILED_STATUS_CONFLICT", "failed/not_interpretable cannot have assay_status PASS")

        pair_id = row["prospective_pair_id"]
        if not row["calibration_pair_id"] and not pair_id:
            error("MISSING_PROSPECTIVE_PAIR_ID", "unseen prospective row requires prospective_pair_id")
        key = (pair_id, row["independent_run"], row["replicate_id"])
        if key in seen_keys:
            error("DUPLICATE_REPLICATE_KEY", f"duplicate key {key}")
        seen_keys.add(key)
        by_pair[pair_id].append(row)

    pair_run_counts: dict[str, int] = {}
    pair_class_sets: dict[str, list[str]] = {}
    for pair_id, pair_rows in by_pair.items():
        runs = {row["independent_run"] for row in pair_rows}
        pair_run_counts[pair_id] = len(runs)
        pair_class_sets[pair_id] = sorted({row["activity_class"].lower() for row in pair_rows})
        if require_two_independent_runs and len(runs) < 2:
            errors.append({
                "row": pair_id,
                "code": "INCOMPLETE_INDEPENDENT_RUNS",
                "message": "at least two independent runs are required before W1 V3 ingestion",
            })

    if errors:
        raise W1V3AdapterError(errors)

    summary = {
        "rows": len(normalized),
        "prospective_pairs": len(by_pair),
        "independent_runs_per_pair": pair_run_counts,
        "activity_classes_by_pair": pair_class_sets,
        "unknown_pair_is_negative": False,
        "w1_test_reuse_forbidden": True,
        "claim_status": "SEMANTIC_W1_APPEND_ONLY_ROWS; NOT_YET_TRAINED_OR_EVALUATED",
    }
    return normalized, summary

def adapt_w1_v3_rows(rows: Iterable[Mapping[str, object]], **kwargs: Any) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Public alias emphasizing that this adapter performs no label inference."""

    return validate_w1_v3_rows(rows, **kwargs)
