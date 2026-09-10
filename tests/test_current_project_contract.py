from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validate_current_project_contract.py"
CONTRACT = ROOT / "configs/biomaster_current_contract_v1.json"


def load_validator():
    spec = importlib.util.spec_from_file_location("validate_current_project_contract", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_contract_internal_consistency() -> None:
    validator = load_validator()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    checks = validator.validate_contract_only(contract)
    assert "current_scored_core_arithmetic" in checks
    assert "registry_arithmetic" in checks
    assert "kirhub_endpoint_boundary" in checks


def test_current_contract_local_audit_artifacts() -> None:
    validator = load_validator()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    checks = validator.validate_artifacts(contract)
    assert "full_fit_training_counts" in checks
    assert "target_routing_counts" in checks
    assert "s5_metrics" in checks
    assert "kirhub_retrospective_metrics" in checks
