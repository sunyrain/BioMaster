from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path
from typing import Any


DIRECTIONS = [
    "oncology",
    "cardiovascular",
    "infectious_disease",
    "neurology_psychiatry",
    "immunology_inflammation",
]


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def import_status(module: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module)
    return {"available": bool(spec), "detail": str(spec.origin or "") if spec else ""}


def gnina_runtime_env() -> tuple[dict[str, str], str]:
    lib_paths = [
        "/root/miniconda3/lib/python3.12/site-packages/nvidia/cudnn/lib",
        "/usr/local/cuda/lib64",
    ]
    existing = [path for path in lib_paths if Path(path).exists()]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    value = ":".join(existing + ([current] if current else []))
    env = os.environ.copy()
    if value:
        env["LD_LIBRARY_PATH"] = value
    return env, value


def binary_status(name: str, candidates: list[Path], min_size_mb: int = 1) -> dict[str, Any]:
    path = shutil.which(name)
    checked: list[str] = []
    min_size = min_size_mb * 1024 * 1024
    if path:
        candidate = Path(path)
        checked.append(str(candidate))
        size = candidate.stat().st_size if candidate.exists() else None
        executable = os.access(candidate, os.X_OK)
        return {
            "available": bool(size and size >= min_size and executable),
            "path": str(candidate),
            "sizeBytes": size,
            "minSizeBytes": min_size,
            "executable": executable,
            "checked": checked,
        }
    for candidate in candidates:
        checked.append(str(candidate))
        if candidate.exists() and candidate.stat().st_size >= min_size and os.access(candidate, os.X_OK):
            return {
                "available": True,
                "path": str(candidate),
                "sizeBytes": candidate.stat().st_size,
                "minSizeBytes": min_size,
                "executable": os.access(candidate, os.X_OK),
                "checked": checked,
            }
    return {"available": False, "path": "", "sizeBytes": None, "minSizeBytes": min_size, "executable": False, "checked": checked}


def gnina_status(candidates: list[Path], min_size_mb: int = 1000) -> dict[str, Any]:
    status = binary_status("gnina", candidates, min_size_mb=min_size_mb)
    if not status["available"]:
        return {**status, "runtimeReady": False, "version": "", "runtimeError": "binary_not_available", "ldLibraryPath": ""}

    env, ld_path = gnina_runtime_env()
    try:
        result = subprocess.run(
            [status["path"], "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
            env=env,
        )
    except Exception as exc:  # noqa: BLE001 - preserve runtime evidence for audit.
        return {
            **status,
            "available": False,
            "runtimeReady": False,
            "version": "",
            "runtimeError": f"{type(exc).__name__}: {exc}",
            "ldLibraryPath": ld_path,
        }
    version = (result.stdout or result.stderr or "").strip().splitlines()[0] if (result.stdout or result.stderr) else ""
    runtime_ready = result.returncode == 0
    return {
        **status,
        "available": runtime_ready,
        "runtimeReady": runtime_ready,
        "version": version,
        "runtimeError": "" if runtime_ready else (result.stderr or result.stdout or "")[:500],
        "ldLibraryPath": ld_path,
    }


def matching_files(root: Path, patterns: list[str], max_files: int = 40) -> list[Path]:
    matches: list[Path] = []
    temporary_suffixes = (".part", ".tmp", ".download", ".crdownload")
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file() and not any(path.name.endswith(suffix) for suffix in temporary_suffixes):
                matches.append(path)
            if len(matches) >= max_files:
                return sorted(set(matches))
    return sorted(set(matches))


def checkpoint_files(paths: list[Path]) -> list[Path]:
    patterns = ["**/*.pt", "**/*.pth", "**/*.ckpt", "**/*.h5", "**/*.keras", "**/*.pkl", "**/*.pickle"]
    found: list[Path] = []
    for path in paths:
        if path.exists():
            found.extend(matching_files(path, patterns, 80))
    return sorted(set(found))


def file_names(files: list[Path], root: Path, max_items: int = 20) -> list[str]:
    return [rel(path, root) for path in files[:max_items]]


def status_row(
    layer: str,
    priority: str,
    status: str,
    ready: bool,
    expected_path: str,
    evidence: str,
    next_action: str,
) -> dict[str, Any]:
    return {
        "layer": layer,
        "priority": priority,
        "status": status,
        "ready": bool(ready),
        "expectedPath": expected_path,
        "evidence": evidence,
        "nextAction": next_action,
    }


def build_audit(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    autodl = Path("/root/autodl-tmp")
    gnina_candidates = [
        autodl / "tools/gnina/gnina.1.3.2",
        autodl / "tools/gnina/gnina.1.3.2.cuda12.8",
        root / "tools/gnina/gnina",
        root / "tools/gnina/gnina.1.3.2",
    ]
    gnina = gnina_status(gnina_candidates, min_size_mb=1000)

    lincs_dir = root / "data/external/lincs_cmap"
    lincs_matrix = matching_files(
        lincs_dir,
        ["**/*.gctx", "**/*.gct", "**/*level5*", "**/*Level5*", "**/*LEVEL5*", "**/*signature*"],
    )
    siginfo = matching_files(lincs_dir, ["**/*siginfo*", "**/*signature_info*"])
    geneinfo = matching_files(lincs_dir, ["**/*geneinfo*", "**/*gene_info*"])
    compoundinfo = matching_files(lincs_dir, ["**/*compoundinfo*", "**/*pertinfo*", "**/*compound_info*"])
    lincs_ready = bool(lincs_matrix and siginfo and geneinfo and compoundinfo)

    disease_root = root / "data/external/disease_signatures"
    direction_files: dict[str, list[Path]] = {}
    for direction in DIRECTIONS:
        direction_dir = disease_root / direction
        direction_files[direction] = matching_files(
            direction_dir,
            ["**/*.csv", "**/*.tsv", "**/*.txt", "**/*.gmt", "**/*.json"],
            20,
        )
    disease_ready = all(direction_files[direction] for direction in DIRECTIONS)

    drugban_code = root / "third_party/DrugBAN"
    drugban_models = root / "models/drugban"
    drugban_ckpt = checkpoint_files([drugban_models, drugban_code])
    dgl = import_status("dgl")
    dgllife = import_status("dgllife")

    deepdta_code = root / "third_party/DeepDTA"
    deepdta_models = root / "models/deepdta"
    deepdta_ckpt = checkpoint_files([deepdta_models, deepdta_code])
    tensorflow = import_status("tensorflow")
    keras = import_status("keras")

    graphdta_code = root / "third_party/GraphDTA"
    graphdta_models = root / "models/graphdta"
    graphdta_ckpt = checkpoint_files([graphdta_models, graphdta_code])

    chai_downloads = Path(os.environ.get("CHAI_DOWNLOADS_DIR", str(autodl / "chai_downloads")))
    chai_weights = matching_files(chai_downloads, ["**/*.pt", "**/*.safetensors", "**/*.bin", "**/*.ckpt"], 80)
    chai_lab = import_status("chai_lab")

    string_dir = root / "data/external/string"
    biogrid_dir = root / "data/external/biogrid"
    huri_dir = root / "data/external/huri"
    string_files = matching_files(string_dir, ["**/*links*", "**/*protein*info*", "**/*aliases*", "**/*.gz", "**/*.txt"], 60)
    biogrid_files = matching_files(biogrid_dir, ["**/*.zip", "**/*.tab*", "**/*.txt", "**/*.tsv"], 60)
    huri_files = matching_files(huri_dir, ["**/*.tsv", "**/*.txt", "**/*.csv"], 60)
    network_ready = bool(string_files and (biogrid_files or huri_files))

    depmap_dir = root / "data/external/depmap"
    depmap_files = matching_files(depmap_dir, ["**/*CRISPR*", "**/*Achilles*", "**/*OmicsExpression*", "**/*.csv"], 60)
    depmap_ready = bool(depmap_files)

    rows = [
        status_row(
            "GNINA CNN structural rescoring",
            "P0",
            "ready" if gnina["available"] else "missing_binary",
            bool(gnina["available"]),
            "/root/autodl-tmp/tools/gnina/gnina.1.3.2",
            f"binary={gnina}",
            "GNINA binary is available and runtime-ready; no download is required. Extend rescoring only if a broader structural stress test is requested."
            if gnina["available"]
            else "Install a compatible GNINA binary, then rerun this audit with the recorded LD_LIBRARY_PATH.",
        ),
        status_row(
            "LINCS/CMap compound perturbation signatures",
            "P0",
            "ready" if lincs_ready else "missing_lincs_matrix_or_metadata",
            lincs_ready,
            "data/external/lincs_cmap/",
            f"matrix={file_names(lincs_matrix, root)}; siginfo={file_names(siginfo, root)}; geneinfo={file_names(geneinfo, root)}; compoundinfo={file_names(compoundinfo, root)}",
            "LINCS/CMap Level 5 signatures and metadata are available; no download is required. Extend only with disease-subtype signatures if needed."
            if lincs_ready
            else "Download CLUE/LINCS Level 5 compound signatures plus siginfo/geneinfo/compoundinfo metadata.",
        ),
        status_row(
            "Disease expression signatures",
            "P0",
            "ready" if disease_ready else "missing_direction_signatures",
            disease_ready,
            "data/external/disease_signatures/<direction>/",
            "; ".join(f"{direction}={file_names(files, root, 5)}" for direction, files in direction_files.items()),
            "Per-direction disease signatures are available; no download is required. Extend only with narrower disease subtypes if requested."
            if disease_ready
            else "Add per-direction DEG/signature files with gene, logFC, pvalue/padj or explicit up/down gene lists.",
        ),
        status_row(
            "DrugBAN formal DTI model",
            "P1",
            "ready" if drugban_code.exists() and dgl["available"] and dgllife["available"] and drugban_ckpt else "missing_dependencies_or_checkpoint",
            bool(drugban_code.exists() and dgl["available"] and dgllife["available"] and drugban_ckpt),
            "third_party/DrugBAN/ and models/drugban/",
            f"code={drugban_code.exists()}; dgl={dgl}; dgllife={dgllife}; checkpoints={file_names(drugban_ckpt, root)}",
            "Install/containerize DGL and DGLLife, then add validated DrugBAN weights or retrain reproducibly.",
        ),
        status_row(
            "DeepDTA formal DTI model",
            "P1",
            "ready" if deepdta_code.exists() and deepdta_ckpt and (tensorflow["available"] or keras["available"]) else "missing_runtime_or_checkpoint",
            bool(deepdta_code.exists() and deepdta_ckpt and (tensorflow["available"] or keras["available"])),
            "third_party/DeepDTA/ and models/deepdta/",
            f"code={deepdta_code.exists()}; tensorflow={tensorflow}; keras={keras}; checkpoints={file_names(deepdta_ckpt, root)}",
            "Add validated DeepDTA weights or retrain on Davis/KIBA in an isolated legacy TensorFlow/Keras environment.",
        ),
        status_row(
            "GraphDTA optional DTI model",
            "P1",
            "ready" if graphdta_code.exists() and graphdta_ckpt else "missing_code_or_checkpoint",
            bool(graphdta_code.exists() and graphdta_ckpt),
            "third_party/GraphDTA/ and models/graphdta/",
            f"code={graphdta_code.exists()}; checkpoints={file_names(graphdta_ckpt, root)}",
            "Add GraphDTA code and validated Davis/KIBA weights if a third DTI model is needed.",
        ),
        status_row(
            "Chai-1 optional second complex model",
            "P2",
            "ready" if chai_lab["available"] and chai_weights else "missing_package_or_weights",
            bool(chai_lab["available"] and chai_weights),
            "/root/autodl-tmp/chai_downloads/ and Python env with chai_lab",
            f"chai_lab={chai_lab}; downloads={chai_downloads}; weights={file_names(chai_weights, root)}",
            "Install chai_lab in a separate environment and set CHAI_DOWNLOADS_DIR=/root/autodl-tmp/chai_downloads before running finalist complexes.",
        ),
        status_row(
            "Expanded PPI/network medicine data",
            "P2",
            "ready" if network_ready else "partial_or_missing_network_sources",
            network_ready,
            "data/external/string/, data/external/biogrid/, data/external/huri/",
            f"STRING={file_names(string_files, root)}; BioGRID={file_names(biogrid_files, root)}; HuRI={file_names(huri_files, root)}",
            "STRING plus HuRI/BioGRID network evidence is available; no required download remains. BioGRID can still be added as an optional coverage expansion."
            if network_ready
            else "Download STRING human physical links/protein info/aliases and optionally BioGRID or HuRI for broader network coverage.",
        ),
        status_row(
            "Expanded DepMap/CCLE context",
            "P2",
            "ready" if depmap_ready else "optional_raw_files_missing_or_not_indexed",
            depmap_ready,
            "data/external/depmap/",
            f"files={file_names(depmap_files, root)}",
            "DepMap dependency files are available; no required download remains. Add CCLE expression/copy-number files only for deeper oncology context."
            if depmap_ready
            else "Optional: add raw DepMap/CCLE releases if deeper oncology dependency or expression context is needed.",
        ),
    ]

    summary = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "readyCount": sum(1 for row in rows if row["ready"]),
        "missingCount": sum(1 for row in rows if not row["ready"]),
        "priorityMissing": {
            priority: [row["layer"] for row in rows if row["priority"] == priority and not row["ready"]]
            for priority in ["P0", "P1", "P2"]
        },
        "rows": rows,
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["layer", "priority", "status", "ready", "expectedPath", "evidence", "nextAction"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# SOTA External Dependency Audit",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        f"- Ready layers: {summary['readyCount']}",
        f"- Missing or partial layers: {summary['missingCount']}",
        "",
        "## Priority Missing",
        "",
    ]
    for priority, layers in summary["priorityMissing"].items():
        lines.append(f"- {priority}: {layers}")
    lines.extend(["", "## Layer Details", ""])
    for row in summary["rows"]:
        lines.append(f"### {row['layer']}")
        lines.append(f"- Priority: {row['priority']}")
        lines.append(f"- Status: {row['status']}")
        lines.append(f"- Ready: {row['ready']}")
        lines.append(f"- Expected path: `{row['expectedPath']}`")
        lines.append(f"- Evidence: {row['evidence']}")
        lines.append(f"- Next action: {row['nextAction']}")
        lines.append("")
    return "\n".join(lines)


def layer(summary: dict[str, Any], name: str) -> dict[str, Any]:
    for row in summary["rows"]:
        if row["layer"] == name:
            return row
    return {}


def status_label(row: dict[str, Any]) -> str:
    if not row:
        return "not_audited"
    return "ready" if row.get("ready") else str(row.get("status") or "missing")


def download_plan_markdown(summary: dict[str, Any]) -> str:
    lincs = layer(summary, "LINCS/CMap compound perturbation signatures")
    disease = layer(summary, "Disease expression signatures")
    gnina = layer(summary, "GNINA CNN structural rescoring")
    drugban = layer(summary, "DrugBAN formal DTI model")
    deepdta = layer(summary, "DeepDTA formal DTI model")
    graphdta = layer(summary, "GraphDTA optional DTI model")
    chai = layer(summary, "Chai-1 optional second complex model")
    network = layer(summary, "Expanded PPI/network medicine data")
    depmap = layer(summary, "Expanded DepMap/CCLE context")
    p0_missing = summary["priorityMissing"].get("P0", [])
    lines = [
        "# BioMaster External Dependency Download Plan",
        "",
        f"Generated: {summary['created_utc']}",
        "",
        "This plan is generated from the current local dependency audit. Store large assets under "
        "`/root/autodl-tmp/BioMaster/data/external/` or `/root/autodl-tmp/`; do not place large downloads under `/`.",
        "",
        "## Current Status",
        "",
        f"- P0 required missing layers: {p0_missing}",
        f"- Ready layers: {summary['readyCount']}",
        f"- Missing or partial optional layers: {summary['missingCount']}",
        "",
        "## Ready Locally",
        "",
        f"- GNINA CNN structural rescoring: {status_label(gnina)}; expected path `{gnina.get('expectedPath', '')}`.",
        f"- LINCS/CMap Level 5 perturbation signatures: {status_label(lincs)}; expected path `{lincs.get('expectedPath', '')}`.",
        f"- Per-direction disease expression signatures: {status_label(disease)}; expected path `{disease.get('expectedPath', '')}`.",
        f"- Expanded PPI/network medicine data: {status_label(network)}; expected path `{network.get('expectedPath', '')}`.",
        f"- Expanded DepMap/CCLE dependency context: {status_label(depmap)}; expected path `{depmap.get('expectedPath', '')}`.",
        "",
        "## Required Downloads",
        "",
    ]
    if p0_missing:
        lines.append("The following P0 layers are still required before the current evidence stack can be considered closed:")
        lines.append("")
        for name in p0_missing:
            row = layer(summary, name)
            lines.append(f"- {name}: {row.get('nextAction', '')}")
    else:
        lines.append("No P0 download is currently required. LINCS/CMap, disease signatures, GNINA, core network, and DepMap dependency context are available locally.")

    lines.extend(
        [
            "",
            "## Optional P1 Extensions",
            "",
            "These are useful only if we want formal named-model corroboration beyond the already completed local supervised DTI audit.",
            "",
            "### DrugBAN",
            "",
            f"- Current status: {status_label(drugban)}",
            "- Source: `https://github.com/peizhenbai/DrugBAN`",
            "- Need: isolated DGL/DGLLife-compatible environment plus validated DrugBAN checkpoint or reproducible retraining assets.",
            "- Place checkpoints under: `models/drugban/`",
            "",
            "### DeepDTA",
            "",
            f"- Current status: {status_label(deepdta)}",
            "- Source: `https://github.com/hkmztrk/DeepDTA`",
            "- Need: isolated legacy TensorFlow/Keras environment plus validated checkpoint or Davis/KIBA retraining assets.",
            "- Place checkpoints under: `models/deepdta/`",
            "",
            "### GraphDTA",
            "",
            f"- Current status: {status_label(graphdta)}",
            "- Common source: `https://github.com/thinng/GraphDTA`",
            "- Need: code under `third_party/GraphDTA/` plus validated Davis/KIBA weights or retraining assets.",
            "- Place checkpoints under: `models/graphdta/`",
            "",
            "## Optional P2 Extensions",
            "",
            "### Chai-1",
            "",
            f"- Current status: {status_label(chai)}",
            "- Source: `https://github.com/chaidiscovery/chai-lab`",
            "- Need: separate Chai-1 environment and model cache under `/root/autodl-tmp/chai_downloads/`.",
            "- Recommended scope: run only on a small finalist subset after the full DiffDock queue releases GPU capacity.",
            "",
            "### Additional Raw Context Files",
            "",
            "- BioGRID can still be added as an optional network coverage expansion.",
            "- CCLE expression/copy-number files can still be added for deeper oncology context.",
            "",
            "## Verification",
            "",
            "Run:",
            "",
            "```bash",
            "cd /root/autodl-tmp/BioMaster",
            "export TMPDIR=/root/autodl-tmp/tmp",
            "python scripts/audit_sota_external_dependencies.py --root . --chmod-gnina",
            "python scripts/build_sota_compute_closure_summary.py --root .",
            "python scripts/build_sota_artifact_manifest.py --root .",
            "```",
            "",
            "Expected audit outputs:",
            "",
            "- `outputs/sota_validation/external_dependency_audit/sota_external_dependency_audit.csv`",
            "- `outputs/sota_validation/external_dependency_audit/sota_external_dependency_audit.json`",
            "- `outputs/sota_validation/external_dependency_audit/SOTA_EXTERNAL_DEPENDENCY_AUDIT.md`",
            "- `outputs/sota_validation/SOTA_EXTERNAL_DEPENDENCY_DOWNLOAD_PLAN.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def ensure_gnina_executable(root: Path) -> None:
    for candidate in [
        Path("/root/autodl-tmp/tools/gnina/gnina.1.3.2"),
        Path("/root/autodl-tmp/tools/gnina/gnina.1.3.2.cuda12.8"),
        root / "tools/gnina/gnina",
        root / "tools/gnina/gnina.1.3.2",
    ]:
        if candidate.exists():
            mode = candidate.stat().st_mode
            candidate.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit external dependencies needed for SOTA strengthening.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out-dir", default="outputs/sota_validation/external_dependency_audit")
    parser.add_argument("--download-plan-out", default="outputs/sota_validation/SOTA_EXTERNAL_DEPENDENCY_DOWNLOAD_PLAN.md")
    parser.add_argument("--chmod-gnina", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if args.chmod_gnina:
        ensure_gnina_executable(root)
    rows, summary = build_audit(root)
    out_dir = root / args.out_dir
    write_csv(out_dir / "sota_external_dependency_audit.csv", rows)
    write_json(out_dir / "sota_external_dependency_audit.json", summary)
    (out_dir / "SOTA_EXTERNAL_DEPENDENCY_AUDIT.md").write_text(markdown(summary), encoding="utf-8")
    download_plan_out = root / args.download_plan_out
    download_plan_out.parent.mkdir(parents=True, exist_ok=True)
    download_plan_out.write_text(download_plan_markdown(summary), encoding="utf-8")
    print(json.dumps({"outDir": str(out_dir), "downloadPlan": str(download_plan_out), "readyCount": summary["readyCount"], "missingCount": summary["missingCount"], "priorityMissing": summary["priorityMissing"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
