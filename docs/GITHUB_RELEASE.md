# GitHub Release Checklist

Target repository:

```text
https://github.com/sunyrain/BioMaster
```

Required visibility:

```text
private
```

The current remote was verified as public on 2026-08-03. Do not push the V8
package until the repository is changed to private or the complete public
release boundary is explicitly approved.

## Release Boundary

Commit only:

- `biomaster/`
- promoted, reproducible entry points under `scripts/`
- versioned workflow configuration under `configs/`
- reusable molecular-dynamics tooling under `md/`
- `examples/`
- `tests/`
- `docs/`
- current, compact teacher-facing PDF deliverables in the repository root
- compact audited output tables explicitly allow-listed by `.gitignore`
- `README.md`
- `pyproject.toml`
- `.gitignore`

Do not commit:

- `data/`
- `downloads/`
- `third_party/`
- local environments such as `.venvs/`, `.conda_envs/`, and `.external/`
- caches such as `.cache/`, `.tmp/`, and package caches
- local historical archives under `archive/` or `archives/`
- model weights
- raw graph dumps
- AlphaFold tarballs
- FDA or target workbooks
- broad generated `outputs/` outside the explicit compact allow-list

## Pre-push Checks

Run from the repository root:

```bash
git status --short --branch
git diff --check
python -m compileall -q biomaster scripts md
bash -n scripts/*.sh
pytest -q
git fsck --no-progress --connectivity-only
```

Verify that the staged change set contains no unexpected large file:

```bash
git diff --cached --numstat
```

## Push the Reviewed Branch

The `origin` remote is already configured. Push the reviewed feature branch,
not `main`, so the complete 21-commit project history and the current package
can be reviewed together:

```bash
git fetch origin --prune
