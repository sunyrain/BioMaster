# GitHub Release Checklist

Target repository:

```text
https://github.com/sunyrain/BioMaster
```

Required visibility:

```text
private
```

## Release Boundary

Commit only:

- `biomaster/`
- `scripts/`
- `examples/`
- `tests/`
- `docs/`
- `README.md`
- `pyproject.toml`
- `.gitignore`

Do not commit:

- `data/`
- `downloads/`
- `outputs/`
- `third_party/`
- model weights
- raw graph dumps
- AlphaFold tarballs
- FDA workbook

## Create Private Repo Manually

If GitHub CLI is unavailable, create the private repository in the GitHub web UI:

1. Log in as `sunyrain`.
2. Create repository `BioMaster`.
3. Set visibility to `Private`.
4. Do not initialize with README, license, or `.gitignore`.

Then push from this workspace:

```bash
git remote add origin git@github.com:sunyrain/BioMaster.git
git branch -M main
git push -u origin main
```

If using HTTPS and a personal access token:

```bash
git remote add origin https://github.com/sunyrain/BioMaster.git
git branch -M main
git push -u origin main
```

The token needs repo creation/push permission for private repositories.
