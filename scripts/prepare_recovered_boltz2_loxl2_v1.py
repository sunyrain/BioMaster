#!/usr/bin/env python3
"""Prepare pocket-constrained Boltz-2 inputs for the two formal LOXL2 candidates."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "outputs/recovered_gnina_candidate_docking_v1/RECOVERED_GNINA_CANDIDATE_EVIDENCE_V1.csv"
UNIVERSE = ROOT / "outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv"
P2RANK = ROOT / "outputs/chembl37_known_pocket_atlas/final_atlas/P2RANK_ALL_PREDICTED_POCKETS_875.csv.gz"
OUT = ROOT / "outputs/recovered_boltz2_loxl2_candidates_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    input_dir = OUT / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    evidence = pd.read_csv(CANDIDATES)
    selected = evidence[
        evidence["candidate_triage_status"].eq("COMPUTATIONAL_TRIAGE_PASS_REQUIRES_EXPERIMENT")
    ].copy()
    if len(selected) != 2 or set(selected["gene_symbol"]) != {"LOXL2"}:
        raise ValueError("Expected exactly two formal LOXL2 computational triage candidates")
    universe = pd.read_csv(UNIVERSE, dtype=str).fillna("")
    target = universe[universe["target_chembl_id"].eq("CHEMBL3714029")].iloc[0]
    pockets = pd.read_csv(P2RANK, dtype=str).fillna("")
    pocket = pockets[
        pockets["uniprot_accession"].eq(target["uniprot_accession"])
        & pd.to_numeric(pockets["p2rank_rank"], errors="coerce").eq(3)
    ].iloc[0]
    contacts = [["A", int(position)] for position in pocket["p2rank_residue_positions"].split(";") if position]
    rows = []
    for rank, row in enumerate(selected.sort_values("drug_names", kind="mergesort").itertuples(index=False), start=1):
        pair_id = f"LOXL2_{row.drug_names}_{row.ligand_inchikey}".replace(" ", "_")
        payload = {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": target["sequence"], "msa": "empty"}},
                {"ligand": {"id": "B", "smiles": row.ligand_smiles}},
            ],
            "properties": [{"affinity": {"binder": "B"}}],
            "constraints": [{
                "pocket": {"binder": "B", "contacts": contacts, "max_distance": 6.0, "force": False}
            }],
        }
        yaml_name = f"{rank:02d}_{pair_id}.yaml"
        yaml_path = input_dir / yaml_name
        yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False, width=120), encoding="utf-8")
        rows.append({
            "pair_id": pair_id, "target_chembl_id": row.target_chembl_id, "gene_symbol": row.gene_symbol,
            "drug_names": row.drug_names, "ligand_inchikey": row.ligand_inchikey,
            "ligand_smiles": row.ligand_smiles, "p2rank_matched_rank": 3,
            "p2rank_probability": pocket["p2rank_probability"], "pocket_contacts": len(contacts),
            "yaml_path": str(yaml_path), "yaml_sha256": sha256(yaml_path),
            "gpu": rank - 1, "seed": 2026080900 + rank,
        })
    manifest = pd.DataFrame(rows)
    manifest_path = OUT / "RECOVERED_BOLTZ2_LOXL2_INPUT_MANIFEST_V1.csv"
    manifest.to_csv(manifest_path, index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "PASS", "pairs": len(manifest), "targets": manifest["target_chembl_id"].nunique(),
        "pocket_policy": "LOXL2 P2Rank rank-3 site, the site actually matched by fpocket; 18 residue contacts; force=false",
        "sampling_policy": "Boltz-2, empty MSA, 3 recycles, 200 structure steps, 5 diffusion samples, 200 affinity steps, 5 affinity samples",
        "limitations": "No target-specific Boltz control calibration and no copper/LTQ cofactor context; outputs are orthogonal structural hypotheses, not binding validation.",
        "manifest": str(manifest_path.relative_to(ROOT)),
    }
    (OUT / "RECOVERED_BOLTZ2_LOXL2_PREPARATION_SUMMARY_V1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
