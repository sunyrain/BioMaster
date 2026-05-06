from __future__ import annotations

import csv
import math
import re
from pathlib import Path

from .io import as_float, as_int, first_present, read_csv_rows, read_table_rows, write_csv_rows
from .pubchem import download_sdf, fetch_compound_properties
from .uniprot import fetch_alphafold_prediction, fetch_uniprot_entry, iter_human_reviewed_proteins


DRUG_FIELDS = [
    "drug_id",
    "drug_name",
    "brand_name",
    "approval_year",
    "drugbank_id",
    "pubchem_cid",
    "chembl_id",
    "canonical_smiles",
    "isomeric_smiles",
    "inchikey",
    "molecular_formula",
    "molecular_weight",
    "sdf_path",
    "source",
    "route",
    "therapeutic_area",
    "indication",
    "target_name",
    "target_chembl_id",
    "action_type",
    "mechanism_of_action",
    "notes",
]

PROTEIN_FIELDS = [
    "protein_id",
    "entry_name",
    "gene_name",
    "protein_name",
    "organism",
    "length",
    "sequence",
    "alphafold_pdb_url",
    "alphafold_cif_url",
    "alphafold_bcif_url",
    "alphafold_global_plddt",
    "alphafold_model_created_date",
    "alphafold_latest_version",
    "source",
    "notes",
]

MANIFEST_FIELDS = [
    "pair_id",
    "drug_id",
    "drug_name",
    "protein_id",
    "gene_name",
    "protein_name",
    "ligand_sdf_path",
    "ligand_smiles",
    "receptor_pdb_url",
    "receptor_cif_url",
    "protein_sequence",
    "diffdock_output_dir",
    "status",
]

AFFINITY_FIELDS = [
    "stage4_rank",
    "pair_id",
    "drug_id",
    "drug_name",
    "protein_id",
    "gene_name",
    "protein_name",
    "diffdock_confidence",
    "docking_score",
    "docking_component",
    "affinity_model",
    "affinity_score",
    "kd_nm",
    "affinity_component",
    "combined_ai_score",
]

RANKING_FIELDS = [
    "stage5_rank",
    "pair_id",
    "drug_id",
    "drug_name",
    "protein_id",
    "gene_name",
    "protein_name",
    "combined_ai_score",
    "disease_id",
    "disease_name",
    "direct_disease_score",
    "network_disease_score",
    "final_priority_score",
]

CONPLEX_INPUT_FIELDS = ["protein_id", "drug_id", "protein_sequence", "ligand_smiles"]
CONPLEX_AFFINITY_FIELDS = ["pair_id", "model", "affinity_score", "kd_nm"]


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()
    return slug or "UNKNOWN"


def build_drug_library(
    seed_csv: str | Path,
    out_csv: str | Path,
    sdf_dir: str | Path | None = None,
    year_min: int | None = None,
    fetch_pubchem: bool = True,
    limit: int | None = None,
    strict_external: bool = False,
) -> list[dict[str, object]]:
    rows = read_table_rows(seed_csv)
    output: list[dict[str, object]] = []
    seen_drug_ids: dict[str, int] = {}
    used_drug_ids: set[str] = set()
    for seed in rows:
        if limit is not None and len(output) >= limit:
            break
        approval_year = as_int(first_present(seed, ["approval_year", "Approval Year", "year"]))
        if year_min is not None and approval_year is not None and approval_year < year_min:
            continue

        row = {field: seed.get(field, "") for field in DRUG_FIELDS}
        row["brand_name"] = first_present(seed, ["brand_name", "Brand Name", "brand"])
        row["drug_name"] = first_present(
            seed,
            ["drug_name", "Generic Name (INN)", "generic_name", "Generic Name", "name", "Brand Name"],
        )
        row["approval_year"] = "" if approval_year is None else str(approval_year)
        row["drugbank_id"] = first_present(seed, ["drugbank_id", "DrugBank ID", "drugbank"])
        row["pubchem_cid"] = first_present(seed, ["pubchem_cid", "PubChem CID", "cid"])
        row["chembl_id"] = first_present(seed, ["chembl_id", "ChEMBL ID", "Molecule ChEMBL ID"])
        row["canonical_smiles"] = first_present(seed, ["canonical_smiles", "Canonical SMILES", "SMILES", "smiles"])
        row["isomeric_smiles"] = first_present(seed, ["isomeric_smiles", "Isomeric SMILES", "SMILES", "smiles"])
        row["molecular_formula"] = first_present(seed, ["molecular_formula", "Molecular Formula", "formula"])
        row["molecular_weight"] = first_present(seed, ["molecular_weight", "Molecular Weight (Da)", "Molecular Weight", "mw"])
        row["route"] = first_present(seed, ["route", "Route"])
        row["therapeutic_area"] = first_present(seed, ["therapeutic_area", "Therapeutic Area"])
        row["indication"] = first_present(seed, ["indication", "Indication"])
        row["target_name"] = first_present(seed, ["target_name", "Target Name"])
        row["target_chembl_id"] = first_present(seed, ["target_chembl_id", "Target ChEMBL ID"])
        row["action_type"] = first_present(seed, ["action_type", "Action Type"])
        row["mechanism_of_action"] = first_present(seed, ["mechanism_of_action", "Mechanism of Action"])
        row["source"] = first_present(seed, ["source"], "seed")
        row["notes"] = first_present(seed, ["notes"])

        if fetch_pubchem and row["drug_name"]:
            try:
                namespace = "cid" if row["pubchem_cid"] else "name"
                identifier = row["pubchem_cid"] or row["drug_name"]
                properties = fetch_compound_properties(str(identifier), namespace=namespace)
                for key, value in properties.items():
                    if value and not row.get(key):
                        row[key] = value
            except Exception as exc:
                if strict_external:
                    raise
                row["notes"] = append_note(str(row.get("notes", "")), f"PubChem lookup failed: {exc}")

        if sdf_dir and row.get("pubchem_cid"):
            drug_id_for_path = row.get("drugbank_id") or f"PUBCHEM_{row['pubchem_cid']}"
            target = Path(sdf_dir) / f"{slugify(str(drug_id_for_path))}.sdf"
            try:
                download_sdf(str(row["pubchem_cid"]), target)
                row["sdf_path"] = str(target)
            except Exception as exc:
                if strict_external:
                    raise
                row["notes"] = append_note(str(row.get("notes", "")), f"SDF download failed: {exc}")

        row["drug_id"] = first_present(
            row,
            ["drugbank_id", "pubchem_cid", "chembl_id", "drug_name"],
            default="UNKNOWN_DRUG",
        )
        if row["drug_id"] == row.get("pubchem_cid"):
            row["drug_id"] = f"PUBCHEM:{row['pubchem_cid']}"
        row["drug_id"] = unique_drug_id(row, seen_drug_ids, used_drug_ids)
        output.append(row)

    write_csv_rows(out_csv, output, DRUG_FIELDS)
    return output


def build_protein_library(
    out_csv: str | Path,
    source_csv: str | Path | None = None,
    limit: int | None = None,
    fetch_uniprot: bool = True,
    include_alphafold: bool = False,
    strict_external: bool = False,
) -> list[dict[str, object]]:
    if source_csv:
        seeds = read_csv_rows(source_csv)
    else:
        seeds = list(iter_human_reviewed_proteins(limit=limit))
        limit = None

    output: list[dict[str, object]] = []
    for seed in seeds:
        if limit is not None and len(output) >= limit:
            break
        accession = first_present(seed, ["protein_id", "accession", "Entry"])
        row = {field: seed.get(field, "") for field in PROTEIN_FIELDS}
        row["protein_id"] = accession
        row["entry_name"] = first_present(seed, ["entry_name", "Entry Name"])
        row["gene_name"] = first_present(seed, ["gene_name", "gene", "Gene Names (primary)"])
        row["protein_name"] = first_present(seed, ["protein_name", "Protein names"])
        row["organism"] = first_present(seed, ["organism", "Organism"])
        row["length"] = first_present(seed, ["length", "Length"])
        row["sequence"] = first_present(seed, ["sequence", "Sequence"])
        row["source"] = first_present(seed, ["source"], "seed" if source_csv else "UniProt")
        row["notes"] = first_present(seed, ["notes"])

        if fetch_uniprot and accession and (not row["sequence"] or not row["gene_name"] or not row["protein_name"]):
            try:
                details = fetch_uniprot_entry(accession)
                for key, value in details.items():
                    if value and not row.get(key):
                        row[key] = value
            except Exception as exc:
                if strict_external:
                    raise
                row["notes"] = append_note(str(row.get("notes", "")), f"UniProt lookup failed: {exc}")

        if include_alphafold and accession:
            try:
                prediction = fetch_alphafold_prediction(accession)
                for key, value in prediction.items():
                    if value and (key != "sequence" or not row.get("sequence")):
                        row[key] = value
            except Exception as exc:
                if strict_external:
                    raise
                row["notes"] = append_note(str(row.get("notes", "")), f"AlphaFold lookup failed: {exc}")

        output.append(row)

    write_csv_rows(out_csv, output, PROTEIN_FIELDS)
    return output


def make_screening_manifest(
    drugs_csv: str | Path,
    proteins_csv: str | Path,
    out_csv: str | Path,
    output_prefix: str = "runs/diffdock",
    limit: int | None = None,
) -> list[dict[str, object]]:
    drugs = read_csv_rows(drugs_csv)
    proteins = read_csv_rows(proteins_csv)
    rows: list[dict[str, object]] = []
    for drug in drugs:
        for protein in proteins:
            if limit is not None and len(rows) >= limit:
                write_csv_rows(out_csv, rows, MANIFEST_FIELDS)
                return rows
            pair_id = f"{drug['drug_id']}__{protein['protein_id']}"
            rows.append(
                {
                    "pair_id": pair_id,
                    "drug_id": drug.get("drug_id", ""),
                    "drug_name": drug.get("drug_name", ""),
                    "protein_id": protein.get("protein_id", ""),
                    "gene_name": protein.get("gene_name", ""),
                    "protein_name": protein.get("protein_name", ""),
                    "ligand_sdf_path": drug.get("sdf_path", ""),
                    "ligand_smiles": first_present(drug, ["isomeric_smiles", "canonical_smiles"]),
                    "receptor_pdb_url": protein.get("alphafold_pdb_url", ""),
                    "receptor_cif_url": protein.get("alphafold_cif_url", ""),
                    "protein_sequence": protein.get("sequence", ""),
                    "diffdock_output_dir": str(Path(output_prefix) / pair_id),
                    "status": "pending",
                }
            )
    write_csv_rows(out_csv, rows, MANIFEST_FIELDS)
    return rows


def make_conplex_input(
    manifest_csv: str | Path,
    out_tsv: str | Path,
    limit: int | None = None,
    skip_missing: bool = True,
) -> list[dict[str, object]]:
    manifest = read_csv_rows(manifest_csv)
    rows: list[dict[str, object]] = []
    for row in manifest:
        if limit is not None and len(rows) >= limit:
            break
        protein_id = first_present(row, ["protein_id"])
        drug_id = first_present(row, ["drug_id"])
        sequence = first_present(row, ["protein_sequence", "sequence"])
        smiles = first_present(row, ["ligand_smiles", "canonical_smiles", "isomeric_smiles"])
        if skip_missing and (not protein_id or not drug_id or not sequence or not smiles):
            continue
        rows.append(
            {
                "protein_id": protein_id,
                "drug_id": drug_id,
                "protein_sequence": sequence,
                "ligand_smiles": smiles,
            }
        )

    target = Path(out_tsv)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for row in rows:
            writer.writerow([row[field] for field in CONPLEX_INPUT_FIELDS])
    return rows


def convert_conplex_predictions(
    predictions_tsv: str | Path,
    out_csv: str | Path,
    model: str = "ConPLex",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(predictions_tsv).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for values in reader:
            if not values or all(not value.strip() for value in values):
                continue
            if len(values) < 3:
                raise ValueError("ConPLex prediction rows must contain moleculeID, proteinID, and Prediction")
            molecule_id, protein_id, prediction = [value.strip() for value in values[:3]]
            if molecule_id.lower() in {"moleculeid", "molecule_id"}:
                continue
            rows.append(
                {
                    "pair_id": f"{molecule_id}__{protein_id}",
                    "model": model,
                    "affinity_score": prediction,
                    "kd_nm": "",
                }
            )
    write_csv_rows(out_csv, rows, CONPLEX_AFFINITY_FIELDS)
    return rows


def merge_affinity_scores(
    manifest_csv: str | Path,
    docking_scores_csv: str | Path,
    affinity_scores_csv: str | Path,
    out_csv: str | Path,
    top_n: int | None = None,
) -> list[dict[str, object]]:
    manifest_by_pair = {row["pair_id"]: row for row in read_csv_rows(manifest_csv)}
    docking_by_pair = {row["pair_id"]: row for row in read_csv_rows(docking_scores_csv)}
    affinity_by_pair = {row["pair_id"]: row for row in read_csv_rows(affinity_scores_csv)}
    pair_ids = sorted((set(docking_by_pair) | set(affinity_by_pair)) & set(manifest_by_pair))

    rows: list[dict[str, object]] = []
    for pair_id in pair_ids:
        base = manifest_by_pair[pair_id]
        docking = docking_by_pair.get(pair_id, {})
        affinity = affinity_by_pair.get(pair_id, {})
        raw_docking = raw_docking_component(docking)
        raw_affinity = raw_affinity_component(affinity)
        rows.append(
            {
                "pair_id": pair_id,
                "drug_id": base.get("drug_id", ""),
                "drug_name": base.get("drug_name", ""),
                "protein_id": base.get("protein_id", ""),
                "gene_name": base.get("gene_name", ""),
                "protein_name": base.get("protein_name", ""),
                "diffdock_confidence": docking.get("diffdock_confidence", ""),
                "docking_score": docking.get("docking_score", ""),
                "_raw_docking": raw_docking,
                "affinity_model": affinity.get("model", affinity.get("affinity_model", "")),
                "affinity_score": affinity.get("affinity_score", ""),
                "kd_nm": affinity.get("kd_nm", ""),
                "_raw_affinity": raw_affinity,
            }
        )

    docking_norm = normalize([row["_raw_docking"] for row in rows])
    affinity_norm = normalize([row["_raw_affinity"] for row in rows])
    for idx, row in enumerate(rows):
        row["docking_component"] = docking_norm[idx]
        row["affinity_component"] = affinity_norm[idx]
        components = []
        if row["_raw_docking"] is not None:
            components.append((0.45, docking_norm[idx]))
        if row["_raw_affinity"] is not None:
            components.append((0.55, affinity_norm[idx]))
        if components:
            weight_sum = sum(weight for weight, _ in components)
            row["combined_ai_score"] = round(sum(weight * value for weight, value in components) / weight_sum, 6)
        else:
            row["combined_ai_score"] = 0.0

    rows.sort(key=lambda item: as_float(item["combined_ai_score"], 0.0) or 0.0, reverse=True)
    if top_n is not None:
        rows = rows[:top_n]
    for rank, row in enumerate(rows, start=1):
        row["stage4_rank"] = rank
        row.pop("_raw_docking", None)
        row.pop("_raw_affinity", None)
    write_csv_rows(out_csv, rows, AFFINITY_FIELDS)
    return rows


def rank_disease_relevance(
    candidates_csv: str | Path,
    string_csv: str | Path,
    disgenet_csv: str | Path,
    out_csv: str | Path,
    disease_id: str | None = None,
    top_n: int | None = 100,
) -> list[dict[str, object]]:
    candidates = read_csv_rows(candidates_csv)
    disgenet_rows = read_csv_rows(disgenet_csv)
    string_rows = read_csv_rows(string_csv)

    disease_by_gene: dict[str, dict[str, object]] = {}
    for row in disgenet_rows:
        if disease_id and row.get("disease_id") != disease_id:
            continue
        gene = first_present(row, ["gene_name", "gene", "geneSymbol"]).upper()
        score = as_float(first_present(row, ["disgenet_score", "score", "ScoreGDA"]), 0.0) or 0.0
        current = disease_by_gene.get(gene)
        if current is None or score > as_float(current.get("score"), 0.0):
            disease_by_gene[gene] = {
                "score": score,
                "disease_id": row.get("disease_id", ""),
                "disease_name": row.get("disease_name", ""),
            }

    network_by_gene = compute_network_scores(string_rows, disease_by_gene)
    ai_scores = normalize([as_float(row.get("combined_ai_score"), 0.0) for row in candidates])

    ranked: list[dict[str, object]] = []
    for idx, row in enumerate(candidates):
        gene = first_present(row, ["gene_name"]).upper()
        direct_info = disease_by_gene.get(gene, {})
        direct = as_float(direct_info.get("score"), 0.0) or 0.0
        network = network_by_gene.get(gene, 0.0)
        final_score = (0.55 * ai_scores[idx]) + (0.30 * direct) + (0.15 * network)
        ranked.append(
            {
                "pair_id": row.get("pair_id", ""),
                "drug_id": row.get("drug_id", ""),
                "drug_name": row.get("drug_name", ""),
                "protein_id": row.get("protein_id", ""),
                "gene_name": row.get("gene_name", ""),
                "protein_name": row.get("protein_name", ""),
                "combined_ai_score": row.get("combined_ai_score", ""),
                "disease_id": direct_info.get("disease_id", disease_id or ""),
                "disease_name": direct_info.get("disease_name", ""),
                "direct_disease_score": round(direct, 6),
                "network_disease_score": round(network, 6),
                "final_priority_score": round(final_score, 6),
            }
        )

    ranked.sort(key=lambda item: as_float(item["final_priority_score"], 0.0) or 0.0, reverse=True)
    if top_n is not None:
        ranked = ranked[:top_n]
    for rank, row in enumerate(ranked, start=1):
        row["stage5_rank"] = rank
    write_csv_rows(out_csv, ranked, RANKING_FIELDS)
    return ranked


def compute_network_scores(
    string_rows: list[dict[str, str]],
    disease_by_gene: dict[str, dict[str, object]],
) -> dict[str, float]:
    network_by_gene: dict[str, float] = {}
    disease_genes = set(disease_by_gene)
    for row in string_rows:
        gene_a = first_present(row, ["protein1", "gene_a", "node1", "preferredName_A"]).upper()
        gene_b = first_present(row, ["protein2", "gene_b", "node2", "preferredName_B"]).upper()
        raw_score = as_float(first_present(row, ["string_score", "combined_score", "score"]), 0.0) or 0.0
        string_score = raw_score / 1000.0 if raw_score > 1 else raw_score
        if gene_a in disease_genes and gene_b:
            score = string_score * (as_float(disease_by_gene[gene_a].get("score"), 0.0) or 0.0)
            network_by_gene[gene_b] = max(network_by_gene.get(gene_b, 0.0), score)
        if gene_b in disease_genes and gene_a:
            score = string_score * (as_float(disease_by_gene[gene_b].get("score"), 0.0) or 0.0)
            network_by_gene[gene_a] = max(network_by_gene.get(gene_a, 0.0), score)
    return network_by_gene


def raw_docking_component(row: dict[str, str]) -> float | None:
    confidence = as_float(row.get("diffdock_confidence"))
    if confidence is not None:
        return confidence
    docking_score = as_float(row.get("docking_score"))
    if docking_score is not None:
        return -docking_score
    return None


def raw_affinity_component(row: dict[str, str]) -> float | None:
    affinity_score = as_float(row.get("affinity_score"))
    if affinity_score is not None:
        return affinity_score
    kd_nm = as_float(row.get("kd_nm"))
    if kd_nm is not None and kd_nm > 0:
        return 9.0 - math.log10(kd_nm)
    return None


def normalize(values: list[float | None]) -> list[float]:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return [0.0 for _ in values]
    minimum = min(numeric)
    maximum = max(numeric)
    if math.isclose(minimum, maximum):
        return [1.0 if value is not None else 0.0 for value in values]
    return [round((value - minimum) / (maximum - minimum), 6) if value is not None else 0.0 for value in values]


def append_note(existing: str, note: str) -> str:
    return f"{existing}; {note}" if existing else note


def unique_drug_id(row: dict[str, object], seen: dict[str, int], used: set[str]) -> str:
    base_id = str(row.get("drug_id") or "UNKNOWN_DRUG")
    seen[base_id] = seen.get(base_id, 0) + 1
    candidate = base_id
    if seen[base_id] > 1 or candidate in used:
        suffix_source = first_present(row, ["drug_name", "brand_name", "approval_year"], default=str(seen[base_id]))
        candidate = f"{base_id}__{slugify(suffix_source)}"
        if candidate in used:
            candidate = f"{candidate}_{seen[base_id]}"
    used.add(candidate)
    return candidate
