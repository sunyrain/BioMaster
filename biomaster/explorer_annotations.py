"""Local, traceable disease and pathway annotations for the project explorer.

These readers never turn a predicted association into an approved indication.
Drug identity follows the reviewed FDA entity-to-model-species mapping; all
external databases are opened read-only and remain optional.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import math
import re
import sqlite3
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator


FDA_DIR = "outputs/fda_drug_universe_2005_2026_v1/10_nda_window_mechanism_active_species"
SPECIES_PATH = f"{FDA_DIR}/FDA_NDA_WINDOW_CORE_DISCOVERY_MODEL_LIGANDS_2005_2026.csv"
ENTITY_PATH = f"{FDA_DIR}/FDA_NDA_WINDOW_STANDARD_DRUG_ENTITY_MECHANISM_ACTIVE_SPECIES_AUDIT.csv"
CHEMBL_PATH = "downloads/chembl_37/chembl_37/chembl_37_sqlite/chembl_37.db"
OT_PATH = "data/external/opentargets_26_06/target"
TX_PATH = "data/processed/txgnn_drug_disease_scores.csv"
GTEX_PATH = "data/external/gtex/GTEx_Analysis_v10_RNASeQCv2.4.2_gene_median_tpm.gct.gz"
HPA_PATH = "data/external/hpa/rna_tissue_consensus.tsv.zip"
OT_DISEASE_PATHS = [
    "outputs/retargetmap_spr64_target_review_20260909/OPENTARGETS_64_AVAILABLE_ASSOCIATIONS.csv.gz",
    "outputs/current_production_package_v2/full_untruncated_universe_v4/opentargets_top3000_target_completion_v4/opentargets_final1000_target_disease_long.csv",
    "outputs/chembl_moa_enhanced_information_package_v1/opentargets_final1000_full_disease_completion/opentargets_final1000_target_disease_long.csv",
    "data/processed/opentargets_target_disease_scores.csv",
]


def _rows(path: Path) -> Iterator[dict[str, str]]:
    if not path.is_file():
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _text(value: Any) -> str:
    result = str(value).strip() if value is not None else ""
    return "" if result.lower() in {"nan", "none", "null"} else result


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _parts(value: Any) -> list[str]:
    return [item.strip() for item in _text(value).split(";") if item.strip()]


def _normal_name(value: Any) -> str:
    # Punctuation/case normalization only: never remove salt or metabolite names.
    return re.sub(r"[^a-z0-9]", "", _text(value).lower())


def _new_drug() -> dict[str, Any]:
    return {"known_diseases": [], "txgnn_diseases": [], "known_targets": [],
            "properties": {}, "identifiers": {}}


def _new_target() -> dict[str, Any]:
    return {"pathways": [], "target_diseases": [], "properties": {}, "identifiers": {}}


def load_annotations(root: Path, drugs: dict, targets: dict) -> dict[str, Any]:
    """Return annotations keyed by model InChIKey and target ChEMBL identifier.

    Lists are complete for the local snapshots, with no presentation top-N cut.
    Empty lists mean missing local evidence, never a negative biological claim.
    """
    root = Path(root)
    result: dict[str, Any] = {
        "drugs": {key: _new_drug() for key in drugs},
        "targets": {key: _new_target() for key in targets},
        "sources": [], "counts": {},
    }

    def source(name: str, path: str, **extra: Any) -> dict:
        record = {"name": name, "path": path, "available": (root / path).exists(), **extra}
        result["sources"].append(record)
        return record

    # One model species can represent multiple reviewed marketed entities.
    species_by_drug: dict[str, list[dict]] = defaultdict(list)
    for row in _rows(root / SPECIES_PATH):
        if row.get("ligand_inchikey") in drugs:
            species_by_drug[row["ligand_inchikey"]].append(row)
    entities: dict[str, list[dict]] = defaultdict(list)
    for row in _rows(root / ENTITY_PATH):
        entities[row.get("project_entity_id", "")].append(row)
    source("FDA / RxNorm / GSRS reviewed drug entities", ENTITY_PATH)
    source("Model species ↔ marketed drug identity", SPECIES_PATH)

    chembl_to_drugs: dict[str, set[str]] = defaultdict(set)
    names_to_drugs: dict[str, set[str]] = defaultdict(set)
    for drug_id, data in drugs.items():
        annotation = result["drugs"][drug_id]
        identifiers, properties = annotation["identifiers"], annotation["properties"]
        official_rows = []
        for species in species_by_drug.get(drug_id, []):
            official_rows.extend(entities.get(species.get("project_entity_id", ""), []))
            normalized = _normal_name(species.get("drug_name"))
            if normalized:
                names_to_drugs[normalized].add(drug_id)
        if not official_rows:
            for name in _parts(data.get("name")):
                if _normal_name(name):
                    names_to_drugs[_normal_name(name)].add(drug_id)
        fields = {
            "brand_names": "all_fda_brand_names", "routes": "all_fda_routes",
            "mechanism_of_action": "known_mechanism_of_action",
            "action_types": "known_action_types", "parent_molecular_formula": "gsrs_molecular_formula",
            "fda_application_numbers": "all_fda_nda_application_numbers",
        }
        for destination, field in fields.items():
            values = sorted({_text(row.get(field)) for row in official_rows if _text(row.get(field))})
            if values:
                properties[destination] = "; ".join(values)
        for destination, field in {
            "parent_molecular_weight": "molecular_weight",
            "approval_year": "chembl_first_approval",
        }.items():
            values = [_number(row.get(field)) for row in official_rows]
            values = [value for value in values if value is not None]
            if values:
                properties[destination] = min(values)
        properties["metadata_scope"] = "原上市药物实体；活性代谢物的适应症继承关系需结合 species_role 阅读"
        _add_model_properties(properties, data.get("properties", {}).get("smiles"),
                              [row for row in official_rows if row.get("model_inchikey") == drug_id])
        species_rows = species_by_drug.get(drug_id, [])
        properties["species_role"] = "; ".join(sorted({_text(row.get("ligand_species_role")) for row in species_rows}))
        properties["species_name"] = "; ".join(sorted({_text(row.get("ligand_species_name")) for row in species_rows if _text(row.get("ligand_species_name"))}))
        identifiers["project_entity_ids"] = sorted({row["project_entity_id"] for row in species_rows})
        molecule_ids = set()
        for row in official_rows:
            # Reviewed equivalents may include the parent/active species; retain that provenance.
            for field in ["chembl_full_match_ids", "base_chembl_ids", "chembl_equivalent_entity_ids_for_moa"]:
                molecule_ids.update(_parts(row.get(field)))
        species_molecule_ids = {molecule_id for row in species_rows for molecule_id in _parts(row.get("ligand_species_chembl_id"))}
        molecule_ids.update(species_molecule_ids)
        identifiers["chembl_ids"] = sorted(molecule_ids)
        if molecule_ids:
            identifiers["chembl_id"] = sorted(species_molecule_ids or molecule_ids)[0]
        pubchem = sorted({_text(row.get("gsrs_pubchem_cid")) for row in official_rows if _text(row.get("gsrs_pubchem_cid"))})
        if pubchem:
            identifiers["pubchem_cid"] = pubchem[0]
        for molecule_id in molecule_ids:
            chembl_to_drugs[molecule_id].add(drug_id)

    _load_chembl(root, result, chembl_to_drugs, source)
    _load_txgnn(root, result, names_to_drugs, source)
    _load_opentargets(root, result, targets, source)
    _load_restored_diseases(root, result, source)

    for annotation in result["drugs"].values():
        annotation["known_diseases"] = _deduplicate(annotation["known_diseases"], "phase")
        annotation["known_targets"] = _deduplicate(annotation["known_targets"])
        annotation["txgnn_diseases"] = _deduplicate(annotation["txgnn_diseases"], "score")
    for annotation in result["targets"].values():
        annotation["pathways"] = _deduplicate(annotation["pathways"])
        annotation["target_diseases"].sort(key=lambda item: -(item.get("score") or 0))
    result["counts"] = {
        "known_indications": sum(len(x["known_diseases"]) for x in result["drugs"].values()),
        "approved_indications": sum(item.get("phase") == 4 for x in result["drugs"].values() for item in x["known_diseases"]),
        "clinical_indications": sum(item.get("phase") is not None and 0 < item["phase"] < 4 for x in result["drugs"].values() for item in x["known_diseases"]),
        "known_targets": sum(len(x["known_targets"]) for x in result["drugs"].values()),
        "txgnn_associations": sum(len(x["txgnn_diseases"]) for x in result["drugs"].values()),
        "txgnn_drugs": sum(bool(x["txgnn_diseases"]) for x in result["drugs"].values()),
        "pathways": sum(len(x["pathways"]) for x in result["targets"].values()),
        "pathway_targets": sum(bool(x["pathways"]) for x in result["targets"].values()),
        "target_disease_associations": sum(len(x["target_diseases"]) for x in result["targets"].values()),
        "disease_targets": sum(bool(x["target_diseases"]) for x in result["targets"].values()),
        "go_annotations": sum(len(x["properties"].get("go_annotations", [])) for x in result["targets"].values()),
        "tissue_expression_records": sum(len(x["properties"].get("tissue_expression", [])) for x in result["targets"].values()),
    }
    return result


def _add_model_properties(properties: dict, smiles: str | None, exact_rows: list[dict]) -> None:
    """Keep molecular properties tied to the ranked species, not its parent drug."""
    if exact_rows:
        row = exact_rows[0]
        if row.get("gsrs_molecular_formula"):
            properties["molecular_formula"] = row["gsrs_molecular_formula"]
        for key in ["molecular_weight", "hbd", "hba", "tpsa", "rotatable_bonds"]:
            value = _number(row.get(key))
            if value is not None:
                properties[key] = value
        return
    if not smiles:
        return
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            properties.update(molecular_formula=rdMolDescriptors.CalcMolFormula(molecule),
                              molecular_weight=round(Descriptors.MolWt(molecule), 3),
                              hbd=rdMolDescriptors.CalcNumHBD(molecule), hba=rdMolDescriptors.CalcNumHBA(molecule),
                              tpsa=round(rdMolDescriptors.CalcTPSA(molecule), 3),
                              rotatable_bonds=rdMolDescriptors.CalcNumRotatableBonds(molecule))
    except ImportError:
        pass


def _deduplicate(items: list[dict], score_key: str | None = None) -> list[dict]:
    unique: dict[str, dict] = {}
    for item in items:
        key = item["id"]
        if key not in unique or (score_key and (item.get(score_key) or 0) > (unique[key].get(score_key) or 0)):
            unique[key] = item
    return sorted(unique.values(), key=lambda item: (-(item.get(score_key) or 0) if score_key else 0, item["name"]))


def _load_chembl(root: Path, result: dict, mapping: dict, source: Any) -> None:
    record = source("ChEMBL 37 drug indications and mechanisms", CHEMBL_PATH,
                    note="phase=4 为已批准适应症记录；较低阶段为临床研究，不能称为已批准。")
    if not record["available"] or not mapping:
        return
    try:
        with sqlite3.connect((root / CHEMBL_PATH).resolve().as_uri() + "?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            molecule_ids = sorted(mapping)
            for start in range(0, len(molecule_ids), 400):
                batch = molecule_ids[start:start + 400]
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    f"SELECT md.chembl_id, di.max_phase_for_ind, di.mesh_id, di.mesh_heading, di.efo_id, di.efo_term "
                    f"FROM molecule_dictionary md JOIN drug_indication di ON di.molregno=md.molregno "
                    f"WHERE md.chembl_id IN ({placeholders})", batch)
                for row in rows:
                    item = {"id": _text(row["efo_id"]).replace(":", "_") or f"MESH:{row['mesh_id']}",
                            "name": row["efo_term"] or row["mesh_heading"],
                            "phase": _number(row["max_phase_for_ind"]), "source": "ChEMBL 37 · drug_indication",
                            "evidence": "已登记的药物适应症/临床研究；来自原药及审核等价实体",
                            "molecule_id": row["chembl_id"], "url": f"https://www.ebi.ac.uk/chembl/explore/compound/{row['chembl_id']}"}
                    for drug_id in mapping[row["chembl_id"]]:
                        result["drugs"][drug_id]["known_diseases"].append(dict(item))
                rows = conn.execute(
                    f"SELECT md.chembl_id AS molecule_id, td.chembl_id AS target_id, td.pref_name, td.organism, "
                    f"dm.action_type, dm.mechanism_of_action, dm.direct_interaction "
                    f"FROM molecule_dictionary md JOIN drug_mechanism dm ON dm.molregno=md.molregno "
                    f"JOIN target_dictionary td ON td.tid=dm.tid WHERE md.chembl_id IN ({placeholders})", batch)
                for row in rows:
                    for drug_id in mapping[row["molecule_id"]]:
                        result["drugs"][drug_id]["known_targets"].append({
                            "id": row["target_id"], "name": row["pref_name"], "source": "ChEMBL 37 · drug_mechanism",
                            "evidence": row["mechanism_of_action"], "action_type": row["action_type"],
                            "organism": row["organism"], "direct_interaction": row["direct_interaction"] == 1,
                            "molecule_id": row["molecule_id"], "in_catalog": row["target_id"] in result["targets"],
                        })
    except sqlite3.Error as exc:
        record.update(available=False, error=str(exc))


def _load_restored_diseases(root: Path, result: dict, source: Any) -> None:
    """Prefer the audited full-disease snapshot, joined by exact project IDs."""
    relative = "outputs/biomaster_disease_evidence_720x888_20260909"
    validation = root / relative / "VALIDATION.json"
    if not validation.is_file():
        return
    if json.loads(validation.read_text()).get("all_pass") is not True:
        raise ValueError("Restored disease snapshot did not pass validation")
    import pyarrow.parquet as pq
    for annotation in result["drugs"].values():
        annotation["txgnn_diseases"] = []
    for annotation in result["targets"].values():
        annotation["target_diseases"] = []
    tx_path = f"{relative}/DRUG_ALL_DISEASE_TOP30.csv.gz"
    source("TxGNN 全疾病推理 · 已排除身份暂扣及保守已知关系 · 展示Top30", tx_path,
           note="精确项目结构键连接；sigmoid仅作展示，不是临床获益概率。完整分数矩阵另存。")
    for row in _rows(root / tx_path):
        identifier = row["ligand_inchikey"]
        if identifier in result["drugs"]:
            logit = float(row["logit"])
            result["drugs"][identifier]["txgnn_diseases"].append({
                "id": row["txgnn_disease_id"], "name": row["disease_name"],
                "score": 1 / (1 + math.exp(-max(-80, min(80, logit)))), "logit": logit,
                "source": "TxGNN · full_graph · 2026-09-09 · 全疾病Top30",
                "evidence": "疾病线索，非实验结合结果；合并疾病节点不可作子病种特异预测",
                "source_path": tx_path, "direction": row.get("directions"),
            })
    ot_path = f"{relative}/OPENTARGETS_ALL_TARGET_DISEASE_EVIDENCE.parquet"
    source("Open Targets 26.06 · 886靶点完整疾病证据", ot_path,
           note="完整分页；来源关联不是治疗方向。缺失靶点不回填相近基因。")
    columns = ["target_chembl_id", "disease_id", "disease_name", "overall_score", "datatype_scores_json"]
    for batch in pq.ParquetFile(root / ot_path).iter_batches(columns=columns):
        for row in batch.to_pylist():
            identifier = row["target_chembl_id"]
            if identifier in result["targets"]:
                result["targets"][identifier]["target_diseases"].append({
                    "id": row["disease_id"], "name": row["disease_name"], "score": row["overall_score"],
                    "evidence_scores": json.loads(row["datatype_scores_json"]),
                    "source": "Open Targets 26.06 · 完整分页 · 2026-09-09", "source_path": ot_path,
                    "evidence": "靶点疾病关联；非治疗方向验证",
                    "url": f"https://platform.opentargets.org/disease/{row['disease_id']}",
                })


def _load_txgnn(root: Path, result: dict, names: dict, source: Any) -> None:
    source("TxGNN local full_graph predictions", TX_PATH,
           note="本地已有 cancer / MONDO_0004992 单疾病预测；sigmoid(logit) 不是临床获益概率。未覆盖药物保留为空。")
    for row in _rows(root / TX_PATH):
        name = _normal_name(row.get("txgnn_drug_name"))
        if not name:
            continue
        for drug_id in names.get(name, []):
            score = _number(row.get("txgnn_indication_score"))
            if score is None:
                continue
            result["drugs"][drug_id]["txgnn_diseases"].append({
                "id": row["disease_id"], "name": row["disease_name"], "score": score,
                "logit": _number(row.get("txgnn_indication_logit")), "relation": row.get("relation"),
                "source": "TxGNN · full_graph · 2026-05-06",
                "evidence": "原药名称与 TxGNN DrugBank 节点保守精确匹配；疾病关联预测",
                "mapping": "exact_normalized_reviewed_drug_name", "drugbank_id": row.get("txgnn_drugbank_id"),
            })
            result["drugs"][drug_id]["identifiers"].setdefault("drugbank_id", row.get("txgnn_drugbank_id"))


def _load_opentargets(root: Path, result: dict, targets: dict, source: Any) -> None:
    gene_to_targets: dict[str, set[str]] = defaultdict(set)
    uniprot_to_targets: dict[str, set[str]] = defaultdict(set)
    ensembl_to_targets: dict[str, set[str]] = defaultdict(set)
    for target_id, target in targets.items():
        identifiers = target.get("identifiers", {})
        gene = _text(target.get("gene_symbol") or target.get("name"))
        if gene:
            gene_to_targets[gene].add(target_id)
        uniprot = _text(identifiers.get("uniprot_id") or identifiers.get("uniprot_accession"))
        if uniprot:
            uniprot_to_targets[uniprot].add(target_id)
        ensembl = _text(identifiers.get("ensembl_id"))
        if ensembl:
            ensembl_to_targets[ensembl].add(target_id)

    record = source("OpenTargets 26.06 target annotation / Reactome pathways", OT_PATH)
    parquet_paths = sorted((root / OT_PATH).glob("*.parquet"))
    if parquet_paths:
        try:
            import pyarrow.parquet as parquet

            property_columns = {
                "functionDescriptions": "function_descriptions", "subcellularLocations": "subcellular_locations",
                "tractability": "tractability", "go": "go_annotations", "synonyms": "synonyms",
                "targetClass": "target_classes", "hallmarks": "hallmarks", "chemicalProbes": "chemical_probes",
                "safetyLiabilities": "safety_liabilities", "constraint": "genetic_constraint",
                "genomicLocation": "genomic_location", "canonicalTranscript": "canonical_transcript",
            }
            columns = ["id", "approvedSymbol", "approvedName", "pathways", "proteinIds", *property_columns]
            for path in parquet_paths:
                available = set(parquet.read_schema(path).names)
                table = parquet.read_table(path, columns=[c for c in columns if c in available])
                identity_rows = table.select([c for c in ["id", "approvedSymbol", "proteinIds"] if c in available]).to_pylist()
                selected = [index for index, row in enumerate(identity_rows)
                            if row["id"] in ensembl_to_targets or row.get("approvedSymbol") in gene_to_targets
                            or any(protein.get("id") in uniprot_to_targets for protein in row.get("proteinIds") or [])]
                if not selected:
                    continue
                # Convert rich nested annotation records only for requested entities.
                for row in table.take(selected).to_pylist():
                    matched = set(ensembl_to_targets.get(row["id"], []))
                    for protein in row.get("proteinIds") or []:
                        matched.update(uniprot_to_targets.get(protein.get("id", ""), []))
                    # A unique exact approved symbol is a transparent fallback.
                    if not matched and len(gene_to_targets.get(row.get("approvedSymbol"), [])) == 1:
                        matched.update(gene_to_targets[row["approvedSymbol"]])
                    for target_id in matched:
                        ensembl_to_targets[row["id"]].add(target_id)
                        annotation = result["targets"][target_id]
                        annotation["identifiers"]["ensembl_id"] = row["id"]
                        annotation["description"] = " ".join(row.get("functionDescriptions") or [])
                        annotation["properties"]["approved_name"] = row.get("approvedName")
                        for column, property_name in property_columns.items():
                            if row.get(column) is not None:
                                annotation["properties"][property_name] = row[column]
                        for pathway in row.get("pathways") or []:
                            annotation["pathways"].append({"id": pathway["pathwayId"], "name": pathway["pathway"],
                                "category": pathway.get("topLevelTerm", ""), "source": "OpenTargets 26.06 · Reactome",
                                "evidence": "已注释的靶点通路成员关系", "url": f"https://reactome.org/content/detail/{pathway['pathwayId']}"})
        except (ImportError, OSError, ValueError) as exc:
            record.update(available=False, error=str(exc))
    else:
        record["available"] = False

    # Read newer cache first; avoid silently mixing scores of the same association.
    seen: set[tuple[str, str]] = set()
    for relative in OT_DISEASE_PATHS:
        source("OpenTargets target–disease local snapshot", relative,
               note="靶点–疾病证据关联不直接代表药物治疗方向；本地缓存覆盖范围有限。")
        for row in _rows(root / relative):
            ensembl = row.get("opentargets_target_id") or row.get("ensembl_gene_id") or row.get("query_ensembl_id")
            matched = set(ensembl_to_targets.get(ensembl, []))
            if not matched:
                matched.update(uniprot_to_targets.get(row.get("protein_id", ""), []))
            if not matched:
                gene = row.get("approved_symbol") or row.get("target_gene") or row.get("gene_name")
                if len(gene_to_targets.get(gene, [])) == 1:
                    matched.update(gene_to_targets[gene])
            for target_id in matched:
                key = (target_id, row["disease_id"])
                if key in seen:
                    continue
                seen.add(key)
                try:
                    evidence_scores = json.loads(row.get("datatype_scores_json") or "{}")
                except (ValueError, TypeError):
                    evidence_scores = {}
                result["targets"][target_id]["target_diseases"].append({
                    "id": row["disease_id"], "name": row["disease_name"], "score": _number(row.get("overall_score")),
                    "source": "OpenTargets · " + (row.get("snapshot") or "local association cache"),
                    "evidence": "靶点–疾病关联；非治疗方向验证", "evidence_scores": evidence_scores,
                    "source_path": relative, "url": f"https://platform.opentargets.org/disease/{row['disease_id']}",
                })
    _load_tissue_expression(root, result, ensembl_to_targets, source)


def _load_tissue_expression(root: Path, result: dict, mapping: dict, source: Any) -> None:
    """Retain distinct RNA expression units; GTEx TPM and HPA nTPM are not pooled."""
    source("GTEx v10 tissue median RNA expression", GTEX_PATH,
           note="68 tissues/cell populations; median TPM. Expression is context, not binding evidence.")
    source("Human Protein Atlas local tissue RNA consensus", HPA_PATH,
           note="HPA nTPM; a distinct normalization from GTEx TPM. Snapshot version unavailable in local filename.")
    if (root / GTEX_PATH).is_file():
        with gzip.open(root / GTEX_PATH, "rt", encoding="utf-8") as handle:
            next(handle, None)  # GCT version.
            next(handle, None)  # Matrix dimensions.
            reader = csv.reader(handle, delimiter="\t")
            columns = next(reader, [])[2:]
            for row in reader:
                if not row:
                    continue
                matched = mapping.get(row[0].split(".")[0], [])
                if not matched:
                    continue
                values = [{"id": tissue, "name": tissue.replace("_", " "), "value": _number(raw),
                           "unit": "TPM", "source": "GTEx v10 · median tissue expression"}
                          for tissue, raw in zip(columns, row[2:]) if _number(raw) is not None]
                for target_id in matched:
                    result["targets"][target_id]["properties"].setdefault("tissue_expression", []).extend(values)
    if (root / HPA_PATH).is_file():
        with zipfile.ZipFile(root / HPA_PATH) as archive:
            members = [name for name in archive.namelist() if name.endswith(".tsv")]
            for member in members:
                with archive.open(member) as raw:
                    for row in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8"), delimiter="\t"):
                        matched = mapping.get(row.get("Gene", "").split(".")[0], [])
                        value = _number(row.get("nTPM"))
                        if not matched or value is None:
                            continue
                        for target_id in matched:
                            result["targets"][target_id]["properties"].setdefault("tissue_expression", []).append({
                                "id": row["Tissue"], "name": row["Tissue"], "value": value, "unit": "nTPM",
                                "source": "Human Protein Atlas · local RNA tissue consensus",
                            })
