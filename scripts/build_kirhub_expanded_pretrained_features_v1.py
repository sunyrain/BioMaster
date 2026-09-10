#!/usr/bin/env python3
"""Build an expanded KiRHub matrix with frozen BerMol/ESM2 representations.

The expansion uses exact assay-compound names and exact UniProt accessions from
the KiRHub supplement.  Repeated accessions assayed in multiple complexes are
excluded from the primary scope because a sequence-only input cannot represent
the complex-specific context.  Existing feature caches are reused byte-for-byte;
only cache misses are inferred with the same frozen pretrained checkpoints.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import requests
import torch
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/evidence_routing_compute_execution_20260808_v1"
WORKBOOK = (
    RUN / "drug_centric_cross_target_v1/external_benchmark"
    / "41587_2026_3090_MOESM4_ESM.xlsx"
)
PAIR_BENCHMARK = RUN / "old_drug_innovation_v7/OLD_DRUG_KIRHUB_PAIR_BENCHMARK_V7.csv"
BASE = ROOT / "outputs/old_drug_target_sota_v1"
DEPLOY = BASE / "public_retrained_v1/dtiam_deployment_feature_store_v1"
OFFICIAL = BASE / "public_retrained_v1/dtiam_official_feature_store_v1"
CHEMBL_PAIRS = BASE / "feature_store_v1/CHEMBL37_86674_INDEXED_PAIRS_V1.csv.gz"
BASELINE_SCORES = BASE / "drug_centric_ranker_v1/BIOMASTER_DRUG_TO_TARGET_720X384_V1.csv.gz"
BERMOL_MODEL = ROOT / "third_party/sota_dti_2026/DTIAM/code/BerMolModel_base.pkl"
BERMOL_CODE = ROOT / "third_party/sota_dti_2026/DTIAM/code/BerMol"
ESM_CHECKPOINT = Path("/root/autodl-tmp/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt")
OUT = ROOT / "outputs/kirhub_functional_adaptation_v1/expanded_pretrained_features_v1"
SEED = 20260831
FOLDS = 5
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/accessions"
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/IsomericSMILES,InChIKey/JSON"
ACCESSION_PATTERN = re.compile(
    r"\b(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b"
)
# Three S2 accessions are obsolete/typographical.  The replacements are exact
# reviewed human entries for the S2 HGNC symbols (MAP3K5, NEK3 and PRKG1).
CURRENT_UNIPROT_ACCESSION = {
    "O99683": "Q99683",
    "Q8WUN5": "P51956",
    "P14619": "Q13976",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_name(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def fragment_parent_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid PubChem SMILES: {smiles}")
    parent = rdMolStandardize.FragmentParent(molecule)
    return Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)


def generic_scaffold(smiles: str) -> str:
    from rdkit.Chem.Scaffolds import MurckoScaffold

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold.GetNumAtoms():
        return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(scaffold), canonical=True)
    return "ACYCLIC:" + Chem.MolToSmiles(molecule, canonical=True)


def fetch_pubchem(compound: str) -> dict[str, object]:
    url = PUBCHEM_URL.format(name=urllib.parse.quote(compound, safe=""))
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    properties = response.json()["PropertyTable"]["Properties"][0]
    raw_smiles = properties.get("SMILES") or properties.get("IsomericSMILES")
    if not raw_smiles:
        raise RuntimeError(f"PubChem returned no SMILES for {compound}")
    parent_smiles = fragment_parent_smiles(str(raw_smiles))
    molecule = Chem.MolFromSmiles(parent_smiles)
    return {
        "kirhub_compound": compound,
        "pubchem_cid": int(properties["CID"]),
        "pubchem_raw_smiles": str(raw_smiles),
        "ligand_smiles": parent_smiles,
        "pubchem_inchikey": str(properties.get("InChIKey", "")),
        "ligand_inchikey": Chem.MolToInchiKey(molecule),
        "structure_source_url": url,
    }


def fetch_uniprot(accessions: list[str]) -> pd.DataFrame:
    frames = []
    for start in range(0, len(accessions), 100):
        batch = accessions[start : start + 100]
        response = requests.get(
            UNIPROT_URL,
            params={
                "accessions": ",".join(batch), "format": "tsv",
                "fields": "accession,gene_names,protein_name,sequence,length,reviewed",
            },
            timeout=120,
        )
        response.raise_for_status()
        current = pd.read_csv(io.StringIO(response.text), sep="\t")
        frames.append(current)
    result = pd.concat(frames, ignore_index=True)
    result = result.rename(columns={
        "Entry": "uniprot_accession", "Gene Names": "uniprot_gene_names",
        "Protein names": "uniprot_protein_name", "Sequence": "protein_sequence",
        "Length": "sequence_length", "Reviewed": "uniprot_reviewed",
    })
    result["uniprot_accession"] = result["uniprot_accession"].astype(str).str.upper()
    return result


def encode_bermol(smiles: list[str], device: torch.device) -> np.ndarray:
    sys.path.insert(0, str(BERMOL_CODE))
    from bermol.tokenizer import BerMolTokenizer  # noqa: E402

    with BERMOL_MODEL.open("rb") as handle:
        predictor = dill.load(handle)
    predictor.model.to(device).eval()
    tokenizer = BerMolTokenizer(predictor.vocab)
    tokens = [tokenizer.encode(value).squeeze(0) for value in smiles]
    output = np.empty((len(smiles), 768), dtype=np.float32)
    ordered = sorted(enumerate(tokens), key=lambda item: len(item[1]))
    with torch.inference_mode():
        cursor = 0
        while cursor < len(ordered):
            maximum = len(ordered[min(len(ordered) - 1, cursor + 63)][1])
            batch_size = min(64, max(1, 1_000_000 // max(1, maximum**2)))
            selected = ordered[cursor : cursor + batch_size]
            maximum = max(len(token) for _, token in selected)
            token_ids = torch.zeros((len(selected), maximum), dtype=torch.long)
            attention = torch.full(
                (len(selected), maximum, maximum), -10000.0, dtype=torch.float32
            )
            for row, (_, token) in enumerate(selected):
                length = len(token)
                token_ids[row, :length] = token
                attention[row, :, :length] = 0.0
            _, pooled = predictor.model.encoder(token_ids.to(device), attention.to(device))
            for row, (position, _) in enumerate(selected):
                output[position] = pooled[row].float().cpu().numpy()
            cursor += len(selected)
    del predictor
    torch.cuda.empty_cache()
    return output


def encode_esm2(
    accessions: list[str], sequences: list[str], device: torch.device
) -> np.ndarray:
    os.environ["TORCH_HOME"] = "/root/autodl-tmp/.cache/torch"
    import esm  # noqa: E402

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.to(device).eval()
    converter = alphabet.get_batch_converter()
    layer = model.num_layers
    output = np.empty((len(sequences), 1280), dtype=np.float32)
    truncated = [str(sequence)[:1022] for sequence in sequences]
    ordered = sorted(range(len(truncated)), key=lambda index: len(truncated[index]))
    with torch.inference_mode():
        cursor = 0
        completed = 0
        while cursor < len(ordered):
            maximum = len(truncated[ordered[min(len(ordered) - 1, cursor + 7)]]) + 2
            batch_size = min(8, max(1, 4_000_000 // max(1, maximum**2)))
            selected = ordered[cursor : cursor + batch_size]
            batch = [(accessions[index], truncated[index]) for index in selected]
            _, _, tokens = converter(batch)
            result = model(tokens.to(device), repr_layers=[layer], return_contacts=False)
            representation = result["representations"][layer]
            for row, position in enumerate(selected):
                length = len(truncated[position])
                output[position] = representation[row, 1 : length + 2].mean(0).float().cpu().numpy()
            cursor += len(selected)
            completed += len(selected)
            if completed % 25 < len(selected) or completed == len(sequences):
                print(json.dumps({"esm2_expanded_completed": completed, "total": len(sequences)}), flush=True)
    del model
    torch.cuda.empty_cache()
    return output


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if a < b:
            self.parent[b] = a
        else:
            self.parent[a] = b


def homology_clusters(targets: pd.DataFrame) -> dict[str, str]:
    accessions = targets["uniprot_accession"].tolist()
    union = UnionFind(accessions)
    with tempfile.TemporaryDirectory(prefix="kirhub_mmseqs_") as temporary:
        temp = Path(temporary)
        fasta = temp / "targets.fasta"
        result = temp / "all_vs_all.tsv"
        with fasta.open("w") as handle:
            for row in targets.itertuples(index=False):
                handle.write(f">{row.uniprot_accession}\n{row.protein_sequence}\n")
        subprocess.run([
            "mmseqs", "easy-search", str(fasta), str(fasta), str(result), str(temp / "work"),
            "--format-output", "query,target,fident,alnlen,qlen,tlen",
            "--min-seq-id", "0.20", "-c", "0.25", "--cov-mode", "0",
            "-s", "7.5", "--threads", "16", "--remove-tmp-files", "1",
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        alignments = pd.read_csv(
            result, sep="\t", header=None,
            names=["query", "target", "fident", "alnlen", "qlen", "tlen"],
        )
    alignments["shorter_coverage"] = alignments["alnlen"] / alignments[["qlen", "tlen"]].min(axis=1)
    edges = alignments[
        alignments["query"].ne(alignments["target"])
        & alignments["fident"].ge(0.40)
        & alignments["shorter_coverage"].ge(0.80)
    ]
    for row in edges.itertuples(index=False):
        union.union(str(row.query), str(row.target))
    members: dict[str, list[str]] = defaultdict(list)
    for accession in accessions:
        members[union.find(accession)].append(accession)
    return {
        accession: "KH40_" + min(component)
        for component in members.values() for accession in component
    }


def grouped_folds(data: pd.DataFrame, group_column: str) -> dict[str, int]:
    groups = data[group_column].astype(str).to_numpy()
    y = data["strong_inhibition_label"].to_numpy(dtype=np.int8)
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    fold_by_group: dict[str, int] = {}
    for fold, (_, test) in enumerate(splitter.split(np.zeros((len(data), 1)), y, groups)):
        for group in np.unique(groups[test]):
            if str(group) in fold_by_group:
                raise RuntimeError(f"Group assigned twice: {group}")
            fold_by_group[str(group)] = fold
    return fold_by_group


def main() -> None:
    required = [
        WORKBOOK, PAIR_BENCHMARK, CHEMBL_PAIRS, BASELINE_SCORES,
        DEPLOY / "DTIAM_OLD_DRUG720_BERMOL_INDEX_V1.csv.gz",
        DEPLOY / "DTIAM_OLD_DRUG720_BERMOL768_FLOAT32_V1.npy",
        DEPLOY / "DTIAM_PROJECT384_ESM2_INDEX_V1.csv.gz",
        DEPLOY / "DTIAM_PROJECT384_ESM2_T33_650M_1280_FLOAT32_V1.npy",
        OFFICIAL / "DTIAM_BERMOL768_FLOAT32_V1.npy",
        OFFICIAL / "DTIAM_ESM2_T33_650M_1280_FLOAT32_V1.npy",
        OFFICIAL / "DTIAM_ESM2_TARGET_INDEX_V1.csv.gz",
        BERMOL_MODEL, ESM_CHECKPOINT,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for missing pretrained embeddings")
    OUT.mkdir(parents=True, exist_ok=True)

    raw = pd.read_excel(WORKBOOK, sheet_name="Table S4", header=8)
    target_meta = pd.read_excel(WORKBOOK, sheet_name="Table S2", header=23)
    benchmark = pd.read_csv(PAIR_BENCHMARK, low_memory=False)
    chembl = pd.read_csv(CHEMBL_PAIRS, low_memory=False)
    deploy_drug_index = pd.read_csv(DEPLOY / "DTIAM_OLD_DRUG720_BERMOL_INDEX_V1.csv.gz")
    deploy_drug_features = np.load(DEPLOY / "DTIAM_OLD_DRUG720_BERMOL768_FLOAT32_V1.npy", mmap_mode="r")
    official_drug_features = np.load(OFFICIAL / "DTIAM_BERMOL768_FLOAT32_V1.npy", mmap_mode="r")

    # Drug mapping: 720 deployment cache first, full ChEMBL cache second,
    # PubChem structure + frozen BerMol only for the remaining exact names.
    existing_map = benchmark[["kirhub_compound", "ligand_inchikey"]].drop_duplicates()
    existing_map = existing_map.merge(
        deploy_drug_index[["ligand_inchikey", "ligand_smiles", "drug_feature_index"]],
        on="ligand_inchikey", how="left", validate="one_to_one",
    )
    chembl_drugs = chembl[[
        "parent_molecule_name", "parent_standard_inchi_key", "model_ligand_smiles",
        "drug_feature_index",
    ]].drop_duplicates()
    chembl_drugs["name_key"] = chembl_drugs["parent_molecule_name"].map(normalize_name)
    drug_rows = []
    missing_pubchem = []
    for compound in raw["Compound"].astype(str):
        deployment = existing_map[existing_map["kirhub_compound"].eq(compound)]
        if len(deployment) == 1:
            row = deployment.iloc[0]
            drug_rows.append({
                "kirhub_compound": compound, "ligand_inchikey": row.ligand_inchikey,
                "ligand_smiles": row.ligand_smiles, "feature_source": "DEPLOYMENT_720_CACHE",
                "source_feature_index": int(row.drug_feature_index),
            })
            continue
        full = chembl_drugs[chembl_drugs["name_key"].eq(normalize_name(compound))]
        if len(full) == 1:
            row = full.iloc[0]
            drug_rows.append({
                "kirhub_compound": compound,
                "ligand_inchikey": row.parent_standard_inchi_key,
                "ligand_smiles": row.model_ligand_smiles,
                "feature_source": "CHEMBL37_86674_CACHE",
                "source_feature_index": int(row.drug_feature_index),
            })
            continue
        missing_pubchem.append(compound)
        fetched = fetch_pubchem(compound)
        drug_rows.append({
            **fetched, "feature_source": "PUBCHEM_THEN_FROZEN_BERMOL",
            "source_feature_index": -1,
        })
    drugs = pd.DataFrame(drug_rows)
    if len(drugs) != 92 or drugs["kirhub_compound"].nunique() != 92:
        raise RuntimeError("Expanded drug mapping is not exactly 92")
    drugs["drug_feature_index"] = np.arange(len(drugs), dtype=np.int64)
    drug_features = np.empty((len(drugs), 768), dtype=np.float32)
    for row in drugs.itertuples(index=False):
        if row.feature_source == "DEPLOYMENT_720_CACHE":
            drug_features[row.drug_feature_index] = deploy_drug_features[row.source_feature_index]
        elif row.feature_source == "CHEMBL37_86674_CACHE":
            drug_features[row.drug_feature_index] = official_drug_features[row.source_feature_index]
    inferred_drugs = drugs[drugs["feature_source"].eq("PUBCHEM_THEN_FROZEN_BERMOL")]
    if len(inferred_drugs):
        values = encode_bermol(inferred_drugs["ligand_smiles"].tolist(), device)
        drug_features[inferred_drugs["drug_feature_index"].to_numpy(dtype=np.int64)] = values
    drugs["drug_scaffold_group"] = drugs["ligand_smiles"].map(generic_scaffold)

    # Target mapping: exact S2 RBC-name row and first explicitly listed UniProt
    # accession.  Repeated accession means multiple assay constructs and is
    # excluded from the sequence-only primary scope.
    meta_lookup = {
        normalize_name(row["RBC Name"]): row
        for _, row in target_meta.dropna(subset=["RBC Name"]).iterrows()
    }
    target_rows = []
    for construct in raw.columns[1:].astype(str):
        meta = meta_lookup.get(normalize_name(construct))
        accessions = ACCESSION_PATTERN.findall(
            str(meta["Protein Accession #"]).upper() if meta is not None else ""
        )
        target_rows.append({
            "kirhub_wt_construct": construct,
            "supplement_uniprot_accession": accessions[0] if accessions else None,
            "uniprot_accession": CURRENT_UNIPROT_ACCESSION.get(
                accessions[0], accessions[0]
            ) if accessions else None,
            "supplement_accession_count": len(accessions),
            "supplement_hugo_symbol": str(meta["HUGO symbol"]) if meta is not None else None,
        })
    targets_all = pd.DataFrame(target_rows)
    accession_counts = targets_all["uniprot_accession"].value_counts()
    targets_all["single_construct_accession_scope"] = (
        targets_all["uniprot_accession"].notna()
        & targets_all["uniprot_accession"].map(accession_counts).eq(1)
    )
    targets = targets_all[targets_all["single_construct_accession_scope"]].copy().reset_index(drop=True)
    uniprot = fetch_uniprot(sorted(targets["uniprot_accession"].unique()))
    targets = targets.merge(uniprot, on="uniprot_accession", how="left", validate="one_to_one")
    if targets["protein_sequence"].isna().any():
        unavailable = targets.loc[targets["protein_sequence"].isna(), "uniprot_accession"].tolist()
        raise RuntimeError(f"UniProt accessions unavailable: {unavailable}")
    targets["target_feature_index"] = np.arange(len(targets), dtype=np.int64)

    deploy_target_index = pd.read_csv(DEPLOY / "DTIAM_PROJECT384_ESM2_INDEX_V1.csv.gz")
    deploy_target_features = np.load(
        DEPLOY / "DTIAM_PROJECT384_ESM2_T33_650M_1280_FLOAT32_V1.npy", mmap_mode="r"
    )
    official_target_index = pd.read_csv(OFFICIAL / "DTIAM_ESM2_TARGET_INDEX_V1.csv.gz")
    official_target_features = np.load(
        OFFICIAL / "DTIAM_ESM2_T33_650M_1280_FLOAT32_V1.npy", mmap_mode="r"
    )
    chembl_target_meta = chembl[[
        "query_accession", "target_feature_index", "sequence_key"
    ]].drop_duplicates()
    official_target_index = official_target_index.merge(
        chembl_target_meta, on=["target_feature_index", "sequence_key"], how="left",
        validate="one_to_one",
    )
    target_features = np.empty((len(targets), 1280), dtype=np.float32)
    target_feature_source = []
    missing_target_positions = []
    for row in targets.itertuples(index=False):
        sequence = str(row.protein_sequence)
        deployment = deploy_target_index[
            deploy_target_index["uniprot_accession"].astype(str).str.upper().eq(row.uniprot_accession)
            & deploy_target_index["sequence"].astype(str).eq(sequence)
        ]
        if len(deployment) == 1:
            target_features[row.target_feature_index] = deploy_target_features[
                int(deployment.iloc[0].target_feature_index)
            ]
            target_feature_source.append("DEPLOYMENT_384_EXACT_SEQUENCE_CACHE")
            continue
        official = official_target_index[
            official_target_index["query_accession"].astype(str).str.split("-").str[0].str.upper().eq(
                row.uniprot_accession
            )
            & official_target_index["protein_sequence"].astype(str).eq(sequence)
        ]
        if len(official) == 1:
            target_features[row.target_feature_index] = official_target_features[
                int(official.iloc[0].target_feature_index)
            ]
            target_feature_source.append("CHEMBL37_428_EXACT_SEQUENCE_CACHE")
            continue
        target_feature_source.append("UNIPROT_THEN_FROZEN_ESM2")
        missing_target_positions.append(row.target_feature_index)
    targets["feature_source"] = target_feature_source
    if missing_target_positions:
        missing_targets = targets.iloc[missing_target_positions]
        inferred = encode_esm2(
            missing_targets["uniprot_accession"].tolist(),
            missing_targets["protein_sequence"].tolist(), device,
        )
        target_features[np.asarray(missing_target_positions, dtype=np.int64)] = inferred

    cluster_lookup = homology_clusters(targets)
    targets["target_homology_cluster"] = targets["uniprot_accession"].map(cluster_lookup)
    project_construct_map = benchmark[
        benchmark["kirhub_wt_construct_count"].eq(1)
    ][["kirhub_wt_columns", "target_chembl_id"]].drop_duplicates()
    project_construct_map = project_construct_map.rename(
        columns={"kirhub_wt_columns": "kirhub_wt_construct"}
    )
    targets = targets.merge(
        project_construct_map, on="kirhub_wt_construct", how="left", validate="one_to_one"
    )

    # Materialize only measured values; NaN remains unknown, never negative.
    long = raw.melt(id_vars="Compound", var_name="kirhub_wt_construct", value_name="residual_activity_pct_1uM")
    long = long[long["residual_activity_pct_1uM"].notna()].copy()
    long = long.merge(
        drugs, left_on="Compound", right_on="kirhub_compound", how="inner", validate="many_to_one"
    )
    long = long.merge(
        targets[[
            "kirhub_wt_construct", "uniprot_accession", "uniprot_gene_names",
            "uniprot_protein_name", "target_feature_index", "target_homology_cluster",
            "target_chembl_id",
        ]],
        on="kirhub_wt_construct", how="inner", validate="many_to_one",
    )
    long["residual_activity_pct_1uM"] = pd.to_numeric(
        long["residual_activity_pct_1uM"], errors="raise"
    )
    long["inhibition_fraction_1uM"] = 1.0 - long["residual_activity_pct_1uM"] / 100.0
    long["strong_inhibition_label"] = long["inhibition_fraction_1uM"].ge(0.70).astype(np.int8)
    long["pairId"] = long["ligand_inchikey"] + "__UNIPROT_" + long["uniprot_accession"]
    drug_fold_lookup = grouped_folds(long, "drug_scaffold_group")
    target_fold_lookup = grouped_folds(long, "target_homology_cluster")
    long["drug_scaffold_cold_fold"] = long["drug_scaffold_group"].map(drug_fold_lookup).astype(np.int8)
    long["target_homology_cold_fold"] = long["target_homology_cluster"].map(
        target_fold_lookup
    ).astype(np.int8)
    long["assay_type"] = "HotSpot biochemical residual kinase activity"
    long["assay_concentration_uM"] = 1.0
    long["label_semantics"] = "functional_inhibition_not_direct_binding_affinity"
    long["source_doi"] = "10.1038/s41587-026-03090-8"

    baselines = pd.read_csv(BASELINE_SCORES, low_memory=False)
    baseline_columns = [
        "old_drug_leakage_safe_score_v10", "biomaster_full_fit_v10_borda_score",
        "dtiam_probability", "ensemble_drug_to_target_logit",
    ]
    baselines = baselines[["ligand_inchikey", "target_chembl_id"] + baseline_columns]
    long = long.merge(
        baselines, on=["ligand_inchikey", "target_chembl_id"], how="left",
        validate="many_to_one",
    )
    long["deployment_720x384_comparison_scope"] = long[baseline_columns].notna().all(axis=1)

    checks = {
        "exact_92_drugs": len(drugs) == 92,
        "exact_79_deployment_8_chembl_5_pubchem": drugs["feature_source"].value_counts().to_dict()
        == {
            "DEPLOYMENT_720_CACHE": 79, "CHEMBL37_86674_CACHE": 8,
            "PUBCHEM_THEN_FROZEN_BERMOL": 5,
        },
        "exact_345_single_construct_accession_targets": len(targets) == 345,
        "all_features_finite": bool(np.isfinite(drug_features).all() and np.isfinite(target_features).all()),
        "all_labels_bounded": long["inhibition_fraction_1uM"].between(0, 1).all(),
        "no_duplicate_pairs": not long.duplicated(["ligand_inchikey", "uniprot_accession"]).any(),
        "unknown_values_not_materialized": long["residual_activity_pct_1uM"].notna().all(),
        "five_drug_scaffold_folds": set(long["drug_scaffold_cold_fold"]) == set(range(5)),
        "five_target_homology_folds": set(long["target_homology_cold_fold"]) == set(range(5)),
        "scaffolds_do_not_cross_folds": long.groupby("drug_scaffold_group")[
            "drug_scaffold_cold_fold"
        ].nunique().eq(1).all(),
        "target_clusters_do_not_cross_folds": long.groupby("target_homology_cluster")[
            "target_homology_cold_fold"
        ].nunique().eq(1).all(),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks, indent=2))

    drug_path = OUT / "KIRHUB_EXPANDED_DRUG92_INDEX_V1.csv"
    drug_feature_path = OUT / "KIRHUB_EXPANDED_DRUG92_BERMOL768_FLOAT32_V1.npy"
    target_all_path = OUT / "KIRHUB_ALL409_CONSTRUCT_ACCESSION_AUDIT_V1.csv"
    target_path = OUT / "KIRHUB_EXPANDED_TARGET345_INDEX_V1.csv.gz"
    target_feature_path = OUT / "KIRHUB_EXPANDED_TARGET345_ESM2_1280_FLOAT32_V1.npy"
    pair_path = OUT / "KIRHUB_EXPANDED_FUNCTIONAL_PAIRS_V1.csv.gz"
    drugs.to_csv(drug_path, index=False)
    np.save(drug_feature_path, drug_features, allow_pickle=False)
    targets_all.to_csv(target_all_path, index=False)
    targets.to_csv(target_path, index=False, compression="gzip")
    np.save(target_feature_path, target_features, allow_pickle=False)
    long.to_csv(pair_path, index=False, compression="gzip")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "counts": {
            "drugs": len(drugs), "single_construct_accession_targets": len(targets),
            "measured_pairs": len(long), "strong_pairs_ge70pct": int(long["strong_inhibition_label"].sum()),
            "comparison_pairs_720x384": int(long["deployment_720x384_comparison_scope"].sum()),
        },
        "drug_feature_routes": drugs["feature_source"].value_counts().to_dict(),
        "target_feature_routes": targets["feature_source"].value_counts().to_dict(),
        "pretrained_policy": "frozen BerMol768 and frozen ESM2-650M1280; no backbone fine-tuning",
        "exclusion_policy": (
            "constructs without an exact S2 UniProt accession and accessions repeated across "
            "multiple complexes are excluded; missing assay cells remain unknown"
        ),
        "checks": checks,
        "inputs_sha256": {path.name: sha256(path) for path in required},
        "artifacts": {
            path.name: sha256(path) for path in [
                drug_path, drug_feature_path, target_all_path, target_path,
                target_feature_path, pair_path,
            ]
        },
    }
    report_path = OUT / "KIRHUB_EXPANDED_PRETRAINED_FEATURE_SUMMARY_V1.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
