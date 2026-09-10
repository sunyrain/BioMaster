#!/usr/bin/env python3
"""Build a signed Boltz-2 calibration package from the frozen 338-target protocols.

The package preserves the existing development/locked-evaluation split and uses
the exact experimental construct chains and crystal-defined pocket for every
target.  It intentionally does not invent controls for targets without measured
positive/negative panels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import gemmi
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "outputs/gnina_calibration_338_v1/GNINA_CALIBRATION_CONTROL_PANEL_338_V1.csv.gz"
QUALIFICATION = (
    ROOT
    / "outputs/gnina_calibration_338_v1/evaluation/GNINA_TARGET_QUALIFICATION_338_FULL_V1.csv"
)
DEFAULT_OUT = ROOT / "outputs/boltz2_calibration_338_v1"

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def safe_token(value: Any) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_")
    return token[:180] or "pair"


def resolve_protocol_path(protocol: dict[str, Any], key: str) -> Path:
    value = str(protocol.get("files", {}).get(key, ""))
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_pdb_chains(path: Path) -> dict[str, list[dict[str, Any]]]:
    chains: dict[str, list[dict[str, Any]]] = {}
    residue_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM  "):
                continue
            residue_name = line[17:20].strip().upper()
            if residue_name not in AA3_TO_1:
                continue
            chain_id = line[21].strip() or "_"
            residue_number = line[22:26].strip()
            insertion_code = line[26].strip()
            key = (chain_id, residue_number, insertion_code)
            residue = residue_lookup.get(key)
            if residue is None:
                residue = {
                    "chain_id": chain_id,
                    "residue_number": residue_number,
                    "insertion_code": insertion_code,
                    "residue_name": residue_name,
                    "sequence_code": AA3_TO_1[residue_name],
                    "coordinates": [],
                }
                residue_lookup[key] = residue
                chains.setdefault(chain_id, []).append(residue)
            element = line[76:78].strip().upper()
            atom_name = line[12:16].strip().upper()
            if element == "H" or (not element and atom_name.startswith("H")):
                continue
            try:
                residue["coordinates"].append(
                    (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                )
            except ValueError:
                continue
    return chains


def ligand_heavy_coordinates(path: Path) -> np.ndarray:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    molecule = next((mol for mol in supplier if mol is not None), None)
    if molecule is None or molecule.GetNumConformers() == 0:
        raise ValueError(f"Cannot read crystal ligand coordinates: {path}")
    conformer = molecule.GetConformer()
    coordinates = [
        tuple(conformer.GetAtomPosition(atom.GetIdx()))
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    if not coordinates:
        raise ValueError(f"Crystal ligand has no heavy coordinates: {path}")
    return np.asarray(coordinates, dtype=float)


def select_construct_chains(
    protocol: dict[str, Any], chains: dict[str, list[dict[str, Any]]]
) -> list[str]:
    requested = [str(chain) for chain in protocol.get("construct_chain_ids", [])]
    selected = [chain for chain in requested if chain in chains]
    target_chain = str(protocol.get("target_chain_id", ""))
    if target_chain in chains and target_chain not in selected:
        selected.insert(0, target_chain)
    if not selected and chains:
        selected = [max(chains, key=lambda chain: len(chains[chain]))]
    return selected


def pocket_contacts(
    chains: dict[str, list[dict[str, Any]]],
    selected_chains: list[str],
    input_ids: dict[str, str],
    ligand_coordinates: np.ndarray,
    cutoff: float,
    max_contacts: int,
) -> list[list[Any]]:
    candidates: list[tuple[float, str, int]] = []
    for chain_id in selected_chains:
        for position, residue in enumerate(chains[chain_id], start=1):
            coordinates = residue["coordinates"]
            if not coordinates:
                continue
            atom_coordinates = np.asarray(coordinates, dtype=float)
            distances = np.linalg.norm(
                atom_coordinates[:, None, :] - ligand_coordinates[None, :, :], axis=2
            )
            minimum = float(distances.min())
            if minimum <= cutoff:
                candidates.append((minimum, input_ids[chain_id], position))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return [[chain, position] for _, chain, position in candidates[:max_contacts]]


def write_stripped_template_mmcif(
    source: Path,
    selected_chains: list[str],
    destination: Path,
) -> dict[str, str]:
    structure = gemmi.read_structure(str(source))
    structure.remove_ligands_and_waters()
    selected = set(selected_chains)
    for model in structure:
        remove_names = [chain.name for chain in model if chain.name not in selected]
        for chain_name in remove_names:
            model.remove_chain(chain_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    structure.make_mmcif_document().write_file(str(destination))
    block = gemmi.cif.read_file(str(destination)).sole_block()
    atom_site = block.find(
        [
            "_atom_site.group_PDB",
            "_atom_site.label_asym_id",
            "_atom_site.auth_asym_id",
        ]
    )
    auth_to_label: dict[str, str] = {}
    for row in atom_site:
        if str(row[0]) != "ATOM":
            continue
        label_id = str(row[1])
        auth_id = str(row[2])
        previous = auth_to_label.setdefault(auth_id, label_id)
        if previous != label_id:
            raise ValueError(
                f"Ambiguous mmCIF auth-to-label chain mapping for {auth_id}: "
                f"{previous}, {label_id}"
            )
    missing = sorted(set(selected_chains) - set(auth_to_label))
    if missing:
        raise ValueError(f"Selected template chains absent from stripped mmCIF: {missing}")
    return auth_to_label


def target_template(
    protocol_path: Path,
    cutoff: float,
    max_contacts: int,
    template_dir: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    # Use the project-generated construct PDB only to recover the exact chain
    # sequences and crystal-pocket contacts. Boltz 2.2.1 parses the original
    # mmCIF template more reliably than PDB files with incomplete SEQRES data.
    coordinate_receptor = resolve_protocol_path(protocol, "construct_raw_pdb")
    template_source = resolve_protocol_path(protocol, "raw_mmcif")
    ligand = resolve_protocol_path(protocol, "reference_ligand_crystal_sdf")
    if not coordinate_receptor.is_file() or not template_source.is_file() or not ligand.is_file():
        raise FileNotFoundError(f"Missing receptor or reference ligand for {protocol_path}")
    chains = parse_pdb_chains(coordinate_receptor)
    selected_chains = select_construct_chains(protocol, chains)
    if not selected_chains:
        raise ValueError(f"No protein construct chains in {coordinate_receptor}")
    available_input_ids = [chr(code) for code in range(ord("A"), ord("Z"))]
    if len(selected_chains) > len(available_input_ids):
        raise ValueError(f"Too many construct chains for Boltz input: {len(selected_chains)}")
    input_ids = dict(
        zip(selected_chains, available_input_ids[: len(selected_chains)], strict=True)
    )
    sequences = {
        input_ids[chain]: "".join(residue["sequence_code"] for residue in chains[chain])
        for chain in selected_chains
    }
    contacts = pocket_contacts(
        chains,
        selected_chains,
        input_ids,
        ligand_heavy_coordinates(ligand),
        cutoff,
        max_contacts,
    )
    if not contacts:
        raise ValueError(f"No crystal-defined pocket contacts for {protocol_path}")
    original_total_residues = sum(len(sequence) for sequence in sequences.values())
    crop_windows: dict[str, list[int]] = {}
    if original_total_residues > 1800:
        reverse_ids = {input_id: chain for chain, input_id in input_ids.items()}
        contact_positions: dict[str, list[int]] = {}
        for input_id, position in contacts:
            contact_positions.setdefault(reverse_ids[input_id], []).append(int(position))
        cropped_chains = [chain for chain in selected_chains if contact_positions.get(chain)]
        cropped_input_ids = dict(
            zip(
                cropped_chains,
                available_input_ids[: len(cropped_chains)],
                strict=True,
            )
        )
        cropped_sequences: dict[str, str] = {}
        contact_remap: dict[tuple[str, int], tuple[str, int]] = {}
        for chain in cropped_chains:
            positions = contact_positions[chain]
            start = max(1, min(positions) - 200)
            end = min(len(chains[chain]), max(positions) + 200)
            crop_windows[chain] = [start, end]
            cropped_sequences[cropped_input_ids[chain]] = "".join(
                residue["sequence_code"] for residue in chains[chain][start - 1 : end]
            )
            for position in positions:
                contact_remap[(chain, position)] = (
                    cropped_input_ids[chain],
                    position - start + 1,
                )
        contacts = [
            list(contact_remap[(reverse_ids[input_id], int(position))])
            for input_id, position in contacts
            if (reverse_ids[input_id], int(position)) in contact_remap
        ]
        selected_chains = cropped_chains
        input_ids = cropped_input_ids
        sequences = cropped_sequences
    template_structure = template_dir / f"{protocol['target_chembl_id']}_protein_template.cif"
    template_ids = write_stripped_template_mmcif(
        template_source, selected_chains, template_structure
    )
    return {
        "protocol": protocol,
        "protocol_path": protocol_path,
        "coordinate_receptor_path": coordinate_receptor,
        "template_source_path": template_source,
        "template_structure_path": template_structure,
        "reference_ligand_path": ligand,
        "selected_template_chains": selected_chains,
        "template_label_chain_ids": template_ids,
        "input_ids": input_ids,
        "sequences": sequences,
        "contacts": contacts,
        "original_total_residues": original_total_residues,
        "sequence_crop_applied": bool(crop_windows),
        "sequence_crop_windows": crop_windows,
        "coordinate_receptor_sha256": sha256(coordinate_receptor),
        "template_source_sha256": sha256(template_source),
        "template_structure_sha256": sha256(template_structure),
        "reference_ligand_sha256": sha256(ligand),
        "protocol_sha256": sha256(protocol_path),
    }


def yaml_payload(template: dict[str, Any], smiles: str, pocket_distance: float) -> dict[str, Any]:
    protein_entries = [
        {"protein": {"id": input_id, "sequence": sequence, "msa": "empty"}}
        for input_id, sequence in template["sequences"].items()
    ]
    input_chain_ids = [template["input_ids"][chain] for chain in template["selected_template_chains"]]
    return {
        "version": 1,
        "sequences": protein_entries + [{"ligand": {"id": "Z", "smiles": smiles}}],
        "properties": [{"affinity": {"binder": "Z"}}],
        "templates": [
            {
                "cif": str(template["template_structure_path"].resolve()),
                "chain_id": input_chain_ids,
                "template_id": [
                    template["template_label_chain_ids"][chain]
                    for chain in template["selected_template_chains"]
                ],
            }
        ],
        "constraints": [
            {
                "pocket": {
                    "binder": "Z",
                    "contacts": template["contacts"],
                    "max_distance": float(pocket_distance),
                    "force": False,
                }
            }
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--qualification", type=Path, default=QUALIFICATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--contact-cutoff", type=float, default=6.0)
    parser.add_argument("--max-contacts", type=int, default=48)
    parser.add_argument("--pocket-max-distance", type=float, default=6.0)
    parser.add_argument(
        "--expected-panel-targets",
        type=int,
        default=290,
        help="Expected number of targets represented by the supplied control panel.",
    )
    args = parser.parse_args()
    panel_path = args.panel if args.panel.is_absolute() else ROOT / args.panel
    qualification_path = (
        args.qualification if args.qualification.is_absolute() else ROOT / args.qualification
    )
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    input_dir = output_dir / "inputs"
    template_dir = output_dir / "target_templates"
    input_dir.mkdir(parents=True, exist_ok=True)
    template_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(panel_path, low_memory=False).fillna("")
    qualification = pd.read_csv(qualification_path, low_memory=False).fillna("")
    qualification_lookup = qualification.set_index("target_chembl_id").to_dict(orient="index")
    if panel["control_pair_id"].duplicated().any():
        raise ValueError("Calibration control pair IDs are not unique")
    if set(panel["control_class"]) != {"positive", "negative"}:
        raise ValueError("Calibration panel must contain positive and negative controls")
    if set(panel["calibration_split"]) != {"PROTOCOL_DEVELOPMENT", "LOCKED_EVALUATION"}:
        raise ValueError("Calibration panel split is not frozen as expected")
    remote_ranks = panel.loc[
        panel["gate_b_remote_candidate"].astype(bool)
        & panel["control_class"].eq("positive"),
        ["target_chembl_id", "control_rank_within_target_class"],
    ].drop_duplicates()
    remote_rank_keys = set(map(tuple, remote_ranks.itertuples(index=False, name=None)))
    panel["gate_b_remote_matched_negative"] = [
        row.control_class == "negative"
        and (row.target_chembl_id, row.control_rank_within_target_class) in remote_rank_keys
        for row in panel.itertuples(index=False)
    ]
    panel["gate_b_evaluation_member"] = (
        panel["gate_b_remote_candidate"].astype(bool)
        | panel["gate_b_remote_matched_negative"].astype(bool)
    )

    templates: dict[str, dict[str, Any]] = {}
    target_audit: list[dict[str, Any]] = []
    for target_id, group in panel.groupby("target_chembl_id", sort=True):
        protocol_values = group["protocol_json"].astype(str).drop_duplicates().tolist()
        if len(protocol_values) != 1:
            raise ValueError(f"Target {target_id} has {len(protocol_values)} protocol paths")
        protocol_path = Path(protocol_values[0])
        if not protocol_path.is_absolute():
            protocol_path = ROOT / protocol_path
        try:
            template = target_template(
                protocol_path,
                args.contact_cutoff,
                args.max_contacts,
                template_dir,
            )
            templates[target_id] = template
            status = "READY"
            error = ""
        except Exception as exc:  # noqa: BLE001
            template = {}
            status = "FAILED_INPUT_AUDIT"
            error = f"{type(exc).__name__}:{exc}"
        q = qualification_lookup.get(target_id, {})
        target_audit.append(
            {
                "target_chembl_id": target_id,
                "gene_symbol": str(group.iloc[0]["gene_symbol"]),
                "control_rows": int(len(group)),
                "development_rows": int(group["calibration_split"].eq("PROTOCOL_DEVELOPMENT").sum()),
                "locked_evaluation_rows": int(group["calibration_split"].eq("LOCKED_EVALUATION").sum()),
                "positive_rows": int(group["control_class"].eq("positive").sum()),
                "negative_rows": int(group["control_class"].eq("negative").sum()),
                "gnina_target_qualification": q.get("gnina_target_qualification", ""),
                "gnina_redock_status": q.get("redock_status_final", ""),
                "boltz_input_status": status,
                "boltz_input_error": error,
                "construct_chain_count": len(template.get("selected_template_chains", [])),
                "construct_template_chains": ";".join(template.get("selected_template_chains", [])),
                "template_mmcif_label_chains": ";".join(
                    template.get("template_label_chain_ids", {}).get(chain, "")
                    for chain in template.get("selected_template_chains", [])
                ),
                "boltz_input_chain_ids": ";".join(template.get("sequences", {}).keys()),
                "protein_total_residues": sum(len(seq) for seq in template.get("sequences", {}).values()),
                "protein_original_total_residues": template.get("original_total_residues", 0),
                "sequence_crop_applied": bool(template.get("sequence_crop_applied", False)),
                "sequence_crop_windows": json.dumps(
                    template.get("sequence_crop_windows", {}), sort_keys=True
                ),
                "pocket_contact_count": len(template.get("contacts", [])),
                "docking_context_included": bool(template.get("protocol", {}).get("docking_context_included", False)),
                "boltz_context_limitation": "METAL_OR_COFACTOR_NOT_EXPLICITLY_MODELLED"
                if template.get("protocol", {}).get("docking_context_included", False)
                else "",
                "protocol_path": str(protocol_path),
                "protocol_sha256": template.get("protocol_sha256", ""),
                "coordinate_receptor_path": str(template.get("coordinate_receptor_path", "")),
                "coordinate_receptor_sha256": template.get("coordinate_receptor_sha256", ""),
                "template_structure_path": str(template.get("template_structure_path", "")),
                "template_source_path": str(template.get("template_source_path", "")),
                "template_source_sha256": template.get("template_source_sha256", ""),
                "template_structure_sha256": template.get("template_structure_sha256", ""),
                "reference_ligand_sha256": template.get("reference_ligand_sha256", ""),
            }
        )

    failed_targets = {row["target_chembl_id"] for row in target_audit if row["boltz_input_status"] != "READY"}
    if failed_targets:
        raise RuntimeError(f"Boltz target input audit failed for {len(failed_targets)} targets")

    split_order = {"PROTOCOL_DEVELOPMENT": 0, "LOCKED_EVALUATION": 1}
    class_order = {"positive": 0, "negative": 1}
    panel["_split_order"] = panel["calibration_split"].map(split_order)
    panel["_class_order"] = panel["control_class"].map(class_order)
    panel = panel.sort_values(
        ["target_chembl_id", "_split_order", "_class_order", "control_rank_within_target_class", "control_pair_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    source_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(panel.iterrows(), start=1):
        target_id = str(row["target_chembl_id"])
        template = templates[target_id]
        smiles = str(row["canonical_control_smiles"]).strip()
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid control SMILES for {row['control_pair_id']}")
        canonical_smiles = Chem.MolToSmiles(molecule, isomericSmiles=True)
        payload = yaml_payload(template, canonical_smiles, args.pocket_max_distance)
        yaml_name = f"{rank:06d}_{safe_token(row['control_pair_id'])}.yaml"
        yaml_path = input_dir / yaml_name
        yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False, width=120), encoding="utf-8")
        yaml_hash = sha256(yaml_path)
        input_signature = payload_sha256(
            {
                "yamlSha256": yaml_hash,
                "protocolSha256": template["protocol_sha256"],
                "coordinateReceptorSha256": template["coordinate_receptor_sha256"],
                "templateStructureSha256": template["template_structure_sha256"],
                "templateSourceSha256": template["template_source_sha256"],
                "referenceLigandSha256": template["reference_ligand_sha256"],
                "controlPairId": row["control_pair_id"],
                "calibrationSplit": row["calibration_split"],
                "controlClass": row["control_class"],
            }
        )
        q = qualification_lookup.get(target_id, {})
        common = {
            "externalQueueRank": rank,
            "pairId": row["control_pair_id"],
            "target_chembl_id": target_id,
            "gene_symbol": row["gene_symbol"],
            "drugId": row["parent_molecule_chembl_id"],
            "drug": row["parent_molecule_name"] or row["parent_molecule_chembl_id"],
            "target": row["gene_symbol"],
            "canonicalSmiles": canonical_smiles,
            "knownDrugTargetPair": row["control_class"] == "positive",
            "control_class": row["control_class"],
            "pair_label": row["pair_label"],
            "calibration_split": row["calibration_split"],
            "gate_b_remote_candidate": bool(row["gate_b_remote_candidate"]),
            "gate_b_evaluation_member": bool(
                row["gate_b_evaluation_member"]
            ),
            "gate_b_remote_matched_negative": bool(
                row["gate_b_remote_matched_negative"]
            ),
            "max_tanimoto_to_development_positives": row["max_tanimoto_to_development_positives"],
            "panel_quality": row["panel_quality"],
            "gnina_target_qualification": q.get("gnina_target_qualification", ""),
            "gnina_redock_status": q.get("redock_status_final", ""),
            "pdb_id": row["pdb_id"],
            "construct_template_chains": ";".join(template["selected_template_chains"]),
            "template_mmcif_label_chains": ";".join(
                template["template_label_chain_ids"][chain]
                for chain in template["selected_template_chains"]
            ),
            "boltz_input_chain_ids": ";".join(template["sequences"].keys()),
            "protein_total_residues": sum(len(seq) for seq in template["sequences"].values()),
            "protein_original_total_residues": template["original_total_residues"],
            "sequence_crop_applied": bool(template["sequence_crop_applied"]),
            "sequence_crop_windows": json.dumps(template["sequence_crop_windows"], sort_keys=True),
            "pocket_contact_count": len(template["contacts"]),
            "docking_context_included": bool(template["protocol"].get("docking_context_included", False)),
            "protocolPath": str(template["protocol_path"]),
            "protocolSha256": template["protocol_sha256"],
            "coordinateReceptorPdbPath": str(template["coordinate_receptor_path"]),
            "coordinateReceptorPdbSha256": template["coordinate_receptor_sha256"],
            "templateStructurePath": str(template["template_structure_path"]),
            "templateStructureSha256": template["template_structure_sha256"],
            "templateSourcePath": str(template["template_source_path"]),
            "templateSourceSha256": template["template_source_sha256"],
            "referenceLigandSha256": template["reference_ligand_sha256"],
            "yamlFile": yaml_name,
            "yamlPath": str(yaml_path),
            "yamlSha256": yaml_hash,
            "inputSignatureSha256": input_signature,
            "inputSignatureVersion": "boltz2_calibration_multichain_template_pocket_v1",
        }
        source_rows.append(common)
        manifest_rows.append(common)

    source = pd.DataFrame(source_rows)
    manifest = pd.DataFrame(manifest_rows)
    target_audit_frame = pd.DataFrame(target_audit)
    source_path = output_dir / "BOLTZ2_CALIBRATION_CONTROL_SOURCE_V1.csv.gz"
    manifest_path = output_dir / "BOLTZ2_CALIBRATION_INPUT_MANIFEST_V1.csv"
    target_audit_path = output_dir / "BOLTZ2_CALIBRATION_TARGET_INPUT_AUDIT_V1.csv"
    source.to_csv(source_path, index=False, compression="gzip")
    manifest.to_csv(manifest_path, index=False)
    target_audit_frame.to_csv(target_audit_path, index=False)

    all_target_ids = set(qualification["target_chembl_id"])
    panel_target_ids = set(source["target_chembl_id"])
    no_control = qualification[~qualification["target_chembl_id"].isin(panel_target_ids)].copy()
    no_control_path = output_dir / "BOLTZ2_TARGETS_WITHOUT_FORMAL_CONTROL_PANEL_V1.csv"
    no_control.to_csv(no_control_path, index=False)
    expected_without_panel = len(all_target_ids) - args.expected_panel_targets
    if (
        len(all_target_ids) != 338
        or len(panel_target_ids) != args.expected_panel_targets
        or len(no_control) != expected_without_panel
    ):
        raise ValueError(
            f"Unexpected target counts: all={len(all_target_ids)} panel={len(panel_target_ids)} no_control={len(no_control)}"
        )

    summary = {
        "created_utc": now_utc(),
        "status": "PASS",
        "boltz_version": "2.2.1",
        "formal_targets": 338,
        "targets_with_formal_control_panel": int(source["target_chembl_id"].nunique()),
        "targets_without_formal_control_panel": int(len(no_control)),
        "control_rows": int(len(source)),
        "development_rows": int(source["calibration_split"].eq("PROTOCOL_DEVELOPMENT").sum()),
        "locked_evaluation_rows": int(source["calibration_split"].eq("LOCKED_EVALUATION").sum()),
        "positive_rows": int(source["control_class"].eq("positive").sum()),
        "negative_rows": int(source["control_class"].eq("negative").sum()),
        "multi_chain_targets": int(target_audit_frame["construct_chain_count"].gt(1).sum()),
        "pocket_domain_cropped_targets": int(
            target_audit_frame["sequence_crop_applied"].astype(bool).sum()
        ),
        "context_limited_targets": int(target_audit_frame["docking_context_included"].astype(bool).sum()),
        "pocket_contact_min": int(target_audit_frame["pocket_contact_count"].min()),
        "pocket_contact_median": float(target_audit_frame["pocket_contact_count"].median()),
        "pocket_contact_max": int(target_audit_frame["pocket_contact_count"].max()),
        "method": {
            "protein_state": "experimental construct chains with original signed mmCIF template",
            "template_mapping": "explicit Boltz input-chain to experimental-template-chain mapping",
            "pocket": "heavy-atom residues within 6 A of the crystallographic reference ligand",
            "msa": "empty; experimental template conditioned",
            "primary_score": "affinity_probability_binary",
            "evaluation": "development distribution plus untouched locked evaluation split",
        },
        "inputs": {
            "panel": str(panel_path),
            "panel_sha256": sha256(panel_path),
            "qualification": str(qualification_path),
            "qualification_sha256": sha256(qualification_path),
        },
        "outputs": {
            "source": str(source_path),
            "manifest": str(manifest_path),
            "target_input_audit": str(target_audit_path),
            "targets_without_formal_controls": str(no_control_path),
            "input_directory": str(input_dir),
        },
    }
    summary_path = output_dir / "BOLTZ2_CALIBRATION_PREPARATION_SUMMARY_V1.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
