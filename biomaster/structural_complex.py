"""Experimental complex mapping with independent ligand-input conformers."""
from __future__ import annotations
import ast
import hashlib
from pathlib import Path

from Bio import SeqIO
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from biomaster.odti_local_features_v3 import molecular_graph
from biomaster.odti_pockets_v3 import THREE_TO_ONE


SIDECHAIN = {
    'A': 'CB', 'R': 'CB CG CD NE CZ NH1 NH2', 'N': 'CB CG OD1 ND2', 'D': 'CB CG OD1 OD2',
    'C': 'CB SG', 'Q': 'CB CG CD OE1 NE2', 'E': 'CB CG CD OE1 OE2', 'G': '',
    'H': 'CB CG ND1 CD2 CE1 NE2', 'I': 'CB CG1 CG2 CD1', 'L': 'CB CG CD1 CD2',
    'K': 'CB CG CD CE NZ', 'M': 'CB CG SD CE', 'F': 'CB CG CD1 CD2 CE1 CE2 CZ',
    'P': 'CB CG CD', 'S': 'CB OG', 'T': 'CB OG1 CG2',
    'W': 'CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2', 'Y': 'CB CG CD1 CD2 CE1 CE2 CZ OH',
    'V': 'CB CG1 CG2',
}


def canonical_key(mol):
    return Chem.MolToInchiKey(Chem.RemoveHs(mol))


def mapped_ligand(folder, smiles, seed):
    expected = Chem.MolFromSmiles(smiles)
    if expected is None:
        raise ValueError('invalid_annotation_ligand')
    target = canonical_key(expected)
    matches = []
    for path in sorted((Path(folder) / 'ligand_files').glob('*.sdf')):
        mol = Chem.MolFromMolFile(str(path), sanitize=True, removeHs=True)
        if mol is not None and canonical_key(mol) == target:
            matches.append((path, mol))
    if len(matches) != 1:
        raise ValueError('ambiguous_or_unmatched_sdf_identity_and_stereochemistry')
    path, original = matches[0]
    # Serialize without atom maps, then reorder the bound conformer by RDKit's
    # explicit SMILES output order so labels, graph and encoder share atom IDs.
    smiles = Chem.MolToSmiles(original, canonical=False, isomericSmiles=True)
    order = ast.literal_eval(original.GetProp('_smilesAtomOutputOrder'))
    bound = Chem.RenumberAtoms(original, order)
    graph_mol = Chem.MolFromSmiles(smiles)
    if graph_mol is None or graph_mol.GetNumAtoms() != bound.GetNumAtoms():
        raise ValueError('ligand_order_mapping_failed')
    if not np.array_equal(Chem.GetAdjacencyMatrix(graph_mol, useBO=True), Chem.GetAdjacencyMatrix(bound, useBO=True)):
        raise ValueError('ligand_bond_order_mapping_failed')
    if [a.GetSymbol() for a in graph_mol.GetAtoms()] != [a.GetSymbol() for a in bound.GetAtoms()]:
        raise ValueError('ligand_element_order_mapping_failed')
    coords = np.asarray(bound.GetConformer().GetPositions(), np.float32)
    for bond in bound.GetBonds():
        d = np.linalg.norm(coords[bond.GetBeginAtomIdx()] - coords[bond.GetEndAtomIdx()])
        if not 0.6 < d < 2.6:
            raise ValueError('experimental_ligand_bond_length_outlier')
    # Model input uses an independent conformer. Bound coordinates remain labels.
    generated = Chem.AddHs(graph_mol)
    params = AllChem.ETKDGv3(); params.randomSeed = seed; params.numThreads = 1
    params.timeout = 15; params.maxIterations = 200
    if AllChem.EmbedMolecule(generated, params) != 0:
        raise ValueError('independent_conformer_generation_failed')
    if AllChem.MMFFHasAllMoleculeParams(generated):
        optimized = AllChem.MMFFOptimizeMolecule(generated, maxIters=200)
    else:
        optimized = AllChem.UFFOptimizeMolecule(generated, maxIters=200)
    free = np.asarray(generated.GetConformer().GetPositions()[:len(coords)], np.float32)
    graph = molecular_graph(smiles, max_atoms=128)
    if not graph['available'] or not np.isfinite(free).all():
        raise ValueError('invalid_molecule_graph_or_conformer')
    return dict(atoms=[a.GetSymbol() for a in graph_mol.GetAtoms()], coordinates=free,
                bound_coordinates=coords, atom_chemistry=graph['atom_features'], bond=graph['bond_type'],
                stereo=graph['bond_stereo'], smiles=smiles, inchikey=target,
                sdf_path=str(path), input_geometry='INDEPENDENT_ETKDG_MMFF_OR_UFF', optimize_code=int(optimized))


def mapped_receptor(folder):
    sequences = {r.id: str(r.seq) for r in SeqIO.parse(str(Path(folder) / 'sequences.fasta'), 'fasta')}
    d = MMCIF2Dict(str(Path(folder) / 'receptor.cif'))
    n = len(d['_atom_site.Cartn_x'])
    atoms, skipped_alt = {}, 0
    for i in range(n):
        if d.get('_atom_site.pdbx_PDB_model_num', ['1'] * n)[i] != '1':
            continue
        chain = d['_atom_site.label_asym_id'][i]
        if chain not in sequences:
            continue  # water/nonprotein chains do not receive residue labels
        label = d['_atom_site.label_seq_id'][i]
        if label in ['.', '?']:
            continue
        index = int(label) - 1
        aa = THREE_TO_ONE.get(d['_atom_site.label_comp_id'][i], 'X')
        if not 0 <= index < len(sequences[chain]) or sequences[chain][index] != aa:
            raise ValueError('cif_fasta_residue_mismatch')
        alternate = d.get('_atom_site.label_alt_id', ['.'] * n)[i]
        if alternate not in ['.', '?', 'A']:
            skipped_alt += 1; continue
        element = d['_atom_site.type_symbol'][i].upper()
        if element in ['H', 'D']:
            continue
        occupancy = float(d['_atom_site.occupancy'][i])
        if occupancy <= 0:
            continue
        name = d['_atom_site.label_atom_id'][i]
        key = (chain, index, name)
        if key in atoms:
            raise ValueError('ambiguous_alternate_atom')
        xyz = np.array([float(d['_atom_site.Cartn_' + axis][i]) for axis in ['x', 'y', 'z']], np.float32)
        if not np.isfinite(xyz).all():
            raise ValueError('invalid_protein_coordinate')
        atoms[key] = (element, xyz, np.clip(occupancy, 0, 1))
    if not atoms:
        raise ValueError('no_mapped_protein_atoms')
    return sequences, atoms, skipped_alt


def region_record(atoms, sequences, keys):
    xyz, symbols, residue, names, quality, ca, frames = [], [], [], [], [], [], []
    chains, indices, sequence_hashes = [], [], []
    for i, (chain, index) in enumerate(sorted(keys)):
        aa = sequences[chain][index]
        selected = {name: value for (ch, j, name), value in atoms.items() if ch == chain and j == index}
        expected = set(('N CA C O ' + SIDECHAIN.get(aa, '')).split())
        if aa not in SIDECHAIN or not expected.issubset(selected):
            raise ValueError('pocket_residue_missing_heavy_atoms_or_modified')
        if any(v[0] not in ['C', 'N', 'O', 'S'] for v in selected.values()):
            raise ValueError('pocket_contains_pretrained_unsupported_element')
        origin = selected['CA'][1]
        x, u = selected['C'][1] - origin, selected['N'][1] - origin
        x /= max(np.linalg.norm(x), 1e-8); y = u - x * np.dot(u, x)
        if np.linalg.norm(y) < 1e-6:
            raise ValueError('degenerate_backbone_frame')
        y /= np.linalg.norm(y)
        ca.append(origin); frames.append(np.stack([x, y, np.cross(x, y)], axis=1))
        chains.append(chain); indices.append(index); sequence_hashes.append(hashlib.sha256(sequences[chain].encode()).hexdigest())
        for name, (element, position, q) in sorted(selected.items()):
            xyz.append(position); symbols.append(element); residue.append(i); names.append(name); quality.append(q)
    return dict(atoms=symbols, coordinates=np.array(xyz, np.float32), atom_residue=np.array(residue, np.int32),
                atom_names=np.array(names), atom_quality=np.array(quality, np.float32), ca=np.array(ca, np.float32),
                frames=np.array(frames, np.float32), frame_mask=np.ones(len(ca), bool),
                residue_chains=chains, residue_indices=np.array(indices, np.int32), sequence_hashes=sequence_hashes)


def minimum_distances(ligand, region):
    d = np.linalg.norm(ligand[:, None] - region['coordinates'][None], axis=-1)
    return np.stack([d[:, region['atom_residue'] == i].min(1) for i in range(len(region['ca']))], axis=1).astype(np.float32)


def map_complex(folder, annotation):
    seed = int(hashlib.sha256(annotation['system_id'].encode()).hexdigest()[:8], 16) % 2147483647
    ligand = mapped_ligand(folder, annotation['ligand_rdkit_canonical_smiles'], seed)
    sequences, atoms, alt = mapped_receptor(folder)
    xyz = np.array([v[1] for v in atoms.values()]); atom_keys = list(atoms)
    d = np.linalg.norm(ligand['bound_coordinates'][:, None] - xyz[None], axis=-1).min(0)
    native_keys = {(ch, j) for (ch, j, _), distance in zip(atom_keys, d) if distance <= 6.0}
    if not 5 <= len(native_keys) <= 256:
        raise ValueError('native_pocket_size_outside_5_256_no_truncation')
    native = region_record(atoms, sequences, native_keys)
    if len(native['atoms']) > 2048:
        raise ValueError('native_pocket_encoder_budget_no_truncation')
    truth = minimum_distances(ligand['bound_coordinates'], native)
    if truth.min() < 1.5 or not (truth < 4.5).any():
        raise ValueError('protein_ligand_clash_or_no_contacts')
    regions, labels = [native], [truth]
    # One non-overlapping alternative region in the SAME experimental receptor.
    # Its zeros mean no contacts in this observed pose, not biochemical inactivity.
    centers = [(key, value[1]) for key, value in atoms.items() if key[2] == 'CA' and key[:2] not in native_keys]
    centers.sort(key=lambda kv: hashlib.sha256((annotation['system_id'] + str(kv[0])).encode()).hexdigest())
    for key, center in centers:
        if np.linalg.norm(ligand['bound_coordinates'] - center, axis=-1).min() < 16:
            continue
        keys = {(ch, j) for (ch, j, _), distance in zip(atom_keys, np.linalg.norm(xyz - center, axis=-1)) if distance <= 8}
        if keys & native_keys or not 5 <= len(keys) <= 128:
            continue
        try:
            other = region_record(atoms, sequences, keys)
        except ValueError:
            continue
        if len(other['atoms']) > 2048:
            continue
        distance = minimum_distances(ligand['bound_coordinates'], other)
        if (distance < 4.5).any():
            continue
        regions.append(other); labels.append(distance); break
    return dict(system_id=annotation['system_id'], annotation=annotation, sequences=sequences,
                ligand=ligand, regions=regions, distance_labels=labels, native_pocket_index=0,
                alternate_atoms_skipped=alt, mapping='label_asym_id_and_label_seq_id_with_exact_fasta_AA',
                label_geometry='EXPERIMENTAL_BOUND_POSE', input_ligand_geometry='INDEPENDENT_ETKDG',
                multi_chain_protein=len(sequences)>1)
