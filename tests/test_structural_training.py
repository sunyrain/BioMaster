"""Identity, geometry and label semantics for experimental structural training."""
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from biomaster.structural_complex import mapped_ligand, minimum_distances, region_record
from biomaster.structural_training import complex_losses
from biomaster.pocket_precision import PocketPrecisionConfig


def test_ligand_atom_order_and_independent_input_coordinates(tmp_path):
    folder = tmp_path / 'ligand_files'; folder.mkdir()
    mol = Chem.AddHs(Chem.MolFromSmiles('CC[C@H](O)COc1ccccc1'))
    params = AllChem.ETKDGv3(); params.randomSeed = 41
    assert AllChem.EmbedMolecule(mol, params) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    Chem.MolToMolFile(mol, str(folder / 'ligand.sdf'))
    record = mapped_ligand(tmp_path, Chem.MolToSmiles(mol), 83)
    assert record['input_geometry'] == 'INDEPENDENT_ETKDG_MMFF_OR_UFF'
    assert record['coordinates'].shape == record['bound_coordinates'].shape
    bound = record['bound_coordinates']; free = record['coordinates']
    assert not np.allclose(bound, free)
    bonds = record['bond'] > 0
    distances = np.linalg.norm(bound[:, None] - bound[None], axis=-1)
    assert np.all((distances[bonds] > 0.6) & (distances[bonds] < 2.6))


def test_multichain_mapping_does_not_collapse_identical_residue_numbers():
    atoms = {}
    coords = {'N': [-1, 1, 0], 'CA': [0, 0, 0], 'C': [1, 0, 0], 'O': [2, 0, 0], 'CB': [0, -1, 1]}
    for chain, shift in [('1.A', 0), ('2.A', 10)]:
        for name, value in coords.items():
            atoms[chain, 0, name] = (name[0], np.array(value, np.float32) + shift, 1.)
    r = region_record(atoms, {'1.A': 'A', '2.A': 'A'}, {('1.A', 0), ('2.A', 0)})
    assert r['residue_chains'] == ['1.A', '2.A']
    assert r['residue_indices'].tolist() == [0, 0]
    d = minimum_distances(np.array([[2., 0., 1.]], np.float32), r)
    assert d.shape == (1, 2) and d[0, 0] < d[0, 1]


def test_supervised_losses_ignore_padding_and_weight_complexes_equally():
    cfg = PocketPrecisionConfig()
    torch.manual_seed(83)
    aux = dict(pair_mask=torch.tensor([[[True, True, False]], [[True, True, True]]]),
               distance_logits=torch.randn(2, 1, 3, 64, requires_grad=True),
               contact_logits=torch.randn(2, 1, 3, requires_grad=True),
               pocket_gate=torch.tensor([1., 0.], requires_grad=True))
    b = dict(owner=torch.tensor([0, 0]))
    labels = dict(distance=torch.tensor([[[3., 8., float('nan')]], [[20., 30., 40.]]]), native=torch.tensor([0]))
    loss = complex_losses(aux, b, labels, cfg)
    assert torch.isfinite(loss).all()
    loss.sum().backward()
    assert aux['distance_logits'].grad[0, 0, 2].abs().sum() == 0
    assert aux['contact_logits'].grad[0, 0, 2] == 0
    assert aux['pocket_gate'].grad[0] < 0 and aux['pocket_gate'].grad[1] > 0
