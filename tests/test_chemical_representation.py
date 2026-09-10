import torch
import pytest
from biomaster.unified_interaction import UnifiedInteraction,UnifiedConfig
from biomaster.chemical_representation import ChemicalConfig,ChemicalInteraction
from biomaster.molecular_controls import MolecularControlConfig,MolecularControlInteraction


def test_input_replacement_preserves_shared_trunk_initialization():
    for name in ['bermol','bermol_morgan']:
        torch.manual_seed(41)
        baseline=UnifiedInteraction(UnifiedConfig(variant='global',width=24,dropout=0))
        torch.manual_seed(41)
        model=ChemicalInteraction(ChemicalConfig(variant='global',width=24,dropout=0,drug_representation=name))
        for key,value in baseline.state_dict().items():
            if not key.startswith('drug_global.'):
                torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)


def test_each_declared_chemical_input_and_protein_input_influences_scores():
    torch.set_num_threads(2)
    for name in ['bermol','bermol_morgan']:
        width=768+(2048 if name=='bermol_morgan' else 0)
        drug=torch.randn(4,width,requires_grad=True)
        target=torch.randn(4,1280,requires_grad=True)
        batch=dict(drug_global=drug,drug_graph_mean=torch.randn(4,40),
                   pretrained_available=torch.ones(4),target_global=target)
        model=ChemicalInteraction(ChemicalConfig(variant='global',width=24,dropout=0,drug_representation=name))
        model(batch).square().mean().backward()
        assert drug.grad[:,:768].abs().sum()>0 and target.grad.abs().sum()>0
        if name=='bermol_morgan':assert drug.grad[:,768:].abs().sum()>0
        assert all(p.grad is not None for p in model.parameters())
    with pytest.raises(ValueError):ChemicalInteraction(ChemicalConfig(variant='global',drug_representation='undeclared'))


def test_fingerprint_controls_preserve_shared_trunk_and_use_all_declared_inputs():
    for representation,width in [('morgan',2048),('drugclip_morgan',2560)]:
        torch.manual_seed(17);baseline=UnifiedInteraction(UnifiedConfig(variant='global',width=24,dropout=0))
        torch.manual_seed(17);model=MolecularControlInteraction(MolecularControlConfig(variant='global',width=24,dropout=0,
            drug_representation=representation))
        for key,value in baseline.state_dict().items():
            if not key.startswith('drug_global.'):torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        x=torch.randn(4,width,requires_grad=True)
        batch=dict(drug_global=x,drug_graph_mean=torch.randn(4,40),pretrained_available=torch.ones(4),target_global=torch.randn(4,1280))
        model(batch).square().mean().backward()
        assert x.grad[:,-2048:].abs().sum()>0
        if representation=='drugclip_morgan':assert x.grad[:,:512].abs().sum()>0
