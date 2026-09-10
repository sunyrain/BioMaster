"""Instantiate an explicitly recorded architecture; exports retain one family only."""
from .unified_interaction import UnifiedConfig,UnifiedInteraction
from .refined_interaction import RefinedConfig,RefinedInteraction
from .anchored_interaction import AnchoredConfig,AnchoredInteraction
from .chemical_representation import ChemicalConfig,ChemicalInteraction
from .molecular_controls import MolecularControlConfig,MolecularControlInteraction


FAMILIES={
    'unified':(UnifiedConfig,UnifiedInteraction),
    'refined':(RefinedConfig,RefinedInteraction),
    'anchored':(AnchoredConfig,AnchoredInteraction),
    'chemical':(ChemicalConfig,ChemicalInteraction),
    'molecular_control':(MolecularControlConfig,MolecularControlInteraction),
}


def infer_family(state):
    if state.get('family'):return state['family']
    if state.get('architecture'):return state['architecture']
    config=state['config']
    if 'parent_variant' in config:return 'anchored'
    if config.get('drug_representation') in ['morgan','drugclip_morgan']:return 'molecular_control'
    if config.get('drug_representation') in ['bermol','bermol_morgan']:return 'chemical'
    return 'refined' if state.get('refinement') else 'unified'


def build_model(family,config):
    if family not in FAMILIES:raise ValueError('unrecognized model family')
    cfg,model=FAMILIES[family]
    return model(cfg(**config))
