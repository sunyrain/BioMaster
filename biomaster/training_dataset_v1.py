"""Identity, censoring and evidence rules for the unrestricted human training set."""
from __future__ import annotations
import hashlib
import math
import re
from functools import lru_cache

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')
ENDPOINT_TASK = {'Kd':'AFFINITY_KD_KI','Ki':'AFFINITY_KD_KI',
                 'IC50':'ACTIVITY_IC50','EC50':'ACTIVITY_EC50',
                 'INACTIVE':'ANNOTATED_INACTIVITY'}
MUTANT = re.compile(r'\bmutants?\b|\bmutations?\b|\bchimer(?:a|ic)\b|\bfusion\b|\b[A-Z]\d{2,4}[A-Z]\b',re.I)


def sha_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def concentration(raw, unit='nM', relation=''):
    text=str(raw or '').strip().replace('≤','<=').replace('≥','>=')
    match=re.fullmatch(r'\s*([<>=~≈]{0,2})\s*([+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*',text)
    if not match:return None, None
    embedded=match[1] or '='; rel=str(relation or embedded).strip()
    if match[1] and relation and embedded!=relation:return None,None
    factor={'m':1e9,'mm':1e6,'um':1e3,'nm':1.,'pm':.001,'fm':.000001}.get(str(unit).strip().lower().replace('μ','u').replace('µ','u'))
    if factor is None or rel not in ['=','<','<=','>','>=']:return None,None
    value=float(match[2])*factor
    if not math.isfinite(value) or value<=0:return None,None
    return rel,value


def observation_label(value, relation, explicit_inactive=False):
    if value is None:return 0 if explicit_inactive else None
    positive=(relation=='=' and value<=1000) or (relation in ['<','<='] and value<=1000)
    negative=(relation=='=' and value>=10000) or (relation in ['>','>='] and value>=10000)
    if positive and explicit_inactive:return -1
    if positive:return 1
    if negative or explicit_inactive:return 0
    return None


@lru_cache(maxsize=2000000)
def molecule_identity(smiles, reported_key=''):
    """No neutralization, tautomer collapse, metabolite substitution or stereochemistry loss."""
    try:
        return _molecule_identity(smiles,reported_key)
    except (ValueError,RuntimeError,OverflowError):
        return None,'rdkit_structure_processing_error'


def _molecule_identity(smiles, reported_key):
    mol=Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:return None,'invalid_smiles'
    if any(a.GetAtomicNum()==0 for a in mol.GetAtoms()):return None,'unspecified_atom'
    raw_key=Chem.MolToInchiKey(mol)
    if not raw_key:return None,'missing_computable_inchikey'
    if reported_key and raw_key!=reported_key:return None,'reported_inchikey_smiles_mismatch'
    fragments=Chem.GetMolFrags(mol,asMols=True,sanitizeFrags=True)
    organic=[f for f in fragments if any(a.GetAtomicNum()==6 for a in f.GetAtoms())]
    if len(organic)!=1:return None,'multiple_or_no_organic_components'
    parent=organic[0]; key=Chem.MolToInchiKey(parent)
    if not key:return None,'missing_parent_inchikey'
    scaffold_mol=Chem.Mol(parent)
    Chem.RemoveStereochemistry(scaffold_mol)
    scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=scaffold_mol,includeChirality=False)
    return dict(molecule_id=key,connectivity_key=key.split('-')[0],
                smiles=Chem.MolToSmiles(parent,isomericSmiles=True),
                scaffold=scaffold or 'ACYCLIC:'+key.split('-')[0],
                heavy_atoms=parent.GetNumHeavyAtoms(),formal_charge=Chem.GetFormalCharge(parent),
                counterions_removed=len(fragments)-1),''


def classify_construct(sequence, canonical):
    if not sequence:return 'accession_resolved_construct_unverified'
    if sequence==canonical:return 'exact_canonical_sequence'
    if len(sequence)>=20 and sequence in canonical:return 'exact_wildtype_fragment'
    return 'sequence_mismatch_or_mutant'


def split_for_group(group):
    bucket=int(sha_text('biomaster-final-20260910:'+group)[:8],16)%100
    return 'train' if bucket<80 else 'validation' if bucket<90 else 'test'


def record_quality(observations, rejected_bindingdb_ids):
    """An inconsistent source ID is never resolved by first-file or majority preference."""
    import numpy as np
    variants=observations.groupby(['source','source_record_id']).variant_hash.transform('nunique')
    tainted=observations.source.eq('BindingDB_202609') & observations.source_record_id.str.split(':').str[0].isin(rejected_bindingdb_ids)
    return np.select([variants.gt(1),tainted],['SOURCE_RECORD_VARIANT_CONFLICT','SOURCE_IDENTITY_REJECTED_IN_ANOTHER_VARIANT'],default='ACCEPTED')


def summarize_observations(group):
    """One task/pair label; duplicate source counts never vote away a contradiction."""
    labels=set(group.observation_label.dropna().astype(int))
    conflict=-1 in labels or {0,1}<=labels
    label=None if conflict or not labels else (1 if 1 in labels else 0)
    values=group.loc[group.relation.eq('=') & group.value_nM.notna(),'value_nM'].drop_duplicates()
    # Censored observations have no exact regression target.
    import numpy as np
    exact=-np.log10(values.to_numpy(float)*1e-9)
    return dict(binary_label=label,label_status='CONFLICT' if conflict else 'GREY_OR_UNRESOLVED' if label is None else 'POSITIVE' if label else 'NEGATIVE',
                source_records=len(group),unique_numeric_evidence=group[['endpoint','relation','value_nM']].drop_duplicates().shape[0],
                exact_unique_values=len(values),median_p_activity_unique=float(np.median(exact)) if len(exact) else None,
                exact_p_activity_range=float(exact.max()-exact.min()) if len(exact) else None)
