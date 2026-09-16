#!/usr/bin/env python3
"""Freeze identities and evaluation scope before off-the-shelf model inference."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import yaml
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/frontier_dti_20260916'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    final_path = ROOT / 'outputs/spr384_final_experiment_table_20260910/SPR384_FINAL_DETAILED.csv'
    panel_path = ROOT / 'outputs/palinova_ab_same_task_20260914/ALL_PANEL_PREDICTIONS_AND_OVERLAP.csv'
    final = pd.read_csv(final_path)
    panels = pd.read_csv(panel_path, low_memory=False)
    # BindingDB479 is nested in the merged panel. Use its original exact identities,
    # and report the prespecified 175 A/B-train+validation-unseen subset separately.
    panel = panels[panels.is_bindingdb479.eq(True)].drop_duplicates('pair_id').copy()
    assert len(panel) == 479, len(panel)
    drugs = pd.read_csv(ROOT / 'outputs/biomaster_matrix_720x890_20260910/DRUG_INDEX.csv').set_index('drug_id')
    targets = pd.read_csv(ROOT / 'outputs/target_universe_ch37_v2/TARGET_UNIVERSE_OFFICIAL_888_V2.csv', low_memory=False).set_index('target_chembl_id')
    rows = []
    for cohort, data in [('SPR384', final), ('BINDINGDB479', panel)]:
        for _, row in data.iterrows():
            drug_id = row['药物完整InChIKey'] if cohort == 'SPR384' else row['ligand_inchikey']
            target_id = row['新靶点ChEMBL编号'] if cohort == 'SPR384' else row['target_chembl_id']
            smiles = drugs.loc[drug_id, 'smiles']
            molecule = Chem.MolFromSmiles(smiles)
            assert molecule is not None and Chem.MolToInchiKey(molecule) == drug_id, drug_id
            target = targets.loc[target_id]
            rows.append(dict(pair_id=f'{drug_id}__{target_id}', cohort=cohort,
                drug_id=drug_id, target_id=target_id, smiles=smiles, sequence=target['sequence'],
                uniprot_id=target['uniprot_accession'], gene=target['gene_symbol'],
                drug_name=drugs.loc[drug_id, 'name'],
                candidate_id=row.get('原候选编号', ''), priority=row.get('排序', None),
                label=None if cohort == 'SPR384' else row['binary_label'],
                ab_unseen=False if cohort == 'SPR384' else bool(row['both_AB_train_validation_pair_unseen']),
                protein_length=len(target['sequence']),
                protein_sha256=hashlib.sha256(target['sequence'].encode()).hexdigest()))
    manifest = pd.DataFrame(rows)
    assert manifest.loc[manifest.cohort.eq('BINDINGDB479'),'label'].notna().all()
    assert int(manifest.ab_unseen.sum()) == 175
    manifest.to_csv(OUT / 'INPUT_MANIFEST.csv', index=False)
    unique = manifest.drop_duplicates('pair_id')
    unique.to_csv(OUT / 'UNIQUE_PAIRS.csv', index=False)
    # The pool is fixed. SPR scores are predictions, never truth labels.
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        cohorts={'SPR384':384, 'BINDINGDB479':479, 'BINDINGDB_AB_UNSEEN175':175},
        unique_pairs=len(unique), original_spr_sha256=hashlib.sha256(final_path.read_bytes()).hexdigest(),
        historical_panel_sha256=hashlib.sha256(panel_path.read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256((OUT/'INPUT_MANIFEST.csv').read_bytes()).hexdigest(),
        primary_scores={'nesso':'affinity_probability_binary', 'probematch':'official ensemble positive probability', 'dtbind':'occurrence sigmoid'},
        secondary_scores={'nesso_pic50':'6 - affinity_pred_value; predicted pIC50, not measured Kd'},
        evaluation='AP, AUROC, prevalence; identical successfully-scored subsets for comparisons; no test-set threshold fitting',
        caveat='Public checkpoints training overlap with BindingDB is unknown; A/B-unseen does not establish new-model-unseen status.',
        ranks='Within frozen SPR384 pool only; no new full 720x384 matrix ranking.',
        frozen_experiment='Unchanged, controls separate; no new scores determine experimental truth.',
        model_revisions={'ProbeMatchDTI':'2b9c0dcb40bcc4e5482d6ea98aa1419538ecdd12','DTBind':'08983e476760fbe5b5dc62be06fd2e087b31ac0c'},
        nesso_settings={'recycling_steps':5,'precision':'bf16-mixed','seed':20260916,'pocket_crop':True})
    (OUT/'PROTOCOL.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n')
    inputs=OUT/'nesso/inputs'; inputs.mkdir(parents=True,exist_ok=True)
    # IDs preserve exact joins while YAML sort order interleaves target lengths.
    for row in unique.itertuples(index=False):
        payload={'sequences':[{'protein':{'id':'A','sequence':row.sequence}}, {'ligand':{'id':'B','smiles':row.smiles}}], 'properties':[{'affinity':{'binder':'B'}}]}
        (inputs/f'{row.pair_id}.yaml').write_text(yaml.safe_dump(payload,sort_keys=False))
    print(json.dumps(protocol,ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
