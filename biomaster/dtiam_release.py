"""Single deployment identity for DTIAM catalog, review panels and exports."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RELEASE_ID = 'dtiam_a_kdki_inactive_20260917'
NAME = 'DTIAM A（九月加强版）'
DIRECTORY = 'outputs/dtiam_a_catalog_20260917'
SCORES = DIRECTORY + '/DTIAM_A_720X384_SCORES.csv.gz'
MANIFEST = DIRECTORY + '/MANIFEST.json'
KEYS = ['ligand_inchikey', 'target_chembl_id']


def load_scores(root: str | Path) -> pd.DataFrame:
    """Missing releases stay missing; historical scores never fill their place."""
    root = Path(root)
    path = root / SCORES
    if not path.is_file():
        return pd.DataFrame(columns=KEYS + ['dtiam_probability'])
    metadata = json.loads((root / MANIFEST).read_text())
    if metadata.get('status') != 'COMPLETE' or metadata.get('release_id') != RELEASE_ID:
        raise ValueError('DTIAM A release is incomplete or has the wrong identity')
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['scores_sha256']:
        raise ValueError('DTIAM A score artifact checksum mismatch')
    frame = pd.read_csv(path, usecols=KEYS + ['dtiam_probability', 'dtiam_model_version'])
    values = pd.to_numeric(frame.dtiam_probability, errors='coerce')
    if (len(frame) != metadata['rows'] or frame.duplicated(KEYS).any()
            or not frame.dtiam_model_version.eq(RELEASE_ID).all()
            or not np.isfinite(values).all() or not values.between(0, 1).all()):
        raise ValueError('Invalid DTIAM A score rows')
    frame['dtiam_probability'] = values
    return frame[KEYS + ['dtiam_probability']]


def overlay_baseline(root: str | Path, baseline: pd.DataFrame) -> pd.DataFrame:
    """Replace only the DTIAM channel of cached review baselines by exact pair ID."""
    scores = load_scores(root)
    scores['pair_id'] = scores.ligand_inchikey + '__' + scores.target_chembl_id
    result = baseline.copy()
    result['dtiam'] = result.pair_id.map(scores.set_index('pair_id').dtiam_probability)
    result['dtiam_model_version'] = RELEASE_ID
    return result
