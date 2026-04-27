from config import PATHS, ANALYSIS_OPTIONS
from utils.filing import get_session_dirs_by_animal, load_fr_matrix
from data.session import Session

import numpy as np
import pandas as pd
import pickle
import gc
from pathlib import Path


#%% Load movement-potent/null dims and project single-trial activity

def _load_move_dims(session_dir: Path) -> dict | None:
    path = session_dir / 'move_dims.pkl'
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def _project_onto_move_dims(fr: pd.DataFrame, move_dims: dict) -> dict:
    """
    project FR matrix onto movement-potent/null dims per area
    returns dict[area] -> {'pot': (n_pot, nT), 'null': (n_null, nT)}
    """
    projs = {}
    for area, ws in move_dims.items():
        cids = ws['cids']
        missing = [c for c in cids if c not in fr.index]
        if missing:
            print(f'  {area}: {len(missing)}/{len(cids)} cids missing in FR matrix, skipping')
            continue
        X = fr.loc[cids].to_numpy()  # nN x nT, rows aligned with weight cols
        projs[area] = dict(
            pot  = ws['pot']  @ X,
            null = ws['null'] @ X,
        )
    return projs


#%%

def visualize_single_trial_activity():
    """
    visualize individual licks, baseline trajectories, and tf pulse responses along
    movement-potent/null dimensions for single session
    """
    pass


def visualize_prelick_activity_vs_evidence():
    """
    visualize relationship between movement-null projection and 'amount of evidence' in
    defined window preceding licks
    """
    pass


#%% project all sessions

animal_sessions = get_session_dirs_by_animal(npx_dir=PATHS['npx_dir_local'])

for animal, sess_dirs in animal_sessions.items():
    for sess_dir in sess_dirs:
        print(f'{animal}/{sess_dir.name}')

        move_dims = _load_move_dims(sess_dir)
        if move_dims is None:
            print('  no move_dims.pkl')
            continue

        fr = load_fr_matrix(str(sess_dir / 'FR_matrix_ds.parquet'))
        proj = _project_onto_move_dims(fr, move_dims)
        del fr
        gc.collect()

        print(f'  projected {len(proj)} areas')