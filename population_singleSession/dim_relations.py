"""
Relations between already-extracted dimensions: combine tf_dims + move_dims to get
TF-coding directions orthogonalised against movement space (per area / block).
"""

from config import PATHS
from utils.filing import get_session_dirs_by_animal

import pickle
import numpy as np


def project_tf_into_movenull(tf_dims: dict, move_dims: dict) -> dict:
    """
    project tf_pot onto move_null subspace, per area / block.
    out[area][block]['movenull'] = (1, nN), unit-normalised
    """
    out = {}
    for area, tf_area in tf_dims.items():
        if area not in move_dims:
            continue
        blocks = [k for k in tf_area if k not in ('cids', 'delay')]
        area_out = {}
        for block in blocks:
            if block not in move_dims[area]:
                continue
            tf_pot  = tf_area[block]['pot']             # (1, nN)
            mv_null = move_dims[area][block]['null']    # (n_null, nN)
            proj = tf_pot @ mv_null.T @ mv_null         # (1, nN)
            n = np.linalg.norm(proj)
            if n > 0:
                proj = proj / n
            area_out[block] = {'movenull': proj}
        if area_out:
            area_out['cids'] = tf_area['cids']
            out[area] = area_out
    return out


def fit_tf_in_movenull_per_session(npx_dir: str = PATHS['npx_dir_local']):
    """
    runner: per session, load tf_dims.pkl + move_dims.pkl, save tf_in_movenull.pkl
    """
    animal_sessions = get_session_dirs_by_animal(npx_dir)
    for animal, sessions in animal_sessions.items():
        print(f'Computing tf-in-movenull for {animal}')
        for session_dir in sessions:
            print(f'    {session_dir}')
            with open(session_dir / 'tf_dims.pkl', 'rb') as f:
                tf_dims = pickle.load(f)
            with open(session_dir / 'move_dims.pkl', 'rb') as f:
                move_dims = pickle.load(f)
            out = project_tf_into_movenull(tf_dims, move_dims)
            with open(session_dir / 'tf_in_movenull.pkl', 'wb') as f:
                pickle.dump(out, f)


if __name__ == '__main__':
    fit_tf_in_movenull_per_session()