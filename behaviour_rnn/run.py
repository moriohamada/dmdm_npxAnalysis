"""
train the behaviour RNN for one mouse, picked by array index.
mirrors lick_pred/run.py: load dfs_processed.pkl from npx_dir/behaviour,
fit fit_subj on that mouse's df, save weights to npx_dir/behaviour_rnn/models.

after the array job finishes, run train_rnns_all_subj(dfs, overwrite=False)
locally to aggregate the simulated mirror observables across mice.
"""
import os
import sys
import pickle
from pathlib import Path

import torch

from behaviour_rnn.train import fit_subj, save_model, _model_path


def _load_dfs(npx_dir):
    path = os.path.join(npx_dir, 'behaviour', 'dfs_processed.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def run_single_mouse(subj, df, data_dir, device='cpu'):
    path = _model_path(subj, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'\n===== training RNN for {subj} on {device} =====')
    result = fit_subj(df, verbose=True, device=device)
    save_model(result, path)


if __name__ == '__main__':
    npx_dir   = sys.argv[1]
    mouse_idx = int(sys.argv[2])

    dfs = _load_dfs(npx_dir)
    subjs = sorted(dfs.keys())
    print(f'Found {len(subjs)} mice: {subjs}')

    if mouse_idx >= len(subjs):
        print(f'Mouse index {mouse_idx} out of range')
        sys.exit(1)

    subj = subjs[mouse_idx]
    data_dir = Path(npx_dir) / 'behaviour_rnn'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running mouse {mouse_idx}: {subj}')
    run_single_mouse(subj, dfs[subj], data_dir, device=device)
