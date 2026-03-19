"""
run lick prediction model for all mice
groups sessions by mouse, runs hyperparameter sweep and evaluation
"""
import os
import pickle
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

from config import PATHS, LICK_PRED_OPS
from data.session import Session
from data.lick_features import build_session_features
from utils.filing import get_session_files
from analyses.lick_prediction import run_sweep_and_ablation


def _group_sessions_by_mouse(npx_dir, npx_only=False):
    paths = get_session_files(npx_dir, npx_only=npx_only)
    grouped = defaultdict(list)
    for p in paths:
        animal = Path(p).parent.parent.name
        grouped[animal].append(p)
    return dict(grouped)


def run_single_mouse(animal, sess_paths, save_dir, ops=LICK_PRED_OPS):
    """run lick prediction for one mouse"""
    print(f'\n===== {animal} ({len(sess_paths)} sessions) =====')

    sessions_data = []
    session_names = []
    for path in sess_paths:
        sess = Session.load(path)
        X, y, trial_ids = build_session_features(sess, ops)
        if len(X) > 0:
            sessions_data.append((X, y, trial_ids))
            session_names.append(sess.name)
            print(f'  {sess.name}: {X.shape[0]} bins, '
                  f'{(y > 0).sum()} lick bins')

    if len(sessions_data) < 3:
        print(f'  Skipping {animal}: too few sessions')
        return

    loss_curve_dir = os.path.join(save_dir, animal, 'loss_curves')
    os.makedirs(loss_curve_dir, exist_ok=True)

    sweep_results, full_results = run_sweep_and_ablation(
        sessions_data, ops, save_dir=loss_curve_dir)

    result = dict(
        animal=animal,
        session_names=session_names,
        sweep_results=sweep_results,
        full_results=full_results,
    )

    save_path = os.path.join(save_dir, f'{animal}_lick_pred.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(result, f)

    # save per-architecture model weights (from each fold)
    for arch, fr in full_results.items():
        for i, state_dict in enumerate(fr['fold_models']):
            model_path = os.path.join(save_dir, f'{animal}_{arch}_fold{i}.pt')
            torch.save(state_dict, model_path)

    print(f'  Saved to {save_path}')


def run_lick_prediction(npx_dir=PATHS['npx_dir_local'],
                        ops=LICK_PRED_OPS,
                        save_dir=None,
                        npx_only=False,
                        overwrite=True):
    """
    run lick prediction for all mice
    saves results and model weights per mouse as pickle + .pt files
    """
    if save_dir is None:
        save_dir = os.path.join(npx_dir, 'lick_prediction')
    os.makedirs(save_dir, exist_ok=True)

    grouped = _group_sessions_by_mouse(npx_dir, npx_only=npx_only)

    for animal, sess_paths in grouped.items():
        save_path = os.path.join(save_dir, f'{animal}_lick_pred.pkl')
        if not overwrite and os.path.exists(save_path):
            print(f'\n===== {animal}: already done, skipping =====')
            continue
        run_single_mouse(animal, sess_paths, save_dir, ops)


if __name__ == '__main__':
    import sys

    npx_dir = sys.argv[1]
    mouse_idx = int(sys.argv[2])

    grouped = _group_sessions_by_mouse(npx_dir, npx_only=False)
    animals = sorted(grouped.keys())
    print(f'Found {len(animals)} mice: {animals}')

    if mouse_idx >= len(animals):
        print(f'Mouse index {mouse_idx} out of range')
        sys.exit(1)

    animal = animals[mouse_idx]
    save_dir = os.path.join(npx_dir, 'lick_prediction')
    os.makedirs(save_dir, exist_ok=True)

    print(f'Running mouse {mouse_idx}: {animal}')
    run_single_mouse(animal, grouped[animal], save_dir)
