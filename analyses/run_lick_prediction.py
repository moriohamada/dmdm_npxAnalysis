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
from analyses.lick_prediction import (
    hyperparameter_sweep, leave_one_out_cv,
    LinearLickModel, NetworkLickModel,
    fit_best_model, ablation_analysis, extract_stimulus_filter,
)


def _group_sessions_by_mouse(npx_dir, npx_only=False):
    paths = get_session_files(npx_dir, npx_only=npx_only)
    grouped = defaultdict(list)
    for p in paths:
        animal = Path(p).parent.parent.name
        grouped[animal].append(p)
    return dict(grouped)


def run_lick_prediction(npx_dir=PATHS['npx_dir_local'],
                        ops=LICK_PRED_OPS,
                        save_dir=None,
                        npx_only=False,
                        overwrite=False):
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
            continue

        print(f'  Running hyperparameter sweep...')
        sweep_results = hyperparameter_sweep(sessions_data, ops)

        model, mu, sd, best_key = fit_best_model(sessions_data, sweep_results, ops)

        print(f'  Running ablation analysis...')
        ablation, baseline_losses = ablation_analysis(
            model, sessions_data, mu, sd, ops)

        result = dict(
            animal=animal,
            session_names=session_names,
            sweep_results=sweep_results,
            best_config=best_key,
            ablation=ablation,
            baseline_losses=baseline_losses,
            norm_mu=mu,
            norm_sd=sd,
        )

        save_path = os.path.join(save_dir, f'{animal}_lick_pred.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)

        model_path = os.path.join(save_dir, f'{animal}_model.pt')
        torch.save(model.state_dict(), model_path)
        print(f'  Saved to {save_path}')
