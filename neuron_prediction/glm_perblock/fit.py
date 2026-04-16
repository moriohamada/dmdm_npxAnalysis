"""per-block ridge poisson glm: fit glm_ridge separately on early and late block trials"""
import numpy as np
import pickle
from pathlib import Path

from config import ANALYSIS_OPTIONS, GLM_OPTIONS
from data.session import Session
from neuron_prediction.data import (
    load_glm_inputs, get_trial_fold_indices, neuron_seed,
)
from neuron_prediction.glm_ridge.fit import fit_neuron


BLOCKS = ('early', 'late')


def drop_predictor(X, col_map, name):
    """remove a named predictor from X and reindex col_map"""
    if name not in col_map:
        return X, col_map

    col_slice, _ = col_map[name]
    n_drop = col_slice.stop - col_slice.start

    keep = np.ones(X.shape[1], dtype=bool)
    keep[col_slice] = False
    X_new = X[:, keep]

    col_map_new = {}
    for k, (cs, lags) in col_map.items():
        if k == name:
            continue
        if cs.start >= col_slice.stop:
            new_slice = slice(cs.start - n_drop, cs.stop - n_drop)
        else:
            new_slice = cs
        col_map_new[k] = (new_slice, lags)

    return X_new, col_map_new


def get_block_fold_indices(trials_df, t_ax, n_folds, block, seed, ignore_first_n):
    """assign CV fold ids to bins inside trials of a single block

    bins outside this block or in transition trials get -1
    """
    sub = trials_df[trials_df['hazardblock'] == block]
    return get_trial_fold_indices(sub, t_ax, n_folds, seed=seed,
                                  ignore_first_n=ignore_first_n)


def fit_neuron_perblock_from_disk(sess_dir, neuron_idx, ops=GLM_OPTIONS):
    """fit ridge GLM for one neuron separately per block, save results"""
    sess_dir = Path(sess_dir)
    counts, X, col_map, t_ax, valid_mask = load_glm_inputs(str(sess_dir))

    # block predictor is constant within each block - drop it
    X, col_map = drop_predictor(X, col_map, 'block')

    sess = Session.load(str(sess_dir / 'session.pkl'))
    ignore_n = ANALYSIS_OPTIONS['ignore_first_trials_in_block']
    seed = neuron_seed(str(sess_dir), neuron_idx)

    results_dir = sess_dir / 'glm_perblock_results'
    results_dir.mkdir(exist_ok=True)

    col_map_path = results_dir / 'col_map.pkl'
    if not col_map_path.exists():
        with open(col_map_path, 'wb') as f:
            pickle.dump(col_map, f)

    for block in BLOCKS:
        fold_ids = get_block_fold_indices(
            sess.trials, t_ax, ops['n_folds'], block, seed, ignore_n)
        n_valid = (fold_ids >= 0).sum()
        n_folds_actual = len(set(fold_ids[fold_ids >= 0]))
        print(f'  {block}: {n_valid} valid bins, {n_folds_actual} folds')

        result = fit_neuron(counts[neuron_idx], X, col_map, fold_ids, ops)
        if result is None:
            print(f'  {block}: skipped (no valid data)')
            continue

        out_path = results_dir / f'neuron_{neuron_idx}_{block}.npz'
        np.savez(out_path, **result)
        print(f'  {block}: r={np.nanmean(result["full_r"]):.4f}, '
              f'lambda={float(result["best_lambda"]):.0e}, '
              f'saved to {out_path.name}')
