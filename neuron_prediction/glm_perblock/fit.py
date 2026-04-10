"""
per-block poisson glm: refit the existing glm separately on early-block and
late-block trials, at lambda=0 (no regularisation), to get block-conditional
kernels for downstream TDR analysis
"""
import numpy as np
import pickle
from pathlib import Path

from config import ANALYSIS_OPTIONS, GLM_OPTIONS
from data.session import Session
from neuron_prediction.data import (
    load_glm_inputs, get_trial_fold_indices, normalise_design_matrix,
)
from neuron_prediction.evaluate import pearson_r, lesion_design_matrix
from neuron_prediction.glm.fit import _fit_poisson_glm, _predict_glm


BLOCKS = ('early', 'late')


#%% drop block column

def drop_predictor(X, col_map, name):
    """remove a single predictor column from X and reindex col_map

    returns (X_new, col_map_new). raises if predictor is not single-column
    """
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


#%% per-block fold ids

def get_block_fold_indices(trials_df, t_ax, n_folds, block, ignore_first_n):
    """assign 10-fold CV ids to bins inside trials of a single block

    bins outside this block (or in transition / invalid trials) get -1
    """
    sub = trials_df[trials_df['hazardblock'] == block]
    return get_trial_fold_indices(sub, t_ax, n_folds, seed=0,
                                  ignore_first_n=ignore_first_n)


#%% per-block fit

def fit_neuron_one_block(counts_1d, X, col_map, fold_ids, ops=GLM_OPTIONS):
    """fit poisson glm for one neuron on one block subset, lambda=0

    1. cv folds at lambda=0 (no grid search)
    2. lesion analysis using fold models
    3. refit on all valid in-block bins for kernel extraction

    returns dict matching the existing per-neuron glm format (no best_lambda).
    returns None if no valid data
    """
    lesion_groups = ops['lesion_groups']
    group_names = list(lesion_groups.keys())
    n_folds = ops['n_folds']

    fit_kw = {k: ops[k] for k in ('max_iter', 'tol') if k in ops}
    coarse_kw = {**fit_kw, 'max_iter': ops.get('cv_max_iter', 200),
                 'tol': ops.get('cv_tol', 1e-4)}

    valid = fold_ids >= 0
    if valid.sum() < n_folds:
        return None

    X_v = X[valid]
    y_v = counts_1d[valid].astype(np.float64)
    folds_v = fold_ids[valid]

    # group masks: bins where each lesion group's predictors are non-zero
    group_masks = {}
    for gname, pred_list in lesion_groups.items():
        mask = np.zeros(X_v.shape[0], dtype=bool)
        for pred_name in pred_list:
            if pred_name in col_map:
                col_slice, _ = col_map[pred_name]
                mask |= np.any(X_v[:, col_slice] != 0, axis=1)
        group_masks[gname] = mask

    #%% cv at lambda=0
    fold_wb = {}
    for k in range(n_folds):
        test_mask = folds_v == k
        train_mask = ~test_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        X_train, _, _, _ = normalise_design_matrix(
            X_v[train_mask], X_v[test_mask], col_map)
        y_train = y_v[train_mask]

        if y_train.sum() == 0:
            continue

        w, b = _fit_poisson_glm(X_train, y_train, col_map,
                                lambda_gl=0.0, **coarse_kw)
        fold_wb[k] = (w, b)

    if not fold_wb:
        return None

    #%% evaluate fold models: full + lesioned
    full_r = np.full(n_folds, np.nan)
    full_r_group = {g: np.full(n_folds, np.nan) for g in group_names}
    lesioned_r = {g: np.full(n_folds, np.nan) for g in group_names}

    for k, (w, b) in fold_wb.items():
        test_mask = folds_v == k
        X_train_raw, X_test_raw = X_v[~test_mask], X_v[test_mask]
        _, X_test, _, _ = normalise_design_matrix(
            X_train_raw, X_test_raw, col_map)
        y_test = y_v[test_mask]

        y_pred = _predict_glm(X_test, w, b)
        full_r[k] = pearson_r(y_test, y_pred)

        for gname, pred_list in lesion_groups.items():
            win = group_masks[gname][test_mask]
            if win.sum() < 5:
                continue

            full_r_group[gname][k] = pearson_r(y_test[win], y_pred[win])

            X_les = lesion_design_matrix(X_test, pred_list, col_map)
            y_les = _predict_glm(X_les, w, b)
            lesioned_r[gname][k] = pearson_r(y_test[win], y_les[win])

    #%% refit on all in-block valid bins for kernels
    X_all, _, _, _ = normalise_design_matrix(X_v, X_v, col_map)
    w_final, b_final = _fit_poisson_glm(X_all, y_v, col_map,
                                         lambda_gl=0.0, **fit_kw)

    result = {
        'weights': w_final,
        'bias': np.array([b_final]),
        'fold_ids': fold_ids,
        'full_r': full_r,
    }
    for gname in group_names:
        result[f'full_r_group_{gname}'] = full_r_group[gname]
        result[f'lesioned_r_{gname}'] = lesioned_r[gname]

    return result


def fit_neuron_perblock_from_disk(sess_dir, neuron_idx, ops=GLM_OPTIONS):
    """load prepped data, fit one neuron per block, save two npz files"""
    sess_dir = Path(sess_dir)
    counts, X, col_map, t_ax, valid_mask = load_glm_inputs(str(sess_dir))

    # drop block predictor (constant within each block subset)
    X, col_map = drop_predictor(X, col_map, 'block')

    sess = Session.load(str(sess_dir / 'session.pkl'))
    ignore_n = ANALYSIS_OPTIONS['ignore_first_trials_in_block']

    results_dir = sess_dir / 'glm_perblock_results'
    results_dir.mkdir(exist_ok=True)

    # save col_map once per session (same for all neurons)
    col_map_path = results_dir / 'col_map.pkl'
    if not col_map_path.exists():
        with open(col_map_path, 'wb') as f:
            pickle.dump(col_map, f)

    for block in BLOCKS:
        fold_ids = get_block_fold_indices(
            sess.trials, t_ax, ops['n_folds'], block, ignore_n)
        n_valid = (fold_ids >= 0).sum()
        print(f'  {block}: {n_valid} valid bins, '
              f'{len(set(fold_ids[fold_ids >= 0]))} folds')

        result = fit_neuron_one_block(counts[neuron_idx], X, col_map, fold_ids, ops)
        if result is None:
            print(f'  {block}: skipped (no valid data)')
            continue

        out_path = results_dir / f'neuron_{neuron_idx}_{block}.npz'
        np.savez(out_path, **result)
        print(f'  {block}: saved to {out_path.name}')
