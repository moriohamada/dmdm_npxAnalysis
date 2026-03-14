"""
Input-driven linear dynamical systems analysis in PC space.

Fit x_{t+1} = Ax_t + Bu_t, where:
    x_t: neural state in PC space (n_pcs,)
    u_t: stimulus input — log2(TF) at each time bin
    A: dynamics matrix (n_pcs x n_pcs)
    B: input matrix (n_pcs x 1)

Fit separately per condition (earlyBlock_early, lateBlock_early, lateBlock_late)
on baseline period only, excluding time bins near licks and transition trials.
"""
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

from config import ANALYSIS_OPTIONS, PATHS
from data.session import Session
from utils.filing import get_response_files

CONDITIONS = {
    'earlyBlock_early': dict(block='early', time='early'),
    'lateBlock_early':  dict(block='late',  time='early'),
    'lateBlock_late':   dict(block='late',  time='late'),
}


def fit_lds(Z, U, valid):
    """
    Fit LDS w least-squares.

    Z: (n_pcs, T) state trajectory in PC space
    U: (n_inputs, T) input (log2 TF)
    valid: (T,) boolean mask for usable bins
    """
    # only use pairs where both t and t+1 are valid, avoiding spanning trial boundaries
    consecutive = valid[:-1] & valid[1:]
    idx = np.where(consecutive)[0]

    Z_curr = Z[:, idx]
    Z_next = Z[:, idx + 1]
    U_curr = U[:, idx]

    # solve Z_next = [A B] @ [Z_curr; U_curr] for [A B]
    predictors = np.vstack([Z_curr, U_curr])
    M, _, _, _ = np.linalg.lstsq(predictors.T, Z_next.T, rcond=None)
    M = M.T

    n_pcs = Z.shape[0]
    A = M[:, :n_pcs]
    B = M[:, n_pcs:]

    Z_pred = A @ Z_curr + B @ U_curr
    ss_res = np.sum((Z_next - Z_pred) ** 2)
    ss_tot = np.sum((Z_next - Z_next.mean(axis=1, keepdims=True)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return A, B, r2, len(idx)


def _build_input_vector(session, t_ax):
    """
    Build full-session stimulus vector U (1, T) by mapping per-trial TF onto FR matrix
    time bins. Bins outside any trial get 0
    """
    T = len(t_ax)
    U = np.zeros((1, T))

    for _, row in session.trials.iterrows():
        tf_raw = np.array(row['TF'])
        ft_raw = np.array(row['frame_time'])

        # remove zero/nan entries, subsample ::3 to get 20Hz update rate
        nonzero = tf_raw != 0
        valid_ft = ~np.isnan(ft_raw)
        keep = nonzero & valid_ft
        tf_20hz = tf_raw[keep][::3]
        ft_20hz = ft_raw[keep][::3]

        if len(ft_20hz) == 0:
            continue

        # find which FR bins fall within this trial's stimulus period
        mask = (t_ax >= ft_20hz[0]) & (t_ax <= ft_20hz[-1])
        bin_idx = np.where(mask)[0]
        if len(bin_idx) == 0:
            continue

        # zero-order hold: assign each bin the last stimulus value <= bin time
        insert_idx = np.searchsorted(ft_20hz, t_ax[bin_idx], side='right') - 1
        insert_idx = np.clip(insert_idx, 0, len(tf_20hz - 1)
        U[0, bin_idx] = np.log2(tf_20hz[insert_idx])

    return U


def _get_lick_mask(session, t_ax, buffer):
    """Boolean mask (T,): True for bins NOT within `buffer` of any lick."""
    mask = np.ones(len(t_ax), dtype=bool)
    if session.move is not None and 'licks' in session.move:
        lick_times = session.move['licks']
        for lt in lick_times:
            mask &= np.abs(t_ax - lt) > buffer
    return mask


def _get_condition_mask(session, t_ax, condition, ops, trial_indices=None):
    """
    Boolean mask (T,) selecting bins that:
    - belong to trials matching the condition (block + time-in-trial)
    - are not in transition trials
    - are away from licks

    trial_indices: if provided, only include these trials (for train/test)
    """
    cond = CONDITIONS[condition]
    mask = np.zeros(len(t_ax), dtype=bool)

    for tr, row in session.trials.iterrows():
        if trial_indices is not None and tr not in trial_indices:
            continue
        if row['tr_in_block'] <= ops['ignore_first_trials_in_block']:
            continue
        if row['hazardblock'] != cond['block']:
            continue

        # time window within trial
        bl_on = row['Baseline_ON_rise']
        tr_time_start = bl_on + ops['rmv_time_around']

        if cond['time'] == 'early':
            tr_time_end = bl_on + ops['tr_split_time']
        else:  # late — go until trial end
            tr_time_end = np.nanmax([row['Baseline_ON_fall'],
                                     row['Change_ON_fall']])

        if tr_time_end <= tr_time_start:
            continue

        mask |= (t_ax >= tr_time_start) & (t_ax < tr_time_end)

    # exclude bins near licks
    mask &= _get_lick_mask(session, t_ax, ops['rmv_time_around'])
    return mask


def _get_fold_splits(session, condition, ops, n_folds):
    """
    k-fold CV splits over trials. Returns list of (train_trials, test_trials).
    Only includes non-transition trials in the correct block.
    """
    cond = CONDITIONS[condition]
    valid_trials = []
    for tr, row in session.trials.iterrows():
        if row['tr_in_block'] <= ops['ignore_first_trials_in_block']:
            continue
        if row['hazardblock'] != cond['block']:
            continue
        valid_trials.append(tr)

    valid_trials = np.array(valid_trials)
    n = len(valid_trials)
    if n < n_folds:
        return []

    fold_size = n // n_folds
    folds = []
    for k in range(n_folds):
        test_idx = valid_trials[k * fold_size:(k + 1) * fold_size]
        train_idx = np.setdiff1d(valid_trials, test_idx)
        folds.append((train_idx, test_idx))
    return folds


def fit_session_lds(session, fr_matrix, weights, ops=ANALYSIS_OPTIONS):
    """
    Fit input-driven LDS per condition for one session.

    Args:
        session: Session object (with trials, move)
        fr_matrix: pd.DataFrame (nN x T)
        weights: (nN, n_pcs) PCA weights
    """
    t_ax = fr_matrix.columns.values

    # project full session into PC space
    Z = weights.T @ fr_matrix.values  # (n_pcs, T)
    U = _build_input_vector(session, t_ax)

    n_folds = ops['lds_n_folds']
    results = {}

    for cond_name in CONDITIONS:
        # fit on all data for this condition
        full_mask = _get_condition_mask(session, t_ax, cond_name, ops)
        A, B, r2_full, n_samples = fit_lds(Z, U, full_mask)

        eigenvalues = np.linalg.eigvals(A)

        # cross-validation
        folds = _get_fold_splits(session, cond_name, ops, n_folds)
        r2_train = np.full(n_folds, np.nan)
        r2_test = np.full(n_folds, np.nan)

        for k, (train_trials, test_trials) in enumerate(folds):
            train_mask = _get_condition_mask(session, t_ax, cond_name, ops,
                                             trial_indices=train_trials)
            test_mask = _get_condition_mask(session, t_ax, cond_name, ops,
                                            trial_indices=test_trials)

            A_k, B_k, r2_tr, _ = fit_lds(Z, U, train_mask)
            if A_k is None:
                continue
            r2_train[k] = r2_tr

            # test R2
            consec = test_mask[:-1] & test_mask[1:]
            idx = np.where(consec)[0]
            if len(idx) == 0:
                continue

            Z_pred = A_k @ Z[:, idx] + B_k @ U[:, idx]
            Z_actual = Z[:, idx + 1]
            ss_res = np.sum((Z_actual - Z_pred) ** 2)
            ss_tot = np.sum((Z_actual - Z_actual.mean(axis=1, keepdims=True)) ** 2)
            r2_test[k] = 1 - ss_res / ss_tot

        results[cond_name] = dict(
            A=A, B=B,
            eigenvalues=eigenvalues,
            r2_full=r2_full,
            r2_train=r2_train,
            r2_test=r2_test,
            n_samples=n_samples,
        )

    return results


def _save_lds_results(results, save_path):
    """
    Save to hdf5
        lds_<pca_key>.h5
        -<condition>/
        --A, B, eigenvalues_real, eigenvalues_imag
        --r2_full, r2_train, r2_test
        --n_samples (attr)
        - ...
    """
    with h5py.File(save_path, 'w') as f:
        for cond_name, res in results.items():
            cond = f.create_group(cond_name)
            cond.create_dataset('A', data=res['A'])
            cond.create_dataset('B', data=res['B'])
            cond.create_dataset('eigenvalues_real',
                                data=res['eigenvalues'].real)
            cond.create_dataset('eigenvalues_imag',
                                data=res['eigenvalues'].imag)
            cond.create_dataset('r2_full', data=res['r2_full'])
            cond.create_dataset('r2_train', data=res['r2_train'])
            cond.create_dataset('r2_test', data=res['r2_test'])
            cond.attrs['n_samples'] = res['n_samples']


def run_lds_analysis(npx_dir=PATHS['npx_dir_local'],
                     ops=ANALYSIS_OPTIONS,
                     pca_key='event_all'):
    """
    Fit input-driven LDS for all sessions.
    pca_key selects which PCA to use (e.g. 'event_all', 'event_V1', 'session_all').
    Saves per-session results to <session_dir>/lds_<pca_key>.h5.
    """
    psth_paths = get_response_files(npx_dir)

    for i, psth_path in enumerate(psth_paths):
        sess_dir = Path(psth_path).parent
        sess_data = Session.load(str(sess_dir / 'session.pkl'))
        print(f'{sess_data.animal}_{sess_data.name} ({i + 1}/{len(psth_paths)})')

        pca_path = sess_dir / 'pca.h5'
        if not pca_path.exists():
            print('  No pca.h5, skipping')
            continue

        with h5py.File(pca_path, 'r') as f:
            if pca_key not in f:
                print(f'  pca_key "{pca_key}" not found, skipping')
                continue
            weights = f[pca_key]['weights'][:]  # (nN, n_pcs)

        fr_path = sess_dir / 'FR_matrix.parquet'
        if not fr_path.exists():
            print('  No FR_matrix, skipping')
            continue
        fr_matrix = pd.read_parquet(fr_path)

        results = fit_session_lds(sess_data, fr_matrix, weights, ops)

        if results:
            save_path = str(sess_dir / f'lds_{pca_key}.h5')
            _save_lds_results(results, save_path)
            for cond_name, res in results.items():
                print(f'  {cond_name}: R²={res["r2_full"]:.3f}, '
                      f'R2_test={np.nanmean(res["r2_test"]):.3f}, '
                      f'n={res["n_samples"]}')
