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
from utils.filing import get_response_files, load_fr_matrix
from utils.rois import in_any_area, in_group
from utils.smoothing import downsample_bins

CONDITIONS = {
    'earlyBlock_early': dict(block='early', time='early'),
    'lateBlock_early':  dict(block='late',  time='early'),
    'lateBlock_late':   dict(block='late',  time='late'),
}


def fit_lds(Z, U, valid):
    """
    Fit LDS w least-squares: z(t+1) = Az(t) + Bu(t)

    Z: (n_pcs, T) state trajectory in PC space
    U: (n_inputs, T) input (log2 TF)
    valid: (T,) boolean mask for usable bins
    """
    # only use pairs where both t and t+1 are valid, avoiding spanning trial boundaries
    consecutive = valid[:-1] & valid[1:]
    idx = np.where(consecutive)[0]
    if len(idx) < Z.shape[0] + 1:
        return None, None, np.nan, 0

    Z_curr = Z[:, idx]
    Z_next = Z[:, idx + 1]
    U_curr = U[:, idx]

    # solve Z_next = [A B] @ [Z_curr; U_curr] for [A B]
    predictors = np.vstack([Z_curr, U_curr])
    M, _, _, _ = np.linalg.lstsq(predictors.T, Z_next.T)
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
        insert_idx = np.clip(insert_idx, 0, len(tf_20hz - 1))
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


def fit_empirical_flow(Z, valid_mask, n_bins=15, grid_edges=None):
    """
    Estimate empirical flow field by binning state space and averaging dZ.

    Z: (n_dims, T) trajectory in PC space
    valid_mask: (T,) boolean — which bins to use
    n_bins: bins per dimension
    grid_edges: list of arrays, one per dimension. If None, computed from data.

    Returns:
        grid_edges: list of (n_bins+1,) arrays
        bin_centers: list of (n_bins,) arrays
        mean_flow: (n_bins, n_bins, ..., n_dims) mean dZ per bin
        counts: (n_bins, n_bins, ...) number of observations per bin
    """
    n_dims = Z.shape[0]

    # consecutive valid pairs
    consec = valid_mask[:-1] & valid_mask[1:]
    idx = np.where(consec)[0]
    if len(idx) == 0:
        return None

    Z_start = Z[:, idx].T      # (n_pairs, n_dims)
    dZ = (Z[:, idx + 1] - Z[:, idx]).T  # (n_pairs, n_dims)

    # build grid
    if grid_edges is None:
        grid_edges = []
        for d in range(n_dims):
            lo, hi = np.percentile(Z_start[:, d], [2, 98])
            grid_edges.append(np.linspace(lo, hi, n_bins + 1))

    bin_centers = [(e[:-1] + e[1:]) / 2 for e in grid_edges]

    # assign each point to a bin
    bin_idx = np.zeros((len(Z_start), n_dims), dtype=int)
    in_range = np.ones(len(Z_start), dtype=bool)
    for d in range(n_dims):
        bi = np.digitize(Z_start[:, d], grid_edges[d]) - 1
        bi = np.clip(bi, 0, n_bins - 1)
        # exclude points outside the grid
        in_range &= (Z_start[:, d] >= grid_edges[d][0])
        in_range &= (Z_start[:, d] <= grid_edges[d][-1])
        bin_idx[:, d] = bi

    bin_idx = bin_idx[in_range]
    dZ = dZ[in_range]

    # accumulate
    shape = tuple([n_bins] * n_dims)
    mean_flow = np.full(shape + (n_dims,), np.nan)
    counts = np.zeros(shape, dtype=int)

    # flatten bin indices to 1D for efficient accumulation
    strides = np.array([n_bins ** (n_dims - 1 - d) for d in range(n_dims)])
    flat_idx = (bin_idx * strides).sum(axis=1)

    for flat_bin in np.unique(flat_idx):
        mask = flat_idx == flat_bin
        multi_idx = tuple((flat_bin // strides) % n_bins)
        counts[multi_idx] = mask.sum()
        mean_flow[multi_idx] = dZ[mask].mean(axis=0)

    return grid_edges, bin_centers, mean_flow, counts


def fit_session_lds(session, Z, t_ax, U, ops=ANALYSIS_OPTIONS):
    """
    Fit input-driven LDS per condition for one session.

    Args:
        session: Session object (with trials, move)
        Z: (n_lds, T) PC trajectory
        t_ax: (T,) time axis
        U: (n_inputs, T) input vector
    """

    n_folds = ops['lds_n_folds']
    results = {}

    for cond_name in CONDITIONS:
        # fit on all data for this condition
        full_mask = _get_condition_mask(session, t_ax, cond_name, ops)
        A, B, r2_full, n_samples = fit_lds(Z, U, full_mask)
        if A is None:
            continue

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


def _load_session_fr(sess_dir, ops):
    """Load FR matrix, downsample, and return with session data and PCA weights."""
    sess_dir = Path(sess_dir)
    pca_path = sess_dir / 'pca.h5'
    fr_path = sess_dir / 'FR_matrix.parquet'
    if not pca_path.exists() or not fr_path.exists():
        return None

    sess_data = Session.load(str(sess_dir / 'session.pkl'))
    ds_factor = round(ops['pop_bin_width'] / ops['sp_bin_width'])
    fr_full = load_fr_matrix(fr_path)
    fr_full = downsample_bins(fr_full, ds_factor)
    areas = sess_data.unit_info['brain_region_comb'].values

    return sess_data, fr_full, areas, pca_path


def _get_fr_for_key(fr_full, areas, pca_key, pca_path):
    """Filter FR matrix and load weights for a pca_key."""
    with h5py.File(pca_path, 'r') as f:
        if pca_key not in f:
            return None, None
        weights = f[pca_key]['weights'][:]

    group_name = pca_key.split('_', 1)[1]
    if group_name == 'all':
        mask = in_any_area(areas)
    else:
        mask = in_group(areas, group_name)
    return fr_full.iloc[mask], weights


def lds_single_session(sess_dir, ops, pca_keys):
    """Fit LDS for one session across all pca_keys."""
    loaded = _load_session_fr(sess_dir, ops)
    if loaded is None:
        return
    sess_data, fr_full, areas, pca_path = loaded

    for pca_key in pca_keys:
        lds_path = Path(sess_dir) / f'lds_{pca_key}.h5'
        if lds_path.exists():
            continue

        fr_matrix, weights = _get_fr_for_key(fr_full, areas, pca_key, pca_path)
        if fr_matrix is None:
            continue

        t_ax = fr_matrix.columns.values
        n_lds = ops['lds_n_dims']
        Z = weights[:, :n_lds].T @ fr_matrix.values
        U = _build_input_vector(sess_data, t_ax)
        del fr_matrix

        results = fit_session_lds(sess_data, Z, t_ax, U, ops)
        if results:
            _save_lds_results(results, str(lds_path))
            for cond_name, res in results.items():
                print(f'  {pca_key}/{cond_name}: R2={res["r2_full"]:.3f}, '
                      f'R2_test={np.nanmean(res["r2_test"]):.3f}, '
                      f'n={res["n_samples"]}')

    del fr_full


def flow_single_session(sess_dir, ops, pca_keys):
    """Estimate empirical flow for one session across all pca_keys."""
    loaded = _load_session_fr(sess_dir, ops)
    if loaded is None:
        return
    sess_data, fr_full, areas, pca_path = loaded

    for pca_key in pca_keys:
        flow_path = Path(sess_dir) / f'flow_{pca_key}.h5'
        if flow_path.exists():
            continue

        fr_matrix, weights = _get_fr_for_key(fr_full, areas, pca_key, pca_path)
        if fr_matrix is None:
            continue

        t_ax = fr_matrix.columns.values
        n_flow = ops['flow_n_dims']
        Z_flow = weights[:, :n_flow].T @ fr_matrix.values
        del fr_matrix

        # shared grid from all conditions pooled
        all_valid = np.zeros(len(t_ax), dtype=bool)
        for cond_name in CONDITIONS:
            all_valid |= _get_condition_mask(sess_data, t_ax, cond_name, ops)

        shared_result = fit_empirical_flow(Z_flow, all_valid,
                                            n_bins=ops['flow_n_bins'])
        if shared_result is None:
            continue
        shared_edges = shared_result[0]

        with h5py.File(flow_path, 'w') as f:
            for d, edges in enumerate(shared_edges):
                f.create_dataset(f'grid_edges/{d}', data=edges)
                f.create_dataset(f'bin_centers/{d}',
                                 data=(edges[:-1] + edges[1:]) / 2)
            for cond_name in CONDITIONS:
                cond_mask = _get_condition_mask(sess_data, t_ax, cond_name, ops)
                result = fit_empirical_flow(Z_flow, cond_mask,
                                             n_bins=ops['flow_n_bins'],
                                             grid_edges=shared_edges)
                if result is None:
                    continue
                _, _, mean_flow, counts = result
                grp = f.create_group(cond_name)
                grp.create_dataset('mean_flow', data=mean_flow)
                grp.create_dataset('counts', data=counts)
        print(f'  {pca_key}: flow saved')

    del fr_full


def run_lds_analysis(npx_dir=PATHS['npx_dir_local'],
                     ops=ANALYSIS_OPTIONS,
                     pca_keys=None,
                     n_workers=2):
    """Fit linear LDS for all sessions and pca_keys."""
    from utils.rois import AREA_GROUPS
    if pca_keys is None:
        pca_keys = ['event_all'] + [f'event_{g}' for g in AREA_GROUPS]

    sess_dirs = [str(Path(p).parent) for p in get_response_files(npx_dir)]
    from multiprocessing import Pool
    from functools import partial
    with Pool(n_workers) as pool:
        pool.map(partial(lds_single_session, ops=ops, pca_keys=pca_keys), sess_dirs)


def run_flow_analysis(npx_dir=PATHS['npx_dir_local'],
                      ops=ANALYSIS_OPTIONS,
                      pca_keys=None,
                      n_workers=4):
    """Estimate empirical flow fields for all sessions and pca_keys."""
    from utils.rois import AREA_GROUPS
    if pca_keys is None:
        pca_keys = ['event_all'] + [f'event_{g}' for g in AREA_GROUPS]

    sess_dirs = [str(Path(p).parent) for p in get_response_files(npx_dir)]
    from multiprocessing import Pool
    from functools import partial
    with Pool(n_workers) as pool:
        pool.map(partial(flow_single_session, ops=ops, pca_keys=pca_keys), sess_dirs)
