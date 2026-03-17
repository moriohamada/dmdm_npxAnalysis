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
import gc
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

from scipy.stats import wilcoxon
from config import ANALYSIS_OPTIONS, PATHS
from data.session import Session
from utils.filing import get_response_files
from utils.rois import in_any_area, in_group
from utils.downsampling import downsample_bins

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

        # clean following stimulus.py: remove TF zeros, truncate to change frame,
        # remove frame_time NaNs, subsample ::3 for 20Hz
        tf_seq = tf_raw[tf_raw.nonzero()]
        ch_fr = round(row['stimT'] * 60)
        tf_20hz = np.log2(tf_seq[:ch_fr:3])
        ft_20hz = ft_raw[~np.isnan(ft_raw)][:ch_fr:3]

        if len(tf_20hz) == 0 or len(ft_20hz) == 0:
            continue
        if len(tf_20hz) != len(ft_20hz):
            continue

        # find which FR bins fall within this trial's stimulus period
        mask = (t_ax >= ft_20hz[0]) & (t_ax <= ft_20hz[-1])
        bin_idx = np.where(mask)[0]
        if len(bin_idx) == 0:
            continue

        # zero-order hold: assign each bin the last stimulus value <= bin time
        insert_idx = np.searchsorted(ft_20hz, t_ax[bin_idx], side='right') - 1
        insert_idx = np.clip(insert_idx, 0, len(tf_20hz) - 1)
        U[0, bin_idx] = tf_20hz[insert_idx]

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
        else:  # late — start at split time, go until trial end
            tr_time_start = bl_on + ops['tr_split_time']
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


def fit_empirical_flow(Z, valid_mask, n_bins=15, grid_edges=None, min_count=5000):
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
            lo, hi = np.percentile(Z_start[:, d], [2.5, 97.5])
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
        n = mask.sum()
        counts[multi_idx] = n
        if n >= min_count:
            mean_flow[multi_idx] = dZ[mask].mean(axis=0)

    return grid_edges, bin_centers, mean_flow, counts


def _flow_r2(Z, valid_mask, grid_edges, mean_flow, n_bins):
    """
    Compute R2 of flow field predictions on held-out data.
    Null model is dZ=0, so R2 > 0 means the flow field is informative.
    """
    n_dims = Z.shape[0]
    consec = valid_mask[:-1] & valid_mask[1:]
    idx = np.where(consec)[0]
    if len(idx) == 0:
        return np.nan

    Z_start = Z[:, idx].T
    dZ = (Z[:, idx + 1] - Z[:, idx]).T

    strides = np.array([n_bins ** (n_dims - 1 - d) for d in range(n_dims)])
    bin_idx = np.zeros((len(Z_start), n_dims), dtype=int)
    in_range = np.ones(len(Z_start), dtype=bool)
    for d in range(n_dims):
        bi = np.digitize(Z_start[:, d], grid_edges[d]) - 1
        bi = np.clip(bi, 0, n_bins - 1)
        in_range &= (Z_start[:, d] >= grid_edges[d][0])
        in_range &= (Z_start[:, d] <= grid_edges[d][-1])
        bin_idx[:, d] = bi

    bin_idx = bin_idx[in_range]
    dZ = dZ[in_range]
    if len(dZ) == 0:
        return np.nan

    flat_idx = (bin_idx * strides).sum(axis=1)
    dZ_pred = np.zeros_like(dZ)
    for i, fi in enumerate(flat_idx):
        multi_idx = tuple((fi // strides) % n_bins)
        flow_val = mean_flow[multi_idx]
        if not np.any(np.isnan(flow_val)):
            dZ_pred[i] = flow_val

    ss_res = np.sum((dZ - dZ_pred) ** 2)
    ss_tot = np.sum(dZ ** 2)  # null is dZ=0
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


def fit_session_lds(session, Z, t_ax, U, ops=ANALYSIS_OPTIONS,
                    compute_cv_r2=False):
    """
    Fit input-driven LDS per condition for one session.

    args:
        session: Session object (with trials, move)
        Z: (n_lds, T) PC trajectory
        t_ax: (T,) time axis
        U: (n_inputs, T) input vector
        compute_cv_r2: if True, compute R2 vs identity-A null and cross-block R2
    """
    results = {}

    for cond_name in CONDITIONS:
        # fit on all data for this condition
        full_mask = _get_condition_mask(session, t_ax, cond_name, ops)
        A, B, r2_full, n_samples = fit_lds(Z, U, full_mask)
        if A is None:
            continue

        eigenvalues = np.linalg.eigvals(A)

        res = dict(
            A=A, B=B,
            eigenvalues=eigenvalues,
            r2_full=r2_full,
            n_samples=n_samples,
        )

        if compute_cv_r2:
            # null model: A=I, refit B on full data
            consec = full_mask[:-1] & full_mask[1:]
            idx = np.where(consec)[0]
            dZ = Z[:, idx + 1] - Z[:, idx]
            B_null, _, _, _ = np.linalg.lstsq(U[:, idx].T, dZ.T)
            B_null = B_null.T

            Z_actual = Z[:, idx + 1]
            ss_tot = np.sum((Z_actual - Z_actual.mean(axis=1, keepdims=True)) ** 2)
            Z_pred_null = Z[:, idx] + B_null @ U[:, idx]
            r2_null = 1 - np.sum((Z_actual - Z_pred_null) ** 2) / ss_tot

            res.update(r2_null=r2_null, delta_r2=r2_full - r2_null)

        results[cond_name] = res

    # cross-block prediction on held-out data
    eB_key, lB_key = 'earlyBlock_early', 'lateBlock_early'
    if compute_cv_r2 and eB_key in results and lB_key in results:
        A_eB_full, B_eB_full = results[eB_key]['A'], results[eB_key]['B']
        A_lB_full, B_lB_full = results[lB_key]['A'], results[lB_key]['B']
        n_folds = ops['lds_n_folds']

        for cond_key, A_cross, B_cross in [(eB_key, A_lB_full, B_lB_full),
                                            (lB_key, A_eB_full, B_eB_full)]:
            folds = _get_fold_splits(session, cond_key, ops, n_folds)
            r2_within = np.full(n_folds, np.nan)
            r2_cross = np.full(n_folds, np.nan)

            for k, (train_trials, test_trials) in enumerate(folds):
                train_mask = _get_condition_mask(session, t_ax, cond_key, ops,
                                                 trial_indices=train_trials)
                test_mask = _get_condition_mask(session, t_ax, cond_key, ops,
                                                trial_indices=test_trials)
                A_k, B_k, _, _ = fit_lds(Z, U, train_mask)
                if A_k is None:
                    continue

                consec = test_mask[:-1] & test_mask[1:]
                idx = np.where(consec)[0]
                if len(idx) == 0:
                    continue

                Z_actual = Z[:, idx + 1]
                ss_tot = np.sum((Z_actual - Z_actual.mean(axis=1, keepdims=True)) ** 2)

                Z_pred_w = A_k @ Z[:, idx] + B_k @ U[:, idx]
                r2_within[k] = 1 - np.sum((Z_actual - Z_pred_w) ** 2) / ss_tot

                Z_pred_c = A_cross @ Z[:, idx] + B_cross @ U[:, idx]
                r2_cross[k] = 1 - np.sum((Z_actual - Z_pred_c) ** 2) / ss_tot

            results[cond_key]['r2_within_cv'] = r2_within
            results[cond_key]['r2_cross_cv'] = r2_cross
            results[cond_key]['delta_cross'] = r2_within - r2_cross

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
            cond.attrs['n_samples'] = res['n_samples']
            if 'r2_null' in res:
                cond.attrs['r2_null'] = res['r2_null']
                cond.attrs['delta_r2'] = res['delta_r2']
            if 'r2_within_cv' in res:
                cond.create_dataset('r2_within_cv', data=res['r2_within_cv'])
                cond.create_dataset('r2_cross_cv', data=res['r2_cross_cv'])
                cond.create_dataset('delta_cross', data=res['delta_cross'])


def _load_session_fr(sess_dir, ops):
    """load downsampled FR matrix and return with session data."""
    import pandas as pd
    sess_dir = Path(sess_dir)
    pca_path = sess_dir / 'pca.h5'
    fr_path = sess_dir / 'FR_matrix_ds.parquet'
    if not pca_path.exists():
        print(f'    PCA data not found in {sess_dir}!')
        return None
    if not fr_path.exists():
        print(f'    Downsampled FR data not found in {sess_dir}!')
        return None
    sess_data = Session.load(str(sess_dir / 'session.pkl'))
    fr_full = pd.read_parquet(fr_path)
    areas = sess_data.unit_info['brain_region_comb'].values

    return sess_data, fr_full, areas, pca_path


def _get_fr_for_key(fr_full, areas, pca_key, pca_path):
    with h5py.File(pca_path, 'r') as f:
        if pca_key not in f:
            return None, None
        weights = f[pca_key]['weights'][:]

    group_name = pca_key.split('_', 1)[1]
    if group_name == 'all':
        mask = in_any_area(areas)
    else:
        mask = in_group(areas, group_name)
    fr_filtered = fr_full.iloc[mask]
    if fr_filtered.shape[0] != weights.shape[0]:
        print(f'    WARNING: {pca_key} unit count mismatch '
              f'(FR={fr_filtered.shape[0]}, pca weights={weights.shape[0]}) '
              f'— pca.h5 may be stale, re-run extract_pcs()')
        return None, None
    return fr_filtered, weights


def lds_single_session(sess_dir, ops, pca_keys, compute_cv_r2=False):
    """Fit LDS for one session across all pca_keys."""
    print(sess_dir)
    loaded = _load_session_fr(sess_dir, ops)
    if loaded is None:
        return
    sess_data, fr_full, areas, pca_path = loaded
    del loaded; gc.collect()
    for pca_key in pca_keys:
        lds_path = Path(sess_dir) / f'lds_{pca_key}.h5'
        # if lds_path.exists():
        #     continue

        fr_matrix, weights = _get_fr_for_key(fr_full, areas, pca_key, pca_path)
        if fr_matrix is None:
            continue

        t_ax = fr_matrix.columns.values
        n_lds = ops['lds_n_dims']
        Z = weights[:, :n_lds].T @ fr_matrix.values
        U = _build_input_vector(sess_data, t_ax)
        del fr_matrix; gc.collect()

        results = fit_session_lds(sess_data, Z, t_ax, U, ops,
                                  compute_cv_r2=compute_cv_r2)
        if results:
            _save_lds_results(results, str(lds_path))
            for cond_name, res in results.items():
                msg = (f'  {pca_key}/{cond_name}: '
                       f'R2={res["r2_full"]:.3f}, n={res["n_samples"]}')
                if 'r2_null' in res:
                    msg += f', R2_null={res["r2_null"]:.3f}, dR2={res["delta_r2"]:.3f}'
                if 'delta_cross' in res:
                    msg += (f', R2_within_cv={np.nanmean(res["r2_within_cv"]):.3f}'
                            f', R2_cross_cv={np.nanmean(res["r2_cross_cv"]):.3f}'
                            f', dR2_cross={np.nanmean(res["delta_cross"]):.3f}')
                print(msg)

    del fr_full; gc.collect()


def flow_single_session(sess_dir, ops, pca_keys, compute_cv_r2=False):
    """Estimate empirical flow for one session across all pca_keys."""
    loaded = _load_session_fr(sess_dir, ops)
    if loaded is None:
        return
    sess_data, fr_full, areas, pca_path = loaded
    n_folds = ops['lds_n_folds']
    n_bins = ops['flow_n_bins']

    for pca_key in pca_keys:
        flow_path = Path(sess_dir) / f'flow_{pca_key}.h5'
        # if flow_path.exists():
        #     continue

        fr_matrix, weights = _get_fr_for_key(fr_full, areas, pca_key, pca_path)
        if fr_matrix is None:
            continue

        t_ax = fr_matrix.columns.values
        n_flow = ops['flow_n_dims']
        Z_flow = weights[:, :n_flow].T @ fr_matrix.values
        del fr_matrix; gc.collect()

        # shared grid from all conditions pooled
        all_valid = np.zeros(len(t_ax), dtype=bool)
        for cond_name in CONDITIONS:
            all_valid |= _get_condition_mask(sess_data, t_ax, cond_name, ops)

        min_count = ops['flow_min_count']
        shared_result = fit_empirical_flow(Z_flow, all_valid, n_bins=n_bins,
                                            min_count=min_count)
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
                                             n_bins=n_bins,
                                             grid_edges=shared_edges,
                                             min_count=min_count)
                if result is None:
                    continue
                _, _, mean_flow, counts = result
                grp = f.create_group(cond_name)
                grp.create_dataset('mean_flow', data=mean_flow)
                grp.create_dataset('counts', data=counts)

                if compute_cv_r2:
                    folds = _get_fold_splits(sess_data, cond_name, ops, n_folds)
                    cv_min_count = max(1, int(min_count * (n_folds - 1) / n_folds))
                    r2_cv = np.full(n_folds, np.nan)
                    for k, (train_trials, test_trials) in enumerate(folds):
                        train_mask = _get_condition_mask(
                            sess_data, t_ax, cond_name, ops,
                            trial_indices=train_trials)
                        test_mask = _get_condition_mask(
                            sess_data, t_ax, cond_name, ops,
                            trial_indices=test_trials)

                        train_result = fit_empirical_flow(
                            Z_flow, train_mask, n_bins=n_bins,
                            grid_edges=shared_edges,
                            min_count=cv_min_count)
                        if train_result is None:
                            continue
                        _, _, train_flow, _ = train_result
                        r2_cv[k] = _flow_r2(Z_flow, test_mask, shared_edges,
                                            train_flow, n_bins)

                    valid_folds = ~np.isnan(r2_cv)
                    if valid_folds.sum() >= 3:
                        stat, pval = wilcoxon(r2_cv[valid_folds],
                                              alternative='greater')
                    else:
                        stat, pval = np.nan, np.nan

                    grp.create_dataset('r2_cv', data=r2_cv)
                    grp.attrs['wilcoxon_stat'] = stat
                    grp.attrs['wilcoxon_p'] = pval

        print(f'{sess_data.animal}, {sess_data.name}')
        print(f'  {pca_key}: flow saved')
        if compute_cv_r2:
            with h5py.File(flow_path, 'r') as f:
                for cond_name in CONDITIONS:
                    if cond_name in f and 'r2_cv' in f[cond_name]:
                        pval = f[cond_name].attrs.get('wilcoxon_p', np.nan)
                        r2m = np.nanmean(f[cond_name]['r2_cv'][:])
                        print(f'    {cond_name}: R2_cv={r2m:.3f}, p={pval:.4f}')

    del fr_full; gc.collect()


def run_lds_analysis(npx_dir=PATHS['npx_dir_local'],
                     ops=ANALYSIS_OPTIONS,
                     pca_keys=None,
                     n_workers=1,
                     compute_cv_r2=False):
    """Fit linear LDS for all sessions and pca_keys."""
    from utils.rois import AREA_GROUPS
    if pca_keys is None:
        pca_keys = ['event_all'] + [f'event_{g}' for g in AREA_GROUPS]

    sess_dirs = [str(Path(p).parent) for p in get_response_files(npx_dir)]
    if n_workers is None or n_workers <= 1:
        for sd in sess_dirs:
            lds_single_session(sd, ops, pca_keys, compute_cv_r2=compute_cv_r2)
    else:
        from multiprocessing import Pool
        from functools import partial
        with Pool(n_workers) as pool:
            pool.map(partial(lds_single_session, ops=ops, pca_keys=pca_keys,
                             compute_cv_r2=compute_cv_r2), sess_dirs)


def run_flow_analysis(npx_dir=PATHS['npx_dir_local'],
                      ops=ANALYSIS_OPTIONS,
                      pca_keys=None,
                      n_workers=1,
                      compute_cv_r2=False):
    """Estimate empirical flow fields for all sessions and pca_keys."""
    from utils.rois import AREA_GROUPS
    if pca_keys is None:
        pca_keys = ['event_all'] + [f'event_{g}' for g in AREA_GROUPS]

    sess_dirs = [str(Path(p).parent) for p in get_response_files(npx_dir)]
    if n_workers is None or n_workers <= 1:
        for sd in sess_dirs:
            flow_single_session(sd, ops, pca_keys, compute_cv_r2=compute_cv_r2)
    else:
        from multiprocessing import Pool
        from functools import partial
        with Pool(n_workers) as pool:
            pool.map(partial(flow_single_session, ops=ops, pca_keys=pca_keys,
                             compute_cv_r2=compute_cv_r2), sess_dirs)
