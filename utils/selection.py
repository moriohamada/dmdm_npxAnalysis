"""
Functions for selecting/filtering units, trials, time
"""
import numpy as np
import pandas as pd
from data.session import Session
from utils.rois import AREA_GROUPS, in_group


def filter_units(fr_stats: pd.DataFrame,
                 min_fr: float = 1.0,
                 min_fr_sd: float = 0.5) -> np.ndarray:
    """
    Return boolean mask (nN,) for units passing FR criteria based on fr mean and sd
    """
    return ((fr_stats['mean'] >= min_fr) &
            (fr_stats['sd'] >= min_fr_sd)).values

def trim_fr_to_periods(session: Session,
                       fr_matrix: pd.DataFrame,
                       include: str = 'trial',
                       buffer: float = 1):
    """
    select specific time periods to keep in fr matrix specified by 'include'.
        'trial' - keep only in-trial data +/- buffer seconds
        'baseline' - keep only baseline-period data (from -buffer seconds)

    """
    bl_starts = session.trials['Baseline_ON_rise'].values - buffer
    bl_ends   = session.trials['Baseline_ON_fall'].values
    tr_ends   = np.nanmax(session.trials[['Baseline_ON_fall',
                                          'Change_ON_fall']].values, axis=1) + buffer
    t_ax = fr_matrix.columns.values
    starts = bl_starts
    ends = tr_ends if include == 'trial' else bl_ends

    valid = np.any((t_ax[None, :] >= starts[:, None]) &
                   (t_ax[None, :] < ends[:, None]), axis=0)

    return fr_matrix.loc[:, valid]


CONDITIONS = {
    'earlyBlock_early': dict(block='early', time='early'),
    'lateBlock_early':  dict(block='late',  time='early'),
    'lateBlock_late':   dict(block='late',  time='late'),
}


def _get_lick_mask_old(session, t_ax, buffer):
    """Boolean mask (T,): True for bins NOT within `buffer` of any lick.
    Old version using session.move['licks'] — kept for comparison."""
    mask = np.ones(len(t_ax), dtype=bool)
    if session.move is not None and 'licks' in session.move:
        lick_times = session.move['licks']
        for lt in lick_times:
            mask &= np.abs(t_ax - lt) > buffer
    return mask


def _get_exclusion_mask(session, t_ax, buffer_bl, buffer_move):
    """Boolean mask (T,): False for bins within buffer_bl of any baseline onset
    or within buffer_move of any lick/abort"""
    mask = np.ones(len(t_ax), dtype=bool)

    for _, row in session.trials.iterrows():
        bl_on = row['Baseline_ON_rise']

        # exclude around baseline onset
        mask &= np.abs(t_ax - bl_on) > buffer_bl

        # exclude around lick (rt_FA relative to baseline onset)
        if not np.isnan(row.get('rt_FA', np.nan)):
            lick_t = bl_on + row['rt_FA']
            mask &= np.abs(t_ax - lick_t) > buffer_move

        # exclude around abort (rt_abort relative to baseline onset)
        if not np.isnan(row.get('rt_abort', np.nan)):
            abort_t = bl_on + row['rt_abort']
            mask &= np.abs(t_ax - abort_t) > buffer_move

    return mask


def get_condition_mask(session, t_ax, condition, ops, trial_indices=None):
    """
    Boolean mask (T,) selecting bins that:
    - belong to trials matching the condition (block + time-in-trial)
    - are not in transition trials (first few trials of block)
    - are away from licks/aborts/baseline onsets

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

        bl_on = row['Baseline_ON_rise']
        tr_time_start = bl_on + ops['rmv_time_around_bl']

        if cond['time'] == 'early':
            tr_time_end = bl_on + ops['tr_split_time']
        else:
            tr_time_start = bl_on + ops['tr_split_time']
            tr_time_end = np.nanmax([row['Baseline_ON_fall'],
                                     row['Change_ON_fall']])

        if tr_time_end <= tr_time_start:
            continue

        mask |= (t_ax >= tr_time_start) & (t_ax < tr_time_end)

    mask &= _get_exclusion_mask(session, t_ax,
                                ops['rmv_time_around_bl'],
                                ops['rmv_time_around_move'])
    return mask


def get_neuron_mask(sess_dir, area=None, unit_filter=None):
    """boolean mask for neurons matching area and/or GLM classification (OR logic)"""
    session = Session.load(str(sess_dir / 'session.pkl'))
    regions = session.unit_info['brain_region_comb'].values
    n = len(regions)

    if area is None or area == 'all':
        mask = np.ones(n, dtype=bool)
    elif area in AREA_GROUPS:
        mask = in_group(regions, area)
    else:
        mask = np.array([r == area for r in regions])

    if unit_filter is not None:
        glm_path = sess_dir / 'glm_ridge_classifications.csv'
        if not glm_path.exists():
            return np.zeros(n, dtype=bool)
        glm = pd.read_csv(glm_path)
        glm_mask = np.zeros(n, dtype=bool)
        n_glm = min(len(glm), n)
        for f in unit_filter:
            col = f'{f}_sig'
            if col in glm.columns:
                glm_mask[:n_glm] |= glm[col].values[:n_glm].astype(bool)
        mask &= glm_mask

    return mask


def get_window_bins(ops, dt):
    return max(1, int(round(ops['sliding_window_ms'] / 1000 / dt)))
