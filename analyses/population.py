"""
Functions for population analyses
"""
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from sklearn.decomposition import PCA

from config import ANALYSIS_OPTIONS, PATHS
from data.session import Session
from analyses.load_responses import load_psth_mean
from utils.filing import get_response_files, load_fr_matrix
from utils.rois import AREA_GROUPS, in_any_area, in_group
from utils.smoothing import downsample_bins

# default event selection: event_type: list of conditions
DEFAULT_EVENT_SELECTION = {
    'tf':   ['earlyBlock_early_pos', 'earlyBlock_early_neg',
             'lateBlock_early_pos',  'lateBlock_early_neg',
             'lateBlock_late_pos',   'lateBlock_late_neg'],
    'lick': ['earlyBlock_early_fa', 'lateBlock_early_fa', 'lateBlock_late_fa'],
}

# event type: conditions to project PC weights onto
DEFAULT_PROJECTION_EVENTS = {
    'tf':   ['earlyBlock_early_pos', 'earlyBlock_early_neg',
             'lateBlock_early_pos',  'lateBlock_early_neg',
             'lateBlock_late_pos',   'lateBlock_late_neg'],
    'blOn': ['early', 'late'],
    'bl':   ['early', 'late'],
    'lick': ['earlyBlock_early_fa',  'earlyBlock_early_hit',
             'lateBlock_early_fa',   'lateBlock_early_hit',
             'lateBlock_late_fa',    'lateBlock_late_hit'],
    'ch':   ['early_hit_tf*', 'late_hit_tf*',
             'early_miss_tf*', 'late_miss_tf*'],
}



def print_pca_contents(pca_path: str):
    """Print keys and shapes in a pca.h5 file."""
    with h5py.File(pca_path, 'r') as f:
        for name in f:
            grp = f[name]
            n_neurons, n_pcs = grp['weights'].shape
            var = grp['var_explained'][:3]
            proj_keys = list(grp['projections'].keys()) if 'projections' in grp else []
            print(f'{name}: {n_neurons} neurons, {n_pcs} PCs, '
                  f'top3 var={var.round(3)}, {len(proj_keys)} projections')


def _run_pca(X: np.ndarray, n_components: int = 10):
    """
    Run PCA on (nN x nT) matrix.
    Returns explained variance ratio and weights (nN x nPC).
    """
    n_components = min(n_components, *X.shape)
    pca = PCA(n_components=n_components)
    weights = pca.fit_transform(X)  # nN x nPC
    return pca.explained_variance_ratio_, weights


def _project_events(psth_path: str,
                    weights: np.ndarray,
                    projection_events: dict[str, list[str]] = DEFAULT_PROJECTION_EVENTS,
                    area_mask: np.ndarray = None,
                    ds_factor: int = 1):
    """
    Project mean PSTHs through PC weights for specified event/condition pairs.
    Returns dict of {event_type/condition: nPC x nT} and corresponding t_ax dict.
    """
    import fnmatch
    projections = {}
    t_axes = {}

    with h5py.File(psth_path, 'r') as f:
        for et, conditions in projection_events.items():
            mean_key = f'{et}_mean'
            if mean_key not in f or f't_ax/{et}' not in f:
                continue
            t_axes[et] = downsample_bins(f[f't_ax/{et}'][:], ds_factor, axis=0)
            available = list(f[mean_key].keys())

            for cond_pattern in conditions:
                if '*' in cond_pattern:
                    matched = sorted(c for c in available
                                     if fnmatch.fnmatch(c, cond_pattern))
                else:
                    matched = [cond_pattern] if cond_pattern in available else []

                for cond in matched:
                    mean_resp = f[f'{mean_key}/{cond}'][:]  # nN x nT
                    if area_mask is not None:
                        mean_resp = mean_resp[area_mask]
                    mean_resp = downsample_bins(mean_resp, ds_factor)
                    projections[f'{et}/{cond}'] = weights.T @ mean_resp  # nPC x nT

    return projections, t_axes


def _extract_trial_bins(fr_matrix: pd.DataFrame,
                        session: Session,
                        trial_buffer: float = 1.0,
                        ds_factor: int = 1):
    """
    Mask FR matrix to only include time bins around trials (±trial_buffer from
    trial start/end). Optionally downsamples. Returns (nN x nT_valid) array.
    """
    t_ax = fr_matrix.columns.values
    valid = np.zeros(len(t_ax), dtype=bool)

    for _, row in session.trials.iterrows():
        t_start = row['Baseline_ON_rise'] - trial_buffer
        t_end = np.nanmax([row['Baseline_ON_fall'], row['Change_ON_fall']]) + trial_buffer
        valid |= (t_ax >= t_start) & (t_ax < t_end)

    out = fr_matrix.values[:, valid]
    return downsample_bins(out, ds_factor)


def _build_concat_matrix(psth_path: str,
                         event_selection: dict[str, list[str]],
                         unit_mask: np.ndarray = None,
                         ds_factor: int = 1):
    """
    Load mean PSTHs for selected event/condition pairs and concatenate.
    Uses load_psth_mean for each (event_type, condition), then filters to unit_mask.
    Returns (nN x nConditions*nT) matrix and list of condition labels.
    """
    blocks = []
    labels = []
    for event_type, conditions in event_selection.items():
        for cond in conditions:
            mean, _, _ = load_psth_mean(psth_path, event_type=event_type,
                                        condition=cond, baseline_subtract=False)
            if mean is None:
                continue
            if unit_mask is not None:
                mean = mean[unit_mask]
            if mean.shape[0] == 0:
                continue
            blocks.append(downsample_bins(mean, ds_factor))
            labels.append(f'{event_type}/{cond}')

    if not blocks:
        return None, []
    return np.concatenate(blocks, axis=1), labels


def pca_by_session(psth_path: str,
                   areas: np.ndarray,
                   ops: dict = ANALYSIS_OPTIONS,
                   event_selection: dict[str, list[str]] = None,
                   n_components: int = 10,
                   fr_matrix: np.ndarray = None):
    """
    Run PCA for each area group defined in AREA_GROUPS, plus 'all' (any known area).
    For event-aligned and (optionally) whole-session FR matrix.
    areas: brain_region_comb values for all units in the session (unfiltered).
    Returns dict of results keyed by e.g. 'event_all', 'event_thalamus', 'session_all'.
    """
    if event_selection is None:
        event_selection = DEFAULT_EVENT_SELECTION

    ds_factor = round(ops['pop_bin_width'] / ops['sp_bin_width'])
    results = {}

    for group_name in ['all', *AREA_GROUPS]:
        mask = in_any_area(areas) if group_name == 'all' else in_group(areas, group_name)
        if mask.sum() < 2:
            continue

        # event-aligned PCA
        X_ev, _ = _build_concat_matrix(psth_path, event_selection,
                                        unit_mask=mask, ds_factor=ds_factor)
        if X_ev is not None:
            var_exp, weights = _run_pca(X_ev, n_components)
            projections, t_axes = _project_events(psth_path, weights,
                                                   area_mask=mask, ds_factor=ds_factor)
            results[f'event_{group_name}'] = dict(var_explained=var_exp, weights=weights,
                                                   projections=projections, t_axes=t_axes)

        # whole-session PCA
        if fr_matrix is not None:
            fr_sub = fr_matrix[mask]
            if fr_sub.shape[0] >= 2:
                var_exp, weights = _run_pca(fr_sub, n_components)
                projections, t_axes = _project_events(psth_path, weights,
                                                       area_mask=mask, ds_factor=ds_factor)
                results[f'session_{group_name}'] = dict(var_explained=var_exp, weights=weights,
                                                         projections=projections, t_axes=t_axes)

    return results


def _save_pca_results(results: dict, save_path: str):
    with h5py.File(save_path, 'w') as f:
        for name, res in results.items():
            analysis = f.create_group(name)
            analysis.create_dataset('var_explained', data=res['var_explained'])
            analysis.create_dataset('weights', data=res['weights'])

            proj = analysis.create_group('projections')
            for label, arr in res['projections'].items():
                proj.create_dataset(label, data=arr)

            t_ax = analysis.create_group('t_ax')
            for et, arr in res['t_axes'].items():
                t_ax.create_dataset(et, data=arr)


def extract_pcs(npx_dir: str = PATHS['npx_dir_local'],
                ops: dict = ANALYSIS_OPTIONS,
                event_selection: dict[str, list[str]] = None,
                 include_whole_session: bool = True):
    """
    Extract PCs in several ways:
    1) event_aligned, all units, per session
    2) event_aligned, by area, per session
    3) whole session, all units (if include_whole_session)
    4) whole session, by area (if include_whole_session)
    5) event_aligned, all units, combined across sessions
    6) event_aligned, by area, combined across sessions

    Saves results to <session_dir>/pca.h5 and combined results
    to <npx_dir>/pca_combined.h5. hdf5 layout:

    pca.h5
    --- <analysis>                      e.g. event_all, event_V1, session_all
        - var_explained                 (nPC,)
        - weights                       (nN x nPC)
        --- projections
            - <event_type>/<cond>       (nPC x nT) for every condition in psths.h5
        --- t_ax
            - <event_type>              (nT,)
    """
    if event_selection is None:
        event_selection = DEFAULT_EVENT_SELECTION

    psth_paths = get_response_files(npx_dir)

    combined_blocks = {}  # area_key -> list of X matrices
    combined_labels = None

    n_components = ops['n_pcs']
    ds_factor = round(ops['pop_bin_width'] / ops['sp_bin_width'])

    for i, psth_path in enumerate(psth_paths):
        sess_data = Session.load(psth_path.replace('psths.h5', 'session.pkl'))
        print(f'{sess_data.animal}_{sess_data.name} ({i + 1}/{len(psth_paths)})')
        save_dir = Path(npx_dir) / sess_data.animal / sess_data.name

        areas = sess_data.unit_info['brain_region_comb'].values
        if not in_any_area(areas).any():
            continue

        # load whole-session FR matrix (trimmed to trial bins, downsampled)
        fr_matrix = None
        if include_whole_session:
            fr_path = save_dir / 'FR_matrix.parquet'
            if fr_path.exists():
                fr_df = load_fr_matrix(fr_path)
                fr_matrix = _extract_trial_bins(fr_df, sess_data,
                                                ds_factor=ds_factor)
                del fr_df

        # per-session PCA (loops through all area groups internally)
        results = pca_by_session(psth_path, areas, ops, event_selection,
                                 n_components, fr_matrix)
        if results:
            _save_pca_results(results, str(save_dir / 'pca.h5'))

        # accumulate for combined cross-session analysis
        for group_name in ['all', *AREA_GROUPS]:
            mask = in_any_area(areas) if group_name == 'all' else in_group(areas, group_name)
            if mask.sum() < 2:
                continue
            X_ev, cond_labels = _build_concat_matrix(psth_path, event_selection,
                                                      unit_mask=mask,
                                                      ds_factor=ds_factor)
            if X_ev is not None:
                combined_blocks.setdefault(group_name, []).append(X_ev)
                if combined_labels is None:
                    combined_labels = cond_labels

    # Combined PCA across all sessions
    combined_save_path = str(Path(npx_dir) / 'pca_combined.h5')
    with h5py.File(combined_save_path, 'w') as f:
        for key, blocks in combined_blocks.items():
            X = np.concatenate(blocks, axis=0)
            if X.shape[0] < n_components:
                continue
            var_exp, weights = _run_pca(X, n_components)
            grp = f.create_group(f'event_combined_{key}')
            grp.create_dataset('var_explained', data=var_exp)
            grp.create_dataset('weights', data=weights)
            if combined_labels is not None:
                grp.create_dataset('condition_labels',
                                   data=np.array(combined_labels, dtype='S'))


def extract_coding_dimensions():
    pass

def extract_lda():
    pass
