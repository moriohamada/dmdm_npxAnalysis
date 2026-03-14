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
from utils.filing import get_response_files

# default event selection: event_type -> list of conditions
DEFAULT_EVENT_SELECTION = {
    'tf':   ['earlyBlock_early_pos', 'earlyBlock_early_neg',
             'lateBlock_early_pos',  'lateBlock_early_neg',
             'lateBlock_late_pos',   'lateBlock_late_neg'],
    'lick': ['earlyBlock_early_fa', 'lateBlock_early_fa', 'lateBlock_late_fa'],
}

# event type -> conditions to project PC weights onto
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


def _run_pca(X: np.ndarray, n_components: int = 10):
    """
    Run PCA on (nN x nT) matrix.
    Returns explained variance ratio and weights (nN x nPC).
    """
    n_components = min(n_components, *X.shape)
    pca = PCA(n_components=n_components)
    weights = pca.fit_transform(X)  # nN x nPC
    return pca.explained_variance_ratio_, weights


def _project_events(psth_path: str, weights: np.ndarray,
                    projection_events: dict[str, list[str]] = DEFAULT_PROJECTION_EVENTS,
                    neuron_mask: np.ndarray = None):
    """
    Project mean PSTHs through PC weights for specified event/condition pairs.
    Conditions support wildcards (e.g. 'early_hit_tf*').
    Returns dict of {event_type/condition: nPC x nT} and corresponding t_ax dict.
    """
    import fnmatch
    W = weights if neuron_mask is None else weights[neuron_mask]
    projections = {}
    t_axes = {}

    with h5py.File(psth_path, 'r') as f:
        for et, conditions in projection_events.items():
            mean_key = f'{et}_mean'
            if mean_key not in f or f't_ax/{et}' not in f:
                continue
            t_axes[et] = f[f't_ax/{et}'][:]
            available = list(f[mean_key].keys())

            for cond_pattern in conditions:
                if '*' in cond_pattern:
                    matched = sorted(c for c in available
                                     if fnmatch.fnmatch(c, cond_pattern))
                else:
                    matched = [cond_pattern] if cond_pattern in available else []

                for cond in matched:
                    mean_resp = f[f'{mean_key}/{cond}'][:]  # nN x nT
                    if neuron_mask is not None:
                        mean_resp = mean_resp[neuron_mask]
                    projections[f'{et}/{cond}'] = W.T @ mean_resp  # nPC x nT

    return projections, t_axes


def _extract_trial_bins(fr_matrix: pd.DataFrame,
                        session: Session,
                        trial_buffer: float = 1.0):
    """
    Mask FR matrix to only include time bins around trials (±trial_buffer from
    trial start/end). Returns (nN x nT_valid) array.
    """
    t_ax = fr_matrix.columns.values
    valid = np.zeros(len(t_ax), dtype=bool)

    for _, row in session.trials.iterrows():
        t_start = row['Baseline_ON_rise'] - trial_buffer
        t_end = np.nanmax([row['Baseline_ON_fall'], row['Change_ON_fall']]) + trial_buffer
        valid |= (t_ax >= t_start) & (t_ax < t_end)

    return fr_matrix.values[:, valid]


def _build_concat_matrix(psth_path: str,
                         event_selection: dict[str, list[str]]):
    """
    Load mean PSTHs for selected event/condition pairs and concatenate.
    Uses load_psth_mean for each (event_type, condition).
    Returns (nN x nConditions*nT) matrix and list of condition labels.
    """
    blocks = []
    labels = []
    for event_type, conditions in event_selection.items():
        for cond in conditions:
            mean, _, _ = load_psth_mean(psth_path, event_type=event_type,
                                        condition=cond, baseline_subtract=True)
            if mean.shape[0] == 0:
                continue
            blocks.append(mean)  # nN x nT
            labels.append(f'{event_type}/{cond}')

    if not blocks:
        return None, []
    return np.concatenate(blocks, axis=1), labels


def pca_by_session(psth_path: str,
                   sess_data: Session,
                   event_selection: dict[str, list[str]] = None,
                   n_components: int = 10,
                   fr_matrix: np.ndarray = None,
                   X_event: np.ndarray = None,
                   cond_labels: list[str] = None):
    """
    Run PCA on a single session:
    - event-aligned mean PSTHs (all units + per area)
    - whole session FR matrix (all units + per area), if provided
    Returns dict of results keyed by analysis name.
    """
    if event_selection is None:
        event_selection = DEFAULT_EVENT_SELECTION

    results = {}
    areas = sess_data.unit_info['brain_region_comb'].values
    unique_areas = np.unique(areas)

    # Event-aligned PCA
    if X_event is None:
        X_event, cond_labels = _build_concat_matrix(psth_path, event_selection)

    if X_event is not None:
        var_exp, weights = _run_pca(X_event, n_components)
        projections, t_axes = _project_events(psth_path, weights)
        results['event_all'] = dict(var_explained=var_exp, weights=weights,
                                    projections=projections, t_axes=t_axes)

        for area in unique_areas:
            mask = areas == area
            if mask.sum() < n_components:
                continue
            var_exp, weights = _run_pca(X_event[mask], n_components)
            projections, t_axes = _project_events(psth_path, weights,
                                                      neuron_mask=mask)
            results[f'event_{area}'] = dict(var_explained=var_exp, weights=weights,
                                            projections=projections, t_axes=t_axes)

    # whole-session PCA
    if fr_matrix is not None:
        var_exp, weights = _run_pca(fr_matrix, n_components)
        projections, t_axes = _project_events(psth_path, weights)
        results['session_all'] = dict(var_explained=var_exp, weights=weights,
                                      projections=projections, t_axes=t_axes)

        for area in unique_areas:
            mask = areas == area
            if mask.sum() < n_components:
                continue
            var_exp, weights = _run_pca(fr_matrix[mask], n_components)
            projections, t_axes = _project_events(psth_path, weights,
                                                      neuron_mask=mask)
            results[f'session_{area}'] = dict(var_explained=var_exp, weights=weights,
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
                n_components: int = 10,
                include_whole_session: bool = False):
    """
    Extract PCs in several ways:
    1) event_aligned, all units, per session
    2) event_aligned, by area, per session
    3) whole session, all units (if include_whole_session)
    4) whole session, by area (if include_whole_session)
    5) event_aligned, all units, combined across sessions
    6) event_aligned, by area, combined across sessions

    Saves per-session results to <session_dir>/pca.h5 and combined results
    to <npx_dir>/pca_combined.h5. HDF5 layout:

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

    for i, psth_path in enumerate(psth_paths):
        sess_data = Session.load(psth_path.replace('psths.h5', 'session.pkl'))
        print(f'{sess_data.animal}_{sess_data.name} ({i + 1}/{len(psth_paths)})')
        save_dir = Path(npx_dir) / sess_data.animal / sess_data.name

        # load event-aligned data (reuse for per-session PCA + accumulation)
        X_ev, cond_labels = _build_concat_matrix(psth_path, event_selection)

        # load whole-session FR matrix (trimmed to trial bins)
        fr_matrix = None
        if include_whole_session:
            fr_path = save_dir / 'FR_matrix.parquet'
            if fr_path.exists():
                fr_df = pd.read_parquet(fr_path)
                fr_matrix = _extract_trial_bins(fr_df, sess_data)

        # per-session PCA
        results = pca_by_session(psth_path, sess_data, event_selection,
                                 n_components, fr_matrix, X_ev, cond_labels)
        if results:
            _save_pca_results(results, str(save_dir / 'pca.h5'))

        # accumulate for combined analysis
        if X_ev is not None:
            if combined_labels is None:
                combined_labels = cond_labels
            areas = sess_data.unit_info['brain_region_comb'].values
            combined_blocks.setdefault('all', []).append(X_ev)
            for area in np.unique(areas):
                mask = areas == area
                combined_blocks.setdefault(area, []).append(X_ev[mask])

    # Combined PCA across all sessions
    combined_results = {}
    for key, blocks in combined_blocks.items():
        X = np.concatenate(blocks, axis=0)  # stack neurons across sessions
        if X.shape[0] < n_components:
            continue
        comps, var_exp, scores = _run_pca(X, n_components)
        name = 'event_combined_all' if key == 'all' else f'event_combined_{key}'
        combined_results[name] = dict(components=comps, var_explained=var_exp,
                                      scores=scores, condition_labels=combined_labels)

    if combined_results:
        _save_pca_results(combined_results, str(Path(npx_dir) / 'pca_combined.h5'))


def extract_coding_dimensions():
    pass

def extract_lda():
    pass
