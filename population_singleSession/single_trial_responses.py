"""
Visualize single trial/event responses along extracted axes - per session/area"""

from config import PATHS, ANALYSIS_OPTIONS
from utils.filing import get_session_dirs_by_animal, load_fr_matrix
from utils.rois import AREA_GROUPS
AREA_NAMES = AREA_GROUPS.keys()
from data.session import Session
from population_singleSession.single_trial_plotting import visualize_responses

EVENT_TYPES = ['bl', 'tf', 'ch', 'lick']
PRE_BASELINE = 1.0   # s; pre-baseline window for full-trial trajectories

from pathlib import Path, PosixPath
import numpy as np
import pandas as pd
import pickle
import h5py

#%%

def _load_dims(session_dir):
    mv_dims = pickle.load(open(session_dir/'move_dims.pkl', 'rb'))
    tf_dims = pickle.load(open(session_dir/'tf_dims.pkl', 'rb'))
    movenull_path = session_dir / 'tf_in_movenull.pkl'
    if movenull_path.exists():
        movenull_dims = pickle.load(open(movenull_path, 'rb'))
        for area, area_dims in movenull_dims.items():
            if area not in tf_dims:
                continue
            for block, block_dims in area_dims.items():
                if block == 'cids' or block not in tf_dims[area]:
                    continue
                tf_dims[area][block].update(block_dims)
    return {'mv': mv_dims, 'tf': tf_dims}

def _block_keys(area_dims):
    return [k for k in area_dims if k not in ('cids', 'delay')]


def _all_areas(dim_sources):
    out = set()
    for d in dim_sources.values():
        out |= set(d.keys())
    return out


def _all_blocks(dim_sources, area):
    out = set()
    for d in dim_sources.values():
        if area in d:
            out |= set(_block_keys(d[area]))
    return out


def _project_psths(h5_file, ev_types, w, in_area):
    """project all PSTH conditions through w (n_dim, nN_area)"""
    out = {}
    for ev in ev_types:
        ev_grp = h5_file.get(ev)
        if ev_grp is None:
            continue
        out[ev] = {}
        for cond in ev_grp.keys():
            psth = ev_grp[cond][:]
            if psth.shape[0] == 0:
                continue
            psth_area = psth[:, in_area, :]
            if psth_area.shape[1] != w.shape[1]:
                continue
            out[ev][cond] = np.einsum('dn,ent->edt', w, psth_area)
    return out


def _load_psth_projections(psth_path: PosixPath|str,
                           ev_types: list[str],
                           dim_sources: dict,
                           session: Session) -> dict:
    """
    project event-aligned psths onto each dim source's axes (per area / block).
    axis names are prefixed with the source name (e.g. 'mv_pot', 'tf_null').
    """
    proj = {'t_ax': {}, 'data': {}}
    with h5py.File(str(psth_path), 'r') as f:
        for ev in ev_types:
            if f't_ax/{ev}' in f:
                proj['t_ax'][ev] = f[f't_ax/{ev}'][:]

        for area in _all_areas(dim_sources):
            in_area = session.area_mask(AREA_GROUPS[area])
            proj['data'][area] = {}
            for block in _all_blocks(dim_sources, area):
                proj['data'][area][block] = {}
                for prefix, dims in dim_sources.items():
                    if area not in dims or block not in dims[area]:
                        continue
                    for axis, w in dims[area][block].items():
                        proj['data'][area][block][f'{prefix}_{axis}'] = (
                            _project_psths(f, ev_types, w, in_area)
                        )
    return proj


def _outcome_key(row):
    if row['IsHit']:   return 'hit'
    if row['IsMiss']:  return 'miss'
    if row['IsFA']:    return 'fa'
    if row['IsAbort']: return 'abort'
    return None


def _trial_window(row):
    """(bl_on, tr_end) full-trial slice; tr_end = nanmax(Baseline_ON_fall, Change_ON_fall)"""
    bl_on = row['Baseline_ON_rise']
    tr_end = np.nanmax([row['Baseline_ON_fall'], row['Change_ON_fall']])
    if pd.isna(bl_on) or pd.isna(tr_end):
        return None
    return bl_on, tr_end


def _trial_end_info(row):
    """(end_t, end_event) where end_t is relative to baseline onset"""
    bl_on = row['Baseline_ON_rise']
    if (row['IsHit'] or row['IsFA']) and not pd.isna(row['motion_onset']):
        return row['motion_onset'] - bl_on, 'lick'
    if row['IsAbort']:
        return row['rt_abort'], 'abort'
    if row['IsHit'] or row['IsMiss']:
        return row['Change_ON_rise'] - bl_on, 'change'
    return np.nan, None


def _change_t(row):
    if pd.isna(row['Change_ON_rise']) or pd.isna(row['Baseline_ON_rise']):
        return np.nan
    return row['Change_ON_rise'] - row['Baseline_ON_rise']


def _add_full_trial_trajectories(proj: dict,
                                 session_dir: PosixPath|str,
                                 dim_sources: dict,
                                 session: Session,
                                 ops: dict = ANALYSIS_OPTIONS) -> dict:
    """
    project full-trial activity (downsampled, NaN-padded to common length) onto each
    dim source's axes. saved under event type 'bl_traj', cond = trial outcome.
    populates proj['info'][area][block][outcome] with trial DataFrames.
    """
    fr_ds = load_fr_matrix(str(Path(session_dir) / 'FR_matrix_ds.parquet'))
    t_ax = fr_ds.columns.values
    bin_size = float(np.median(np.diff(t_ax)))
    trials = session.trials
    non_trans = (trials['tr_in_block'] > ops['ignore_first_trials_in_block']).values

    durs = []
    for _, row in trials.iterrows():
        win = _trial_window(row)
        if win is not None:
            durs.append(win[1] - win[0])
    if not durs:
        return proj

    n_pre  = int(round(PRE_BASELINE / bin_size))
    n_post = int(np.ceil(max(durs) / bin_size))
    t_rel  = np.arange(-n_pre, n_post) * bin_size
    nT     = len(t_rel)
    proj['t_ax']['bl_traj'] = t_rel
    proj.setdefault('info', {})

    for area in _all_areas(dim_sources):
        in_area = session.area_mask(AREA_GROUPS[area])
        fr_area = fr_ds.values[in_area, :]
        proj['info'].setdefault(area, {})

        for block in _all_blocks(dim_sources, area):
            trial_slices, trial_info = [], []
            for trial_idx, row in trials.iterrows():
                if block != 'all' and (not non_trans[trial_idx]
                                       or row['hazardblock'] != block):
                    continue
                outcome = _outcome_key(row)
                if outcome is None:
                    continue
                win = _trial_window(row)
                if win is None:
                    continue
                bl_on, tr_end = win
                t_in_trial = np.where((t_ax >= bl_on - PRE_BASELINE) &
                                      (t_ax < tr_end))[0]
                if len(t_in_trial) < 2:
                    continue
                start_idx = max(0, int(round(
                    (t_ax[t_in_trial[0]] - bl_on + PRE_BASELINE) / bin_size)))
                end_t, end_event = _trial_end_info(row)
                trial_slices.append((outcome, start_idx, fr_area[:, t_in_trial]))
                trial_info.append({
                    'trial': trial_idx,
                    'block': row['hazardblock'],
                    'outcome': outcome,
                    'change_t': _change_t(row),
                    'end_t': end_t,
                    'end_event': end_event,
                })
            if not trial_slices:
                continue

            idxs_by_outcome = {}
            for i, (outcome, _, _) in enumerate(trial_slices):
                idxs_by_outcome.setdefault(outcome, []).append(i)

            proj['info'][area].setdefault(block, {})
            for outcome, idxs in idxs_by_outcome.items():
                proj['info'][area][block][outcome] = pd.DataFrame(
                    [trial_info[i] for i in idxs])

            proj['data'].setdefault(area, {}).setdefault(block, {})
            for prefix, dims in dim_sources.items():
                if area not in dims or block not in dims[area]:
                    continue
                for axis, w in dims[area][block].items():
                    n_dim = w.shape[0]
                    axis_key = f'{prefix}_{axis}'
                    proj['data'][area][block].setdefault(axis_key, {})
                    proj['data'][area][block][axis_key]['bl_traj'] = {}
                    for outcome, idxs in idxs_by_outcome.items():
                        traj = np.full((len(idxs), n_dim, nT), np.nan)
                        for i, slice_idx in enumerate(idxs):
                            _, start_idx, neural = trial_slices[slice_idx]
                            proj_traj = w @ neural
                            end_idx = min(nT, start_idx + proj_traj.shape[1])
                            traj[i, :, start_idx:end_idx] = (
                                proj_traj[:, :end_idx - start_idx])
                        proj['data'][area][block][axis_key]['bl_traj'][outcome] = traj
    return proj


def project_and_visualize(npx_dir: str = PATHS["npx_dir_local"],
                          areas: list[str] = AREA_NAMES,
                          ops: dict = ANALYSIS_OPTIONS):
    """load psths + full-trial activity, project onto dims, plot per session/area"""
    animal_sessions = get_session_dirs_by_animal(npx_dir=npx_dir)

    for animal in animal_sessions.keys():
        print(f'Running single-trial analyses for {animal}')
        for session_dir in animal_sessions[animal]:
            print(f'    {session_dir}')

            dim_sources = _load_dims(session_dir)
            if all(len(d) == 0 for d in dim_sources.values()):
                print('        skipping - no dims extracted')
                continue

            session = Session.load(session_dir / 'session.pkl')

            proj = _load_psth_projections(session_dir / 'psths.h5',
                                          EVENT_TYPES, dim_sources, session)
            proj = _add_full_trial_trajectories(proj, session_dir, dim_sources,
                                                session, ops)

            visualize_responses(proj, session.animal, session.name)


#%%

if __name__ == '__main__':
    project_and_visualize()