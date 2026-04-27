"""
Get averaged lick-aligned activity (by area, by session inc orofacial nuclei) to compute
movement potent and null spaces - single sessions.
"""

from config import PATHS, ANALYSIS_OPTIONS, MOVEDIM_OPTIONS
from utils.rois import AREA_GROUPS
AREA_NAMES = list(AREA_GROUPS.keys())
from utils.filing import get_session_dirs_by_animal, load_fr_matrix
from data.session import Session
from data.responses import compute_psth
from utils.time import time_mask

from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.linalg import null_space

def get_lick_activity_by_area(session: Session,
                              areas: list[str],
                              move_ops: dict) -> dict:
    """
    Iterate through areas and extract lick-aligned activity.
    Returns dict keyed by area, each value also a dict with:
    ['cids']: list of unit ids,
    ['X']: nEv x nN x nT np array containing peri-lick activity
    ['t']: time axis for X
    """
    lick_aligned_sp = dict()
    for group in areas:
        in_area = session.area_mask(AREA_GROUPS[group])
        if not any(in_area):
            continue
        lick_aligned_sp[group] = dict()
        lick_aligned_sp[group]['cids'] = session.unit_info.cluster_id.values[in_area]
        lick_aligned_sp[group]['X'], lick_aligned_sp[group]['t'] = (
            compute_psth(X=session.fr_matrix.values[in_area,:],
                         t_ax=session.fr_matrix.columns.values,
                         event_times = session.lick_times.time.values,
                         resp_win = move_ops['full_lick_period'])
        )

    return lick_aligned_sp

def get_lick_aligned_motion(session: Session,
                            eye_cam_times: np.ndarray,
                            move_ops: dict) -> dict:
    """
    Get lick-aligned movement signals (mouth/whisker motion energy, pupil area, running
    speed), resampled onto the fr_matrix time axis.

    Returns dict keyed by signal name, each value also a dict with:
    ['E']: nEv x nT np array containing peri-lick movement energy/pupil size/speed
    ['t']: time axis for X
    """
    t_ax = session.fr_matrix.columns.values
    lick_times = session.lick_times.time.values
    video = session.move['video']
    running = session.move['running']

    signals = {
        'mouth_me':   (video['mouth_me'],    eye_cam_times),
        'whisker_me': (video['whisker_me'],  eye_cam_times),
        # 'pupil_area': (video['pupil_area'],  eye_cam_times),
        'wheel_speed':(running['speed'],     running['time']),
    }

    lick_aligned_mv = {}
    for name, (sig, times) in signals.items():
        sig = np.asarray(sig, dtype=float)
        times = np.asarray(times)
        n = min(len(sig), len(times))
        aligned = np.interp(t_ax, times[:n], sig[:n])  # nT
        E, ev_t = compute_psth(
            X=aligned[None, :],
            t_ax=t_ax,
            event_times=lick_times,
            resp_win=move_ops['full_lick_period'],
        )
        lick_aligned_mv[name] = {'E': E[:, 0, :], 't': ev_t}

    return lick_aligned_mv

def _centre_mov(lick_aligned_mv):
    for move_type in lick_aligned_mv.keys():
        lick_aligned_mv[move_type]['E_mean'] = (
            np.nanmean(lick_aligned_mv[move_type]['E'], axis=0)
        )
        lick_aligned_mv[move_type]['E_z'] = (
            (lick_aligned_mv[move_type]['E_mean'] -
             np.nanmean(lick_aligned_mv[move_type]['E_mean'])) /
            np.nanstd(lick_aligned_mv[move_type]['E_mean'])
        )
    return lick_aligned_mv

def _centre_neural(lick_aligned_sp,
                   bl_t: tuple[float,float] = (-2, -1.5)):
    for area in lick_aligned_sp.keys():
        lick_aligned_sp[area]['X_mean'] = np.nanmean(lick_aligned_sp[area]['X'], axis=0)
        # bl_subtract
        in_bl = ((lick_aligned_sp[area]['t'] >= bl_t[0]) &
                 (lick_aligned_sp[area]['t'] <= bl_t[1]))

        lick_aligned_sp[area]['X_centred'] = (
            lick_aligned_sp[area]['X_mean'] -
            np.nanmean(lick_aligned_sp[area]['X_mean'][:, in_bl], axis=1, keepdims=True)
        )

    return lick_aligned_sp

def _centre_signals(lick_aligned_mv, lick_aligned_sp):
    # z-score motion energy
    lick_aligned_mv = _centre_mov(lick_aligned_mv)
    # baseline subtract neural
    lick_aligned_sp = _centre_neural(lick_aligned_sp)
    return lick_aligned_mv, lick_aligned_sp

def calculate_movement_dims_by_area(mv_by_block: dict,
                                    sp_by_block: dict,
                                    move_ops: dict = MOVEDIM_OPTIONS):
    """
    regress motion energy versus neural activity by area, trying delays from 0
    to 100ms; pick the delay with best R2 on 'all' (per area), reuse for early/late.
    motion + neural PCA bases fit on 'all'; regression (on 'lick_period'), null
    space and null rotation done per block at the chosen delay.
    Returns: move_dims[area], dict with 'cids', 'delay' (s), and per-block:
        pot: n_pot_dims x nN
        null: n_null_dims x nN
    """

    blocks = list(mv_by_block.keys())  # e.g. ['all', 'early', 'late']

    # PCA of motor (fit on 'all', applied per block)
    mv_stack = {b: np.stack([v['E_z'] for _, v in mv_by_block[b].items()], axis=1)
                for b in blocks}
    mv_pca = PCA(n_components=2) # basiaclly seemed two dimensional - face motE, wheel spd
    mv_pca.fit(mv_stack['all'])

    # delay grid: 0..100ms in FR bin steps (motion side - always populated)
    any_t = next(iter(mv_by_block['all'].values()))['t']
    bin_size = float(np.median(np.diff(any_t)))
    delay_bins = np.arange(int(round(0.1 / bin_size)) + 1)

    move_dims = dict()
    for area in sp_by_block['all'].keys():

        X_fulldim = sp_by_block['all'][area]['X_centred']
        t = sp_by_block['all'][area]['t']
        if X_fulldim.shape[0] < move_ops['min_neurons']:
            continue
        move_dims[area] = {'cids': sp_by_block['all'][area]['cids']}

        # neural PCA (fit on 'all', applied per block)
        sp_pca = PCA(n_components=6)
        sp_pca.fit(X_fulldim.T)
        pre_lick_t = time_mask(t, move_ops['prelick_period'])
        lick_t = time_mask(t, move_ops['lick_period'])

        # PC scores + motor targets per block
        Xs = {b: sp_pca.transform(sp_by_block[b][area]['X_centred'].T) for b in blocks}
        ys = {b: (mv_pca.components_ @ mv_stack[b].T).T for b in blocks}  # nT x n_motor

        # pick delay on 'all' by max R2 within lick_period
        r2s = []
        for k in delay_bins:
            X_lag = Xs['all'][:len(t)-k]
            y_lag = ys['all'][k:]
            mask = lick_t[k:]
            mdl = LinearRegression(fit_intercept=True).fit(X_lag[mask], y_lag[mask])
            r2s.append(mdl.score(X_lag[mask], y_lag[mask]))
        best_k = int(delay_bins[np.argmax(r2s)])
        move_dims[area]['delay'] = best_k * bin_size

        for block in blocks:
            X_lag = Xs[block][:len(t)-best_k]
            y_lag = ys[block][best_k:]
            mask = lick_t[best_k:]

            # linear regression to get w_pot
            pot_mdl = LinearRegression(fit_intercept=True)
            pot_mdl.fit(X_lag[mask], y_lag[mask])
            # w_null
            null_pc = null_space(pot_mdl.coef_)

            # rotate w_null to capture pre-lick
            rot_null_pc = PCA()
            rot_null_pc.fit((null_pc.T @ Xs[block][pre_lick_t,:].T).T)

            # compute full weighting to go from neurons to movement_potent/-null spaces
            move_dims[area][block] = dict()
            move_dims[area][block]['pot'] = pot_mdl.coef_ @ sp_pca.components_
            move_dims[area][block]['null']= rot_null_pc.components_ @ null_pc.T @ sp_pca.components_

    return move_dims

def fit_movespace_per_session(npx_dir: str = PATHS['npx_dir_local'],
                              ops: dict = ANALYSIS_OPTIONS,
                              move_ops: dict = MOVEDIM_OPTIONS,
                              areas: list[str] = AREA_NAMES,
                              ):
    """
    Runner to loop through all sessions, extract lick-aligned neural activity and motion
    energy, calculate movement-potent dim.
    """
    animal_sessions = get_session_dirs_by_animal(npx_dir)

    for animal in animal_sessions.keys():
        print(f'Fitting movement space for {animal}')
        sessions = animal_sessions[animal]
        for session_dir in sessions:
            print(f'    {session_dir}')

            session = Session.load(session_dir/'session.pkl')
            # load FR matrix
            fr = load_fr_matrix(str(session_dir / 'FR_matrix.parquet'))
            session.fr_matrix = fr
            # eye-cam timestamps from ceph -
            # TODO: change intiial preprocessing to have these locally! no point having
            #  movement signals without time ax...
            ceph_sess = Path(PATHS['ceph_dir']) / session.animal / session.name
            eye_cam_times = pd.read_csv(ceph_sess / 'daq_Eye_cam.csv')['rise_t'].values

            # drop licks too close to baseline onset (applies to all blocks)
            valid_t = (session.lick_times['tr_time'] > ops['rmv_time_around_bl']).values
            session.lick_times = session.lick_times[valid_t].reset_index(drop=True)

            # create [area](fr) key for lick-aligned movement
            lick_aligned_sp  = get_lick_activity_by_area(session, areas, move_ops)
            lick_aligned_mv = get_lick_aligned_motion(session, eye_cam_times, move_ops)

            # event-axis masks per block; transition trials excluded from early/late
            non_trans = (session.lick_times['tr_in_block'] > ops['ignore_first_trials_in_block']).values
            block_masks = {
                'all':   np.ones(len(session.lick_times), dtype=bool),
                'early': (session.lick_times['block'] == 'early').values & non_trans,
                'late':  (session.lick_times['block'] == 'late').values  & non_trans,
            }
            # skip session if 'all' under threshold; drop early/late if either under
            min_licks = move_ops['min_licks']
            if block_masks['all'].sum() < min_licks:
                print(f'      skipping - {block_masks["all"].sum()} licks (< {min_licks})')
                continue
            if (block_masks['early'].sum() < min_licks or
                block_masks['late'].sum() < min_licks):
                print(f'      early/late under {min_licks} licks - "all" only')
                block_masks = {'all': block_masks['all']}

            sp_by_block, mv_by_block = dict(), dict()
            for block, mask in block_masks.items():
                sp_block = {area: {'cids': v['cids'], 'X': v['X'][mask], 't': v['t']}
                            for area, v in lick_aligned_sp.items()}
                mv_block = {sig: {'E': v['E'][mask], 't': v['t']}
                            for sig, v in lick_aligned_mv.items()}
                mv_by_block[block], sp_by_block[block] = _centre_signals(mv_block, sp_block)

            move_dims = calculate_movement_dims_by_area(mv_by_block, sp_by_block, move_ops)

            # save
            save_path = session_dir / 'move_dims.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(move_dims, f)

if __name__ == '__main__':
    fit_movespace_per_session()
