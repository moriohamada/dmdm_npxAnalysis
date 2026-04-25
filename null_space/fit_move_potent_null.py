"""
Get averaged lick-aligned activity (by area, by session inc orofacial nuclei) to compute
movement potent dimension per area.
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
    Get lick-aligned movement signals (mouth/whisker motion energy, pupil area,
    running wheel speed), resampled onto the fr_matrix time axis.
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

def calculate_movement_dims_by_area(lick_aligned_mv: dict,
                                    lick_aligned_sp: dict,
                                    move_ops: dict = MOVEDIM_OPTIONS):
    """
    regress motion energy versus neural activity by area (at different delays),
    to identify movement-potent and -null dimensions.
    Returns: move_dims, dict with k: v:
        w_pot: n_pot_dims x nN
        w_null: n_null_dims x nN
    """

    # PCA of motor
    mv_stack = np.stack([v['E_z'] for _, v in lick_aligned_mv.items()], axis=1)
    mv_pca = PCA(n_components=2) # basiaclly seemed two dimensional - face motE, wheel spd
    mv_pca.fit(mv_stack)
    y = mv_pca.components_ @ mv_stack.T # target for prediction

    move_dims = dict()
    for area in lick_aligned_sp.keys():

        X_fulldim = lick_aligned_sp[area]['X_centred']
        t = lick_aligned_sp[area]['t']
        if X_fulldim.shape[0] < 10:
            continue
        move_dims[area] = dict()

        sp_pca = PCA(n_components=6)
        sp_pca.fit(X_fulldim.T)
        X = sp_pca.transform(X_fulldim.T)

        # linear regression to get w_pot
        pot_mdl = LinearRegression(fit_intercept=True)
        pot_mdl.fit(X, y.T)
        # w_null
        null_pc = null_space(pot_mdl.coef_)

        # rotate w_null to capture pre-lick
        rot_null_pc = PCA()
        pre_lick_t = time_mask(t, move_ops['prelick_period'])
        rot_null_pc.fit((null_pc.T @ X[pre_lick_t,:].T).T)

        # compute full weighting to go from neurons to movement_potent/-null spaces
        move_dims[area]['pot'] = pot_mdl.coef_ @ sp_pca.components_
        move_dims[area]['null']= rot_null_pc.components_ @ null_pc.T @ sp_pca.components_

    return move_dims

def fit_movespace_per_session(npx_dir: str = PATHS['npx_dir_local'],
                              ops: dict = ANALYSIS_OPTIONS,
                              move_ops: dict = MOVEDIM_OPTIONS,
                              areas: list[str] = AREA_NAMES,
                              ):
    """
    Runner to loop through all sessions, extract lick-aligned neural activity and
    motion energy, calculate movement-potent dim.
    """
    animal_sessions = get_session_dirs_by_animal(npx_dir)

    for animal in animal_sessions.keys():
        print(f'Fitting movement space for {animal}')
        sessions = animal_sessions[animal]
        for session_dir in sessions:
            print(f'    {session_dir}')

            session = Session.load(session_dir/'session.pkl')
            # load FR matrix
            fr = load_fr_matrix(str(session_dir / 'FR_matrix_ds.parquet'))
            session.fr_matrix = fr
            # eye-cam timestamps from ceph -
            # TODO: change intiial preprocessing to have these locally! no point having
            #  movement signals without time ax...
            ceph_sess = Path(PATHS['ceph_dir']) / session.animal / session.name
            eye_cam_times = pd.read_csv(ceph_sess / 'daq_Eye_cam.csv')['rise_t'].values

            # create [area](fr) key for lick-aligned movement
            lick_aligned_sp  = get_lick_activity_by_area(session, areas, move_ops)
            lick_aligned_mv = get_lick_aligned_motion(session, eye_cam_times, move_ops)
            lick_aligned_mv, lick_aligned_sp = _centre_signals(lick_aligned_mv, lick_aligned_sp)
            move_dims = calculate_movement_dims_by_area(lick_aligned_mv, lick_aligned_sp, move_ops)

            # save
            save_path = session_dir / 'move_dims.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(move_dims, f)

if __name__ == '__main__':
    fit_movespace_per_session()