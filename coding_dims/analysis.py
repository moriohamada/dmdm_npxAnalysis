"""coding dimension rotation and motor projection analyses"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.decomposition import PCA

from config import PATHS, ANALYSIS_OPTIONS, CODING_DIM_OPS
from data.load_responses import load_psth
from data.responses import compute_psth
from data.session import Session
from utils.filing import get_session_dirs_by_animal, load_fr_matrix
from utils.smoothing import causal_boxcar


#%% shared utilities

def _get_window_bins(bm_ops):
    dt = ANALYSIS_OPTIONS['pop_bin_width'] # loading downsampled data for memory saving
    return max(1, int(round(bm_ops['sliding_window_ms'] / 1000 / dt)))


def _load_tf_trials_with_values(sess_dir, ops=ANALYSIS_OPTIONS):
    """
    load per-trial TF responses for early-trial conditions and match
    to actual TF values from session.tf_pulses
    """
    psth_path = str(sess_dir / 'psths.h5')
    session = Session.load(str(sess_dir / 'session.pkl'))

    # reconstruct the same filters used during extraction
    tf = session.tf_pulses
    non_trans = tf['tr_in_block'] > ops['ignore_first_trials_in_block']
    early_tr = ((tf['tr_time'] > ops['rmv_time_around_bl']) &
                (tf['tr_time'] < ops['tr_split_time']))
    t_to_event = np.fmin(tf['time_to_lick'], tf['time_to_abort'])
    valid = ((tf['tr_time'] > ops['rmv_time_around_bl']) &
             (t_to_event > ops['rmv_time_around_move']))

    e_block = (tf['block'] == 'early') & non_trans
    l_block = (tf['block'] == 'late') & non_trans
    pos = tf['tf'] > 0

    # early block early trial: pos then neg (same order as extraction)
    eb_pos_mask = pos & e_block & early_tr & valid
    eb_neg_mask = ~pos & e_block & early_tr & valid
    lb_pos_mask = pos & l_block & early_tr & valid
    lb_neg_mask = ~pos & l_block & early_tr & valid

    t_ax = None
    results = {}
    for block, pos_mask, neg_mask, pos_key, neg_key in [
        ('early', eb_pos_mask, eb_neg_mask,
         'earlyBlock_early_pos', 'earlyBlock_early_neg'),
        ('late', lb_pos_mask, lb_neg_mask,
         'lateBlock_early_pos', 'lateBlock_early_neg'),
    ]:
        pos_resp, t_ax = load_psth(psth_path, 'tf', pos_key)  # (nEv, nN, nT)
        neg_resp, _ = load_psth(psth_path, 'tf', neg_key)

        pos_tf_vals = tf.loc[pos_mask, 'tf'].values
        neg_tf_vals = tf.loc[neg_mask, 'tf'].values

        # concatenate pos and neg
        resp = np.concatenate([pos_resp, neg_resp], axis=0)
        tf_vals = np.concatenate([pos_tf_vals, neg_tf_vals])

        results[block] = dict(resp=resp, tf_vals=tf_vals)

    return results, t_ax, session


def _load_lick_trials(sess_dir):
    """load all early-trial lick-aligned responses (hits and FAs, both blocks)"""
    psth_path = str(sess_dir / 'psths.h5')
    try:
        resp, t_ax = load_psth(psth_path, 'lick', '*_early_*')
        if resp.shape[0] == 0:
            return None, None
        return resp, t_ax
    except (KeyError, ValueError):
        return None, None


def _group_trials_by_tf(resp, tf_vals):
    """group per-trial responses by unique TF value"""
    unique_tfs = np.unique(tf_vals)
    grouped = {}
    for tf_val in unique_tfs:
        mask = tf_vals == tf_val
        grouped[tf_val] = resp[mask]
    return grouped, unique_tfs


#%% analysis 2: time-resolved TF coding dimension rotation

def compute_coding_rotation(npx_dir=PATHS['npx_dir_local'],
                            ops=ANALYSIS_OPTIONS,
                            bm_ops=CODING_DIM_OPS,
                            plot_during_extraction=CODING_DIM_OPS['plot_during_extraction']):
    """
    analysis 2: compute time-resolved TF coding direction per block,
    measure rotation between blocks. done per animal (pseudo-population).
    """
    animal_sessions = get_session_dirs_by_animal(npx_dir)
    window_bins = _get_window_bins(bm_ops)
    n_perm = bm_ops['n_permutations']
    all_results = {}
    save_dir = Path(npx_dir) / 'block_modulation'
    save_dir.mkdir(exist_ok=True)

    for animal, sess_dirs in animal_sessions.items():
        print(f'Coding rotation: {animal}')

        # collect data across sessions for pseudo-population
        block_trials = {'early': [], 'late': []}
        block_tf_vals = {'early': [], 'late': []}
        t_ax = None

        for sess_dir in sess_dirs:
            data, sess_t_ax, _ = _load_tf_trials_with_values(sess_dir, ops)
            if t_ax is None:
                t_ax = sess_t_ax

            n_neurons_offset = sum(
                d.shape[1] for d in block_trials['early']) if block_trials['early'] else 0

            for block in ['early', 'late']:
                resp = data[block]['resp']  # (nEv, nN, nT)
                resp = causal_boxcar(resp, window_bins, axis=-1)
                block_trials[block].append(resp)
                block_tf_vals[block].append(data[block]['tf_vals'])

        # for pseudo-population: we need to match trial counts per TF value
        # across sessions. use trial-averaged responses per TF per session,
        # then concatenate neurons.
        n_time = len(t_ax)

        # get common TF values across all sessions
        all_tf_unique = set()
        for block in ['early', 'late']:
            for tf_arr in block_tf_vals[block]:
                all_tf_unique.update(np.unique(tf_arr))
        all_tf_unique = np.sort(list(all_tf_unique))

        # build pseudo-population mean responses per TF per block
        # shape: (n_tf_values, n_neurons_total, n_time)
        pseudo_pop = {}

        for block in ['early', 'late']:
            session_means = []
            for sess_idx, (resp, tf_vals) in enumerate(
                    zip(block_trials[block], block_tf_vals[block])):
                grouped, unique_tfs = _group_trials_by_tf(resp, tf_vals)
                # mean per TF value for this session
                means = np.zeros((len(all_tf_unique), resp.shape[1], n_time))
                for i, tf_val in enumerate(all_tf_unique):
                    if tf_val in grouped:
                        means[i] = np.nanmean(grouped[tf_val], axis=0)
                    else:
                        means[i] = np.nan
                session_means.append(means)

            # concatenate neurons across sessions: (n_tf, total_nN, nT)
            pseudo_pop[block] = np.concatenate(session_means, axis=1)

        # vectorised coding vector regression across all time bins
        # valid mask is constant across time (NaN pattern is per TF x session)
        n_neurons_total = pseudo_pop['early'].shape[1]
        n_tf = len(all_tf_unique)

        def _coding_vectors_vectorised(pop):
            """
            compute coding vector at each time bin via vectorised OLS.
            pop: (n_tf, nN, nT). returns (nT, nN) coding vectors (gain per neuron).
            """
            valid = ~np.any(np.isnan(pop[:, :, 0]), axis=1)  # same for all t
            if valid.sum() < 3:
                return np.full((n_time, n_neurons_total), np.nan)
            tf_v = all_tf_unique[valid]
            tf_c = tf_v - tf_v.mean()
            ss = tf_c @ tf_c
            if ss == 0:
                return np.full((n_time, n_neurons_total), np.nan)
            # pop[valid] is (n_valid, nN, nT) -> reshape to (n_valid, nN*nT)
            resp_v = pop[valid]
            n_valid = resp_v.shape[0]
            resp_flat = resp_v.reshape(n_valid, -1)  # (n_valid, nN*nT)
            gains_flat = (tf_c @ resp_flat) / ss     # (nN*nT,)
            return gains_flat.reshape(n_neurons_total, n_time).T  # (nT, nN)

        coding_vectors = {}
        for block in ['early', 'late']:
            coding_vectors[block] = _coding_vectors_vectorised(pseudo_pop[block])

        # vectorised cosine similarity and angle between blocks
        # coding_vectors[block] is (nT, nN)
        v_e = coding_vectors['early']
        v_l = coding_vectors['late']
        magnitude = {
            'early': np.linalg.norm(v_e, axis=1),
            'late': np.linalg.norm(v_l, axis=1),
        }

        dot_el = np.sum(v_e * v_l, axis=1)
        denom = magnitude['early'] * magnitude['late']
        valid_t = denom > 0
        cosine_sim = np.full(n_time, np.nan)
        cosine_sim[valid_t] = np.clip(dot_el[valid_t] / denom[valid_t], -1, 1)
        angle = np.full(n_time, np.nan)
        angle[valid_t] = np.degrees(np.arccos(cosine_sim[valid_t]))

        # within-block temporal evolution
        within_block_rotation = {}
        for block in ['early', 'late']:
            v = coding_vectors[block]
            v1, v2 = v[:-1], v[1:]
            n1 = np.linalg.norm(v1, axis=1)
            n2 = np.linalg.norm(v2, axis=1)
            ok = (n1 > 0) & (n2 > 0)
            rot = np.full(n_time - 1, np.nan)
            cos = np.clip(np.sum(v1[ok] * v2[ok], axis=1) / (n1[ok] * n2[ok]),
                          -1, 1)
            rot[ok] = np.degrees(np.arccos(cos))
            within_block_rotation[block] = rot

        # null distribution: shuffle block labels, vectorised
        # for each TF value, randomly swap early/late means
        # swap_mask: (n_perm, n_tf) boolean
        swap_mask = np.random.rand(n_perm, n_tf) > 0.5

        pop_e = pseudo_pop['early']  # (n_tf, nN, nT)
        pop_l = pseudo_pop['late']

        null_cosine = np.full((n_perm, n_time), np.nan)
        for p in range(n_perm):
            swap = swap_mask[p]  # (n_tf,) boolean
            shuf_e = pop_e.copy()
            shuf_l = pop_l.copy()
            shuf_e[swap] = pop_l[swap]
            shuf_l[swap] = pop_e[swap]

            sv_e = _coding_vectors_vectorised(shuf_e)  # (nT, nN)
            sv_l = _coding_vectors_vectorised(shuf_l)

            ne = np.linalg.norm(sv_e, axis=1)
            nl = np.linalg.norm(sv_l, axis=1)
            ok = (ne > 0) & (nl > 0)
            null_cosine[p, ok] = np.clip(
                np.sum(sv_e[ok] * sv_l[ok], axis=1) / (ne[ok] * nl[ok]),
                -1, 1)

        # p-value at each time bin (low cosine = more rotation = interesting)
        valid_null = ~np.isnan(null_cosine)
        null_count = valid_null.sum(axis=0)
        cosine_p = np.full(n_time, np.nan)
        has_null = null_count > 0
        cosine_p[has_null] = np.nansum(
            null_cosine[:, has_null] <= cosine_sim[has_null],
            axis=0) / null_count[has_null]

        all_results[animal] = {
            't_ax': t_ax,
            'coding_vectors': coding_vectors,
            'cosine_sim': cosine_sim,
            'angle': angle,
            'magnitude': magnitude,
            'within_block_rotation': within_block_rotation,
            'null_cosine': null_cosine,
            'cosine_p': cosine_p,
            'tf_values': all_tf_unique,
        }

        with open(save_dir / 'coding_rotation.pkl', 'wb') as f:
            pickle.dump(all_results, f)

        # for debugging
        if plot_during_extraction:
            import matplotlib.pyplot as plt
            from coding_dims.plotting import plot_coding_rotation
            plots_dir = str(Path(PATHS['plots_dir']) / 'block_modulation')
            plot_coding_rotation(npx_dir=npx_dir, save_dir=plots_dir)
            plt.close('all')

    print(f'Saved coding rotation results to {save_dir / "coding_rotation.pkl"}')


#%% analysis 3: motor dimension projection

def _generate_fake_lick_times(session, n_fake, ops=ANALYSIS_OPTIONS):
    """
    sample random times during non-lick baseline periods as fake lick events.
    avoids times within 2s of any real lick.
    """
    real_lick_times = session.lick_times['time'].values
    bl_onsets = session.bl_onsets['time'].values
    tr_durs = session.bl_onsets['tr_dur'].values

    # candidate pool: times during trials, at least 2s from any lick
    candidates = []
    for onset, dur in zip(bl_onsets, tr_durs):
        t_start = onset + 2.0  # skip first 2s of trial
        t_end = onset + min(dur, ops['tr_split_time']) - 0.5
        if t_end <= t_start:
            continue
        times = np.arange(t_start, t_end, 0.1)
        # remove times within 2s of any real lick
        for lt in real_lick_times:
            times = times[np.abs(times - lt) > 2.0]
        candidates.append(times)

    if not candidates:
        return np.array([])

    candidates = np.concatenate(candidates)
    if len(candidates) <= n_fake:
        return candidates

    return np.random.choice(candidates, size=n_fake, replace=False)


def compute_motor_projection(npx_dir=PATHS['npx_dir_local'],
                             ops=ANALYSIS_OPTIONS,
                             bm_ops=CODING_DIM_OPS,
                             plot_during_extraction=CODING_DIM_OPS['plot_during_extraction']):
    """
    analysis 3: define motor subspace from lick-aligned activity,
    project TF responses onto it, compare between blocks.
    done per animal (pseudo-population).
    """
    animal_sessions = get_session_dirs_by_animal(npx_dir)
    window_bins = _get_window_bins(bm_ops)
    all_results = {}
    save_dir = Path(npx_dir) / 'block_modulation'
    save_dir.mkdir(exist_ok=True)

    for animal, sess_dirs in animal_sessions.items():
        print(f'Motor projection: {animal}')

        # collect lick and TF data across sessions
        lick_even = []     # even trial lick responses
        lick_odd = []      # odd trial lick responses
        fake_lick_all = []  # fake lick responses
        tf_data = {'early': [], 'late': []}
        lick_t_ax = None
        tf_t_ax = None

        for sess_dir in sess_dirs:
            session = Session.load(str(sess_dir / 'session.pkl'))

            # lick-aligned responses
            lick_resp, lt_ax = _load_lick_trials(sess_dir)
            if lick_resp is None or lick_resp.shape[0] < 4:
                continue
            if lick_t_ax is None:
                lick_t_ax = lt_ax

            lick_resp = causal_boxcar(lick_resp, window_bins, axis=-1)
            even_idx = np.arange(0, lick_resp.shape[0], 2)
            odd_idx = np.arange(1, lick_resp.shape[0], 2)
            lick_even.append(np.nanmean(lick_resp[even_idx], axis=0))  # (nN, nT)
            lick_odd.append(np.nanmean(lick_resp[odd_idx], axis=0))

            # fake licks
            n_real = lick_resp.shape[0]
            n_fake = bm_ops.get('n_fake_licks_per_session') or n_real
            fake_times = _generate_fake_lick_times(session, n_fake, ops)

            if len(fake_times) > 0:
                # extract fake lick responses from FR matrix if available
                fr_path = sess_dir / 'FR_matrix.parquet'
                if fr_path.exists():
                    fr = load_fr_matrix(str(fr_path))
                    resp_win = (-1.5 - ops.get('resp_buffer', 0), 0.5)
                    fake_resp, _ = compute_psth(
                        fr.values, fr.columns.values, fake_times, resp_win)
                    fake_resp = causal_boxcar(fake_resp, window_bins, axis=-1)
                    fake_lick_all.append(np.nanmean(fake_resp, axis=0))
                else:
                    # if no FR matrix locally, use lick responses with shuffled time
                    shuffled = lick_resp[np.random.permutation(lick_resp.shape[0])]
                    fake_lick_all.append(np.nanmean(shuffled, axis=0))

            # TF data
            tf_trial_data, tt_ax, _ = _load_tf_trials_with_values(sess_dir, ops)
            if tf_t_ax is None:
                tf_t_ax = tt_ax
            for block in ['early', 'late']:
                resp = tf_trial_data[block]['resp']
                resp = causal_boxcar(resp, window_bins, axis=-1)
                tf_data[block].append(np.nanmean(resp, axis=0))  # (nN, nT)

        if not lick_even or not tf_data['early']:
            print(f'  Skipping {animal}: insufficient data')
            continue

        # build pseudo-population: concatenate neurons across sessions
        lick_even_pop = np.concatenate(lick_even, axis=0)  # (nN_total, nT)
        lick_odd_pop = np.concatenate(lick_odd, axis=0)
        tf_pop = {}
        for block in ['early', 'late']:
            tf_pop[block] = np.concatenate(tf_data[block], axis=0)

        fake_pop = None
        if fake_lick_all:
            fake_pop = np.concatenate(fake_lick_all, axis=0)

        # step 3a: define motor subspace from lick-aligned trajectory
        # PCA on even-trial lick trajectory: (nT, nN) matrix
        lick_trajectory_even = lick_even_pop.T  # (nT, nN)
        lick_trajectory_odd = lick_odd_pop.T

        # centre
        mean_even = np.nanmean(lick_trajectory_even, axis=0)
        lick_traj_centred = lick_trajectory_even - mean_even

        max_components = min(lick_traj_centred.shape)
        pca = PCA(n_components=max_components)
        pca.fit(lick_traj_centred)

        # cross-validate on odd trials
        lick_traj_odd_centred = lick_trajectory_odd - mean_even
        total_var_odd = np.sum(lick_traj_odd_centred ** 2)

        var_explained_real = np.zeros(max_components)
        for k in range(1, max_components + 1):
            proj = lick_traj_odd_centred @ pca.components_[:k].T
            recon = proj @ pca.components_[:k]
            var_explained_real[k - 1] = 1 - np.sum(
                (lick_traj_odd_centred - recon) ** 2) / total_var_odd

        # null: fake lick variance explained
        var_explained_fake = None
        if fake_pop is not None:
            fake_trajectory = fake_pop.T  # (nT, nN)
            fake_centred = fake_trajectory - mean_even
            total_var_fake = np.sum(fake_centred ** 2)

            var_explained_fake = np.zeros(max_components)
            for k in range(1, max_components + 1):
                proj = fake_centred @ pca.components_[:k].T
                recon = proj @ pca.components_[:k]
                var_explained_fake[k - 1] = 1 - np.sum(
                    (fake_centred - recon) ** 2) / total_var_fake

        # determine N_motor: PCs where real >> fake
        if var_explained_fake is not None:
            improvement = var_explained_real - var_explained_fake
            n_motor = max(1, np.sum(improvement > 0.01))
        else:
            # fallback: use elbow or fixed number
            n_motor = min(5, max_components)

        motor_components = pca.components_[:n_motor]  # (n_motor, nN)

        # step 3b: non-motor TF subspace
        # project TF responses into orthogonal complement of motor subspace
        tf_mean_all = (tf_pop['early'] + tf_pop['late']) / 2  # (nN, nT)
        tf_trajectory = tf_mean_all.T  # (nT, nN)
        tf_centred = tf_trajectory - np.nanmean(tf_trajectory, axis=0)

        # project out motor subspace
        motor_proj = tf_centred @ motor_components.T @ motor_components
        tf_residual = tf_centred - motor_proj

        # PCA on residual
        max_tf_comp = min(tf_residual.shape)
        pca_tf = PCA(n_components=min(10, max_tf_comp))
        pca_tf.fit(tf_residual)
        n_tf_nonmotor = min(n_motor, pca_tf.n_components_)
        nonmotor_components = pca_tf.components_[:n_tf_nonmotor]

        # step 3c: all-to-all projection
        n_tf_time = len(tf_t_ax)
        n_lick_time = len(lick_t_ax)

        heatmaps_motor = {}
        heatmaps_nonmotor = {}

        # normalise lick directions at each time point: (nN, nT_lick)
        lick_norms = np.linalg.norm(lick_even_pop, axis=0, keepdims=True)
        lick_norms[lick_norms == 0] = 1.0
        lick_normed = lick_even_pop / lick_norms

        # normalise non-motor components: (n_tf_nonmotor, nN)
        nm_norms = np.linalg.norm(nonmotor_components, axis=1, keepdims=True)
        nm_norms[nm_norms == 0] = 1.0
        nm_normed = nonmotor_components / nm_norms

        for block in ['early', 'late']:
            tf_block = tf_pop[block].T  # (nT_tf, nN)

            # motor: (nT_tf, nN) @ (nN, nT_lick) -> (nT_tf, nT_lick)
            heatmaps_motor[block] = tf_block @ lick_normed

            # non-motor: (nT_tf, nN) @ (n_tf_nonmotor, nN).T -> (nT_tf, n_tf_nonmotor)
            heatmaps_nonmotor[block] = tf_block @ nm_normed.T

        all_results[animal] = {
            'tf_t_ax': tf_t_ax,
            'lick_t_ax': lick_t_ax,
            'heatmaps_motor': heatmaps_motor,
            'heatmaps_nonmotor': heatmaps_nonmotor,
            'n_motor': n_motor,
            'n_tf_nonmotor': n_tf_nonmotor,
            'var_explained_real': var_explained_real,
            'var_explained_fake': var_explained_fake,
            'motor_components': motor_components,
            'nonmotor_components': nonmotor_components,
        }

        with open(save_dir / 'motor_projection.pkl', 'wb') as f:
            pickle.dump(all_results, f)

        # for debugging
        if plot_during_extraction:
            import matplotlib.pyplot as plt
            from coding_dims.plotting import plot_motor_projection
            plots_dir = str(Path(PATHS['plots_dir']) / 'block_modulation')
            plot_motor_projection(npx_dir=npx_dir, save_dir=plots_dir)
            plt.close('all')

    print(f'Saved motor projection results to {save_dir / "motor_projection.pkl"}')
