"""
behavioural analysis functions: psychometrics, lick-triggered stimulus,
hazard rates, pulse-aligned lick probability
"""
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.ndimage import uniform_filter1d

from config import PATHS, ANALYSIS_OPTIONS

BEHAVIOUR_DATA_DIR = os.path.join(PATHS['npx_dir_local'], 'behaviour')


def build_dfs_from_sessions(npx_dir, config=ANALYSIS_OPTIONS):
    """
    build {subject: trials_df} dict from npx session files.
    maps column names to match what the analysis functions expect:
      TF -> stim_TF, tag -> stim_tag, IsProbe -> isProbe
    adds sessionID column.
    """
    import os
    from data.session import Session

    dfs_by_subj = {}
    for subj in sorted(os.listdir(npx_dir)):
        subj_dir = os.path.join(npx_dir, subj)
        if not os.path.isdir(subj_dir):
            continue

        sess_dfs = []
        for sess_idx, sess_name in enumerate(sorted(os.listdir(subj_dir))):
            pkl_path = os.path.join(subj_dir, sess_name, 'session.pkl')
            if not os.path.exists(pkl_path):
                continue
            sess = Session.load(pkl_path)
            df = sess.trials.copy()
            df['sessionID'] = sess_idx + 1
            sess_dfs.append(df)

        if sess_dfs:
            combined = pd.concat(sess_dfs, ignore_index=True)
            # column aliases
            if 'TF' in combined.columns and 'stim_TF' not in combined.columns:
                combined['stim_TF'] = combined['TF']
            if 'tag' in combined.columns and 'stim_tag' not in combined.columns:
                combined['stim_tag'] = combined['tag']
            if 'IsProbe' in combined.columns and 'isProbe' not in combined.columns:
                combined['isProbe'] = combined['IsProbe']
            dfs_by_subj[subj] = combined

    return dfs_by_subj


def filter_sessions(dfs, config=ANALYSIS_OPTIONS):
    """remove early sessions and sessions with too few hits"""
    df_filtered = dfs.copy()
    for subj in df_filtered:
        df = df_filtered[subj]
        rmv_idx = []
        for session in df['sessionID'].unique():
            sess_mask = df['sessionID'] == session
            if session < config.get('ignore_first_sessions', 0):
                rmv_idx.extend(df[sess_mask].index)
            if df.loc[sess_mask, 'IsHit'].sum() < config['min_hits_in_session']:
                rmv_idx.extend(df[sess_mask].index)
        rmv_idx.extend(
            df[df['tr_in_block'] < config['ignore_first_trials_in_block']].index)
        df_filtered[subj] = df.drop(set(rmv_idx))
    return df_filtered


def _strip_and_convert_tf(tf_raw):
    """strip grey-screen zeros from raw TF array and convert to log2 octaves"""
    tf_raw = np.array(tf_raw)
    tf_stim = tf_raw[tf_raw > 0]
    return np.log2(tf_stim)


#%% psychometrics

def extract_psychometric(dfs, config=ANALYSIS_OPTIONS):
    """extract hit rate and RTs for changes across conditions"""
    change_tfs = config['change_tfs']
    n_subj, n_chs = len(dfs), len(change_tfs)
    shape = (n_subj, n_chs, 2, 2)  # subj x change x block x probe

    psycho = np.full(shape, np.nan)
    chrono = np.full(shape, np.nan)

    blocks = ['early', 'late']
    probes = [False, True]

    for subj_id, subj in enumerate(dfs):
        df = dfs[subj]
        for ch_id, ch in enumerate(change_tfs):
            for block_id, block in enumerate(blocks):
                for probe_id, probe in enumerate(probes):
                    mask = (
                        (df['Stim2TF'] == ch) &
                        (df['hazardblock'] == block) &
                        (df['isProbe'] == probe)
                    )
                    hits = mask & df['IsHit']
                    miss = mask & df['IsMiss']

                    total = hits.sum() + miss.sum()
                    if total > 0:
                        psycho[subj_id, ch_id, block_id, probe_id] = hits.sum() / total
                    chrono[subj_id, ch_id, block_id, probe_id] = df.loc[hits, 'rt_RT'].mean()

    return psycho, chrono


#%% lick-triggered stimulus

def extract_perilick_info(dfs, config=ANALYSIS_OPTIONS):
    """extract peri-lick TF sequences for each trial across all subjects"""
    n_samples = config.get('n_pre_lick_samples', 40)
    frame_step = config.get('tf_sample_step', 3)
    frame_rate = config.get('frame_rate', 60)
    smooth_size = config['smooth_tf']
    smooth_origin = int(-(smooth_size - 1) / 2)

    all_lick_info = {}

    for subj in dfs:
        df = dfs[subj]
        prev_lick = np.nan
        prev_outcome = ''

        records = []
        for _, row in df.iterrows():
            if row['IsHit']:
                lick_time = row['stimT'] + row['rt_RT']
                lick_type = 'hit'
            elif row['IsFA']:
                lick_time = row['rt_FA']
                lick_type = 'fa'
            else:
                continue

            tf_oct = _strip_and_convert_tf(row['stim_TF'])
            tf_smooth = uniform_filter1d(tf_oct, size=smooth_size,
                                         origin=smooth_origin)

            if not np.isnan(lick_time):
                lick_frame = round(lick_time * frame_rate) - 1
                all_sample_frames = np.arange(0, len(tf_smooth), frame_step)
                valid = all_sample_frames[all_sample_frames <= lick_frame]
                first = max(0, len(valid) - n_samples)
                stim_before = tf_smooth[valid[first:]]
                pad = n_samples - len(stim_before)
                if pad > 0:
                    stim_before = np.concatenate([np.full(pad, np.nan), stim_before])
            else:
                stim_before = np.full(n_samples, np.nan)

            records.append({
                'lickTime': lick_time,
                'lickType': lick_type,
                'stimBefore': stim_before,
                'hazardblock': row['hazardblock'],
                'trialExpect': 'expected' if not row['isProbe'] else 'unexpected',
                'prevOutcome': prev_outcome,
                'prevLickTime': prev_lick,
            })

            prev_outcome = row['trialoutcome']
            prev_lick = lick_time

        all_lick_info[subj] = pd.DataFrame(records)

    return all_lick_info


def extract_elts(lick_triggered_data, config=ANALYSIS_OPTIONS):
    """extract early-lick-triggered TF sequences for 3 conditions"""
    t_early = config['ignore_trial_start']
    t_split = config['tr_split_time']

    cond_names = ['earlyBlock_early', 'lateBlock_early', 'lateBlock_late']
    lts = {cond: {} for cond in cond_names}
    n_samples = config.get('n_pre_lick_samples', 40)

    for subj, df in lick_triggered_data.items():
        if df.empty:
            for cond in cond_names:
                lts[cond][subj] = np.full((0, n_samples), np.nan)
            continue

        is_fa = df['lickType'] == 'fa'
        is_early_block = df['hazardblock'] == 'early'
        is_late_block = df['hazardblock'] == 'late'
        early_in_trial = (df['lickTime'] > t_early) & (df['lickTime'] <= t_split)
        late_in_trial = df['lickTime'] > t_split

        masks = {
            'earlyBlock_early': is_fa & is_early_block & early_in_trial,
            'lateBlock_early':  is_fa & is_late_block & early_in_trial,
            'lateBlock_late':   is_fa & is_late_block & late_in_trial,
        }

        for cond, mask in masks.items():
            trials = df.loc[mask, 'stimBefore'].values
            if len(trials) > 0:
                lts[cond][subj] = np.stack(trials)
            else:
                lts[cond][subj] = np.full((0, n_samples), np.nan)
    return lts


def calculate_elta(lts, config=ANALYSIS_OPTIONS):
    """average early-lick-triggered TF across subjects for each condition"""
    elta = {}
    for cond, subj_data in lts.items():
        subj_means = []
        for subj, trials in subj_data.items():
            if len(trials) > 0:
                subj_means.append(np.nanmean(trials, axis=0))

        subj_means = np.stack(subj_means)
        elta[cond] = {
            'mean': np.nanmean(subj_means, axis=0),
            'sem': np.nanstd(subj_means, axis=0) / np.sqrt(len(subj_means)),
            'subj_means': subj_means,
        }
    return elta


def parallel_analysis(data, actual_ev, config=ANALYSIS_OPTIONS, n_iter=100):
    """parallel analysis using synthetic stimuli matching the generative process"""
    n_trials, n_features = data.shape
    smooth_size = config['smooth_tf']
    smooth_origin = int(-(smooth_size - 1) / 2)

    random_evs = np.zeros((n_iter, min(n_trials, n_features)))
    for i in range(n_iter):
        raw = np.random.normal(loc=0, scale=0.25, size=(n_trials, n_features * 3))
        for row in range(n_trials):
            raw[row] = uniform_filter1d(raw[row], size=smooth_size, origin=smooth_origin)
        raw = raw[:, ::3]
        pca_rand = PCA()
        pca_rand.fit(raw)
        random_evs[i] = pca_rand.explained_variance_ratio_

    random_ev_thresh = np.percentile(random_evs, 100 * (1 - config['sig_thresh']), axis=0)
    n_sig = int(np.sum(actual_ev > random_ev_thresh))

    return n_sig, random_ev_thresh


def calculate_eltc(lts, config=ANALYSIS_OPTIONS):
    """run PCA per subject on lick-triggered stimuli for each condition"""
    eltc = {}
    for cond, subj_data in lts.items():
        eltc[cond] = {}
        for subj, tf in subj_data.items():
            tf = tf[:, 20:]

            valid = ~np.any(np.isnan(tf), axis=1)
            n_invalid = np.sum(~valid)
            if n_invalid > 0:
                print(f'{cond} | {subj}: dropped {n_invalid}/{len(tf)} trials with NaNs')
            tf_clean = tf[valid]

            if len(tf_clean) < 2:
                continue

            pca = PCA()
            scores = pca.fit_transform(tf_clean)
            n_sig, random_ev_thresh = parallel_analysis(
                tf_clean, pca.explained_variance_ratio_, config)

            eltc[cond][subj] = {
                'components': pca.components_,
                'explained_var': pca.explained_variance_,
                'explained_var_ratio': pca.explained_variance_ratio_,
                'scores': scores,
                'mean': pca.mean_,
                'n_sig': n_sig,
                'parallel_ev_thresh': random_ev_thresh,
            }
    return eltc


def extract_baseline_projections(dfs, eltc, config=ANALYSIS_OPTIONS,
                                 n_components=3, lookahead=1.0, stepsize=5):
    """slide 1s windows across pre-change baseline, project onto top PCs"""
    from numpy.lib.stride_tricks import sliding_window_view

    n_samples = 20
    sample_rate = config.get('tf_sample_rate', 20)
    frame_step = config.get('tf_sample_step', 3)
    frame_rate = config.get('frame_rate', 60)
    ignore_start = int(config.get('ignore_trial_start', 2) * sample_rate)
    smooth_size = config['smooth_tf']
    smooth_origin = int(-(smooth_size - 1) / 2)

    projections = {}
    for cond, subj_data in eltc.items():
        projections[cond] = {}
        for subj, pca_data in subj_data.items():
            components = pca_data['components'][:n_components]
            df = dfs[subj]
            all_scores = []
            all_licked = []

            for _, row in df.iterrows():
                tf_oct = _strip_and_convert_tf(row['stim_TF'])
                tf = uniform_filter1d(tf_oct, size=smooth_size,
                                      origin=smooth_origin)
                sampled = tf[::frame_step]

                change_sample = min(int(row['stimT'] * sample_rate), len(sampled))
                max_start = change_sample - n_samples

                if max_start < ignore_start:
                    continue

                windows = sliding_window_view(sampled, n_samples)
                starts = np.arange(ignore_start, max_start + 1, stepsize)
                starts = starts[starts < len(windows)]
                if len(starts) == 0:
                    continue

                segments = windows[starts]
                scores = segments @ components.T
                all_scores.append(scores)

                lick_time = row['rt_FA'] if row['IsFA'] else None
                window_end_times = (starts + n_samples) / sample_rate

                if lick_time is not None:
                    deltas = lick_time - window_end_times
                    licked = (deltas > 0) & (deltas <= lookahead)
                else:
                    licked = np.zeros(len(starts), dtype=bool)
                all_licked.append(licked)

            if all_scores:
                projections[cond][subj] = {
                    'scores': np.concatenate(all_scores),
                    'lick_followed': np.concatenate(all_licked),
                }

    return projections


def align_eltc(eltc, projections):
    """flip PC signs so lick-followed windows score higher than no-lick windows"""
    eltc_aligned = {}
    for cond, subj_data in eltc.items():
        eltc_aligned[cond] = {}
        for subj, data in subj_data.items():
            if subj not in projections.get(cond, {}):
                eltc_aligned[cond][subj] = data
                continue

            proj = projections[cond][subj]
            licked = proj['lick_followed']
            scores_proj = proj['scores']

            components = data['components'].copy()
            scores = data['scores'].copy()

            for i in range(components.shape[0]):
                if i >= scores_proj.shape[1]:
                    break
                mean_lick = np.mean(scores_proj[licked, i]) if np.any(licked) else 0
                mean_no = np.mean(scores_proj[~licked, i]) if np.any(~licked) else 0
                if mean_lick < mean_no:
                    components[i] *= -1
                    scores[:, i] *= -1

            eltc_aligned[cond][subj] = {
                **data,
                'components': components,
                'scores': scores,
            }

    return eltc_aligned


#%% hazard rates

def calculate_el_hazard(dfs, config=ANALYSIS_OPTIONS):
    """FA hazard rate over baseline period, by block"""
    bin_size = config.get('hazard_bin_size', 0.5)
    bin_step = config.get('hazard_bin_step', 0.1)
    half_bin = bin_size / 2

    results = {}
    for subj, df in dfs.items():
        max_stim_t = config.get('hazard_max_time', 15.5)
        bin_centres = np.arange(half_bin, max_stim_t - half_bin + bin_step, bin_step)
        n_bins = len(bin_centres)

        early_licks = np.zeros(n_bins)
        early_at_risk = np.zeros(n_bins)
        late_licks = np.zeros(n_bins)
        late_at_risk = np.zeros(n_bins)

        for _, row in df.iterrows():
            lick_time = row['rt_FA'] if row['IsFA'] else np.inf
            cens_time = min(lick_time, row['stimT'])
            is_early = row['hazardblock'] == 'early'

            licks_arr = early_licks if is_early else late_licks
            at_risk_arr = early_at_risk if is_early else late_at_risk

            for b, bc in enumerate(bin_centres):
                bin_start = bc - half_bin
                if bin_start >= cens_time:
                    break
                at_risk_arr[b] += 1
                if bin_start <= lick_time < bc + half_bin:
                    licks_arr[b] += 1

        results[subj] = {
            'binCentres': bin_centres,
            'earlyBlock': np.where(early_at_risk > 0,
                                   early_licks / early_at_risk, np.nan),
            'lateBlock': np.where(late_at_risk > 0,
                                  late_licks / late_at_risk, np.nan),
            'early_n': early_at_risk,
            'late_n': late_at_risk,
        }

    return results


#%% pulse-aligned lick probability

def _extract_tf_pulses(dfs, config=ANALYSIS_OPTIONS):
    frame_step = config.get('tf_sample_step', 3)
    frame_rate = config.get('frame_rate', 60)
    sample_rate = 1 / (frame_step / frame_rate)

    all_dfs = []

    for subj, df in dfs.items():
        if df.empty:
            continue
        tf_subsampled = [_strip_and_convert_tf(tf)[::frame_step]
                         for tf in df['stim_TF'].values]
        if len(tf_subsampled) == 0:
            continue
        max_len = max(len(t) for t in tf_subsampled)
        tf_mat = np.full((len(df), max_len), np.nan)
        for i, t in enumerate(tf_subsampled):
            tf_mat[i, :len(t)] = t

        tf_dev = tf_mat
        stim_times = np.arange(max_len) / sample_rate

        lick_times = np.where(df['IsFA'].values, df['rt_FA'].values, np.inf)
        cens_times = np.minimum(lick_times, df['stimT'].values)

        valid_mask = (stim_times[None, :] < cens_times[:, None]) & ~np.isnan(tf_dev)
        tf_dev_next = np.full_like(tf_dev, np.nan)
        tf_dev_next[:, :-1] = tf_dev[:, 1:]

        rows, cols = np.where(valid_mask)
        all_dfs.append(pd.DataFrame({
            'subj': subj,
            'block': df['hazardblock'].values[rows],
            'session': df['sessionID'].values[rows],
            'stim_time': stim_times[cols],
            'tf_dev': tf_dev[rows, cols],
            'tf_dev_next': tf_dev_next[rows, cols],
            'lick_time': np.where(np.isfinite(lick_times[rows]),
                                  lick_times[rows], np.nan),
            'cens_time': cens_times[rows],
        }))

        del tf_mat, tf_dev, tf_dev_next, valid_mask

    return pd.concat(all_dfs, ignore_index=True)


def _binomial_ci(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo, hi


def calculate_pulse_lick_prob(dfs, config=ANALYSIS_OPTIONS):
    """for each TF stimulus, ask: did a lick follow within the lick window?"""
    bin_centres = np.array(config.get('tf_pulse_bin_centres',
                                      np.arange(-0.5, 0.55, 0.1)))
    half_width = config.get('tf_pulse_bin_width', 0.2) / 2
    lick_win = config.get('tf_pulse_lick_win', [0.25, 1.0])
    time_split = config.get('tr_split_time', 8.0)
    n_bins = len(bin_centres)

    stim_df = _extract_tf_pulses(dfs, config)

    stim_df['licked'] = (
        stim_df['lick_time'].notna() &
        (stim_df['lick_time'] >= stim_df['stim_time'] + lick_win[0]) &
        (stim_df['lick_time'] < stim_df['stim_time'] + lick_win[1])
    )
    stim_df['early_in_trial'] = stim_df['stim_time'] < time_split

    conditions = {
        'earlyBlock_earlyTrial': (stim_df['block'] == 'early') & stim_df['early_in_trial'],
        'earlyBlock_lateTrial': (stim_df['block'] == 'early') & ~stim_df['early_in_trial'],
        'lateBlock_earlyTrial': (stim_df['block'] == 'late') & stim_df['early_in_trial'],
        'lateBlock_lateTrial': (stim_df['block'] == 'late') & ~stim_df['early_in_trial'],
    }

    results = {}
    for subj in stim_df['subj'].unique():
        subj_df = stim_df[stim_df['subj'] == subj]
        results[subj] = {'binCentres': bin_centres}

        for cond, mask in conditions.items():
            cond_df = subj_df[mask]
            pairs_df = cond_df[cond_df['tf_dev_next'].notna()]

            lick_prob = np.full(n_bins, np.nan)
            lick_prob_ci = np.full((n_bins, 2), np.nan)
            n_stim = np.zeros(n_bins, dtype=int)

            for b, bc in enumerate(bin_centres):
                in_bin = ((cond_df['tf_dev'] >= bc - half_width) &
                          (cond_df['tf_dev'] < bc + half_width))
                n = in_bin.sum()
                k = cond_df.loc[in_bin, 'licked'].sum()
                n_stim[b] = n
                if n > 0:
                    lick_prob[b] = k / n
                    lick_prob_ci[b] = _binomial_ci(k, n)

            lick_prob_2d = np.full((n_bins, n_bins), np.nan)
            n_2d = np.zeros((n_bins, n_bins), dtype=int)

            for b1, bc1 in enumerate(bin_centres):
                in_bin1 = ((pairs_df['tf_dev'] >= bc1 - half_width) &
                           (pairs_df['tf_dev'] < bc1 + half_width))
                for b2, bc2 in enumerate(bin_centres):
                    in_bin2 = ((pairs_df['tf_dev_next'] >= bc2 - half_width) &
                               (pairs_df['tf_dev_next'] < bc2 + half_width))
                    both = in_bin1 & in_bin2
                    n = both.sum()
                    n_2d[b1, b2] = n
                    if n > 0:
                        lick_prob_2d[b1, b2] = pairs_df.loc[both, 'licked'].mean()

            results[subj][cond] = {
                'lickProb': lick_prob,
                'lickProb_ci': lick_prob_ci,
                'n': n_stim,
                'lickProb2D': lick_prob_2d,
                'n2D': n_2d,
            }

    return results


#%% save / load

def _save(obj, name, data_dir=BEHAVIOUR_DATA_DIR):
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(data_dir, f'{name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'Saved {name} to {path}')


def _load(name, data_dir=BEHAVIOUR_DATA_DIR):
    path = os.path.join(data_dir, f'{name}.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


#%% run all

def extract_all_behavioural(npx_dir=PATHS['npx_dir_local'],
                            config=ANALYSIS_OPTIONS,
                            overwrite=False):
    """extract and save all behavioural analyses"""
    data_dir = os.path.join(npx_dir, 'behaviour')

    def _load_or_compute(name):
        """check if cached result exists; return it or None"""
        path = os.path.join(data_dir, f'{name}.pkl')
        if not overwrite and os.path.exists(path):
            print(f'Loading cached {name}')
            return _load(name, data_dir)
        return None

    dfs = _load_or_compute('dfs_processed')
    if dfs is None:
        dfs = filter_sessions(build_dfs_from_sessions(npx_dir, config), config)
        _save(dfs, 'dfs_processed', data_dir)

    psycho_chrono = _load_or_compute('psychometric')
    if psycho_chrono is None:
        psycho_chrono = extract_psychometric(dfs, config)
        _save(psycho_chrono, 'psychometric', data_dir)

    hazard = _load_or_compute('hazard_rates')
    if hazard is None:
        hazard = calculate_el_hazard(dfs, config)
        _save(hazard, 'hazard_rates', data_dir)

    lick_triggered = _load_or_compute('lick_triggered')
    if lick_triggered is None:
        lick_triggered = extract_perilick_info(dfs, config)
        _save(lick_triggered, 'lick_triggered', data_dir)

    lts = _load_or_compute('elts')
    if lts is None:
        lts = extract_elts(lick_triggered, config)
        _save(lts, 'elts', data_dir)

    elta = _load_or_compute('elta')
    if elta is None:
        elta = calculate_elta(lts, config)
        _save(elta, 'elta', data_dir)

    eltc = _load_or_compute('eltc')
    if eltc is None:
        eltc = calculate_eltc(lts, config)
        _save(eltc, 'eltc', data_dir)

    projections = _load_or_compute('eltc_projections')
    if projections is None:
        projections = extract_baseline_projections(dfs, eltc, config)
        _save(projections, 'eltc_projections', data_dir)

    eltc_aligned = _load_or_compute('eltc_aligned')
    if eltc_aligned is None:
        eltc_aligned = align_eltc(eltc, projections)
        _save(eltc_aligned, 'eltc_aligned', data_dir)

    pulse_lick = _load_or_compute('pulse_lick_prob')
    if pulse_lick is None:
        pulse_lick = calculate_pulse_lick_prob(dfs, config)
        _save(pulse_lick, 'pulse_lick_prob', data_dir)

    print('All behavioural analyses extracted')


def load_behavioural(name, npx_dir=PATHS['npx_dir_local']):
    """load a saved behavioural analysis result by name"""
    return _load(name, os.path.join(npx_dir, 'behaviour'))
