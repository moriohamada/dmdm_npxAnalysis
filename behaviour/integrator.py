"""
leaky integrator model for FA behaviour.
simulates a noisy evidence accumulator driven by baseline TF,
fits parameters per subject per block via grid search.
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from itertools import product
from scipy.stats import gaussian_kde
from joblib import Parallel, delayed

from config import BEHAVIOUR_PARAMS


SEARCH_PARAMS = {
    'threshold': np.linspace(.25, 3, 24),
    'gain':      np.array([1.0]),
    'tau':       np.concatenate([[0], np.logspace(-3, np.log10(10), 18), [np.inf]]),
    'sigma':     np.logspace(-3, 1, 20),
}


def clean_df(df, min_rt=BEHAVIOUR_PARAMS['ignore_trial_start'],
             include_non_fa=True):
    """keep FA trials (with RT filter) and optionally non-FA trials"""
    fa_mask = df['IsFA'] & (df['rt_FA'] > min_rt) & (df['rt_FA'] <= 8)

    if include_non_fa:
        non_fa_mask = ~df['IsFA'] & (df['stimT'] > min_rt)
        df_clean = df[fa_mask | non_fa_mask].reset_index(drop=True)
    else:
        df_clean = df[fa_mask].reset_index(drop=True)

    return df_clean


def subsample_df(df, max_trials=10000):
    if len(df) > max_trials:
        print(f'Subsampling {max_trials} from {len(df)} trials')
        df = df.sample(n=max_trials).reset_index(drop=True)
    return df


def get_rt_kernel(df, fast_tfs=(2.0, 4.0)):
    """fit KDE to RT distribution from fast changes"""
    fast_hits = df[df['Stim2TF'].isin(fast_tfs) & df['IsHit']]
    rts = fast_hits['rt_RT'].dropna().values
    if len(rts) < 30:
        raise ValueError(f'Too few fast-change hits to fit RT kernel: n={len(rts)}')
    return gaussian_kde(rts)


def precompute_tf_matrix(df, config=BEHAVIOUR_PARAMS, max_time=8.0):
    """build stimulus matrices for the integrator simulation"""
    frame_step = config.get('tf_sample_step', 3)
    frame_rate = config.get('frame_rate', 60)
    n_trials = len(df)
    max_samples = int(max_time * frame_rate / frame_step)

    stim_end_times = np.where(df['IsFA'].values, df['rt_FA'].values, df['stimT'].values)
    stim_end_times = np.minimum(stim_end_times, max_time)
    stim_frames = np.round(stim_end_times * frame_rate).astype(int)
    stim_samples = np.minimum((stim_frames / frame_step).astype(int), max_samples)

    real_tf_dev = np.full((n_trials, max_samples), np.nan)
    is_synthetic = np.zeros((n_trials, max_samples), dtype=bool)
    trial_lengths = np.full(n_trials, max_samples, dtype=int)

    for i, (_, row) in enumerate(df.iterrows()):
        tf_frames = np.array(row['stim_TF'])
        real = tf_frames[::frame_step]
        n_real = min(len(real), stim_samples[i])
        real_tf_dev[i, :n_real] = real[:n_real]
        is_synthetic[i, n_real:] = True

    return real_tf_dev, is_synthetic, trial_lengths, max_samples


def simulate_behaviour(real_tf_dev, is_synthetic, trial_lengths, max_samples,
                       threshold=1.0, gain=1.0, tau=0.1, sigma=0.0, N=5000,
                       config=BEHAVIOUR_PARAMS):
    """vectorised leaky integrator simulation across all trials"""
    frame_step = config.get('tf_sample_step', 3)
    frame_rate = config.get('frame_rate', 60)
    dt = frame_step / frame_rate
    decay = 0.0 if tau == 0 else (1.0 if np.isinf(tau) else np.exp(-dt / tau))
    n_trials = real_tf_dev.shape[0]

    lick_times = np.full((n_trials, N), np.nan)
    evidence = np.zeros((n_trials, N))
    still_running = np.ones((n_trials, N), dtype=bool)

    min_time = config.get('ignore_trial_start', 2.0)
    min_sample = int(min_time / dt)

    for t in range(max_samples):
        if not np.any(still_running):
            break

        if t < min_sample:
            evidence[:] = 0.0
            continue

        within_trial = (t < trial_lengths)
        active = still_running & within_trial[:, None]
        if not np.any(active):
            break

        real_input = np.where(is_synthetic[:, t], 0.0,
                              np.nan_to_num(real_tf_dev[:, t]))[:, None]

        internal = np.random.normal(0, sigma, (n_trials, N)) if sigma > 0 else 0.0

        evidence = evidence * decay + gain * real_input + internal

        crossed = active & (evidence >= threshold)
        lick_times[crossed] = t * dt
        still_running[crossed] = False

    time_bins = np.arange(0, max_samples * dt, dt)
    bin_edges = np.append(time_bins, max_samples * dt)
    lick_dist = np.zeros((n_trials, len(time_bins)))

    for i in range(n_trials):
        licked = lick_times[i][~np.isnan(lick_times[i])]
        if len(licked) > 0:
            counts, _ = np.histogram(licked, bins=bin_edges)
            lick_dist[i] = counts / N

    return lick_dist, time_bins, N


def calculate_likelihood(df, lick_dist, time_bins, rt_kernel, epsilon=1e-6):
    """log likelihood of observed FA behaviour given simulated lick distributions"""
    log_lik = 0.0
    dt = time_bins[1] - time_bins[0]

    rt_probs = rt_kernel(np.arange(0, 2.0, dt))
    rt_probs /= rt_probs.sum()

    for trial_idx, (_, row) in enumerate(df.iterrows()):
        dist = lick_dist[trial_idx]
        p_licked = dist.sum()

        if row['IsFA']:
            t_real = row['rt_FA']
            if p_licked == 0:
                log_lik += np.log(epsilon)
            else:
                convolved = np.convolve(dist, rt_probs, mode='full')[:len(time_bins)]
                t_idx = np.clip(np.searchsorted(time_bins, t_real), 0, len(time_bins) - 1)
                p = convolved[t_idx] / dt
                log_lik += np.log(p + epsilon)
        else:
            p_no_lick = 1.0 - p_licked
            log_lik += np.log(p_no_lick + epsilon)

    return log_lik / len(df)


def grid_search_params(df, rt_kernel, params=SEARCH_PARAMS, n_jobs=-1, verbose=True):
    param_keys = list(params.keys())
    param_combos = list(product(*params.values()))

    real_tf_dev, is_synthetic, trial_lengths, max_samples = precompute_tf_matrix(df)

    if verbose:
        print(f'Running grid search over {len(param_combos)} parameter combinations...')

    def _run(vals):
        combo = dict(zip(param_keys, vals))
        lick_dist, time_bins, n_runs = simulate_behaviour(
            real_tf_dev, is_synthetic, trial_lengths, max_samples, **combo)
        lik = calculate_likelihood(df, lick_dist, time_bins, rt_kernel)
        return lik, lick_dist, time_bins, combo, n_runs

    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(_run)(vals) for vals in param_combos)

    likelihoods = np.array([r[0] for r in results])
    best_idx = np.argmax(likelihoods)

    all_params = [dict(zip(param_keys, vals)) for vals in param_combos]
    best_lick_dist = results[best_idx][1]
    best_time_bins = results[best_idx][2]
    best_params = results[best_idx][3]
    n_runs = results[best_idx][4]

    if verbose:
        print(f'Best params: {best_params} (ll={likelihoods[best_idx]:.3f})')

    return all_params, likelihoods, best_lick_dist, best_time_bins, best_params, n_runs


def model_integrator_by_subj(dfs, search_params=SEARCH_PARAMS,
                             save_path=None, overwrite=False):
    """run grid search per subject per block"""
    if save_path is not None and Path(save_path).exists() and not overwrite:
        print(f'Loading simulation results from {save_path}')
        return load_results(save_path)

    params = {'early': {}, 'late': {}}
    likelihoods = {'early': {}, 'late': {}}
    lick_dists = {'early': {}, 'late': {}}
    time_bins = {'early': {}, 'late': {}}
    best_params = {'early': {}, 'late': {}}
    n_runs = {'early': {}, 'late': {}}

    for subj, df in dfs.items():
        df_clean = clean_df(df)
        rt_kernel = get_rt_kernel(df)

        for block in ['early', 'late']:
            df_block = df_clean[df_clean['hazardblock'] == block].reset_index(drop=True)

            fa_trials = df_block[df_block['IsFA']]
            non_fa_trials = df_block[~df_block['IsFA']]
            if len(non_fa_trials) > len(fa_trials) * 3:
                non_fa_trials = non_fa_trials.sample(
                    n=len(fa_trials) * 3, random_state=42)
            df_block = pd.concat([fa_trials, non_fa_trials]).reset_index(drop=True)
            df_block = subsample_df(df_block, max_trials=2000)

            print(f'{subj} | {block} n_trials={len(df_block)}')
            (params[block][subj],
             likelihoods[block][subj],
             lick_dists[block][subj],
             time_bins[block][subj],
             best_params[block][subj],
             n_runs[block][subj]) = grid_search_params(df_block, rt_kernel, search_params)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        save_results(save_path, params, likelihoods, lick_dists,
                     time_bins, best_params, n_runs)

    return params, likelihoods, lick_dists, time_bins, best_params, n_runs


def save_results(path, params, likelihoods, lick_dists,
                 time_bins, best_params, n_runs):
    with open(path, 'wb') as f:
        pickle.dump({
            'params': params, 'likelihoods': likelihoods,
            'lick_dists': lick_dists, 'time_bins': time_bins,
            'best_params': best_params, 'n_runs': n_runs,
        }, f)
    print(f'Results saved to {path}')


def load_results(path):
    with open(path, 'rb') as f:
        results = pickle.load(f)
    return (results['params'], results['likelihoods'], results['lick_dists'],
            results['time_bins'], results['best_params'], results['n_runs'])
