"""
combine multiple stochastic runs of the leaky integrator grid search.
averages per-combo log-likelihoods across runs.
"""
import pickle
import numpy as np
from pathlib import Path


def load_runs(results_dir, subj, block):
    results_dir = Path(results_dir)
    paths = sorted(results_dir.glob(f'leaky_int_{subj}_{block}_run*.pkl'))
    if not paths:
        raise FileNotFoundError(f'No run files found for {subj} {block}')

    runs = []
    for path in paths:
        with open(path, 'rb') as f:
            runs.append(pickle.load(f))
    return runs


def combine(results_dir, subj, block):
    runs = load_runs(results_dir, subj, block)

    for r in runs[1:]:
        if r['params'] != runs[0]['params']:
            raise ValueError(f'Param grids differ across runs for {subj} {block}!')

    stacked = np.stack([r['likelihoods'] for r in runs])
    mean_lls = stacked.mean(axis=0)
    sem_lls = stacked.std(axis=0) / np.sqrt(len(runs))

    best_idx = np.argmax(mean_lls)
    best_params = runs[0]['params'][best_idx]
    best_run = int(np.argmin(np.abs(stacked[:, best_idx] - mean_lls[best_idx])))

    print(f'{subj} | {block} | {len(runs)} runs combined '
          f'| best: {best_params}  (ll = {mean_lls[best_idx]:.3f} '
          f'+/- {sem_lls[best_idx]:.3f})')

    return {
        'params': runs[0]['params'],
        'likelihoods': mean_lls,
        'likelihoods_sem': sem_lls,
        'n_runs': len(runs),
        'best_params': best_params,
        'lick_dist': runs[best_run]['lick_dist'],
        'time_bins': runs[best_run]['time_bins'],
    }


def collate_all(results_dir, subjects, blocks=('early', 'late'), save_path=None):
    """combine all subjects and blocks, optionally save"""
    params = {'early': {}, 'late': {}}
    likelihoods = {'early': {}, 'late': {}}
    lick_dists = {'early': {}, 'late': {}}
    time_bins = {'early': {}, 'late': {}}
    best_params = {'early': {}, 'late': {}}

    for subj in subjects:
        for block in blocks:
            try:
                result = combine(results_dir, subj, block)
                params[block][subj] = result['params']
                likelihoods[block][subj] = result['likelihoods']
                lick_dists[block][subj] = result['lick_dist']
                time_bins[block][subj] = result['time_bins']
                best_params[block][subj] = result['best_params']
            except FileNotFoundError as e:
                print(f'  [skip] {e}')

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump({
                'params': params, 'likelihoods': likelihoods,
                'lick_dists': lick_dists, 'time_bins': time_bins,
                'best_params': best_params,
            }, f)
        print(f'Saved combined results -> {save_path}')

    return params, likelihoods, lick_dists, time_bins, best_params
