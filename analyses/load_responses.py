import numpy as np
import h5py

def print_psth_contents(filepath):
    with h5py.File(filepath, 'r') as f:
        for event_type in f.keys():
            if event_type in ('t_ax',) or event_type.endswith('_mean'):
                continue
            conditions = list(f[event_type].keys())
            print(f'\n{event_type}:')
            for cond in conditions:
                n_trials = f[f'{event_type}/{cond}'].shape[0]
                print(f'  {cond}  (n={n_trials})')

def load_psth(filepath: str, event_type: str='tf', condition: str=None):
    with h5py.File(filepath, 'r') as f:
        t_ax = f[f't_ax/{event_type}'][:]
        if condition is None:
            conditions = list(f[event_type].keys())
        else:
            conditions = [condition]

        arr = np.concatenate([f[f'{event_type}/{cond}'][:] for cond in conditions],
                             axis=0)
    return arr, t_ax

def load_psth_mean(filepath: str, event_type: str = 'tf', condition: str = None):
    """
    Return mean & sem across trials (nN x nT)
    if condition is None, returns average across all conditions
    """
    arr, t_ax = load_psth(filepath=filepath, event_type=event_type, condition=condition)

    mean = np.nanmean(arr, axis=0)
    sem = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr[:, 0, 0])))

    return mean, sem, t_ax