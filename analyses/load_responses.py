"""
Functions for loading event-aligned responses
"""
import numpy as np
import h5py
import fnmatch

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

def load_psth(filepath: str,
              event_type: str = 'tf',
              condition: str = None,
              baseline_subtract: bool = False):

    with h5py.File(filepath, 'r') as f:
        t_ax = f[f't_ax/{event_type}'][:]
        all_conditions = list(f[event_type].keys())

        if condition is None:
            conditions = all_conditions
        elif '*' in condition:
            conditions = sorted(k for k in all_conditions if fnmatch.fnmatch(k, condition))
            if not conditions:
                raise ValueError(f"No conditions in '{event_type}' match '{condition}'. "
                                 f"Available: {all_conditions}")
        else:
            conditions = [condition]

        arr = np.concatenate([f[f'{event_type}/{cond}'][:] for cond in conditions],
                             axis=0)

    if baseline_subtract:
        bl_mask = t_ax < 0
        bl_mean = np.nanmean(arr[:, :, bl_mask], axis=(0, 2), keepdims=True)
        arr     = arr - bl_mean

    return arr, t_ax

def load_psth_mean(filepath: str,
                   event_type: str = 'tf',
                   condition: str = None,
                   baseline_subtract: bool = True):
    """
    Return mean & sem across trials (nN x nT).
    If condition is None, averages across all conditions.
    """
    arr, t_ax = load_psth(filepath=filepath, event_type=event_type,
                          condition=condition, baseline_subtract=baseline_subtract)

    mean = np.nanmean(arr, axis=0)
    sem = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])


    return mean, sem, t_ax