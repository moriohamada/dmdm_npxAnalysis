"""
classify units by lesion significance, shared across glm variants
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp

from config import GLM_OPTIONS
from data.session import Session


def classify_units(sess_dir, fit_type, ops=GLM_OPTIONS):
    """classify units from per-neuron results for one session

    fit_type: 'glm', 'glm_ridge', or 'glm_unreg'
    """
    warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)
    sess_dir = Path(sess_dir)
    results_dir = sess_dir / f'{fit_type}_results'
    sess = Session.load(str(sess_dir / 'session.pkl'))
    n_neurons = len(sess.fr_stats)
    group_names = list(ops['lesion_groups'].keys())

    classifications = []
    for i in range(n_neurons):
        res_path = results_dir / f'neuron_{i}.npz'
        if not res_path.exists():
            classifications.append({
                'neuron_idx': i,
                'cluster_id': sess.fr_stats.index[i],
            })
            continue

        res = np.load(res_path, allow_pickle=True)
        full_r = res['full_r']

        ok_r = full_r[~np.isnan(full_r)]
        if len(ok_r) >= 3:
            _, p_full = ttest_1samp(ok_r, 0)
            sig_full = p_full < ops['lesion_alpha'] and np.mean(ok_r) > 0
        else:
            p_full = 1.0
            sig_full = False

        row = {
            'neuron_idx': i,
            'cluster_id': sess.fr_stats.index[i],
            'mean_r': np.nanmean(full_r),
            'is_predictable_p': p_full,
            'is_predictable': sig_full,
        }

        for gname in group_names:
            full_r_g = res[f'full_r_group_{gname}']
            les_r_g = res[f'lesioned_r_{gname}']
            ok = ~(np.isnan(full_r_g) | np.isnan(les_r_g))

            if ok.sum() >= 3:
                _, p = ttest_rel(full_r_g[ok], les_r_g[ok])
                delta_r = np.nanmean(full_r_g[ok]) - np.nanmean(les_r_g[ok])
            else:
                p = 1.0
                delta_r = 0.0

            is_sig = (sig_full and
                      p < ops['lesion_alpha'] and
                      delta_r > 0) if ok.sum() >= 3 else False

            row[f'{gname}_mean_r'] = np.nanmean(full_r_g)
            row[f'{gname}_p'] = p
            row[f'{gname}_delta_r'] = delta_r
            row[f'{gname}_sig'] = is_sig

        classifications.append(row)

    df = pd.DataFrame(classifications)
    df.to_csv(sess_dir / f'{fit_type}_classifications.csv', index=False)
    return df


def extract_kernels(weights, col_map, bin_width=GLM_OPTIONS['bin_width']):
    """reshape flat weight vector into named kernels

    returns dict: predictor_name -> (t_ax_kernel, kernel_values)
    """
    kernels = {}
    for name, (col_slice, lags) in col_map.items():
        t_ax_kernel = lags * bin_width
        kernel_vals = weights[col_slice]
        kernels[name] = (t_ax_kernel, kernel_vals)
    return kernels
