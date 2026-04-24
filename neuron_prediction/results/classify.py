"""
classify units by lesion significance, shared across glm variants

two classification paths per neuron:
- per-bin pearson pipeline (legacy): t-tests on fold-wise correlations
  in predictor-active windows. kept as diagnostic columns.
- PETH pipeline (paper's method, Khilkevich & Lohse 2024): two-criterion
  test on event-averaged PETHs. determines {gname}_sig for tf/lick_prep/
  lick_exec. fold-wise PETHs must be saved in the fit npz.
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp

from config import GLM_OPTIONS
from data.session import Session


# lesion groups with PETH-based sig; others fall back to per-bin sig
PETH_KINDS = ('tf', 'lick_prep', 'lick_exec')


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

        # is_predictable: full model's per-bin correlation significantly > 0
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

        # per-bin pearson per lesion group
        per_bin_sig = {}
        for gname in group_names:
            full_r_g = res[f'full_r_group_{gname}']
            les_r_g = res[f'lesioned_r_{gname}']
            ok = ~(np.isnan(full_r_g) | np.isnan(les_r_g))

            if ok.sum() >= 3:
                _, p = ttest_rel(full_r_g[ok], les_r_g[ok])
                delta_r = np.nanmean(full_r_g[ok]) - np.nanmean(les_r_g[ok])
                per_bin_sig[gname] = (sig_full and p < ops['lesion_alpha']
                                       and delta_r > 0)
            else:
                p = 1.0
                delta_r = 0.0
                per_bin_sig[gname] = False

            row[f'{gname}_mean_r'] = np.nanmean(full_r_g)
            row[f'{gname}_p'] = p
            row[f'{gname}_delta_r'] = delta_r

        # PETH-based classification for tf/lick_prep/lick_exec
        for gname in group_names:
            if gname in PETH_KINDS:
                r_full_mean, r_resid_mean, p_resid, sig = _peth_criteria(
                    res, gname, ops)
                row[f'{gname}_peth_r'] = r_full_mean
                row[f'{gname}_peth_resid_r'] = r_resid_mean
                row[f'{gname}_peth_p'] = p_resid
                row[f'{gname}_sig'] = sig
            else:
                # non-PETH groups fall back to per-bin sig
                row[f'{gname}_sig'] = per_bin_sig[gname]

        classifications.append(row)

    df = pd.DataFrame(classifications)
    df.to_csv(sess_dir / f'{fit_type}_classifications.csv', index=False)
    return df


def _peth_criteria(res, kind, ops, key_prefix=''):
    """apply paper's two PETH criteria

    1: mean over folds of pearson_r(actual, full) > peth_r_thresh
    2: one-sample t-test of pearson_r(actual-reduced, full-reduced) across folds,
    p < peth_alpha and mean > 0

    for tf the signal is (fast PETH - slow PETH). for licks - 'fast' events are just
    licks, no slow events.

    returns (r_full_mean, r_resid_mean, p_resid, sig)
    """
    key_base = f'{key_prefix}peth_{kind}'
    need = f'{key_base}_actual_fast'
    if need not in res.files:
        return np.nan, np.nan, 1.0, False

    a_fast = res[f'{key_base}_actual_fast']
    a_slow = res[f'{key_base}_actual_slow']
    f_fast = res[f'{key_base}_full_fast']
    f_slow = res[f'{key_base}_full_slow']
    r_fast = res[f'{key_base}_reduced_fast']
    r_slow = res[f'{key_base}_reduced_slow']

    if kind == 'tf':
        actual = a_fast - a_slow
        full = f_fast - f_slow
        reduced = r_fast - r_slow
    else:
        actual = a_fast
        full = f_fast
        reduced = r_fast

    n_folds = actual.shape[0]
    r_full = np.full(n_folds, np.nan)
    r_resid = np.full(n_folds, np.nan)

    for k in range(n_folds):
        a, f, r = actual[k], full[k], reduced[k]
        if np.all(np.isfinite(a)) and np.all(np.isfinite(f)) \
                and np.std(a) > 0 and np.std(f) > 0:
            r_full[k] = np.corrcoef(a, f)[0, 1]
        if (np.all(np.isfinite(a)) and np.all(np.isfinite(f))
                and np.all(np.isfinite(r))):
            ar = a - r
            fr = f - r
            if np.std(ar) > 0 and np.std(fr) > 0:
                r_resid[k] = np.corrcoef(ar, fr)[0, 1]

    r_full_mean = np.nanmean(r_full)
    crit1 = r_full_mean > ops['peth_r_thresh']

    valid_resid = r_resid[~np.isnan(r_resid)]
    if len(valid_resid) >= 3:
        _, p = ttest_1samp(valid_resid, 0)
        if np.mean(valid_resid) <= 0:
            p = 1.0
    else:
        p = 1.0
    crit2 = p < ops['peth_alpha']

    r_resid_mean = np.nanmean(r_resid)
    sig = bool(crit1 and crit2)
    return r_full_mean, r_resid_mean, p, sig


def classify_units_perblock(sess_dir, fit_type='glm_perblock',
                             ops=GLM_OPTIONS):
    """classify_units for perblock fit (neuron_{i}_{block}.npz)

    writes one csv per block: {fit_type}_classifications_{block}.csv
    """
    warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)
    sess_dir = Path(sess_dir)
    results_dir = sess_dir / f'{fit_type}_results'
    sess = Session.load(str(sess_dir / 'session.pkl'))
    n_neurons = len(sess.fr_stats)

    for block in ('early', 'late'):
        classifications = _classify_from_files(
            results_dir,
            file_pattern=lambda i, block=block: f'neuron_{i}_{block}.npz',
            n_neurons=n_neurons,
            sess=sess,
            ops=ops,
            key_prefix='',
        )
        df = pd.DataFrame(classifications)
        df.to_csv(sess_dir / f'{fit_type}_classifications_{block}.csv',
                  index=False)


def classify_units_network(sess_dir, hidden_size, fit_type='network',
                            ops=GLM_OPTIONS):
    """classify_units for network fit, picking one hidden size

    network stores per-hidden-size results under 'h{nh}_' prefix
    """
    warnings.filterwarnings('ignore', 'Mean of empty slice', RuntimeWarning)
    sess_dir = Path(sess_dir)
    results_dir = sess_dir / f'{fit_type}_results'
    sess = Session.load(str(sess_dir / 'session.pkl'))
    n_neurons = len(sess.fr_stats)

    classifications = _classify_from_files(
        results_dir,
        file_pattern=lambda i: f'neuron_{i}.npz',
        n_neurons=n_neurons,
        sess=sess,
        ops=ops,
        key_prefix=f'h{hidden_size}_',
    )
    df = pd.DataFrame(classifications)
    df.to_csv(sess_dir / f'{fit_type}_classifications_h{hidden_size}.csv',
              index=False)
    return df


def _classify_from_files(results_dir, file_pattern, n_neurons, sess, ops,
                          key_prefix=''):
    """shared per-neuron classification loop, parametrised by file and key

    file_pattern: callable(i) -> filename. used to locate the neuron's npz
    key_prefix: prefix for all npz keys (e.g. 'h0_' for network)
    """
    group_names = list(ops['lesion_groups'].keys())
    full_r_key = f'{key_prefix}full_r'

    out = []
    for i in range(n_neurons):
        res_path = results_dir / file_pattern(i)
        if not res_path.exists():
            out.append({
                'neuron_idx': i,
                'cluster_id': sess.fr_stats.index[i],
            })
            continue

        res = np.load(res_path, allow_pickle=True)
        if full_r_key not in res.files:
            out.append({
                'neuron_idx': i,
                'cluster_id': sess.fr_stats.index[i],
            })
            continue
        full_r = res[full_r_key]

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
            if gname in PETH_KINDS:
                r_full_mean, r_resid_mean, p_resid, sig = _peth_criteria(
                    res, gname, ops, key_prefix=key_prefix)
                row[f'{gname}_peth_r'] = r_full_mean
                row[f'{gname}_peth_resid_r'] = r_resid_mean
                row[f'{gname}_peth_p'] = p_resid
                row[f'{gname}_sig'] = sig

        out.append(row)
    return out


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
