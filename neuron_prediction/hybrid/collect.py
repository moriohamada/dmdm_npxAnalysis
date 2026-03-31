"""post-hoc: collect hybrid results, classify units by lesion significance"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_rel, ttest_1samp

from config import PATHS, HYBRID_OPTIONS
from data.session import Session


def classify_units(sess_dir, ops=HYBRID_OPTIONS):
    """classify units by skip and interaction lesion significance

    saves hybrid_classifications.csv in hybrid_results/
    """
    results_dir = Path(sess_dir) / 'hybrid_results'
    sess = Session.load(str(Path(sess_dir) / 'session.pkl'))
    n_neurons = len(sess.fr_stats)
    group_names = list(ops['lesion_groups'].keys())
    interactions = ops['interactions']
    interaction_keys = ['_x_'.join(inter) for inter in interactions]
    lesion_alpha = ops.get('lesion_alpha', 0.05)

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
        skip_r = res['skip_r']

        # is this neuron predictable at all?
        ok_r = full_r[~np.isnan(full_r)]
        if len(ok_r) >= 3:
            _, p_full = ttest_1samp(ok_r, 0)
            sig_full = p_full < lesion_alpha and np.mean(ok_r) > 0
        else:
            p_full = 1.0
            sig_full = False

        row = {
            'neuron_idx': i,
            'cluster_id': sess.fr_stats.index[i],
            'mean_r': np.nanmean(full_r),
            'mean_skip_r': np.nanmean(skip_r),
            'is_predictable_p': p_full,
            'is_predictable': sig_full,
        }

        # per-group skip lesion significance
        for gname in group_names:
            key = f'full_r_group_{gname}'
            les_key = f'skip_lesion_r_{gname}'
            if key not in res or les_key not in res:
                continue
            full_r_g = res[key]
            les_r_g = res[les_key]
            ok = ~(np.isnan(full_r_g) | np.isnan(les_r_g))

            if ok.sum() >= 3:
                _, p_val = ttest_rel(full_r_g[ok], les_r_g[ok])
                delta_r = np.nanmean(full_r_g[ok]) - np.nanmean(les_r_g[ok])
            else:
                p_val = 1.0
                delta_r = 0.0

            is_sig = (sig_full and p_val < lesion_alpha and delta_r > 0
                      ) if ok.sum() >= 3 else False

            row[f'{gname}_mean_r'] = np.nanmean(full_r_g)
            row[f'{gname}_p'] = p_val
            row[f'{gname}_delta_r'] = delta_r
            row[f'{gname}_sig'] = is_sig

        # per-interaction lesion significance
        for ik in interaction_keys:
            les_key = f'lesion_r_{ik}'
            if les_key not in res:
                continue
            les_r = res[les_key]
            ok = ~(np.isnan(full_r) | np.isnan(les_r))

            if ok.sum() >= 3:
                _, p_val = ttest_rel(full_r[ok], les_r[ok])
                delta_r = np.nanmean(full_r[ok]) - np.nanmean(les_r[ok])
            else:
                p_val = 1.0
                delta_r = 0.0

            is_sig = (sig_full and p_val < lesion_alpha and delta_r > 0
                      ) if ok.sum() >= 3 else False

            row[f'interaction_{ik}_p'] = p_val
            row[f'interaction_{ik}_delta_r'] = delta_r
            row[f'interaction_{ik}_sig'] = is_sig

        classifications.append(row)

    df = pd.DataFrame(classifications)
    df.to_csv(results_dir / 'hybrid_classifications.csv', index=False)
    n_pred = df['is_predictable'].sum() if 'is_predictable' in df else 0
    print(f'  {n_pred} predictable / {n_neurons} neurons')
    for ik in interaction_keys:
        col = f'interaction_{ik}_sig'
        if col in df:
            print(f'    {ik}: {df[col].sum()} significant')
    return df


def collect_all(npx_dir=PATHS['npx_dir_local']):
    """classify units for all sessions with hybrid results"""
    for subj in sorted(os.listdir(npx_dir)):
        subj_dir = os.path.join(npx_dir, subj)
        if not os.path.isdir(subj_dir):
            continue
        for sess in sorted(os.listdir(subj_dir)):
            sess_dir = os.path.join(subj_dir, sess)
            results_dir = os.path.join(sess_dir, 'hybrid_results')
            if not os.path.isdir(results_dir):
                continue
            print(f'{subj}/{sess}')
            classify_units(sess_dir)
