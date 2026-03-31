"""post-hoc: collect network results, hidden unit lesion analysis"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

from config import PATHS, NETWORK_OPTIONS, ANALYSIS_OPTIONS
from data.session import Session
from neuron_prediction.data import (
    load_glm_inputs, get_trial_fold_indices, neuron_seed,
    normalise_design_matrix,
)
from neuron_prediction.evaluate import pearson_r
from neuron_prediction.network.model import PoissonNet


def _get_hidden_sizes(res):
    """recover fitted hidden sizes from prefixed keys in npz"""
    sizes = set()
    for key in res.files:
        if key.startswith('h') and '_' in key:
            prefix = key.split('_')[0]  # e.g. 'h8'
            try:
                sizes.add(int(prefix[1:]))
            except ValueError:
                continue
    return sorted(sizes)


def lesion_hidden_units(sess_dir, neuron_idx, ops=NETWORK_OPTIONS):
    """knock out each hidden unit, measure r drop

    returns DataFrame with unit_idx, n_hidden, r_full, r_lesioned, delta_r,
    plus one column per predictor group with input weight norms.
    returns empty DataFrame for linear models (no hidden units).
    """
    import torch

    counts, X, col_map, t_ax, _ = load_glm_inputs(sess_dir)

    # use fold_ids for consistent data subsetting with fitting
    sess = Session.load(str(Path(sess_dir) / 'session.pkl'))
    fold_ids = get_trial_fold_indices(
        sess.trials, t_ax, ops['n_outer_folds'],
        seed=neuron_seed(sess_dir, neuron_idx),
        ignore_first_n=ANALYSIS_OPTIONS['ignore_first_trials_in_block'])
    valid = fold_ids >= 0

    y = counts[neuron_idx][valid].astype(np.float64)
    X_v = X[valid]

    # normalise to match final refit (which uses all valid data's statistics)
    X_v, _, _, _ = normalise_design_matrix(X_v, X_v, col_map)

    res_path = Path(sess_dir) / 'network_results' / f'neuron_{neuron_idx}.npz'
    if not res_path.exists():
        return pd.DataFrame()
    res = np.load(res_path, allow_pickle=True)

    hidden_sizes = _get_hidden_sizes(res)
    all_rows = []

    for n_hidden in hidden_sizes:
        if n_hidden == 0:
            continue

        p = f'h{n_hidden}_'
        if f'{p}hidden_weights' not in res:
            continue

        model = PoissonNet(X_v.shape[1], n_hidden)
        model.hidden.weight.data = torch.tensor(
            res[f'{p}hidden_weights'], dtype=torch.float32)
        model.hidden.bias.data = torch.tensor(
            res[f'{p}hidden_bias'], dtype=torch.float32)
        model.output.weight.data = torch.tensor(
            res[f'{p}output_weights'].reshape(1, -1), dtype=torch.float32)
        model.output.bias.data = torch.tensor(
            res[f'{p}output_bias'], dtype=torch.float32)
        model.eval()

        X_t = torch.tensor(X_v, dtype=torch.float32)

        with torch.no_grad():
            y_pred_full = torch.exp(model(X_t)).numpy()
        r_full = pearson_r(y, y_pred_full)

        for unit_idx in range(n_hidden):
            saved_w = model.output.weight.data[0, unit_idx].item()
            model.output.weight.data[0, unit_idx] = 0.0

            with torch.no_grad():
                y_pred_les = torch.exp(model(X_t)).numpy()
            r_les = pearson_r(y, y_pred_les)

            row = {
                'n_hidden': n_hidden,
                'unit_idx': unit_idx,
                'r_full': r_full,
                'r_lesioned': r_les,
                'delta_r': r_full - r_les,
            }
            # weight norms per predictor group
            hidden_weights = res[f'{p}hidden_weights']
            for name, (col_slice, _) in col_map.items():
                row[f'norm_{name}'] = np.linalg.norm(
                    hidden_weights[unit_idx][col_slice])

            all_rows.append(row)
            model.output.weight.data[0, unit_idx] = saved_w

    return pd.DataFrame(all_rows)


def classify_units(sess_dir, ops=NETWORK_OPTIONS):
    """classify units by lesion significance, one CSV per hidden size

    mirrors glm classify_units exactly: same columns, same statistical
    tests. saves h{n}_classifications.csv in network_results/
    """
    from scipy.stats import ttest_rel, ttest_1samp
    from neuron_prediction.evaluate import interaction_combo_key

    results_dir = Path(sess_dir) / 'network_results'
    sess = Session.load(str(Path(sess_dir) / 'session.pkl'))
    n_neurons = len(sess.fr_stats)
    group_names = list(ops['lesion_groups'].keys())
    lesion_alpha = ops['lesion_alpha']

    # find all hidden sizes from first available result file
    hidden_sizes = None
    for i in range(n_neurons):
        res_path = results_dir / f'neuron_{i}.npz'
        if res_path.exists():
            res = np.load(res_path, allow_pickle=True)
            hidden_sizes = _get_hidden_sizes(res)
            break
    if hidden_sizes is None:
        return {}

    all_dfs = {}
    for nh in hidden_sizes:
        p = f'h{nh}_'
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
            if f'{p}full_r' not in res.files:
                classifications.append({
                    'neuron_idx': i,
                    'cluster_id': sess.fr_stats.index[i],
                })
                continue
            full_r = res[f'{p}full_r']

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
                'is_predictable_p': p_full,
                'is_predictable': sig_full,
            }

            for gname in group_names:
                full_r_g = res[f'{p}full_r_group_{gname}']
                les_r_g = res[f'{p}permuted_r_{gname}']
                ok = ~(np.isnan(full_r_g) | np.isnan(les_r_g))

                if ok.sum() >= 3:
                    _, p_val = ttest_rel(full_r_g[ok], les_r_g[ok])
                    delta_r = np.nanmean(full_r_g[ok]) - np.nanmean(les_r_g[ok])
                else:
                    p_val = 1.0
                    delta_r = 0.0

                is_sig = (sig_full and
                          p_val < lesion_alpha and
                          delta_r > 0) if ok.sum() >= 3 else False

                row[f'{gname}_mean_r'] = np.nanmean(full_r_g)
                row[f'{gname}_p'] = p_val
                row[f'{gname}_delta_r'] = delta_r
                row[f'{gname}_sig'] = is_sig

            # interaction significance
            # interaction effect = delta_r(joint) - sum(delta_r(individual))
            combos = ops['interaction_combos']
            for combo in combos:
                ck = interaction_combo_key(combo)
                int_key = f'{p}interaction_r_{ck}'
                if int_key not in res.files:
                    continue

                int_r = res[int_key]

                # sum of individual delta_rs per fold
                sum_individual = np.zeros(len(int_r))
                for gname in combo:
                    fg = res.get(f'{p}full_r_group_{gname}')
                    pg = res.get(f'{p}permuted_r_{gname}')
                    if fg is not None and pg is not None:
                        sum_individual += (fg - pg)

                # joint delta_r per fold
                joint_delta = full_r - int_r

                # interaction effect = joint - sum(individual)
                interaction_effect = joint_delta - sum_individual

                ok = ~np.isnan(interaction_effect)
                if ok.sum() >= 3:
                    _, p_int = ttest_1samp(interaction_effect[ok], 0)
                    mean_effect = np.nanmean(interaction_effect[ok])
                else:
                    p_int = 1.0
                    mean_effect = 0.0

                row[f'interaction_{ck}_effect'] = mean_effect
                row[f'interaction_{ck}_p'] = p_int
                row[f'interaction_{ck}_sig'] = (
                    sig_full and p_int < lesion_alpha and mean_effect > 0
                ) if ok.sum() >= 3 else False

            classifications.append(row)

        df = pd.DataFrame(classifications)
        df.to_csv(results_dir / f'h{nh}_classifications.csv', index=False)
        all_dfs[nh] = df
        print(f'  h={nh}: {df["is_predictable"].sum() if "is_predictable" in df else 0} '
              f'predictable / {n_neurons} neurons')

    return all_dfs


def collect_all(npx_dir=PATHS['npx_dir_local']):
    """run hidden unit lesion for all fitted neurons, save as CSV"""
    for subj in sorted(os.listdir(npx_dir)):
        subj_dir = os.path.join(npx_dir, subj)
        if not os.path.isdir(subj_dir):
            continue
        for sess in sorted(os.listdir(subj_dir)):
            sess_dir = os.path.join(subj_dir, sess)
            results_dir = os.path.join(sess_dir, 'network_results')
            if not os.path.isdir(results_dir):
                continue
            print(f'{subj}/{sess}')
            neuron_files = sorted(Path(results_dir).glob('neuron_*.npz'))
            for nf in neuron_files:
                neuron_idx = int(nf.stem.split('_')[1])
                df = lesion_hidden_units(sess_dir, neuron_idx)
                if len(df) > 0:
                    df.to_csv(Path(results_dir) / f'hidden_units_{neuron_idx}.csv',
                              index=False)
