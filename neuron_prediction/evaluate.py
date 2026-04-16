"""shared evaluation metrics for neuron prediction models"""
import numpy as np
from itertools import combinations


def pearson_r(y_true, y_pred):
    """pearson correlation, nan if degenerate"""
    from scipy.stats import pearsonr
    if len(y_true) < 5 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return pearsonr(y_true, y_pred.ravel())[0]


def lesion_design_matrix(X, predictor_names, col_map):
    """zero out columns belonging to a list of predictors"""
    X_les = X.copy()
    for name, (col_slice, _) in col_map.items():
        if name in predictor_names:
            X_les[:, col_slice] = 0
    return X_les


def reduce_design_matrix(X, predictor_names, col_map):
    """remove columns belonging to a list of predictors, return reduced X and col_map"""
    col_map_red = {}
    offset = 0
    keep_slices = []
    for name, (col_slice, lags) in col_map.items():
        n_cols = col_slice.stop - col_slice.start
        if name in predictor_names:
            continue
        keep_slices.append(col_slice)
        col_map_red[name] = (slice(offset, offset + n_cols), lags)
        offset += n_cols

    X_red = np.concatenate([X[:, s] for s in keep_slices], axis=1)
    return X_red, col_map_red


def permute_design_matrix(X, predictor_names, col_map, rng=None):
    """shuffle rows of a predictor group's columns

    preserves marginal distribution and within-group column structure
    but breaks relationship with other predictors and the target.
    valid for non-linear models where zeroing creates OOD inputs.
    """
    if rng is None:
        rng = np.random.RandomState()
    X_perm = X.copy()
    perm_idx = rng.permutation(X.shape[0])
    for name, (col_slice, _) in col_map.items():
        if name in predictor_names:
            X_perm[:, col_slice] = X[perm_idx][:, col_slice]
    return X_perm


def get_interaction_combos(group_names, max_order=3):
    """generate all 2-way and 3-way combinations of group names"""
    combos = []
    for order in range(2, max_order + 1):
        combos.extend(combinations(sorted(group_names), order))
    return combos


def interaction_combo_key(combo):
    """e.g. ('block', 'tf') -> 'block_x_tf'"""
    return '_x_'.join(combo)


def compare_models(glm_results_dir, network_results_dir, n_hidden,
                   neuron_indices=None):
    """per-neuron scatter of r_GLM vs r_NN with paired statistics

    n_hidden: which network hidden size to compare against GLM
    returns DataFrame with columns: neuron_idx, r_glm, r_nn
    """
    import pandas as pd
    from pathlib import Path
    from scipy.stats import wilcoxon

    glm_dir = Path(glm_results_dir)
    net_dir = Path(network_results_dir)
    p = f'h{n_hidden}_'

    rows = []
    indices = neuron_indices or sorted(
        int(f.stem.split('_')[1]) for f in glm_dir.glob('neuron_*.npz')
    )

    for i in indices:
        glm_path = glm_dir / f'neuron_{i}.npz'
        net_path = net_dir / f'neuron_{i}.npz'
        if not glm_path.exists() or not net_path.exists():
            continue
        glm_res = np.load(glm_path)
        net_res = np.load(net_path)
        if f'{p}full_r' not in net_res.files:
            continue
        rows.append({
            'neuron_idx': i,
            'r_glm': np.nanmean(glm_res['full_r']),
            'r_nn': np.nanmean(net_res[f'{p}full_r']),
        })

    df = pd.DataFrame(rows)
    if len(df) > 10:
        valid = df.dropna(subset=['r_glm', 'r_nn'])
        stat, pval = wilcoxon(valid['r_nn'], valid['r_glm'])
        df.attrs['wilcoxon_stat'] = stat
        df.attrs['wilcoxon_p'] = pval
    return df
