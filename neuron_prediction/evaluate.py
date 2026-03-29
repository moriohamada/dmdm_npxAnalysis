"""shared evaluation metrics for neuron prediction models"""
import numpy as np


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


def compare_models(glm_results_dir, network_results_dir, neuron_indices=None):
    """per-neuron scatter of r_GLM vs r_NN with paired statistics

    returns DataFrame with columns: neuron_idx, r_glm, r_nn
    """
    import pandas as pd
    from pathlib import Path
    from scipy.stats import wilcoxon

    glm_dir = Path(glm_results_dir)
    net_dir = Path(network_results_dir)

    rows = []
    indices = neuron_indices or sorted(
        int(p.stem.split('_')[1]) for p in glm_dir.glob('neuron_*.npz')
    )

    for i in indices:
        glm_path = glm_dir / f'neuron_{i}.npz'
        net_path = net_dir / f'neuron_{i}.npz'
        if not glm_path.exists() or not net_path.exists():
            continue
        glm_res = np.load(glm_path)
        net_res = np.load(net_path)
        rows.append({
            'neuron_idx': i,
            'r_glm': np.nanmean(glm_res['full_r']),
            'r_nn': np.nanmean(net_res['full_r']),
        })

    df = pd.DataFrame(rows)
    if len(df) > 10:
        valid = df.dropna(subset=['r_glm', 'r_nn'])
        stat, p = wilcoxon(valid['r_nn'], valid['r_glm'])
        df.attrs['wilcoxon_stat'] = stat
        df.attrs['wilcoxon_p'] = p
    return df
