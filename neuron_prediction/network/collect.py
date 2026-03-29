"""post-hoc: collect network results, hidden unit lesion analysis"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

from config import PATHS
from neuron_prediction.data import load_glm_inputs
from neuron_prediction.evaluate import pearson_r
from neuron_prediction.network.model import PoissonLinear, PoissonNet


def lesion_hidden_units(sess_dir, neuron_idx):
    """knock out each hidden unit, measure r drop

    returns DataFrame with unit_idx, r_full, r_lesioned, delta_r,
    plus one column per predictor group with input weight norms.
    returns empty DataFrame for linear models (no hidden units).
    """
    import torch

    counts, X, col_map, t_ax, valid_mask = load_glm_inputs(sess_dir)
    y = counts[neuron_idx][valid_mask].astype(np.float64)
    X_v = X[valid_mask]

    res_path = Path(sess_dir) / 'network_results' / f'neuron_{neuron_idx}.npz'
    if not res_path.exists():
        return pd.DataFrame()
    res = np.load(res_path, allow_pickle=True)

    n_hidden = int(res['n_hidden'])
    if n_hidden == 0:
        return pd.DataFrame()

    model = PoissonNet(X_v.shape[1], n_hidden)
    model.hidden.weight.data = torch.tensor(res['hidden_weights'], dtype=torch.float32)
    model.hidden.bias.data = torch.tensor(res['hidden_bias'], dtype=torch.float32)
    model.output.weight.data = torch.tensor(
        res['output_weights'].reshape(1, -1), dtype=torch.float32)
    model.output.bias.data = torch.tensor(res['output_bias'], dtype=torch.float32)
    model.eval()

    X_t = torch.tensor(X_v, dtype=torch.float32)

    with torch.no_grad():
        y_pred_full = torch.exp(model(X_t)).numpy()
    r_full = pearson_r(y, y_pred_full)

    rows = []
    for unit_idx in range(n_hidden):
        saved_w = model.output.weight.data[0, unit_idx].item()
        model.output.weight.data[0, unit_idx] = 0.0

        with torch.no_grad():
            y_pred_les = torch.exp(model(X_t)).numpy()
        r_les = pearson_r(y, y_pred_les)

        row = {
            'unit_idx': unit_idx,
            'r_full': r_full,
            'r_lesioned': r_les,
            'delta_r': r_full - r_les,
        }
        # weight norms per predictor group
        unit_weights = res['hidden_weights'][unit_idx]
        for name, (col_slice, _) in col_map.items():
            row[f'norm_{name}'] = np.linalg.norm(unit_weights[col_slice])

        rows.append(row)
        model.output.weight.data[0, unit_idx] = saved_w

    return pd.DataFrame(rows)


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
