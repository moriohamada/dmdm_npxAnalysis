"""training loop and CV for hybrid poisson model"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from config import HYBRID_OPTIONS, ANALYSIS_OPTIONS, GLM_OPTIONS
from data.session import Session
from neuron_prediction.data import (
    load_glm_inputs, get_trial_fold_indices,
    neuron_seed, normalise_design_matrix,
)
from neuron_prediction.evaluate import pearson_r
from neuron_prediction.hybrid.model import (
    HybridModel, proximal_group_lasso_hybrid,
)
from neuron_prediction.results.peth import build_event_spec, fold_peths


#%%
def _to_tensors(X, y, device):
    return (torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(y, dtype=torch.float32, device=device))


def _make_model(n_inputs, col_map, ops):
    return HybridModel(
        n_inputs, ops['interactions'], col_map, ops['units_per_group'])


#%%
def train_one(model, X_train, y_train, col_map, ops, lambda_gl=0.0):
    """train with SGD + proximal group lasso + early stopping"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    n = len(X_train)
    n_val = max(1, int(n * ops['val_frac']))
    perm = np.random.permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    X_tr, y_tr = _to_tensors(X_train[train_idx], y_train[train_idx], device)
    X_val, y_val = _to_tensors(X_train[val_idx], y_train[val_idx], device)

    lr = ops['lr']
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=ops.get('lr_step_size', 500),
        gamma=ops.get('lr_gamma', 0.5))
    poisson_loss = nn.PoissonNLLLoss(log_input=True)

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    batch_size = ops['batch_size']

    for epoch in range(ops['max_epochs']):
        model.train()
        perm_tr = torch.randperm(len(X_tr), device=device)
        current_lr = optimizer.param_groups[0]['lr']
        for start in range(0, len(X_tr), batch_size):
            idx = perm_tr[start:start + batch_size]
            log_rate = model(X_tr[idx])
            loss = poisson_loss(log_rate, y_tr[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            proximal_group_lasso_hybrid(model, col_map, lambda_gl, current_lr)

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_log_rate = model(X_val)
            val_loss = poisson_loss(val_log_rate, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= ops['patience']:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
    model.eval()
    return model, best_val_loss


#%%
def _predict_numpy(model, X, device):
    """run forward pass, return predicted counts as numpy array"""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        log_rate = model(X_t)
        return torch.exp(log_rate).cpu().numpy()


def _eval_model(model, X_test, y_test, group_masks_test, col_map,
                interactions, lesion_groups, device):
    """evaluate full model, skip-only, and per-interaction lesions

    no permutation needed — interactions are architecturally separated
    """
    y_pred = _predict_numpy(model, X_test, device)
    full_r = pearson_r(y_test, y_pred)

    # skip-only baseline
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_pred_skip = torch.exp(model.forward_skip_only(X_t)).cpu().numpy()
    skip_r = pearson_r(y_test, y_pred_skip)

    # per-group r on bins where that group is active
    r_group = {}
    for gname in lesion_groups:
        win = group_masks_test[gname]
        if win.sum() < 5:
            continue
        r_group[gname] = pearson_r(y_test[win], y_pred[win])

    # per-interaction lesion
    r_lesion = {}
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        for i, interaction in enumerate(interactions):
            key = '_x_'.join(interaction)
            y_pred_les = torch.exp(model.forward_lesion(X_t, i)).cpu().numpy()
            r_lesion[key] = pearson_r(y_test, y_pred_les)

    # per-group skip lesion
    r_skip_lesion = {}
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        for gname, pred_list in lesion_groups.items():
            win = group_masks_test[gname]
            if win.sum() < 5:
                continue
            y_pred_les = torch.exp(
                model.forward_lesion_skip(X_t, col_map, [gname])).cpu().numpy()
            r_skip_lesion[gname] = pearson_r(y_test[win], y_pred_les[win])

    return full_r, skip_r, r_group, r_lesion, r_skip_lesion


#%%
def _cv_select_lambda(X_train, y_train, col_map, fold_ids_train, ops):
    """select best lambda via CV on training data"""
    lambdas = ops['group_lasso_lambdas']
    cv_ops = {**ops,
              'max_epochs': ops.get('cv_max_epochs', 500),
              'patience': ops.get('cv_patience', 30)}

    folds = np.unique(fold_ids_train[fold_ids_train >= 0])
    n_folds = len(folds)

    mean_rs = []
    for lam in lambdas:
        fold_rs = []
        for k in folds:
            val_mask = fold_ids_train == k
            tr_mask = (fold_ids_train >= 0) & ~val_mask
            if val_mask.sum() == 0 or tr_mask.sum() == 0:
                continue

            model = _make_model(X_train.shape[1], col_map, cv_ops)
            model, _ = train_one(
                model, X_train[tr_mask], y_train[tr_mask],
                col_map, cv_ops, lam)

            device = next(model.parameters()).device
            y_pred = _predict_numpy(model, X_train[val_mask], device)
            fold_rs.append(pearson_r(y_train[val_mask], y_pred))

        mean_r = np.nanmean(fold_rs) if fold_rs else np.nan
        mean_rs.append(mean_r)
        print(f'    lambda={lam}: r={mean_r:.4f}')

    best_idx = np.nanargmax(mean_rs)
    return lambdas[best_idx]


#%%
def fit_neuron(counts_1d, X, col_map, fold_ids, trials_df, t_ax,
               event_spec=None, ops=HYBRID_OPTIONS):
    """fit hybrid model for one neuron via outer CV with lambda selection

    event_spec: optional dict {kind: (bin_idx, signs, pre, post)} from
        build_event_spec, for paper-style PETH classification. lesion in
        the hybrid is skip-path zeroing (forward_lesion_skip), not refit.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_folds = ops['n_outer_folds']
    lesion_groups = ops['lesion_groups']
    group_names = list(lesion_groups.keys())
    interactions = ops['interactions']
    interaction_keys = ['_x_'.join(inter) for inter in interactions]

    valid = fold_ids >= 0
    X_v = X[valid]
    y_v = counts_1d[valid].astype(np.float64)
    folds_v = fold_ids[valid]
    T = len(counts_1d)
    valid_T_idx = np.where(valid)[0]

    # full-length CV predictions for PETHs
    y_full_cv = np.full(T, np.nan)
    y_red_cv = {g: np.full(T, np.nan) for g in group_names}

    # evaluation masks: bins where each group's predictors are non-zero
    group_masks = {}
    for gname, pred_list in lesion_groups.items():
        mask = np.zeros(X_v.shape[0], dtype=bool)
        for pred_name in pred_list:
            if pred_name in col_map:
                col_slice, _ = col_map[pred_name]
                mask |= np.any(X_v[:, col_slice] != 0, axis=1)
        group_masks[gname] = mask

    # storage
    full_r = np.full(n_folds, np.nan)
    skip_r = np.full(n_folds, np.nan)
    r_group = {g: np.full(n_folds, np.nan) for g in group_names}
    r_lesion = {k: np.full(n_folds, np.nan) for k in interaction_keys}
    r_skip_lesion = {g: np.full(n_folds, np.nan) for g in group_names}
    lambdas = []

    for k in range(n_folds):
        print(f'Outer fold {k+1}/{n_folds}')
        test_mask = folds_v == k
        train_mask = ~test_mask

        X_train_raw, y_train = X_v[train_mask], y_v[train_mask]
        X_test_raw, y_test = X_v[test_mask], y_v[test_mask]

        if y_train.sum() == 0 or y_test.sum() == 0:
            continue

        X_train, X_test, _, _ = normalise_design_matrix(
            X_train_raw, X_test_raw, col_map)

        # inner CV for lambda
        inner_folds = get_trial_fold_indices(
            trials_df, t_ax, 3, seed=k,
            ignore_first_n=ANALYSIS_OPTIONS['ignore_first_trials_in_block'])
        inner_folds_train = inner_folds[valid][train_mask]

        print(f'  lambda selection...')
        lambda_gl = _cv_select_lambda(
            X_train, y_train, col_map, inner_folds_train, ops)
        lambdas.append(lambda_gl)
        print(f'  best lambda={lambda_gl}')

        # train final model for this fold
        model = _make_model(X_train.shape[1], col_map, ops)
        model, _ = train_one(model, X_train, y_train, col_map, ops, lambda_gl)

        gm_test = {g: group_masks[g][test_mask] for g in group_names}
        r, r_sk, r_g, r_les, r_skl = _eval_model(
            model, X_test, y_test, gm_test, col_map, interactions, lesion_groups,
            device)

        full_r[k] = r
        skip_r[k] = r_sk
        for g in group_names:
            if g in r_g: r_group[g][k] = r_g[g]
            if g in r_skl: r_skip_lesion[g][k] = r_skl[g]
        for ik in interaction_keys:
            if ik in r_les: r_lesion[ik][k] = r_les[ik]
        print(f'  r={r:.3f} (skip={r_sk:.3f})')

        # scatter predictions into full-length CV arrays for PETHs
        if event_spec is not None:
            test_T_idx = valid_T_idx[test_mask]
            with torch.no_grad():
                X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
                y_full_cv[test_T_idx] = torch.exp(
                    model(X_t)).cpu().numpy().ravel()
                for gname in group_names:
                    y_red_cv[gname][test_T_idx] = torch.exp(
                        model.forward_lesion_skip(
                            X_t, col_map, [gname])).cpu().numpy().ravel()

    # per-fold PETHs for paper-style classification
    peth_data = {}
    if event_spec is not None:
        counts_f = counts_1d.astype(np.float64)
        for kind, (bin_idx, signs, pre, post) in event_spec.items():
            if kind not in y_red_cv:
                continue
            n_bins = pre + post
            pa_fast = np.full((n_folds, n_bins), np.nan)
            pa_slow = np.full((n_folds, n_bins), np.nan)
            pf_fast = np.full((n_folds, n_bins), np.nan)
            pf_slow = np.full((n_folds, n_bins), np.nan)
            pr_fast = np.full((n_folds, n_bins), np.nan)
            pr_slow = np.full((n_folds, n_bins), np.nan)

            for k in range(n_folds):
                (pa_fast[k], pa_slow[k],
                 pf_fast[k], pf_slow[k],
                 pr_fast[k], pr_slow[k]) = fold_peths(
                    counts_f, y_full_cv, y_red_cv[kind],
                    bin_idx, signs, fold_ids, k, pre, post)

            peth_data[f'peth_{kind}_actual_fast'] = pa_fast
            peth_data[f'peth_{kind}_actual_slow'] = pa_slow
            peth_data[f'peth_{kind}_full_fast'] = pf_fast
            peth_data[f'peth_{kind}_full_slow'] = pf_slow
            peth_data[f'peth_{kind}_reduced_fast'] = pr_fast
            peth_data[f'peth_{kind}_reduced_slow'] = pr_slow

    # final refit on all data
    from collections import Counter
    X_v_norm, _, _, _ = normalise_design_matrix(X_v, X_v, col_map)
    lambda_gl = Counter(lambdas).most_common(1)[0][0] if lambdas else 0
    print(f'Final refit (lambda={lambda_gl})...')
    final_model = _make_model(X_v.shape[1], col_map, ops)
    final_model, _ = train_one(final_model, X_v_norm, y_v, col_map, ops, lambda_gl)

    result = {
        'fold_ids': fold_ids,
        'full_r': full_r,
        'skip_r': skip_r,
        'lambda_gl': np.array(lambda_gl),
        'skip_weights': final_model.skip.weight.detach().cpu().numpy().ravel(),
        'skip_bias': final_model.skip.bias.detach().cpu().numpy().ravel(),
    }
    for g in group_names:
        result[f'full_r_group_{g}'] = r_group[g]
        result[f'skip_lesion_r_{g}'] = r_skip_lesion[g]
    for i, ik in enumerate(interaction_keys):
        result[f'lesion_r_{ik}'] = r_lesion[ik]
        subnet = final_model.subnets[i]
        result[f'{ik}_hidden_weights'] = subnet.hidden.weight.detach().cpu().numpy()
        result[f'{ik}_hidden_bias'] = subnet.hidden.bias.detach().cpu().numpy()
        result[f'{ik}_output_weights'] = subnet.output.weight.detach().cpu().numpy().ravel()
        result[f'{ik}_output_bias'] = subnet.output.bias.detach().cpu().numpy().ravel()
    result.update(peth_data)

    return result


#%%
def fit_neuron_from_disk(sess_dir, neuron_idx, ops=HYBRID_OPTIONS):
    """load prepped data, fit one neuron, save results"""
    import pickle

    counts, X, col_map, t_ax, valid_mask = load_glm_inputs(sess_dir)
    y = counts[neuron_idx]

    sess = Session.load(str(Path(sess_dir) / 'session.pkl'))
    fold_ids = get_trial_fold_indices(
        sess.trials, t_ax, ops['n_outer_folds'],
        seed=neuron_seed(sess_dir, neuron_idx),
        ignore_first_n=ANALYSIS_OPTIONS['ignore_first_trials_in_block'])

    event_spec = build_event_spec(
        sess,
        kinds=['tf', 'lick_prep', 'lick_exec'],
        t_ax=t_ax,
        bin_width=GLM_OPTIONS['bin_width'],
        tf_sd_threshold=GLM_OPTIONS['tf_sd_threshold'])

    result = fit_neuron(y, X, col_map, fold_ids, sess.trials, t_ax,
                        event_spec=event_spec, ops=ops)

    results_dir = Path(sess_dir) / 'hybrid_results'
    results_dir.mkdir(exist_ok=True)
    np.savez(results_dir / f'neuron_{neuron_idx}.npz', **result)

    col_map_path = results_dir / 'col_map.pkl'
    with open(col_map_path, 'wb') as f:
        pickle.dump(col_map, f)
