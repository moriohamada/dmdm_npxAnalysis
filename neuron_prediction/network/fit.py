"""training loop and CV for poisson linear and network models"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from config import NETWORK_OPTIONS, ANALYSIS_OPTIONS
from data.session import Session
from neuron_prediction.data import (
    load_glm_inputs, get_trial_fold_indices,
    neuron_seed, normalise_design_matrix,
)
from neuron_prediction.evaluate import (
    pearson_r, permute_design_matrix,
    get_interaction_combos, interaction_combo_key,
)
from neuron_prediction.network.model import (
    PoissonLinear, PoissonNet, proximal_group_lasso,
)


def _to_tensors(X, y, device):
    return (torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(y, dtype=torch.float32, device=device))


def _make_model(n_inputs, n_hidden):
    """create PoissonLinear (n_hidden=0) or PoissonNet"""
    if n_hidden == 0:
        return PoissonLinear(n_inputs)
    return PoissonNet(n_inputs, n_hidden)


def train_one(model, X_train, y_train, col_map, ops,
              lambda_gl=0.0):
    """train with plain SGD + proximal group lasso + early stopping

    smooth loss (Poisson NLL) updated via SGD.
    non-smooth penalty (group lasso) applied via proximal operator after each step.
    returns trained model and best validation loss.
    """
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
        optimizer,
        step_size=ops.get('lr_step_size', 500),
        gamma=ops.get('lr_gamma', 0.5),
    )
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

            # Poisson NLL
            loss = poisson_loss(log_rate, y_tr[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # proximal step for group lasso (non-smooth penalty)
            proximal_group_lasso(model, col_map, lambda_gl, current_lr)

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


def _predict_numpy(model, X, device):
    """run forward pass, return predicted counts as numpy array"""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        log_rate = model(X_t)
        return torch.exp(log_rate).cpu().numpy()


def _fit_one_inner(X_train, y_train, col_map, n_hidden, ops,
                   inner_folds, lambda_gl, k):
    """fit and evaluate one (lambda, inner fold) pair — for parallel dispatch"""
    val_mask = inner_folds == k
    tr_mask = (inner_folds >= 0) & ~val_mask

    if val_mask.sum() == 0 or tr_mask.sum() == 0:
        return lambda_gl, k, np.nan, np.inf

    model = _make_model(X_train.shape[1], n_hidden)
    model, _ = train_one(
        model, X_train[tr_mask], y_train[tr_mask],
        col_map, ops, lambda_gl)

    device = next(model.parameters()).device
    poisson_loss_fn = nn.PoissonNLLLoss(log_input=True)
    with torch.no_grad():
        X_val_t = torch.tensor(
            X_train[val_mask], dtype=torch.float32, device=device)
        y_val_t = torch.tensor(
            y_train[val_mask], dtype=torch.float32, device=device)
        log_rate = model(X_val_t)
        val_loss = poisson_loss_fn(log_rate, y_val_t).item()
        y_pred = torch.exp(log_rate).cpu().numpy()

    r = pearson_r(y_train[val_mask], y_pred)
    return lambda_gl, k, r, val_loss


def inner_cv_select(X_train, y_train, col_map, n_hidden, ops,
                    trials_df, t_ax, outer_valid, outer_train_mask, seed=0):
    """select best regularisation for a given hidden size via inner CV

    uses trial-level inner folds for proper temporal independence.
    evaluates on held-out inner fold using both poisson NLL and pearson r.
    selects config with highest mean r. parallelises across (lambda, fold) pairs.

    outer_valid: bool (T_full,) — bins with fold_id >= 0
    outer_train_mask: bool (T_full,) — bins in the outer training fold
    """
    from joblib import Parallel, delayed

    n_inner = ops['n_inner_folds']
    n_jobs = ops.get('n_jobs', 1)

    # build trial-level inner folds on the full time axis, then slice
    inner_fold_ids = get_trial_fold_indices(
        trials_df, t_ax, n_inner, seed=seed,
        ignore_first_n=ANALYSIS_OPTIONS['ignore_first_trials_in_block'])
    inner_folds = inner_fold_ids[outer_valid][outer_train_mask]

    configs = ops['group_lasso_lambdas']

    # parallel across all (lambda, fold) pairs
    results = Parallel(n_jobs=n_jobs)(
        delayed(_fit_one_inner)(
            X_train, y_train, col_map, n_hidden, ops,
            inner_folds, lam, k)
        for lam in configs
        for k in range(n_inner)
    )

    # aggregate by lambda
    best_r = -np.inf
    best_params = 0
    for ci, lambda_gl in enumerate(configs):
        fold_rs = [r for lam, k, r, loss in results if lam == lambda_gl]
        fold_losses = [loss for lam, k, r, loss in results if lam == lambda_gl]
        mean_r = np.nanmean(fold_rs) if fold_rs else np.nan
        mean_loss = np.mean(fold_losses) if fold_losses else np.inf
        print(f'    config {ci+1}/{len(configs)}: gl={lambda_gl} '
              f'-> r={mean_r:.4f}, loss={mean_loss:.4f}')
        if not np.isnan(mean_r) and mean_r > best_r:
            best_r = mean_r
            best_params = lambda_gl

    return best_params


def _eval_model(model, X_test, y_test, group_masks_test, lesion_groups,
                col_map, device, n_perm=10, seed=0):
    """full r and per-group permutation importance

    # shuffle regressors instead - can't just zero for network!

    group_masks_test: dict of bool arrays marking bins where each group's
    predictors are non-zero (already sliced to test fold)
    """
    y_pred = _predict_numpy(model, X_test, device)
    r = pearson_r(y_test, y_pred)

    rng = np.random.RandomState(seed)
    r_group = {}
    r_permuted = {}
    for gname, pred_list in lesion_groups.items():
        win = group_masks_test[gname]
        if win.sum() < 5:
            continue

        r_group[gname] = pearson_r(y_test[win], y_pred[win])

        perm_rs = []
        for _ in range(n_perm):
            X_perm = permute_design_matrix(X_test, pred_list, col_map, rng)
            y_pred_perm = _predict_numpy(model, X_perm, device)
            perm_rs.append(pearson_r(y_test[win], y_pred_perm[win]))
        r_permuted[gname] = np.nanmean(perm_rs)

    # pairwise and three-way interaction permutation
    combos = get_interaction_combos(list(lesion_groups.keys()), max_order=3)
    r_interaction = {}
    for combo in combos:
        # collect all predictor names for this combination
        pred_list = []
        for gname in combo:
            pred_list.extend(lesion_groups[gname])

        # use union of group masks
        win = np.zeros(len(y_test), dtype=bool)
        for gname in combo:
            if gname in group_masks_test:
                win |= group_masks_test[gname]
        if win.sum() < 5:
            continue

        perm_rs = []
        for _ in range(n_perm):
            X_perm = permute_design_matrix(X_test, pred_list, col_map, rng)
            y_pred_perm = _predict_numpy(model, X_perm, device)
            perm_rs.append(pearson_r(y_test[win], y_pred_perm[win]))
        r_interaction[interaction_combo_key(combo)] = np.nanmean(perm_rs)

    return r, r_group, r_permuted, r_interaction


def fit_neuron(counts_1d, X, col_map, fold_ids, trials_df, t_ax,
               ops=NETWORK_OPTIONS):
    """fit poisson models across all hidden sizes for one neuron

    for each hidden size, selects best regularisation via inner CV
    and evaluates on outer folds. returns flat dict with h{n}_ prefixed
    keys matching GLM result format.

    fold_ids: (T,) int array from get_trial_fold_indices. bins with
        fold_id == -1 are excluded from all fitting and evaluation.
    """
    from collections import Counter

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_folds = ops['n_outer_folds']
    lesion_groups = ops['lesion_groups']
    group_names = list(lesion_groups.keys())
    combos = get_interaction_combos(group_names, max_order=3)
    combo_keys = [interaction_combo_key(c) for c in combos]
    hidden_sizes = ops['hidden_sizes']

    valid = fold_ids >= 0
    X_v = X[valid]
    y_v = counts_1d[valid].astype(np.float64)
    folds_v = fold_ids[valid]

    # derive evaluation masks from design matrix
    group_masks = {}
    for gname, pred_list in lesion_groups.items():
        mask = np.zeros(X_v.shape[0], dtype=bool)
        for pred_name in pred_list:
            if pred_name in col_map:
                col_slice, _ = col_map[pred_name]
                mask |= np.any(X_v[:, col_slice] != 0, axis=1)
        group_masks[gname] = mask

    def _empty_results():
        return {
            'full_r': np.full(n_folds, np.nan),
            'full_r_group': {g: np.full(n_folds, np.nan) for g in group_names},
            'permuted_r': {g: np.full(n_folds, np.nan) for g in group_names},
            'interaction_r': {ck: np.full(n_folds, np.nan) for ck in combo_keys},
        }

    all_res = {nh: _empty_results() for nh in hidden_sizes}
    all_params = {nh: [] for nh in hidden_sizes}

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

        gm_test = {g: group_masks[g][test_mask] for g in group_names}

        for nh in hidden_sizes:
            print(f'  h={nh}: inner CV...')
            lambda_gl = inner_cv_select(
                X_train, y_train, col_map, nh, ops,
                trials_df, t_ax, valid, train_mask, seed=k)
            all_params[nh].append(lambda_gl)
            print(f'  h={nh}: best gl={lambda_gl}')

            model = _make_model(X_train.shape[1], nh)
            model, _ = train_one(model, X_train, y_train, col_map, ops,
                                  lambda_gl)
            r, r_g, r_l, r_int = _eval_model(
                model, X_test, y_test, gm_test, lesion_groups, col_map,
                device, n_perm=ops['n_perm_importance'], seed=k)
            all_res[nh]['full_r'][k] = r
            for g in group_names:
                if g in r_g: all_res[nh]['full_r_group'][g][k] = r_g[g]
                if g in r_l: all_res[nh]['permuted_r'][g][k] = r_l[g]
            for ck in combo_keys:
                if ck in r_int: all_res[nh]['interaction_r'][ck][k] = r_int[ck]
            print(f'  h={nh}: r={r:.3f}')

    # final refit per hidden size on all valid data
    X_v_norm, _, _, _ = normalise_design_matrix(X_v, X_v, col_map)
    result = {'fold_ids': fold_ids}

    for nh in hidden_sizes:
        if not all_params[nh]:
            continue

        lambda_gl = Counter(all_params[nh]).most_common(1)[0][0]
        print(f'Final refit h={nh} (gl={lambda_gl})...')
        final_model = _make_model(X_v.shape[1], nh)
        final_model, _ = train_one(final_model, X_v_norm, y_v, col_map, ops,
                                    lambda_gl)

        p = f'h{nh}_'
        result[f'{p}full_r'] = all_res[nh]['full_r']
        result[f'{p}lambda_gl'] = np.array(lambda_gl)

        if nh == 0:
            result[f'{p}weights'] = final_model.linear.weight.detach().cpu().numpy().ravel()
            result[f'{p}bias'] = final_model.linear.bias.detach().cpu().numpy().ravel()
        else:
            result[f'{p}hidden_weights'] = final_model.hidden.weight.detach().cpu().numpy()
            result[f'{p}hidden_bias'] = final_model.hidden.bias.detach().cpu().numpy()
            result[f'{p}output_weights'] = final_model.output.weight.detach().cpu().numpy().ravel()
            result[f'{p}output_bias'] = final_model.output.bias.detach().cpu().numpy().ravel()

        for g in group_names:
            result[f'{p}full_r_group_{g}'] = all_res[nh]['full_r_group'][g]
            result[f'{p}permuted_r_{g}'] = all_res[nh]['permuted_r'][g]
        for ck in combo_keys:
            result[f'{p}interaction_r_{ck}'] = all_res[nh]['interaction_r'][ck]

    return result


def fit_neuron_from_disk(sess_dir, neuron_idx, ops=NETWORK_OPTIONS):
    """load prepped data, fit one neuron, save results"""
    import pickle

    counts, X, col_map, t_ax, valid_mask = load_glm_inputs(sess_dir)
    y = counts[neuron_idx]

    sess = Session.load(str(Path(sess_dir) / 'session.pkl'))
    fold_ids = get_trial_fold_indices(
        sess.trials, t_ax, ops['n_outer_folds'],
        seed=neuron_seed(sess_dir, neuron_idx),
        ignore_first_n=ANALYSIS_OPTIONS['ignore_first_trials_in_block'])

    result = fit_neuron(y, X, col_map, fold_ids, sess.trials, t_ax, ops)

    results_dir = Path(sess_dir) / 'network_results'
    results_dir.mkdir(exist_ok=True)
    np.savez(results_dir / f'neuron_{neuron_idx}.npz', **result)

    # save col_map once per session
    col_map_path = results_dir / 'col_map.pkl'
    if not col_map_path.exists():
        with open(col_map_path, 'wb') as f:
            pickle.dump(col_map, f)
