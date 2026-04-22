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
    pearson_r, permute_design_matrix, interaction_combo_key,
)
from neuron_prediction.network.model import (
    PoissonLinear, PoissonNet, proximal_group_lasso,
)
from neuron_prediction.results.peth import build_event_spec, fold_peths
from config import GLM_OPTIONS


def _to_tensors(X, y, device):
    return (torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(y, dtype=torch.float32, device=device))


def _make_model(n_inputs, n_hidden):
    """create PoissonLinear (n_hidden=0) or PoissonNet"""
    if n_hidden == 0:
        return PoissonLinear(n_inputs)
    return PoissonNet(n_inputs, n_hidden)


def train_one(model, X_train, y_train, col_map, ops,
              lambda_gl=0.0, track_loss=False):
    """train with plain SGD + proximal group lasso + early stopping

    smooth loss (Poisson NLL) updated via SGD.
    non-smooth penalty (group lasso) applied via proximal operator after each step.
    returns (model, best_val_loss) or (model, best_val_loss, loss_history) if track_loss.
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
    train_losses = [] if track_loss else None
    val_losses = [] if track_loss else None

    for epoch in range(ops['max_epochs']):
        model.train()
        perm_tr = torch.randperm(len(X_tr), device=device)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_loss = 0.0
        n_batches = 0
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

            if track_loss:
                epoch_loss += loss.item()
                n_batches += 1

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_log_rate = model(X_val)
            val_loss = poisson_loss(val_log_rate, y_val).item()

        if track_loss:
            train_losses.append(epoch_loss / max(n_batches, 1))
            val_losses.append(val_loss)

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

    if track_loss:
        return model, best_val_loss, {'train': train_losses, 'val': val_losses}
    return model, best_val_loss


def _predict_numpy(model, X, device):
    """run forward pass, return predicted counts as numpy array"""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        log_rate = model(X_t)
        return torch.exp(log_rate).cpu().numpy()


# too slow - drop nested cv
# def _fit_one_inner(X_train, y_train, col_map, n_hidden, ops,
#                    inner_folds, lambda_gl, k):
#     """fit and evaluate one (lambda, inner fold) pair — for parallel dispatch"""
#     val_mask = inner_folds == k
#     tr_mask = (inner_folds >= 0) & ~val_mask
#
#     if val_mask.sum() == 0 or tr_mask.sum() == 0:
#         return lambda_gl, k, np.nan, np.inf
#
#     cv_ops = {**ops,
#               'max_epochs': ops.get('cv_max_epochs', ops['max_epochs']),
#               'patience': ops.get('cv_patience', ops['patience'])}
#
#     model = _make_model(X_train.shape[1], n_hidden)
#     model, _ = train_one(
#         model, X_train[tr_mask], y_train[tr_mask],
#         col_map, cv_ops, lambda_gl)
#
#     device = next(model.parameters()).device
#     poisson_loss_fn = nn.PoissonNLLLoss(log_input=True)
#     with torch.no_grad():
#         X_val_t = torch.tensor(
#             X_train[val_mask], dtype=torch.float32, device=device)
#         y_val_t = torch.tensor(
#             y_train[val_mask], dtype=torch.float32, device=device)
#         log_rate = model(X_val_t)
#         val_loss = poisson_loss_fn(log_rate, y_val_t).item()
#         y_pred = torch.exp(log_rate).cpu().numpy()
#
#     r = pearson_r(y_train[val_mask], y_pred)
#     return lambda_gl, k, r, val_loss
#
#
# def inner_cv_select(X_train, y_train, col_map, n_hidden, ops,
#                     trials_df, t_ax, outer_valid, outer_train_mask, seed=0):
#     """select best regularisation for a given hidden size via inner CV"""
#     from joblib import Parallel, delayed
#
#     n_inner = ops['n_inner_folds']
#     n_jobs = ops.get('n_jobs', 1)
#
#     inner_fold_ids = get_trial_fold_indices(
#         trials_df, t_ax, n_inner, seed=seed,
#         ignore_first_n=ANALYSIS_OPTIONS['ignore_first_trials_in_block'])
#     inner_folds = inner_fold_ids[outer_valid][outer_train_mask]
#
#     configs = ops['group_lasso_lambdas']
#
#     results = Parallel(n_jobs=n_jobs)(
#         delayed(_fit_one_inner)(
#             X_train, y_train, col_map, n_hidden, ops,
#             inner_folds, lam, k)
#         for lam in configs
#         for k in range(n_inner)
#     )
#
#     best_r = -np.inf
#     best_params = 0
#     for ci, lambda_gl in enumerate(configs):
#         fold_rs = [r for lam, k, r, loss in results if lam == lambda_gl]
#         fold_losses = [loss for lam, k, r, loss in results if lam == lambda_gl]
#         mean_r = np.nanmean(fold_rs) if fold_rs else np.nan
#         mean_loss = np.mean(fold_losses) if fold_losses else np.inf
#         print(f'    config {ci+1}/{len(configs)}: gl={lambda_gl} '
#               f'-> r={mean_r:.4f}, loss={mean_loss:.4f}')
#         if not np.isnan(mean_r) and mean_r > best_r:
#             best_r = mean_r
#             best_params = lambda_gl
#
#     return best_params


def _eval_model(model, X_test, y_test, group_masks_test, lesion_groups,
                col_map, interaction_combos, device, n_perm=10, seed=0):
    """full r and per-group permutation importance

    # shuffle regressors instead - can't just zero for network!

    group_masks_test: dict of bool arrays marking bins where each group's
    predictors are non-zero (already sliced to test fold)
    interaction_combos: list of tuples specifying which joint permutations to run
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

    # interaction permutation (joint shuffle of group combinations)
    r_interaction = {}
    for combo in interaction_combos:
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


def _run_one_fold(k, X_v, y_v, folds_v, col_map, group_masks,
                  hidden_sizes, lambdas, lesion_groups, combos, ops):
    """fit all (hidden_size, lambda) pairs for one outer fold

    returns dict with per-hidden-size results: best_lambda, full_r,
    per-group r, permutation r, interaction r.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    group_names = list(lesion_groups.keys())

    test_mask = folds_v == k
    train_mask = ~test_mask

    X_train_raw, y_train = X_v[train_mask], y_v[train_mask]
    X_test_raw, y_test = X_v[test_mask], y_v[test_mask]

    if y_train.sum() == 0 or y_test.sum() == 0:
        return None

    X_train, X_test, _, _ = normalise_design_matrix(
        X_train_raw, X_test_raw, col_map)

    gm_test = {g: group_masks[g][test_mask] for g in group_names}

    fold_out = {}
    for nh in hidden_sizes:
        # too slow - drop nested cv
        # try all lambdas, pick best on test fold
        best_r = -np.inf
        best_lambda = lambdas[0]
        best_model = None

        for lam in lambdas:
            model = _make_model(X_train.shape[1], nh)
            model, _ = train_one(model, X_train, y_train, col_map, ops, lam)
            y_pred = _predict_numpy(model, X_test, device)
            r = pearson_r(y_test, y_pred)
            print(f'  fold {k+1} h={nh} gl={lam}: r={r:.4f}')
            if not np.isnan(r) and r > best_r:
                best_r = r
                best_lambda = lam
                best_model = model

        print(f'  fold {k+1} h={nh}: best gl={best_lambda}, r={best_r:.3f}')

        # permutation importance on best model
        r, r_g, r_l, r_int = _eval_model(
            best_model, X_test, y_test, gm_test, lesion_groups, col_map,
            combos, device, n_perm=ops['n_perm_importance'], seed=k)

        # full + per-group permuted predictions for PETH (single shuffle
        # per group, seeded on k for reproducibility)
        y_pred_full = _predict_numpy(best_model, X_test, device)
        rng_peth = np.random.RandomState(k * 10_007 + nh)
        y_pred_red = {}
        for gname, pred_list in lesion_groups.items():
            X_perm = permute_design_matrix(X_test, pred_list, col_map,
                                           rng_peth)
            y_pred_red[gname] = _predict_numpy(best_model, X_perm, device)

        fold_out[nh] = {
            'best_lambda': best_lambda,
            'full_r': r,
            'r_group': r_g,
            'r_permuted': r_l,
            'r_interaction': r_int,
            'y_pred_full': y_pred_full,
            'y_pred_red': y_pred_red,
        }

    return fold_out


def fit_neuron(counts_1d, X, col_map, fold_ids, trials_df, t_ax,
               event_spec=None, ops=NETWORK_OPTIONS):
    """fit poisson models across all hidden sizes for one neuron

    parallelises across outer folds. for each fold, trains all lambdas
    and selects the best on the test fold directly (no inner CV).
    permutation importance runs on the best model per fold.
    final refit on all data uses the most-common best lambda across folds.

    fold_ids: (T,) int array from get_trial_fold_indices. bins with
        fold_id == -1 are excluded from all fitting and evaluation.
    event_spec: optional dict {kind: (bin_idx, signs, pre, post)} for
        paper-style PETH classification. network uses a single random
        permutation (not refit) as the reduced model.
    """
    from collections import Counter
    from joblib import Parallel, delayed

    n_folds = ops['n_outer_folds']
    n_jobs = ops.get('n_jobs', 1)
    lesion_groups = ops['lesion_groups']
    group_names = list(lesion_groups.keys())
    combos = ops['interaction_combos']
    combo_keys = [interaction_combo_key(c) for c in combos]
    hidden_sizes = ops['hidden_sizes']
    lambdas = ops['group_lasso_lambdas']

    valid = fold_ids >= 0
    X_v = X[valid]
    y_v = counts_1d[valid].astype(np.float64)
    folds_v = fold_ids[valid]
    T = len(counts_1d)
    valid_T_idx = np.where(valid)[0]

    # derive evaluation masks from design matrix
    group_masks = {}
    for gname, pred_list in lesion_groups.items():
        mask = np.zeros(X_v.shape[0], dtype=bool)
        for pred_name in pred_list:
            if pred_name in col_map:
                col_slice, _ = col_map[pred_name]
                mask |= np.any(X_v[:, col_slice] != 0, axis=1)
        group_masks[gname] = mask

    # parallel across outer folds
    fold_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_run_one_fold)(
            k, X_v, y_v, folds_v, col_map, group_masks,
            hidden_sizes, lambdas, lesion_groups, combos, ops)
        for k in range(n_folds)
    )

    # collect fold results
    all_params = {nh: [] for nh in hidden_sizes}
    all_res = {nh: {
        'full_r': np.full(n_folds, np.nan),
        'full_r_group': {g: np.full(n_folds, np.nan) for g in group_names},
        'permuted_r': {g: np.full(n_folds, np.nan) for g in group_names},
        'interaction_r': {ck: np.full(n_folds, np.nan) for ck in combo_keys},
    } for nh in hidden_sizes}

    # full-length CV predictions per hidden size, for PETHs
    y_full_cv = {nh: np.full(T, np.nan) for nh in hidden_sizes}
    y_red_cv = {nh: {g: np.full(T, np.nan) for g in group_names}
                for nh in hidden_sizes}

    for k, fold_out in enumerate(fold_results):
        if fold_out is None:
            continue
        test_mask = folds_v == k
        test_T_idx = valid_T_idx[test_mask]
        for nh in hidden_sizes:
            res_nh = fold_out[nh]
            all_params[nh].append(res_nh['best_lambda'])
            all_res[nh]['full_r'][k] = res_nh['full_r']
            for g in group_names:
                if g in res_nh['r_group']:
                    all_res[nh]['full_r_group'][g][k] = res_nh['r_group'][g]
                if g in res_nh['r_permuted']:
                    all_res[nh]['permuted_r'][g][k] = res_nh['r_permuted'][g]
            for ck in combo_keys:
                if ck in res_nh['r_interaction']:
                    all_res[nh]['interaction_r'][ck][k] = res_nh['r_interaction'][ck]
            # scatter fold predictions for PETH
            if 'y_pred_full' in res_nh:
                y_full_cv[nh][test_T_idx] = res_nh['y_pred_full']
                for g, y_red in res_nh['y_pred_red'].items():
                    y_red_cv[nh][g][test_T_idx] = y_red

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

        # per-fold PETHs for paper-style classification (prefix by hidden)
        if event_spec is not None:
            counts_f = counts_1d.astype(np.float64)
            for kind, (bin_idx, signs, pre, post) in event_spec.items():
                if kind not in y_red_cv[nh]:
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
                        counts_f, y_full_cv[nh], y_red_cv[nh][kind],
                        bin_idx, signs, fold_ids, k, pre, post)

                result[f'{p}peth_{kind}_actual_fast'] = pa_fast
                result[f'{p}peth_{kind}_actual_slow'] = pa_slow
                result[f'{p}peth_{kind}_full_fast'] = pf_fast
                result[f'{p}peth_{kind}_full_slow'] = pf_slow
                result[f'{p}peth_{kind}_reduced_fast'] = pr_fast
                result[f'{p}peth_{kind}_reduced_slow'] = pr_slow

    return result


def fit_neuron_from_disk(sess_dir, neuron_idx, ops=NETWORK_OPTIONS,
                         overwrite=False):
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

    results_dir = Path(sess_dir) / 'network_results'
    results_dir.mkdir(exist_ok=True)

    # merge with existing results so new hidden sizes don't overwrite old ones
    res_path = results_dir / f'neuron_{neuron_idx}.npz'
    if not overwrite and res_path.exists():
        existing = dict(np.load(res_path, allow_pickle=True))
        existing.update(result)
        result = existing

    np.savez(res_path, **result)
    print(f'Saved {res_path}')

    # always rewrite col_map to stay in sync with the weights
    col_map_path = results_dir / 'col_map.pkl'
    with open(col_map_path, 'wb') as f:
        pickle.dump(col_map, f)
