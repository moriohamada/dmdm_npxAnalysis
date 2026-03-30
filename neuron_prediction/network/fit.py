"""training loop and CV for poisson linear and network models"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from config import NETWORK_OPTIONS
from data.session import Session
from neuron_prediction.data import (
    load_glm_inputs, get_fold_indices, neuron_seed, normalise_design_matrix,
)
from neuron_prediction.evaluate import pearson_r, lesion_design_matrix
from neuron_prediction.network.model import (
    PoissonLinear, PoissonNet, proximal_group_lasso, ortho_penalty,
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
              lambda_gl=0.0, lambda_ortho=0.0):
    """train with plain SGD + proximal group lasso + early stopping

    smooth loss (Poisson NLL + ortho penalty) updated via SGD.
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

            # smooth loss: Poisson NLL + ortho penalty
            loss = poisson_loss(log_rate, y_tr[idx])
            if lambda_ortho > 0:
                loss = loss + lambda_ortho * ortho_penalty(model)

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


def inner_cv_select(X_train, y_train, col_map, ops, seed=0):
    """select best hyperparams via inner CV on training fold

    returns best (n_hidden, lambda_gl, lambda_ortho)
    """
    n_inner = ops['n_inner_folds']
    n = len(X_train)
    inner_folds = get_fold_indices(n, n_inner, seed=seed)

    best_loss = float('inf')
    best_params = (ops['hidden_sizes'][0], 0, 0)

    # count total configs for progress
    configs = [(nh, gl, ort)
               for nh in ops['hidden_sizes']
               for gl in ops['group_lasso_lambdas']
               for ort in ops['ortho_lambdas']
               if not (nh == 0 and ort > 0)]
    n_configs = len(configs)

    for ci, (n_hidden, lambda_gl, lambda_ortho) in enumerate(configs):
        fold_losses = []
        for k in range(n_inner):
            val_mask = inner_folds == k
            tr_mask = ~val_mask

            model = _make_model(X_train.shape[1], n_hidden)
            _, val_loss = train_one(
                model, X_train[tr_mask], y_train[tr_mask],
                col_map, ops, lambda_gl, lambda_ortho)
            fold_losses.append(val_loss)

        mean_loss = np.mean(fold_losses)
        print(f'    config {ci+1}/{n_configs}: h={n_hidden} gl={lambda_gl} ort={lambda_ortho} -> loss={mean_loss:.4f}')
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_params = (n_hidden, lambda_gl, lambda_ortho)

    return best_params


def _eval_model(model, X_test, y_test, ev_masks_v, lesion_groups, col_map, device):
    """evaluate a fitted model: full r, window r, lesion r"""
    group_names = list(lesion_groups.keys())
    y_pred = _predict_numpy(model, X_test, device)
    r = pearson_r(y_test, y_pred)

    r_window = {}
    r_lesioned = {}
    r_lesioned_window = {}
    for gname, pred_list in lesion_groups.items():
        if gname in ev_masks_v:
            win_mask = ev_masks_v[gname]
            if win_mask.sum() >= 5:
                r_window[gname] = pearson_r(y_test[win_mask], y_pred[win_mask])

        X_les = lesion_design_matrix(X_test, pred_list, col_map)
        y_pred_les = _predict_numpy(model, X_les, device)
        r_lesioned[gname] = pearson_r(y_test, y_pred_les)

        if gname in ev_masks_v:
            win_mask = ev_masks_v[gname]
            if win_mask.sum() >= 5:
                r_lesioned_window[gname] = pearson_r(
                    y_test[win_mask], y_pred_les[win_mask])

    return r, r_window, r_lesioned, r_lesioned_window


def fit_neuron(counts_1d, X, col_map, valid_mask, event_masks=None,
               ops=NETWORK_OPTIONS, seed=0):
    """fit both linear and network poisson models for one neuron

    always fits a linear model (PoissonLinear) and the best network
    model selected by inner CV. both are evaluated on the same folds.
    returns dict with 'linear' and 'network' sub-dicts.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_folds = ops['n_outer_folds']
    lesion_groups = ops['lesion_groups']
    group_names = list(lesion_groups.keys())

    X_v = X[valid_mask]
    y_v = counts_1d[valid_mask].astype(np.float64)
    T_v = len(y_v)

    fold_ids = get_fold_indices(T_v, n_folds, seed)

    # results for both model types
    def _empty_results():
        return {
            'full_r': np.full(n_folds, np.nan),
            'lesioned_r': {g: np.full(n_folds, np.nan) for g in group_names},
            'full_r_window': {g: np.full(n_folds, np.nan) for g in group_names},
            'lesioned_r_window': {g: np.full(n_folds, np.nan) for g in group_names},
        }
    lin_res = _empty_results()
    net_res = _empty_results()

    ev_masks_v = {}
    if event_masks is not None:
        for gname in group_names:
            if gname in event_masks:
                ev_masks_v[gname] = event_masks[gname][valid_mask]

    # network hidden sizes (exclude 0 - linear is always fit separately)
    net_hidden_sizes = [h for h in ops['hidden_sizes'] if h > 0]
    net_ops = dict(ops, hidden_sizes=net_hidden_sizes)
    has_network = len(net_hidden_sizes) > 0

    best_net_params_per_fold = []

    for k in range(n_folds):
        print(f'Outer fold {k+1}/{n_folds}')
        test_mask = fold_ids == k
        train_mask = ~test_mask

        X_train_raw, y_train = X_v[train_mask], y_v[train_mask]
        X_test_raw, y_test = X_v[test_mask], y_v[test_mask]

        if y_train.sum() == 0 or y_test.sum() == 0:
            continue

        X_train, X_test, _, _ = normalise_design_matrix(
            X_train_raw, X_test_raw, col_map)

        # slice event masks to this fold's test set
        ev_test = {}
        for gname in ev_masks_v:
            ev_test[gname] = ev_masks_v[gname][test_mask]

        # always fit linear
        print(f'  fitting linear...')
        lin_model = PoissonLinear(X_train.shape[1])
        lin_model, _ = train_one(lin_model, X_train, y_train, col_map, ops)
        r, r_w, r_l, r_lw = _eval_model(
            lin_model, X_test, y_test, ev_test, lesion_groups, col_map, device)
        lin_res['full_r'][k] = r
        for g in group_names:
            if g in r_w: lin_res['full_r_window'][g][k] = r_w[g]
            if g in r_l: lin_res['lesioned_r'][g][k] = r_l[g]
            if g in r_lw: lin_res['lesioned_r_window'][g][k] = r_lw[g]
        print(f'  linear r={r:.3f}')

        # fit best network via inner CV
        if has_network:
            print(f'  inner CV for network...')
            n_hidden, lambda_gl, lambda_ortho = inner_cv_select(
                X_train, y_train, col_map, net_ops,
                seed=seed + k * 1000)
            best_net_params_per_fold.append((n_hidden, lambda_gl, lambda_ortho))
            print(f'  best: n_hidden={n_hidden}, gl={lambda_gl}, ortho={lambda_ortho}')

            net_model = PoissonNet(X_train.shape[1], n_hidden)
            net_model, _ = train_one(net_model, X_train, y_train, col_map, ops,
                                     lambda_gl, lambda_ortho)
            r, r_w, r_l, r_lw = _eval_model(
                net_model, X_test, y_test, ev_test, lesion_groups, col_map, device)
            net_res['full_r'][k] = r
            for g in group_names:
                if g in r_w: net_res['full_r_window'][g][k] = r_w[g]
                if g in r_l: net_res['lesioned_r'][g][k] = r_l[g]
                if g in r_lw: net_res['lesioned_r_window'][g][k] = r_lw[g]
            print(f'  network r={r:.3f}')

    # fit final models on all valid data for weight extraction
    X_v_norm, _, _, _ = normalise_design_matrix(X_v, X_v, col_map)

    print('Fitting final linear model on all data...')
    final_lin = PoissonLinear(X_v.shape[1])
    final_lin, _ = train_one(final_lin, X_v_norm, y_v, col_map, ops)

    result = {
        'linear': {
            'full_r': lin_res['full_r'],
            'weights': final_lin.linear.weight.detach().cpu().numpy().ravel(),
            'bias': final_lin.linear.bias.detach().cpu().numpy().ravel(),
        },
    }
    for g in group_names:
        result['linear'][f'lesioned_r_{g}'] = lin_res['lesioned_r'][g]
        result['linear'][f'full_r_window_{g}'] = lin_res['full_r_window'][g]
        result['linear'][f'lesioned_r_window_{g}'] = lin_res['lesioned_r_window'][g]

    if has_network and best_net_params_per_fold:
        from collections import Counter
        n_hidden, lambda_gl, lambda_ortho = Counter(best_net_params_per_fold).most_common(1)[0][0]

        print(f'Fitting final network (h={n_hidden}) on all data...')
        final_net = PoissonNet(X_v.shape[1], n_hidden)
        final_net, _ = train_one(final_net, X_v_norm, y_v, col_map, ops,
                                  lambda_gl, lambda_ortho)

        result['network'] = {
            'full_r': net_res['full_r'],
            'n_hidden': n_hidden,
            'lambda_gl': lambda_gl,
            'lambda_ortho': lambda_ortho,
            'hidden_weights': final_net.hidden.weight.detach().cpu().numpy(),
            'hidden_bias': final_net.hidden.bias.detach().cpu().numpy(),
            'output_weights': final_net.output.weight.detach().cpu().numpy().ravel(),
            'output_bias': final_net.output.bias.detach().cpu().numpy().ravel(),
        }
        for g in group_names:
            result['network'][f'lesioned_r_{g}'] = net_res['lesioned_r'][g]
            result['network'][f'full_r_window_{g}'] = net_res['full_r_window'][g]
            result['network'][f'lesioned_r_window_{g}'] = net_res['lesioned_r_window'][g]

    return result


def fit_neuron_from_disk(sess_dir, neuron_idx, ops=NETWORK_OPTIONS):
    """load prepped data, fit one neuron, save results"""
    from neuron_prediction.glm.fit import build_event_masks

    counts, X, col_map, t_ax, valid_mask = load_glm_inputs(sess_dir)
    y = counts[neuron_idx]

    sess = Session.load(str(Path(sess_dir) / 'session.pkl'))
    event_masks = build_event_masks(sess, t_ax)

    result = fit_neuron(y, X, col_map, valid_mask, event_masks,
                        ops, seed=neuron_seed(sess_dir, neuron_idx))

    results_dir = Path(sess_dir) / 'network_results'
    results_dir.mkdir(exist_ok=True)
    np.savez(results_dir / f'neuron_{neuron_idx}.npz', **result)
