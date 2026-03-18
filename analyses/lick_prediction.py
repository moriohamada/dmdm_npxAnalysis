"""
Lick prediction model: logistic regression and single-hidden-layer network.
Predicts P(lick) at each 50ms bin from stimulus, trial history, and state features.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from config import LICK_PRED_OPS
from data.lick_features import FEATURE_COLS, CONTINUOUS_COLS, N_FEATURES


class LinearLickModel(nn.Module):
    def __init__(self, n_features=N_FEATURES):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class NetworkLickModel(nn.Module):
    def __init__(self, n_features=N_FEATURES, n_hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_class_weight(y):
    n_pos = (y > 0).sum()
    n_neg = (y <= 0).sum()
    return n_neg / n_pos


def _normalise_features(X_train, X_test):
    """
    Z-score continuous features, leave binary/one-hot as-is.
    Which columns are continuous is defined in data.lick_features.CONTINUOUS_COLS.
    """
    mu = X_train[:, CONTINUOUS_COLS].mean(axis=0)
    sd = X_train[:, CONTINUOUS_COLS].std(axis=0)
    sd[sd == 0] = 1.0

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[:, CONTINUOUS_COLS] = (X_train[:, CONTINUOUS_COLS] - mu) / sd
    X_test[:, CONTINUOUS_COLS] = (X_test[:, CONTINUOUS_COLS] - mu) / sd

    return X_train, X_test, mu, sd


def train_model(model, X_train, y_train, pos_weight,
                ops=LICK_PRED_OPS, weight_decay=0.0):
    """
    Train with early stopping on a held-out fraction of training data.
    Returns trained model and training history dict.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    n = len(X_train)
    n_val = max(1, int(n * ops['val_frac']))
    perm = np.random.permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    X_tr = torch.tensor(X_train[train_idx], dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train[train_idx], dtype=torch.float32, device=device)
    X_val = torch.tensor(X_train[val_idx], dtype=torch.float32, device=device)
    y_val = torch.tensor(y_train[val_idx], dtype=torch.float32, device=device)

    pw = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    optimiser = torch.optim.Adam(model.parameters(), lr=ops['lr'],
                                  weight_decay=weight_decay)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    train_losses, val_losses = [], []
    batch_size = ops.get('batch_size', 4096)
    n_tr = len(train_idx)

    for epoch in range(ops['max_epochs']):
        # mini-batch training
        model.train()
        perm_tr = np.random.permutation(n_tr)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_tr, batch_size):
            batch_idx = perm_tr[start:start + batch_size]
            X_batch = X_tr[batch_idx]
            y_batch = y_tr[batch_idx]

            optimiser.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_losses.append(epoch_loss / n_batches)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= ops['patience']:
            break

    model.load_state_dict(best_state)
    model.eval()

    return model, dict(best_val_loss=best_val_loss, epochs=epoch + 1,
                       train_losses=train_losses, val_losses=val_losses)


def predict(model, X):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        logits = model(X_t)
        return torch.sigmoid(logits).cpu().numpy()


def evaluate(model, X_test, y_test, pos_weight):
    """weighted BCE loss on test data"""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_test, dtype=torch.float32, device=device)
        pw = torch.tensor([pos_weight], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        logits = model(X_t)
        return criterion(logits, y_t).item()


#%% cross-validation and hyperparameter sweep

def leave_one_out_cv(sessions_data, model_class, n_hidden=None,
                     weight_decay=0.0, ops=LICK_PRED_OPS):
    """
    Leave-one-session-out CV. Returns test loss per fold.

    sessions_data: list of (X, y, trial_ids) per session
    model_class: LinearLickModel or NetworkLickModel
    """
    n_sessions = len(sessions_data)
    test_losses = np.full(n_sessions, np.nan)
    loss_history = []

    all_y = np.concatenate([d[1] for d in sessions_data])
    pos_weight = compute_class_weight(all_y)

    for i in range(n_sessions):
        X_train = np.vstack([sessions_data[j][0] for j in range(n_sessions) if j != i])
        y_train = np.concatenate([sessions_data[j][1] for j in range(n_sessions) if j != i])
        X_test, y_test = sessions_data[i][0], sessions_data[i][1]

        X_train, X_test, _, _ = _normalise_features(X_train, X_test)

        if model_class == NetworkLickModel:
            model = model_class(n_features=X_train.shape[1], n_hidden=n_hidden)
        else:
            model = model_class(n_features=X_train.shape[1])

        model, hist = train_model(model, X_train, y_train, pos_weight,
                                   ops=ops, weight_decay=weight_decay)
        test_losses[i] = evaluate(model, X_test, y_test, pos_weight)
        loss_history.append(hist)

    return test_losses, loss_history


def _save_loss_curves(loss_history, key, save_dir):
    """save train/val loss curves for all CV folds of one config"""
    import matplotlib.pyplot as plt
    n_folds = len(loss_history)
    fig, axes = plt.subplots(1, n_folds, figsize=(3 * n_folds, 3), squeeze=False)
    for i, hist in enumerate(loss_history):
        ax = axes[0, i]
        ax.plot(hist['train_losses'], label='train', alpha=0.7)
        ax.plot(hist['val_losses'], label='val', alpha=0.7)
        ax.set_title(f'fold {i}')
        if i == 0:
            ax.set_ylabel('Loss')
            ax.legend(fontsize=7)
    axes[0, n_folds // 2].set_xlabel('Epoch')
    fig.suptitle(key, fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{key}.png'), dpi=100)
    plt.close(fig)


def hyperparameter_sweep(sessions_data, ops=LICK_PRED_OPS, save_dir=None):
    """
    sweep hidden sizes and weight decays for the network model
    also fits the linear baseline. returns dict with results per config
    """
    results = {}

    for wd in ops['lambdas']:
        key = f'linear_lambda{wd}'
        losses, loss_history = leave_one_out_cv(sessions_data, LinearLickModel,
                                                 weight_decay=wd, ops=ops)
        results[key] = dict(
            model='linear', weight_decay=wd,
            test_losses=losses, mean_loss=np.nanmean(losses))
        print(f'  {key}: mean_loss={np.nanmean(losses):.4f}')
        if save_dir:
            _save_loss_curves(loss_history, key, save_dir)

    for nh in ops['hidden_sizes']:
        for wd in ops['lambdas']:
            key = f'network_h{nh}_lambda{wd}'
            losses, loss_history = leave_one_out_cv(sessions_data, NetworkLickModel,
                                                     n_hidden=nh, weight_decay=wd, ops=ops)
            results[key] = dict(
                model='network', n_hidden=nh, weight_decay=wd,
                test_losses=losses, mean_loss=np.nanmean(losses))
            print(f'  {key}: mean_loss={np.nanmean(losses):.4f}')
            if save_dir:
                _save_loss_curves(loss_history, key, save_dir)

    return results


def fit_best_model(sessions_data, sweep_results, ops=LICK_PRED_OPS):
    """
    refit the best config on all data.
    returns model, normalisation params (mu, sd), and the best config key.
    """
    best_key = min(sweep_results, key=lambda k: sweep_results[k]['mean_loss'])
    cfg = sweep_results[best_key]
    print(f'Best config: {best_key} (mean_loss={cfg["mean_loss"]:.4f})')

    X_all = np.vstack([d[0] for d in sessions_data])
    y_all = np.concatenate([d[1] for d in sessions_data])
    pos_weight = compute_class_weight(y_all)

    mu = X_all[:, CONTINUOUS_COLS].mean(axis=0)
    sd = X_all[:, CONTINUOUS_COLS].std(axis=0)
    sd[sd == 0] = 1.0
    X_normed = X_all.copy()
    X_normed[:, CONTINUOUS_COLS] = (X_normed[:, CONTINUOUS_COLS] - mu) / sd

    if cfg['model'] == 'network':
        model = NetworkLickModel(n_features=X_normed.shape[1], n_hidden=cfg['n_hidden'])
    else:
        model = LinearLickModel(n_features=X_normed.shape[1])

    model, _ = train_model(model, X_normed, y_all, pos_weight,
                           ops=ops, weight_decay=cfg['weight_decay'])

    return model, mu, sd, best_key


#%% interpretation

def ablation_analysis(model, sessions_data, mu, sd, ops=LICK_PRED_OPS):
    """
    zero out each feature group and measure increase in test loss.
    returns dict mapping group name to (n_sessions,) loss increases.
    """
    n_sessions = len(sessions_data)
    all_y = np.concatenate([d[1] for d in sessions_data])
    pos_weight = compute_class_weight(all_y)

    baseline_losses = np.full(n_sessions, np.nan)
    for i in range(n_sessions):
        X_test = sessions_data[i][0].copy()
        X_test[:, CONTINUOUS_COLS] = (X_test[:, CONTINUOUS_COLS] - mu) / sd
        baseline_losses[i] = evaluate(model, X_test, sessions_data[i][1], pos_weight)

    results = {}
    for group_name, cols in FEATURE_COLS.items():
        ablated_losses = np.full(n_sessions, np.nan)
        for i in range(n_sessions):
            X_test = sessions_data[i][0].copy()
            X_test[:, CONTINUOUS_COLS] = (X_test[:, CONTINUOUS_COLS] - mu) / sd
            X_test[:, cols] = 0.0
            ablated_losses[i] = evaluate(model, X_test, sessions_data[i][1], pos_weight)
        results[group_name] = ablated_losses - baseline_losses

    return results, baseline_losses


def extract_stimulus_filter(model):
    """extract the 40 stimulus history weights from a LinearLickModel."""
    with torch.no_grad():
        w = model.linear.weight[0].cpu().numpy()
    return w[:len(FEATURE_COLS['stimulus'])]
