"""
predict P(lick) from demixing latent activations.
leave-one-session-out CV per mouse.
"""
import os
import pickle
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict

from config import DEMIXING_OPTIONS, LICK_PRED_OPS
from data.session import Session
from utils.filing import get_session_files
from utils.downsampling import downsample_bins
from demixing.analysis import load_latents
from lick_pred.features import _trial_lick_time, _compute_motion_lick_delay
from lick_pred.models import (LinearLickModel, train_model, evaluate,
                              compute_class_weight)


def _build_lick_target(t_ax, lick_time, sigma_s=0.25):
    """gaussian-smoothed P(lick) target on given time axis"""
    y = np.zeros(len(t_ax))
    if not np.isnan(lick_time):
        lick_bin = np.argmin(np.abs(t_ax - lick_time))
        y[lick_bin] = 1.0
        bin_s = np.mean(np.diff(t_ax))
        sigma_bins = max(1, sigma_s / bin_s)
        y = gaussian_filter1d(y, sigma=sigma_bins)
        if y.max() > 0:
            y /= y.max()
    return y


def _downsample_trial(z, t_ax, target_bin_s=0.05):
    """downsample latent + time axis to target bin size"""
    bin_s = np.mean(np.diff(t_ax))
    factor = max(1, round(target_bin_s / bin_s))
    if factor <= 1:
        return z, t_ax
    z_ds = downsample_bins(z, factor, axis=0)
    t_ds = downsample_bins(t_ax, factor, axis=0)
    return z_ds, t_ds


def build_session_latent_features(latent_data, session, target_bin_s=0.05,
                                  sigma_s=0.25, truncate_at_change=True):
    """
    build feature matrix (latent activations) and lick targets for one session.
    returns (X, y, trial_ids) or (None, None, None) if no valid trials
    """
    motion_lick_delay = _compute_motion_lick_delay(session.trials)

    Xs, ys, ids = [], [], []

    for trial_idx, tid in enumerate(latent_data.trial_ids):
        z = latent_data.z_all[trial_idx]
        t_ax = latent_data.get_trial_t_ax(tid)

        # find the matching trial row
        if tid not in session.trials.index:
            continue
        row = session.trials.loc[tid]

        if row.get('IsAbort', False):
            continue
        if row.get('trialoutcome') == 'Ref':
            continue

        bl_on = row['Baseline_ON_rise']
        lick_t = _trial_lick_time(row, motion_lick_delay)

        # truncate at change onset if needed
        if truncate_at_change and (row.get('IsHit', False) or row.get('IsMiss', False)):
            change_t = bl_on + row['stimT']
            keep = t_ax < change_t
            if keep.sum() < 2:
                continue
            z = z[keep]
            t_ax = t_ax[keep]

        # downsample to target bin size
        z_ds, t_ds = _downsample_trial(z, t_ax, target_bin_s)
        if len(t_ds) < 2:
            continue

        y = _build_lick_target(t_ds, lick_t, sigma_s)

        Xs.append(z_ds)
        ys.append(y)
        ids.append(np.full(len(z_ds), tid))

    if not Xs:
        return None, None, None

    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ids)


def _normalise_latent_features(X_train, X_test):
    """z-score all features using training stats"""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    return (X_train - mu) / sd, (X_test - mu) / sd


def latent_leave_one_out_cv(sessions_data, latent_dim, ops=LICK_PRED_OPS):
    """
    leave-one-session-out CV for latent-based lick prediction.
    reuses train_model and evaluate from lick_pred.models.
    """
    n_sessions = len(sessions_data)
    test_losses = np.full(n_sessions, np.nan)

    all_y = np.concatenate([d[1] for d in sessions_data])
    pos_weight = compute_class_weight(all_y)

    for i in range(n_sessions):
        X_train = np.vstack([sessions_data[j][0]
                             for j in range(n_sessions) if j != i])
        y_train = np.concatenate([sessions_data[j][1]
                                  for j in range(n_sessions) if j != i])
        X_test, y_test = sessions_data[i][0], sessions_data[i][1]

        X_train, X_test = _normalise_latent_features(X_train, X_test)

        model = LinearLickModel(n_features=latent_dim)
        model, _ = train_model(model, X_train, y_train, pos_weight, ops=ops)
        test_losses[i] = evaluate(model, X_test, y_test, pos_weight)

    return test_losses


def run_latent_lick_prediction(npx_dir, ops=DEMIXING_OPTIONS,
                               lick_ops=LICK_PRED_OPS, save_dir=None):
    """run latent-based lick prediction for all mice"""
    if save_dir is None:
        save_dir = os.path.join(npx_dir, 'demixing_lick_prediction')
    os.makedirs(save_dir, exist_ok=True)

    model_type = ops['model_type']
    latent_dim = ops['latent_dim']

    # group sessions by mouse
    grouped = defaultdict(list)
    for p in get_session_files(npx_dir):
        animal = Path(p).parent.parent.name
        sess_dir = str(Path(p).parent)
        latent_path = Path(sess_dir) / f'demixing_{model_type}_latents.h5'
        if latent_path.exists():
            grouped[animal].append(sess_dir)

    for animal, sess_dirs in sorted(grouped.items()):
        print(f'\n{animal} ({len(sess_dirs)} sessions)')

        sessions_data = []
        session_names = []

        for sess_dir in sess_dirs:
            latent_data = load_latents(sess_dir, model_type)
            session = Session.load(str(Path(sess_dir) / 'session.pkl'))

            X, y, trial_ids = build_session_latent_features(latent_data, session)
            if X is None:
                continue

            sessions_data.append((X, y, trial_ids))
            session_names.append(session.name)
            print(f'  {session.name}: {X.shape[0]} bins, '
                  f'{(y > 0).sum()} lick bins')

        if len(sessions_data) < 3:
            print(f'  skipping {animal}: too few sessions')
            continue

        test_losses = latent_leave_one_out_cv(sessions_data, latent_dim, lick_ops)
        print(f'  mean test loss: {np.nanmean(test_losses):.4f}')

        result = dict(
            animal=animal,
            session_names=session_names,
            test_losses=test_losses,
            mean_loss=np.nanmean(test_losses),
            model_type=model_type,
            latent_dim=latent_dim,
        )

        save_path = os.path.join(save_dir, f'{animal}_latent_lick.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)
        print(f'  saved to {save_path}')
