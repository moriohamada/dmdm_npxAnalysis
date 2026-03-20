"""
extracting and aligning latent representations from demixing models
"""
import numpy as np
import pandas as pd
import torch
import h5py
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import DataLoader
from demixing.dataset import SpikeData, collate_trials
from demixing.models import SAE


@dataclass
class LatentData:
    """lightweight stand-in for SpikeData when working from saved latents"""
    z_all: list         # list of (T_i, latent_dim) arrays
    trial_ids: list
    t_axes: dict        # trial_id -> (T_i,) time axis
    bin_size_ms: float

    def get_trial_t_ax(self, trial_id):
        return self.t_axes[trial_id]


def extract_latents(dataset, model, batch_size=1):
    """run trained model on all trials, return list of (T_i, latent_dim) arrays"""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, collate_fn=collate_trials)
    z_all = []
    with torch.no_grad():
        for data in loader:
            if isinstance(model, SAE):
                _, z = model(data['input'])
            else:
                _, z, _ = model(data['input'], stimulus=data.get('stimulus'))
            mask = data['mask']
            for i in range(z.shape[0]):
                z_all.append(z[i, mask[i]].numpy())
    return z_all


def save_latents(z_all, dataset, model_type, save_path):
    """save per-trial latents + time axes to HDF5"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(save_path, 'w') as f:
        lat_grp = f.create_group('latents')
        tax_grp = f.create_group('t_ax')
        for i, tid in enumerate(dataset.trial_ids):
            lat_grp.create_dataset(str(tid), data=z_all[i])
            tax_grp.create_dataset(str(tid), data=dataset.get_trial_t_ax(tid))
        f.create_dataset('trial_ids', data=np.array(dataset.trial_ids))
        f.attrs['bin_size_ms'] = dataset.bin_size_ms
        f.attrs['latent_dim'] = z_all[0].shape[1]
        f.attrs['model_type'] = model_type


def load_latents(sess_dir, model_type='sae'):
    """load saved latents into a LatentData object"""
    path = Path(sess_dir) / f'demixing_{model_type}_latents.h5'
    with h5py.File(path, 'r') as f:
        trial_ids = f['trial_ids'][:].tolist()
        z_all = [f[f'latents/{tid}'][:] for tid in trial_ids]
        t_axes = {tid: f[f't_ax/{tid}'][:] for tid in trial_ids}
        bin_size_ms = f.attrs['bin_size_ms']

    return LatentData(z_all=z_all, trial_ids=trial_ids,
                      t_axes=t_axes, bin_size_ms=bin_size_ms)


#%% generic alignment

def align_latents_to_events(z_all, dataset, event_times, event_trials,
                            pre_s=0.5, post_s=1.5):
    """
    align per-trial latent trajectories to arbitrary events.
    dataset can be a SpikeData or LatentData (anything with trial_ids,
    bin_size_ms, get_trial_t_ax).
    returns (n_valid, pre+post bins, latent_dim) array and time axis.
    """
    bin_s = dataset.bin_size_ms / 1000.0
    pre_bins = int(round(pre_s / bin_s))
    post_bins = int(round(post_s / bin_s))

    tid_to_idx = {tid: i for i, tid in enumerate(dataset.trial_ids)}

    aligned = []
    for t_event, tr in zip(event_times, event_trials):
        if tr not in tid_to_idx:
            continue

        z = z_all[tid_to_idx[tr]]
        trial_t = dataset.get_trial_t_ax(tr)

        bin_idx = int(np.searchsorted(trial_t, t_event))
        s = bin_idx - pre_bins
        e = bin_idx + post_bins
        if s < 0 or e > z.shape[0]:
            continue
        aligned.append(z[s:e])

    t_ax = np.arange(-pre_bins, post_bins) * bin_s
    if not aligned:
        return None, t_ax
    return np.stack(aligned), t_ax


#%% event-specific alignment

def align_to_baseline_onset(z_all, dataset, session,
                            pre_s=1.0, post_s=2.0,
                            block=None, tr_in_block_min=None):
    df = _filter_events(session.bl_onsets,
                        block=block, tr_in_block_min=tr_in_block_min)
    return align_latents_to_events(
        z_all, dataset,
        event_times=df['time'].values,
        event_trials=df.index.values,
        pre_s=pre_s, post_s=post_s)


def align_to_change_onset(z_all, dataset, session,
                          pre_s=0.5, post_s=1.5,
                          block=None, tr_in_block_min=None,
                          is_hit=None, ch_tf=None):
    df = _filter_events(session.ch_onsets,
                        block=block, tr_in_block_min=tr_in_block_min,
                        is_hit=is_hit, ch_tf=ch_tf)
    return align_latents_to_events(
        z_all, dataset,
        event_times=df['time'].values,
        event_trials=df['trial'].values,
        pre_s=pre_s, post_s=post_s)


def align_to_tf_outliers(z_all, dataset, session,
                         pre_s=0.5, post_s=1.5,
                         block=None, tr_in_block_min=None,
                         tr_time_range=None, rmv_near_response=None):
    """
    align latents to TF outlier pulses, split by sign.
    returns dict(fast=..., slow=...) and time axis.
    """
    df = _filter_events(session.tf_pulses,
                        block=block, tr_in_block_min=tr_in_block_min)

    if tr_time_range is not None:
        t_min, t_max = tr_time_range
        df = df[(df['tr_time'] > t_min) & (df['tr_time'] < t_max)]

    if rmv_near_response is not None:
        t_to_event = np.fmin(df['time_to_lick'], df['time_to_abort'])
        df = df[t_to_event > rmv_near_response]

    pos = df[df['tf'] > 0]
    neg = df[df['tf'] <= 0]

    fast, t_ax = align_latents_to_events(
        z_all, dataset,
        pos['time'].values, pos['trial'].values,
        pre_s=pre_s, post_s=post_s)
    slow, _ = align_latents_to_events(
        z_all, dataset,
        neg['time'].values, neg['trial'].values,
        pre_s=pre_s, post_s=post_s)

    return dict(fast=fast, slow=slow), t_ax


def align_to_lick_onset(z_all, dataset, session,
                        pre_s=1.5, post_s=0.5,
                        block=None, tr_in_block_min=None,
                        tr_time_range=None,
                        is_hit=None, is_FA=None):
    df = _filter_events(session.lick_times,
                        block=block, tr_in_block_min=tr_in_block_min,
                        is_hit=is_hit, is_FA=is_FA)

    if tr_time_range is not None:
        t_min, t_max = tr_time_range
        df = df[(df['tr_time'] > t_min) & (df['tr_time'] < t_max)]

    return align_latents_to_events(
        z_all, dataset,
        event_times=df['time'].values,
        event_trials=df['trial'].values,
        pre_s=pre_s, post_s=post_s)


#%% helpers

def _filter_events(df, block=None, tr_in_block_min=None, **extra):
    mask = pd.Series(True, index=df.index)
    if block is not None and 'block' in df.columns:
        mask &= df['block'] == block
    if tr_in_block_min is not None and 'tr_in_block' in df.columns:
        mask &= df['tr_in_block'] > tr_in_block_min
    for col, val in extra.items():
        if val is not None and col in df.columns:
            mask &= df[col] == val
    return df[mask]
