import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from data.session import Session


class SpikeData(Dataset):
    """
    wraps a Session + FR matrix into a PyTorch Dataset with trial structure.
    each sample is one trial: input is spikes[:, :-1], target is spikes[:, 1:].
    """

    def __init__(self, session: Session, fr_matrix: pd.DataFrame):
        X = fr_matrix.values
        t_ax = fr_matrix.columns.values.astype(float)
        bin_size_ms = round(np.mean(np.diff(t_ax)) * 1000)

        # assign each time bin to a trial
        trial_num = np.full(len(t_ax), -1, dtype=int)
        for tr, row in session.trials.iterrows():
            t_start = row['Baseline_ON_rise'] - 1.0
            t_end = np.nanmax([row['Baseline_ON_fall'],
                               row.get('Change_ON_fall', np.nan)]) + 1.0
            mask = (t_ax >= t_start) & (t_ax < t_end)
            trial_num[mask] = tr

        valid = trial_num >= 0
        self.X = torch.tensor(X[:, valid], dtype=torch.float32)
        self.t_ax = t_ax[valid]
        self.trial_num = trial_num[valid]
        self.bin_size_ms = bin_size_ms

        # build trial slices
        self.trial_ids = np.unique(self.trial_num).tolist()
        self.trial_slices = {}
        for t in self.trial_ids:
            idxs = np.where(self.trial_num == t)[0]
            self.trial_slices[t] = (int(idxs[0]), int(idxs[-1]) + 1)

        # drop trials too short to form an input/target pair
        self.trial_ids = [t for t in self.trial_ids
                          if self.trial_slices[t][1] - self.trial_slices[t][0] > 2]

    def __len__(self):
        return len(self.trial_ids)

    def __getitem__(self, idx):
        tid = self.trial_ids[idx]
        s, e = self.trial_slices[tid]

        out = dict(
            input=self.X[:, s:e - 1],
            target=self.X[:, s + 1:e],
            trial_num=tid,
            time_in_trial=torch.arange(e - s - 1, dtype=torch.float32) * self.bin_size_ms,
        )
        return out

    def get_trial_t_ax(self, trial_id):
        """absolute time axis for a given trial, trimmed to match __getitem__ (drops last bin)"""
        s, e = self.trial_slices[trial_id]
        return self.t_ax[s:e - 1]


def collate_trials(batch):
    """pad variable-length trials into batched tensors"""
    inputs = [item['input'].T for item in batch]    # each (T, nN)
    targets = [item['target'].T for item in batch]
    lengths = torch.tensor([x.shape[0] for x in inputs])

    input_padded = pad_sequence(inputs, batch_first=True)    # (B, Tmax, nN)
    target_padded = pad_sequence(targets, batch_first=True)

    Tmax = input_padded.shape[1]
    mask = torch.arange(Tmax).unsqueeze(0) < lengths.unsqueeze(1)

    return dict(input=input_padded, target=target_padded, mask=mask)
