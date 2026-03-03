"""
Session dataclass and constructor.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import os
import pickle
import pandas as pd
import numpy as np


@dataclass
class Session:
    trials:      pd.DataFrame
    daq:         pd.DataFrame | None = None
    move:        dict         | None = None
    neural:      pd.DataFrame | None = None
    fr_matrix:   pd.DataFrame | None = None

    # event timings
    bl_onsets:   pd.DataFrame | None = None
    tf_pulses:   pd.DataFrame | None = None
    ch_onsets:   pd.DataFrame | None = None
    lick_times:  pd.DataFrame | None = None
    abort_times: pd.DataFrame | None = None

    @property
    def has_neural(self) -> bool:
        return self.neural is not None

    @property
    def n_neurons(self) -> int:
        return len(self.fr_matrix) if self.fr_matrix is not None else 0

    @property
    def areas(self):
        return self.neural.brain_region_comb.unique()

    @property
    def t_ax(self) -> np.ndarray:
        return self.fr_matrix.columns.values if self.fr_matrix is not None else None

    @property
    def trial_outcomes(self):
        return dict(
            Total = len(self.trials),
            Hits  = self.trials['IsHit'].sum(),
            Miss  = self.trials['IsMiss'].sum(),
            FA    = self.trials['IsFA'].sum(),
            Abort = self.trials['IsAbort'].sum(),
        )

    @classmethod
    def from_folder(cls, sess_folder: str) -> Session:

        trials = pd.read_parquet(os.path.join(sess_folder, 'trials.parquet'))

        if not os.path.isfile(os.path.join(sess_folder, 'neural.parquet')):
            return cls(trials=trials)

        neural = (pd.read_parquet(os.path.join(sess_folder, 'neural.parquet'))
                  .drop(columns=['brain_region', 'x', 'y', 'z']))
        daq    = pd.read_parquet(os.path.join(sess_folder, 'daq.parquet'))
        move   = pickle.load(open(os.path.join(sess_folder, 'movement.pkl'), 'rb'))

        return cls(trials=trials, daq=daq, move=move, neural=neural)