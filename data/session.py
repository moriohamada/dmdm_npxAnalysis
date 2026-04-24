"""
Session dataclass - for storing basic trial/neural info for a single session.

Note that fr_matrix and daq lines are dropped while saving by default to save storage.
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path


@dataclass
class Session:
    animal:      str
    name:        str

    trials:      pd.DataFrame
    daq:         pd.DataFrame | None = None
    move:        dict         | None = None
    neural:      pd.DataFrame | None = None
    fr_matrix:   pd.DataFrame | None = None
    fr_stats:    pd.DataFrame | None = None
    fr_normed:   bool | None = None
    unit_info:   pd.DataFrame | None = None

    # event timings
    bl_onsets:   pd.DataFrame | None = None
    tf_pulses:   pd.DataFrame | None = None
    ch_onsets:   pd.DataFrame | None = None
    lick_times:  pd.DataFrame | None = None

    @property
    def has_neural(self) -> bool:
        return self.neural is not None

    @property
    def n_neurons(self) -> int | None:
        if self.fr_matrix is not None:
            return len(self.fr_matrix)
        elif self.unit_info is not None:
            return len(self.unit_info)
        else:
            return None

    @property
    def areas(self):
        return self.neural.brain_region_comb.unique()

    def area_mask(self, areas: list[str]) -> np.ndarray:
        """mask over unit_info rows for neurons in the given areas"""
        regions = self.unit_info['brain_region_comb'].values
        return np.isin(regions, areas)

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
    def from_folder(cls, sess_folder: str | Path) -> Session:

        # parse animal and session names
        if isinstance(sess_folder, Path):
            sess_folder = str(sess_folder)

        file_parts = sess_folder.split('/')
        animal = file_parts[-2]
        session_name = file_parts[-1]

        trials = pd.read_parquet(os.path.join(sess_folder, 'trials.parquet'))

        if not os.path.isfile(os.path.join(sess_folder, 'neural.parquet')):
            return cls(trials=trials, animal=animal, name=session_name)

        neural = (pd.read_parquet(os.path.join(sess_folder, 'neural.parquet'))
                  .drop(columns=['brain_region', 'x', 'y', 'z']))
        neural['cluster_id'] = neural['cluster_id'] + (neural['probe_id'] - 1) * 10000
        daq    = pd.read_parquet(os.path.join(sess_folder, 'daq.parquet'))
        with open(os.path.join(sess_folder, 'movement.pkl'), 'rb') as f:
            move = pickle.load(f)

        # unit info
        unit_info = (neural.groupby('cluster_id')['brain_region_comb']
                     .first()
                     .reset_index())

        return cls(trials=trials, daq=daq, move=move, neural=neural,
                   animal=animal, name=session_name, unit_info=unit_info)

    def save(self, save_path: str) -> None:
        os.makedirs(save_path, exist_ok=True)
        path = os.path.join(save_path, 'session.pkl')
        slimmed = replace(self, fr_matrix=None, neural=None, daq=None)
        with open(path, 'wb') as f:
            pickle.dump(slimmed, f)
        print(f'Saved session to {path}')

    @classmethod
    def load(cls, path: str) -> Session:
        with open(path, 'rb') as f:
            return pickle.load(f)