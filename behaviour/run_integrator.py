"""
HPC entry point: fit leaky integrator (tau, gain, threshold) for one subject/block.
usage: python -m behaviour.run_integrator <subj> <block>
"""
import sys
import pickle
from pathlib import Path
import pandas as pd

from behaviour.integrator import (
    clean_baseline_trials, grid_search, SEARCH_PARAMS,
)

PATHS = dict(
    data='/ceph/mrsic_flogel/public/projects/MoHa_20260212_dmdmTemporalExpectation'
         '/hpc/analysis_data/df_processed.pkl',
    results='/ceph/mrsic_flogel/public/projects/MoHa_20260212_dmdmTemporalExpectation'
            '/hpc/analysis_data/leakyInt/',
)

subj = sys.argv[1]
block = sys.argv[2]

out_path = Path(PATHS['results']) / f'leaky_int_{subj}_{block}.pkl'
if out_path.is_file():
    sys.exit(0)

dfs = pd.read_pickle(PATHS['data'])
df_clean = clean_baseline_trials(dfs[subj])
df_block = df_clean[df_clean['hazardblock'] == block].reset_index(drop=True)

print(f'{subj} | {block} | n_trials={len(df_block)} '
      f'(FA={df_block["IsFA"].sum()}, non-FA={(~df_block["IsFA"]).sum()})')

result = grid_search(df_block, SEARCH_PARAMS, n_jobs=-1, verbose=True)

out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'wb') as f:
    pickle.dump(result, f)
print(f'saved to {out_path}')
