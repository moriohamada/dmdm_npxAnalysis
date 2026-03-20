"""
HPC entry point: fit leaky integrator for one subject/block.
usage: python -m behaviour.run_integrator <subj> <block> <run>
"""
import sys
import pickle
from pathlib import Path
import pandas as pd
from behaviour.integrator import clean_df, get_rt_kernel, grid_search_params, SEARCH_PARAMS

PATHS = dict(
    data='/ceph/mrsic_flogel/public/projects/MoHa_20260212_dmdmTemporalExpectation'
         '/hpc/analysis_data/df_processed.pkl',
    results='/ceph/mrsic_flogel/public/projects/MoHa_20260212_dmdmTemporalExpectation'
            '/hpc/analysis_data/leakyInt/')

subj = sys.argv[1]
block = sys.argv[2]
run = int(sys.argv[3]) if len(sys.argv) > 3 else 1

out_path = Path(PATHS['results']) / f'leaky_int_{subj}_{block}_run{run}.pkl'
if out_path.is_file():
    sys.exit(0)

dfs = pd.read_pickle(PATHS['data'])
df = dfs[subj]

df_clean = clean_df(df)
rt_kernel = get_rt_kernel(df)
df_block = df_clean[df_clean['hazardblock'] == block].reset_index(drop=True)

fa_trials = df_block[df_block['IsFA']]
non_fa_trials = df_block[~df_block['IsFA']]
if len(non_fa_trials) > len(fa_trials) * 3:
    non_fa_trials = non_fa_trials.sample(n=len(fa_trials) * 3, random_state=42)
df_block = pd.concat([fa_trials, non_fa_trials]).reset_index(drop=True)

print(f'{subj} | {block} | n_trials={len(df_block)} '
      f'(FA={fa_trials.shape[0]}, non-FA={len(non_fa_trials)})')

all_params, likelihoods, best_lick_dist, best_time_bins, best_params, n_runs = (
    grid_search_params(df_block, rt_kernel, SEARCH_PARAMS, n_jobs=-1, verbose=True))

out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'wb') as f:
    pickle.dump({
        'params': all_params, 'likelihoods': likelihoods,
        'lick_dist': best_lick_dist, 'time_bins': best_time_bins,
        'best_params': best_params, 'n_runs': n_runs,
    }, f)
print(f'Saved to {out_path}')
