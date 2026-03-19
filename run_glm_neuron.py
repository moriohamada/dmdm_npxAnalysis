"""
SLURM entry point: fit GLM for one neuron

usage: python run_glm_neuron.py --job-index 42 --job-map /path/to/glm_job_map.csv
"""
import argparse
import pandas as pd
from analyses.glm import fit_neuron_from_disk

parser = argparse.ArgumentParser()
parser.add_argument('--job-index', type=int, required=True)
parser.add_argument('--job-map', type=str, required=True)
args = parser.parse_args()

job_map = pd.read_csv(args.job_map)
row = job_map.loc[job_map['job_idx'] == args.job_index].iloc[0]

print(f'Fitting {row["animal"]}/{row["session"]} neuron {row["neuron_idx"]} '
      f'(cluster {row["cluster_id"]})')

fit_neuron_from_disk(row['sess_dir'], int(row['neuron_idx']))
