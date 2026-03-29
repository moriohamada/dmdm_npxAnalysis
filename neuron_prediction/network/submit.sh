#!/bin/bash
#SBATCH --job-name=nn_pred
#SBATCH --output=logs/nn_pred_%A_%a.out
#SBATCH --error=logs/nn_pred_%A_%a.err
#SBATCH --array=0-NJOBS
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00

python -m neuron_prediction.network.run_neuron --job-index $SLURM_ARRAY_TASK_ID --job-map network_job_map.csv
