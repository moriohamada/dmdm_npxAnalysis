#!/bin/bash
#SBATCH --job-name=glm
#SBATCH --output=logs/glm_%A_%a.out
#SBATCH --error=logs/glm_%A_%a.err
#SBATCH --array=0-NJOBS
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

python -m neuron_prediction.glm.run_neuron --job-index $SLURM_ARRAY_TASK_ID --job-map glm_job_map.csv
