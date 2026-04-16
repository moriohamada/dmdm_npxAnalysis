#!/bin/bash
#SBATCH -J glm_ridge
#SBATCH -o logs/glm_ridge-%A_%a.log
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 0-04:00
#SBATCH --array=0-5405%100


module load miniconda/23.10.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /ceph/mrsic_flogel/public/projects/MoHa_20201102_SwitchChangeDetection/conda_envs/neuro_analysis

CODE_DIR=/nfs/nhome/live/morioh/Documents/PycharmProjects/dmdm_npxAnalysis
cd $CODE_DIR

python -u -m neuron_prediction.glm_ridge.run_neuron \
    --job-index $((SLURM_ARRAY_TASK_ID + 10000)) \
    --job-map /ceph/mrsic_flogel/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx/glm_job_map.csv
