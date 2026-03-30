#!/bin/bash
#SBATCH --job-name=nn_pred
#SBATCH --output=logs/nn_pred_%A_%a.log
#SBATCH --array=0-NJOBS
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00

module load miniconda/23.10.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /ceph/mrsic_flogel/public/projects/MoHa_20201102_SwitchChangeDetection/conda_envs/neuro_analysis

CODE_DIR=/nfs/nhome/live/morioh/Documents/PycharmProjects/dmdm_npxAnalysis
cd $CODE_DIR

python -u -m neuron_prediction.network.run_neuron \
    --job-index $SLURM_ARRAY_TASK_ID \
    --job-map /ceph/mrsic_flogel/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx/network_job_map.csv
