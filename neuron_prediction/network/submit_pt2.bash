#!/bin/bash
#SBATCH -J nn_fit
#SBATCH -o logs/nn-%A_%a.log
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 0-06:00
#SBATCH --array=0-5405%100

module load miniconda/23.10.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /ceph/mrsic_flogel/public/projects/MoHa_20201102_SwitchChangeDetection/conda_envs/neuro_analysis

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

CODE_DIR=/nfs/nhome/live/morioh/Documents/PycharmProjects/dmdm_npxAnalysis
cd $CODE_DIR

python -u -m neuron_prediction.network.run_neuron \
    --job-index $((SLURM_ARRAY_TASK_ID + 10000)) \
    --job-map /ceph/mrsic_flogel/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx/network_job_map.csv
