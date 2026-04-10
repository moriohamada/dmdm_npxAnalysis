#!/bin/bash
#SBATCH -J glm_perblock
#SBATCH -o logs/glm_perblock-%A_%a.log
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 0-04:00
#SBATCH --array=0-9999%100

module load miniconda/23.10.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /ceph/mrsic_flogel/public/projects/MoHa_20201102_SwitchChangeDetection/conda_envs/neuro_analysis

CODE_DIR=/nfs/nhome/live/morioh/Documents/PycharmProjects/dmdm_npxAnalysis
cd $CODE_DIR

python -u -m neuron_prediction.glm_perblock.run_neuron \
    --job-index $SLURM_ARRAY_TASK_ID \
    --job-map /ceph/mrsic_flogel/public/projects/MoHa_20260212_dmdmTemporalExpectation/data/npx/glm_job_map.csv
