#!/bin/bash
#SBATCH -J glm_fit
#SBATCH -o logs/glm-%A_%a.out
#SBATCH -e logs/glm-%A_%a.err
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH -t 0-04:00
#SBATCH --array=0-NJOBS

module load miniconda/23.10.0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /ceph/mrsic_flogel/public/projects/MoHa_20201102_SwitchChangeDetection/conda_envs/neuro_analysis

CODE_DIR=/nfs/nhome/live/morioh/Documents/PycharmProjects/dmdm_npxAnalysis
cd $CODE_DIR

python -m neuron_prediction.glm.run_neuron \
    --job-index $SLURM_ARRAY_TASK_ID \
    --job-map glm_job_map.csv
