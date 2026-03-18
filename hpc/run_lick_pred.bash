#!/usr/bin/env bash
#
#SBATCH -J lick_pred
#SBATCH -o logs/lick_pred-%A_%a.out
#SBATCH -e logs/lick_pred-%A_%a.err
#SBATCH -N 1
#SBATCH -t 0-04:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --array=0-15

module load miniconda/23.10.0
module load cuda/11.8
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /ceph/mrsic_flogel/public/projects/MoHa_20201102_SwitchChangeDetection/conda_envs/neuro_analysis

CODE_DIR=/nfs/nhome/live/morioh/Documents/PycharmProjects/dmdm_npxAnalysis
NPX_DIR=/ceph/mrsic_flogel/public/projects/MoHa_20260212_dmdmTemporalExpectation/hpc/npx_mirror

mkdir -p logs

cd $CODE_DIR
python -m analyses.run_lick_prediction $NPX_DIR $SLURM_ARRAY_TASK_ID
