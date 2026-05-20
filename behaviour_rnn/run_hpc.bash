#!/usr/bin/env bash
#
#SBATCH -J behaviour_rnn
#SBATCH -o logs/behaviour_rnn-%A_%a.out
#SBATCH -e logs/behaviour_rnn-%A_%a.err
#SBATCH -N 1
#SBATCH -t 0-12:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --array=0-14

module load miniconda/23.10.0
module load cuda/11.8
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /ceph/mrsic_flogel/public/projects/MoHa_20201102_SwitchChangeDetection/conda_envs/neuro_analysis

CODE_DIR=/nfs/nhome/live/morioh/Documents/PycharmProjects/dmdm_npxAnalysis
NPX_DIR=/ceph/mrsic_flogel/public/projects/MoHa_20260212_dmdmTemporalExpectation/hpc/npx_mirror

mkdir -p logs

cd $CODE_DIR
python -u -m behaviour_rnn.run $NPX_DIR $SLURM_ARRAY_TASK_ID
