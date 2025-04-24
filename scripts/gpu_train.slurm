#!/bin/bash
#SBATCH -p a100_long,a100_short,radiology,gpu4_medium,gpu4_long,gpu8_medium 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120GB
#SBATCH --time=3-00:00:00
#SBATCH --job-name=b6
#SBATCH --output=/gpfs/scratch/wz1492/MuZero-Go/logs/train-%J.log
#SBATCH --exclude=a100-4020

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi
nvcc --version

# activate conda environment
module load gcc/8.1.0
source ~/.bashrc 
conda activate gymgo

python main.py