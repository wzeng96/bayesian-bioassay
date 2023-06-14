#!/bin/bash
#
#SBATCH --mem=40g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:2

#SBATCH --cpus-per-task=14
#SBATCH --time=168:00:00
#SBATCH --mail-type=END
#SBATCH --output=R-data.out
#SBATCH --error=R-data.err

python Tox21/train_test_data.py
