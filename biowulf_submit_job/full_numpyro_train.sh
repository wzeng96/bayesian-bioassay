#!/bin/bash
#
#SBATCH --mem=40g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:2

#SBATCH --cpus-per-task=14
#SBATCH --time=168:00:00
#SBATCH --mail-type=END
#SBATCH --output=R-0612.%j.out
#SBATCH --error=R-0612.%j.err

python Tox21/PyMC_model_train.py
