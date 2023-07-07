#!/bin/bash
#
#SBATCH --mem=40g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100x:4

#SBATCH --cpus-per-task=14
#SBATCH --time=168:00:00
#SBATCH --mail-type=END
#SBATCH --output=R.%j.out
#SBATCH --error=R.%j.err

python Tox21/PyMC_model_train.py
