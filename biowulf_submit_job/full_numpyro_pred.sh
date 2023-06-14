#!/bin/bash
#
#SBATCH --mem=40g
#SBATCH --partition=gpu
#SBATCH --gres=gpu:p100:2

#SBATCH --cpus-per-task=14
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --output=R-pred.out
#SBATCH --error=R-pred.err

python Tox21/PyMC_model_script_numpyro_prediction.py
