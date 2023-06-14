#!/bin/bash
#
#SBATCH --mem=20g
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:k80:1,lscratch:10 
#SBATCH --cpus-per-task=14
#SBATCH --time=5-12:00:00
#SBATCH --mail-type=END
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

module load pymc/4.1.3
python-pymc PyMC_model_script_numpyro.py 100 

