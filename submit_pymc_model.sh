#!/bin/bash
#
#SBATCH --mem=128g
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mail-type=END

#change this to remove jss
python PyMC_model_script_jss.py 

