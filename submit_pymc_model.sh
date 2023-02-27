#!/bin/bash
#
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --time=72:00:00
#SBATCH --mail-type=END
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

#change this to remove jss
python PyMC_model_script_jss.py $1 

