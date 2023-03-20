#!/bin/bash
#
#SBATCH --mem=16g
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mail-type=END

#change this to remove jss
python preprocess_data.py 

