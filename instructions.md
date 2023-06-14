### Instructions to run model on Biowulf
This is an instruction on how to run the hierarchical bayesian model on Biowulf. The steps are break into three parts, first is generating the training and testing set from the Tox21 database, which will be passed to model for training. Second will be the file to train model, lastly is the prediction file. 

Each step will have a shell script to submit job on Biowulf, a python script, output file, error file, and results that will be needed later on.
#### Connect to conda enviornment
- `source myconda`
- `mamba activate pymc`

now you are in the pymc enviornment and are able to run the files

#### Split data into Training_and_Testing
`nano Tox21/train_test_data.py` to look at the train_test python script.

`sbatch data.sh` submit job to biowulf
- output: R-data.out, R-data.err (These two are used to see whether the python script runs successfully)
- `cat R-data.out` to look at the content inside the file

Output: X_train.csv, X_test.csv, y_train.csv, y_test.csv, assay_info_train.csv, assay_info_test.csv

#### Train model
`nano Tox21/PyMC_model_train.py` to look at the model training python script.

`sbatch full_numpyro_train.sh` submit job to biowulf (this steps take days to finish)
- output: R-train.%j.out, R-train.%j.err

Output: results/tr_assay_model_numpyro.pkl

#### Prediction
`nano Tox21/PyMC_model_script_numpyro_prediction.py` to look at the model prediction python script.

`sbatch full_numpyro_pred.sh` submit job to biowulf (this steps take about 10min to finish)
- output: R-pred.out, R-pred.err

Output: test_pred_full_model.csv, random_seed.csv
