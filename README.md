Hi Everyone, 

This project aims to predict chemical toxicity in the Tox21 assay data using Hierarchical Bayesian model, and compare results with Logistic Regressiona Naive Bayesian model. This is the first project that uses in vitro assays to predict chemical toxicity and we also include assay level biological information in addition to chemical descriptors to make predictions.

The code and results for Logistic Regression and Naive Bayesian are located in the `notebooks` folder.

To run the Hierarchical Bayesian model, one needs to first load the conda environment, then run the python script in `data_preprocessing` folder to prepare data for modeling. The steps to run this model in GPU/faster way are under the `numpyro_GPU` folder. 

### PyMC Notebook

 * Load the conda environment or install the following dependencies:
     * pymc, numpy, scipy, pandas, scikit-learn, graphviz (module load graphviz on biowulf)
    
