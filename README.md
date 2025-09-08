<h2 align="center"><b>ML4transients</b> : <br>Machine Learning for Transients and bogus classification</h2>

<p align="center">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://pytorch.org"><img alt="pytorch" src="https://img.shields.io/badge/PyTorch-2.0-DC583A.svg?style=flat&logo=pytorch"></a>
</p>

ML4transients is a Python library for performing classification between Bogus and Transient using Machine Learning without human labelling.
It includes tools to generate injection catalogue to feed the LSST pipeline, performing cutouts from the Difference Image, train classifier, evaluate them and interpret inferences. 

It is still under active development and will receive frequent updates and bugfixes.


# Installation

You can install it by cloning the repository and install the package in dev mode

```sh
# clone the repository
git clone https://github.com/rbonnetguerrini/ML4transients.git
# then install in dev mode
cd ML4transients
pip install --editable .
```


# Setup

## Injection and Cutouts


For all jobs that requires the LSST Pipeline, the lsst distrib must be loaded. 

At CC: 
```sh
source /cvmfs/sw.lsst.eu/linux-x86_64/lsst_distrib/w_2024_30/loadLSST.bash

setup lsst_distrib

export PYTHONPATH="/sps/lsst/users/rbonnetguerrini/ML4transients/src:$PYTHONPATH"

```


## For the rest of the jobs

For the remaining job you can load your own conda environment at CC using ; 

```sh 

source /usr/share/Modules/init/bash
module load Programming_Languages/anaconda/3.11
conda activate your_env
pip install --editable .
```


# Quick tutorial 

## Generating injection catalogues
The injection module allows you to create fake source catalogs for transient detection training. It supports both galaxy-hosted and hostless transients with realistic magnitude distributions.

- To **configure injection parameters** edit `configs/injection_all_bands.yaml` 
- To **generate catalog for single band**: 
```sh
python scripts/injection/gen_injection_catalogue.py --config configs/injection/injection_all_bands.yaml --band r
```
- To **process multiple bands with job array**:
```sh
./scripts/injection/submit_band_catalogue.sh
```


## Performing cutouts:

- Using `scripts/run_cutout.py` you can produce cutout given a specific yaml, template can be found in `ML4transients/configs/configs_cutout.yaml` . From there you can specify visits inside of the collection and if you only want the features, cutouts, or both.

- Since the main problem of producing those cutouts is that you have to split visit in different job for a given collection,  I made an automatic job submitter, that generates configs with a given amount of visit, and launched them all. You can do that using `scripts/data_preparation/submit_collection.sh configs/data_preparation/configs_cutout.yaml 100` , again selecting your own original config, and specifying how much you want it to be split by (here 100 visit per job max)

- Once the cutouts and features are made, if you created them using the automatic job submitter, you should run `python scripts/data_preparation/create_global_index_post_batch.py configs/data_preparation/configs_cutout.yaml` so that you create an index that list in which visit is each diaSourceId.

## Lightcurve Extraction

The lightcurve extraction module efficiently organizes and indexes time-series data for all detected objects. It groups detections by sky patch, saving each patch as an HDF5 file, and builds cross-reference indices for both `diaObjectId` and `diaSourceId`. This enables fast lookup and retrieval of full lightcurves or all sources belonging to a transient candidate. To extract and index lightcurves, use:

```sh
python scripts/data_preparation/run_lightcurves.py configs/data_preparation/configs_cutout.yaml
```

The resulting files in `lightcurves/` include patch-based HDF5 tables and index files for rapid access.

## Load dataset:
```py
from ML4transients.data_access import DatasetLoader
dataset = DatasetLoader('saved/test')
print(dataset)
``` 
`DatasetLoader` allows you to lazy load the different components of a dataset (cutout, lc, features, inference).
when creating this set, it creates a dictionary that assigned each diaSourceId their visit number


## Lightcurve Inference with SuperNNova

The pipeline for running lightcurve-based inference using SuperNNova (SNN) consists of three main steps, they need specific python env to be ran:

1. **Convert HDF5 lightcurve patches to CSV:**  
   Use `scripts/data_preparation/SNN/convert_lc_for_snn.py` to convert each HDF5 patch file into a CSV format compatible with SuperNNova. (lsst_distrib)

2. **Run SNN inference:**  
   Use `scripts/data_preparation/SNN/infer_snn.py` to run SuperNNova on the CSV files. This produces ensemble and individual prediction CSVs for each patch.(SuperNNova env)

3. **Save SNN results back to HDF5:**  
   Use `scripts/data_preparation/SNN/save_inference_snn.py` to write the SNN ensemble predictions back into the original HDF5 patch files under the `snn_inference` dataset.(lsst_distrib)

A bash script (`scripts/data_preparation/SNN/snn_inference.sh`) is provided to automate this workflow, including environment setup for each step.  
This process avoids code redundancy and ensures efficient batch processing of large datasets. 

To then perform the analysis of the SNN model, please look at the `notebooks/data_loading_example.ipynb`


## Hyperparameter Optimization:

Run Bayesian optimization to find optimal hyperparameters:

```sh
# Standard CNN optimization
sbatch scripts/training/submit_training.sh configs/training/standard_training.yaml "standard_bayes_opt" "--hpo"

# Ensemble optimization  
sbatch scripts/training/submit_training.sh configs/training/ensemble_training.yaml "ensemble_bayes_opt" "--hpo"

# Co-teaching optimization
sbatch scripts/training/submit_training.sh configs/training/coteaching_training.yaml "coteaching_bayes_opt" "--hpo"
```

Configure search space in the config file under `bayes_search` section. Best parameters are saved to `bayes_best_params.yaml` in the output directory. Each optimization runs short trials (max_epochs) to efficiently explore hyperparameter space.

## Launch a training:
```sh
sbatch scripts/training/submit_training.sh configs/training/standard_training.yaml "standard_training"

sbatch scripts/training/submit_training.sh configs/training/ensemble_training.yaml "ensemble_training"

sbatch scripts/training/submit_training.sh configs/training/coteaching_training.yaml "coteaching_training"
```
Allows to submit job to your GPU. The number of workers can be change in the config, as well as all the training parameter. 

## Perform Inference: 

```sh
python scripts/evaluation/run_inference.py \
--dataset-path /sps/lsst/groups/transients/HSC/fouchez/raphael/data/rc2_norm       --weights-path /sps/lsst/groups/transients/HSC/fouchez/raphael/training/ensemble_optimized

```
More details in `notebooks/inference_example.ipynb`

## Perform evaluation: 

```sh
python scripts/evaluation/run_evaluation.py     --config configs/evaluation/evaluation_config.yaml     --data-path /sps/lsst/groups/transients/HSC/fouchez/raphael/data/rc2_norm     --weights-path /sps/lsst/groups/transients/HSC/fouchez/raphael/training/ensemble_optimized     --output-dir saved/test_eval/ensemble_umap_uncertainty     --interpretability     --optimize-umap   --run-inference
```

More details in `notebooks/evaluation_example.ipynb`

