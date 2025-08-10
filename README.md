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

## Performing cutouts:

- Using `scripts/run_cutout.py` you can produce cutout given a specific yaml, template can be found in `ML4transients/configs/configs_cutout.yaml` . From there you can specify visits inside of the collection and if you only want the features, cutouts, or both.

- Since the main problem of producing those cutouts is that you have to split visit in different job for a given collection,  I made an automatic job submitter, that generates configs with a given amount of visit, and launched them all. You can do that using `./scripts/submit_collection.sh configs/configs_cutout.yaml 100` , again selecting your own original config, and specifying how much you want it to be split by (here 100 visit per job max)

## Load dataset:
```py
from ML4transients.data_access import DatasetLoader
dataset = DatasetLoader('saved/test')
print(dataset)
``` 
`DatasetLoader` allows you to lazy load the different components of a dataset (cutout, lc, features, inference).
when creating this set, it creates a dictionary that assigned each diaSourceId their visit number

## Launch a training:
```sh 
sbatch scripts/submit_training.sh
```
Allows to submit job to your GPU. The number of workers can be change in the config, as well as all the training parameter. 

## Hyperparameter Optimization:

Run Bayesian optimization to find optimal hyperparameters:

```sh
# Standard CNN optimization
sbatch scripts/submit_training.sh configs/standard_training.yaml "standard_bayes"

# Ensemble optimization  
sbatch scripts/submit_training.sh configs/ensemble_training.yaml "ensemble_bayes"

# Co-teaching optimization
sbatch scripts/submit_training.sh configs/coteaching_training.yaml "coteaching_bayes"
```

Configure search space in the config file under `bayes_search` section. Best parameters are saved to `bayes_best_params.yaml` in the output directory. Each optimization runs short trials (max_epochs) to efficiently explore hyperparameter space.

## Perform Inference: 

```sh
python .scripts/run_inference.py \
--dataset-path /path/to/dataset/folder/ \ # The folder with features, images etc...
  --weights-path /path/to/weight/folder/

```
More details in `notebooks/inference_example.ipynb`

## Perform evaluation: 

```sh
python scripts/run_evaluation.py \
    --config configs/evaluation_config.yaml \
    --data-path /sps/lsst/groups/transients/HSC/fouchez/raphael/data/rc2_norm \
    --weights-path /sps/lsst/groups/transients/HSC/fouchez/raphael/training/simple_run \
    --output-dir saved/test_eval/hdbscan_highdim_optimized \
    --interpretability \
    --optimize-umap \
    --model-hash "6d5bb4aa"
```

More details in `notebooks/evaluation_example.ipynb`