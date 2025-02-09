_Manuscript Under review by EMBC 2025._

# Food-Leftover-Detection
This is the code implementation for our EMBC 2025 submission: Efficient Cross-Channel Feature Selection for Food Leftover Detection

Corresponding Author: [Sicong Huang](https://stmilab.github.io/team/).

## Description

FoodRemainder offers the potential for free-living food leftover estimation, paving the way for improving automated dietary monitoring solutions, particularly for individuals at risk of obesity and type 2 diabetes (T2D).

## Prerequisites

### Setup Environment
Our experiments are conducted on a Linux-based machine with the following specifications:

* Linux-based OS 
* Python 3.9.15
* conda 4.14.0
* PyTorch 1.11.0
* git 2.34.1
* CUDA 11.4 or 11.6 (for GPU acceleration)


We highly recommend you to use the conda environment ([`environment.yml`](environment.yml)) we shared to avoid potential compatibility issues. To set up Conda for your computer, you can follow the official instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).


Instructions for Unix-based Command Line Input: 

1. `git pull https://github.com/stmilab/Food-Leftover-Detection.git` clones the repository to your local machine

2. `cd Food-Leftover-Detection/` changes the directory to the repository

3. `conda env create -f environment.yml` creates a new conda environment the same as ours ([`environment.yml`](environment.yml) contains the packages used for our experiments.)

4. `conda activate foodremainder` activates the created conda environment you just created

## Implementations
FoodRemainder contains 3 major components:
* [GUI for label annotation](#gui-for-label-annotation)
* [Dinov2 for channel extraction](#dinov2-for-channel-extraction)
* [Model for food leftover detection](#model-for-food-leftover-detection)

![Visualization of FoodRemainder](figures/foodremainder_visual.png)


### GUI for Label Annotation

__TODO: Add GUI code__
![Visualization of GUI](figures/leftover_GUI.png)

### Dinov2 for Channel Extraction

__TODO: Add Dinov2 code__


### Model for Food Leftover Detection


* The [`step3_predict_leftover/`](step3_predict_leftover/) directory contains the main scripts for all experiments conducted and reported in the manuscript
* The [`step3_predict_leftover/utils/`](step3_predict_leftover/models/) directory contains the utility functions used by the main scripts
    * The [`parser.py`](step3_predict_leftover/utils/parser.py) file contains the argument parser for the main scripts
    * The [`data_loader.py`](step3_predict_leftover/utils/data_loader.py) file extracts the images and constructs Pytorch datasets and associated dataloaders
    * The [`helper.py`](step3_predict_leftover/utils/helper.py) file contains the evaluation metrics for the main scripts
* The [`step3_predict_leftover/models/`](step3_predict_leftover/models/) contains the custom functions used in the experiments.
    * The [feature selector](step3_predict_leftover/models/exchange.py) mechanism is based on [CEN](https://github.com/yikaiw/CEN/tree/master) implementation
* [`step3_predict_leftover/exp_detail/`](step3_predict_leftover/exp_detail/) contains the detailed configurations of hyperparameters used by all experiments conducted and reported in the manuscript
    * Hyperparameter tuning is conducted using (Weights & Bias)[https://wandb.ai/] with the following scope: [`hyperparam.txt`](exp_setup/mimic_hyperparam.txt)

## Datasets

### CGMacros Dataset

We used the CGMacros dataset (version 1.0.0). 
* The dataset is publicly available on [PhysioNet](https://physionet.org/content/cgmacros/1.0.0/)

* Remember to modify `--data_path` accordingly, otherwise change the default value in [`parser.py`](step3_predict_leftover/utils/parser.py)
