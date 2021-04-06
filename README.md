# smartest
aSsessMent frAmework foR deTEctors of integrity attacks to Smart meTers

This repository includes an experimental framework for assessing the quality and performance
of detectors of integrity attacks to smart-meters.

## Structure of the repository

- *src*: this folder includes the Python source code (organized in sub-folders):
  - *1_preprocessing*: datasets pre-processing module and synthetic dataset generator
  - *2_detectors*:  software detectors, detector infrastructure 
  - *3_experiments*: the experiments launcher module
  - *4_analytics*: the dashboard module
- *script_results*: outputs from the experiments. 
- *requirements.txt*: the Python3 packages required to run the scripts
- *README*: this file
- *.gitignore*: specifies intentionally untracked files to ignore


## Usage 

The dependencies of the tool are listed in requirements.txt.
You can run the following command for installing them:

$ pip install -r requirements.txt


## Reference

The original input datasets are the electricity and gas dataset from the Irish Social Science Data Archive - Commision for Energy Regulation (ISSDA-CER).
Both are available, for research purposes, upon request, and the information provided is anonymous.
External link: https://www.ucd.ie/issda/data/commissionforenergyregulationcer/