# smartest: aSsessMent frAmework foR deTEctors of integrity attacks to Smart meTers

This repository includes an experimental framework for assessing the quality and performance
of detectors of integrity attacks to smart-meters.

## Structure of the repository

- *src*: this folder includes the Python source code (organized in sub-folders):
  - *preprocessing*: datasets pre-processing module and synthetic dataset generator
  - *detectors*:  software detectors, detector infrastructure 
  - *experiments*: the experiments launcher module
  - *analytics*: the dashboard module
- *script_results*: outputs from the experiments. 
- *requirements.txt*: the Python3 packages required to run the scripts
- *README*: this file
- *.gitignore*: specifies intentionally untracked files to ignore


## Usage 

The dependencies of the tool are listed in requirements.txt.
You can run the following command for installing them:

```$ pip install -r requirements.txt```

You must run all the ```python3``` commands from the root directory.


## Reference

The original input datasets are the electricity and gas dataset from the Irish Social Science Data Archive - Commision for Energy Regulation (ISSDA-CER).
Both are available, for research purposes, upon request, and the information provided is anonymous.

- External link: https://www.ucd.ie/issda/data/commissionforenergyregulationcer/

The experimental framework and experiments results are detailed in:

- S. Bernardi, R. Javierre, J. Merseguer, J. Requeno, *Detectors of Smart Grid Integrity Attacks: An Experimental Assessment*, 17th European Dependable Computing Conference, 13-16 September 2021, Munich, Germany. [See preprint](https://github.com/DiasporeUnizar/smartest/blob/main/biblio/BJMR21preprint.pdf).

