AOCECM
==

Requirements
--
### Necessary Libraries
* skactiveml 
(This library provides the AL methods `McPAL` and `ALCE`)
### Necessary packages
* python 3.6
* xlwt
* xlrd
* numpy
* pandas
* sklearn

Conduct Experiments
--
### experiment preparation: data sets split
From folder `DataPartitions` to run
```Python
CreatPartition.py
```
Create the `Pool Set` and `Testing Set` with 6 times five-fold Stratified Cross-validation. 
Obtain 6*5=30 splittings.
### Run the compared AL methods
All the compared methods are in folder `Methods`.
The critical instances selection results will be recorded in this phase.
Before run AOCECM, please first create clustering structure by `GetKmeansStructure.py`.
### Run the KELMOR model on the selected results
```
RunClassifier_KELMOR.py
```
The classification results in each query iteration will be recorded.
