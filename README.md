# Ctrl4AI 

This is a helper package for Machine Learning and Deep Learning solutions.

For demo on usage, please check [README.ipynb](https://github.com/shajijames/ctrl4ai/blob/master/README.ipynb)

#### Contact Developers: [Shaji](https://www.linkedin.com/in/shaji-james/), [Charu](https://www.linkedin.com/in/charunethragiri/), [Selva](https://www.linkedin.com/in/selva-prasanth-274b66166/)

## Highlights
  - Open Source Machine learning / Deep learning Package - focusing only on data preprocessing as of now.
  - The package has lot of methods that can be used independently, but the major highlight of the package is a method with hyperparameters covering the entire flow of preprocessing.
  - Users can simply experiment by running with default parameters which they can further tune by adjusting the parameters based on the requirements
  - Methods are with proper description to make it friendly for the user
  - Self-intelligent methods that understand the type of data, distribution etc. and compute accordingly
  - Minimises the number of checks that user has to do for preprocessing

## Dependencies

Ctrl4AI requires:

* Python (tested under Python 3.6)

## Installation

The easiest way to install the latest release version of Ctrl4AI is via ```pip```:
```bash
pip install ctrl4ai
```
In case you get ```ERROR: Could not install packages due to an EnvironmentError```, try using
```bash
pip install ctrl4ai --user
```
Check for the latest available version in [Ctrl4AI](https://pypi.org/project/ctrl4ai/)

## Import

Import any module from the package thru the following method:
```bash
from ctrl4ai import preprocessing
```

## Learn to use

Understand what each functions does by using ```help()```:
```bash
help(preprocessing.impute_nulls)
```

## ChangeLog

This is the first official release of the package

## ToDo

  - Support reading multiple file formats like parquet, orc etc.
