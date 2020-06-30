# Ctrl4AI 

This is a helper package for Machine Learning and Deep Learning solutions.

For demo on usage, please check [README.ipynb](https://github.com/vkreat-tech/ctrl4ai/blob/master/README.ipynb)

#### Contact Developers: [Shaji](https://www.linkedin.com/in/shaji-james/), [Charu](https://www.linkedin.com/in/charunethragiri/), [Selva](https://www.linkedin.com/in/selva-prasanth-274b66166/)

## Highlights
- Open Source Machine learning / Deep learning Package - focusing only on data preprocessing as of now.
- Auto-Preprocessing package that can be leveraged at the level of abstraction or at the level of customization.
- The flow of auto-preprocessing is handled in a way that it suits the intended type of learning.
- Hypertuning parameters which allows user to clean data specific to any given model.
- Computations for checking the type of data, distribution, correlation etc. are handled in the background.

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
```bash
from ctrl4ai import automl
```

## Learn to use

Understand what each functions does by using ```help()```:
```bash
help(automl.preprocess)
```

## ChangeLog

This is the first official release of the package

## ToDo

- Model specific pre-processing
- Baggining algoritmns
- Text,Image,Audio,Video Analytics
