# Ctrl4AI 
[![Downloads](http://pepy.tech/badge/ctrl4ai)](http://pepy.tech/project/ctrl4ai)

#### This is a helper package for Machine Learning and Deep Learning solutions.

For documentation, please read [HELP.md](https://github.com/vkreat-tech/ctrl4ai/blob/master/HELP.md)

For demo on usage, please check [README.ipynb](https://github.com/vkreat-tech/ctrl4ai/blob/master/README.ipynb)


#### Contact Developers: [Shaji](https://www.linkedin.com/in/shaji-james/), [Charu](https://www.linkedin.com/in/charunethragiri/), [Selva](https://www.linkedin.com/in/selva-prasanth-274b66166/)

## Highlights
- Open Source Package with emphasis on data preprocessing so far.
- Self intelligent methods that can be employed at the levels of abstraction or customization.
- The flow of auto-preprocessing is orchestrated compatible to the learning type.
- Parameter tuning allows users to transform the data precisely to their specifications.
- Developed computations for inspecting the data to discover its type, distribution, correlation etc. which are handled in the background.

## Dependencies

Ctrl4AI requires:

* Python 3 (tested under Python 3.6)

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
from ctrl4ai import prepdata
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

- Added features to transform ordinal values (from string to integers)

## Depreciation Notice

- All functions in ```ctrl4ai.preprocessing``` will be depreciated by the end of 2021. Please plan to switch to the same functions in ```ctrl4ai.prepdata```

## ToDo

- Model specific pre-processing
- Prepare dataset for bagging Algorithms
- Preprocessing for Text,Image,Audio,Video Analytics
