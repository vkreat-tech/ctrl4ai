from ctrl4ai import datasets
from ctrl4ai import prepdata
from ctrl4ai import automl
from ctrl4ai import helper
import pandas as pd

dataset = datasets.titanic()
median_val=dataset['Age'].median()
dataset['Age']=dataset['Age'].fillna(median_val)

dataset['Age_binned_cut']=prepdata.binning(dataset['Age'])
print(dataset['Age_binned_cut'])
print(type(dataset['Age_binned_cut']))
print(type(dataset['Age_binned_cut'][0]))
print(dataset['Age_binned_cut'].shape)
print(dataset['Age_binned_cut'].value_counts())
