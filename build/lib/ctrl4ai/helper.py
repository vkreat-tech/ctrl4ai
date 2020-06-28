# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 15:27:31 2020

@author: ShajiJamesSelvakumar
"""

import geopy.distance
import numpy as np
import pandas as pd
import scipy


def isNaN(num):
  """
  Usage: [arg1]:[numeric value]
  Description: Checks if the value is null (numpy.NaN)
  Returns: Boolean [True/False]
  """
  return num != num


def distance_calculator(start_latitude,start_longitude,end_latitude,end_longitude):
  """
  Usage: [arg1]:[numeric-start_latitude],[arg2]:[numeric-start_longitude],[arg3]:[numeric-end_latitude],[arg4]:[numeric-end_longitude]
  Returns: Numeric [Distance in kilometers]
  """
  if isNaN(start_latitude)  or isNaN(start_longitude) or isNaN(end_latitude) or isNaN(end_longitude):
    return np.NaN
  else:
    start = (start_latitude,start_longitude)
    end = (end_latitude,end_longitude)
    return geopy.distance.vincenty(start, end).kilometers


def test_numeric(test_string):
  """
  Usage: [arg1]:[String/Number]
  Description: Checks if the value is numeric
  Returns: Boolean [True/False]
  """
  try :
    float(test_string)
    res = True
  except :
    res = False
  return res


def check_numeric_col(col_series):
  """
  Usage: [arg1]:[Pandas Series / Single selected column of a dataframe]
  Description: Checks if all the values in the series are numerical
  Returns: Boolean [True/False]
  """
  if all(col_series.apply(lambda x: test_numeric(x))):
    return True
  else:
    return False


def check_categorical_col(col_series,categorical_threshold=0.3):
  """
  Usage: [arg1]:[Pandas Series / Single selected column of a dataframe],[categorical_threshold(default=0.3)]:[Threshold for determing categorical column based on the percentage of unique values(optional)]
  Description: Breaks the values to chunks and checks if the proportion of unique values is less than the threshold
  Returns: Boolean [True/False]
  """
  col_array=np.array(col_series.apply(lambda x:str(x).strip()).astype(str).fillna(str(0)))
  if col_array.size>1000:
    n=10
    k=100
  elif col_array.size>100:
    n=np.ceil(col_array.size/100)
    k=100
  else:
    n=1
    k=col_array.size
  if n%2==0:
    n-=1
  t=0
  f=0
  for i in range(n):
    sample=np.random.choice(col_array,size=k,replace=False)
    if np.unique(sample).size/sample.size<=categorical_threshold:
      t+=1
    else:
      f+=1
  if t>f:
    return True
  else:
    return False


def single_valued_col(col_series):
  """
  Usage: [arg1]:[Pandas Series / Single selected column of a dataframe]
  Description: Checks if the column has only one value
  Returns: Boolean [True/False]
  """
  if col_series.dropna().unique().shape[0]==1:
    return True
  else:
    return False


def one_hot_encoding(dataset,categorical_cols_list):
  """
  Usage: [arg1]:[pandas dataframe],[arg2]:[list of columns to be encoded]
  Description: Transformation for categorical features by getting dummies
  Returns: Dataframe [with separate column for each categorical values]
  """
  dataset=pd.merge(dataset,pd.get_dummies(dataset[categorical_cols_list],columns=categorical_cols_list), left_index=True, right_index=True)
  dataset=dataset.drop(categorical_cols_list,axis=1)
  return dataset


def cramers_v(x, y):
  """
  Usage: [arg1]:[independent categorical series],[arg2]:[dependent categorical series]
  Description: Cramér's V is a measure of association between two nominal variables
  Returns: A value between 0 and +1
  """
  confusion_matrix = pd.crosstab(x,y)
  chi2 = scipy.stats.chi2_contingency(confusion_matrix)[0]
  n = confusion_matrix.sum().sum()
  phi2 = chi2/n
  r,k = confusion_matrix.shape
  phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
  rcorr = r-((r-1)**2)/(n-1)
  kcorr = k-((k-1)**2)/(n-1)
  return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

