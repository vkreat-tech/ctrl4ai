# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:27:31 2020

@author: Shaji,Charu,Selva
"""

import geopy.distance
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)


def isNaN(num):
  """
  Usage: [arg1]:[numeric value]
  Description: Checks if the value is null (numpy.NaN)
  Returns: Boolean [True/False]
  """
  return num != num


def added_constant_log(dataset,
                       col):
  """
  Usage: [arg1]:[dataset], [arg2]:[column in which log transform should be done]
  Description: Log transforms the specified column
  Returns: DataFrame
  """
  min_value=dataset[col].min()
  if min_value<=0:
    dataset[col]=dataset[col].apply(lambda x: np.log(x+np.abs(min_value)+1))
  else:
    dataset[col]=dataset[col].apply(lambda x: np.log(x))
  return dataset


def yeojohnsonlog(x):
  """
  Usage: [arg1]:[real/float value]
  Description: Log transforms the specified column based on Yeo Joshson Power Transform
  Returns: Log value (numeric)
  """
  if x<0:
    y=-np.log(-x+1)
  else:
    y=np.log(x+1)
  return y


def distance_calculator(start_latitude,
                        start_longitude,
                        end_latitude,
                        end_longitude):
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


def check_categorical_col(col_series,
                          categorical_threshold=0.3):
  """
  Usage: [arg1]:[Pandas Series / Single selected column of a dataframe],[categorical_threshold(default=0.3)]:[Threshold for determining categorical column based on the percentage of unique values(optional)]
  Description: Breaks the values to chunks and checks if the proportion of unique values is less than the threshold
  Returns: Boolean [True/False]
  """
  col_array=np.array(col_series.apply(lambda x:str(x).strip()).astype(str).fillna(str(0)))
  if col_array.size>=1000:
    n=10
    k=100
  elif col_array.size>100:
    n=np.floor(col_array.size/100)
    k=100
  else:
    n=1
    k=col_array.size
  if n%2==0:
    n-=1
  t=0
  f=0
  for i in range(int(n)):
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


def one_hot_encoding(dataset,
                     categorical_cols_list):
  """
  Usage: [arg1]:[pandas dataframe],[arg2]:[list of columns to be encoded]
  Description: Transformation for categorical features by getting dummies
  Returns: Dataframe [with separate column for each categorical values]
  """
  dataset=pd.merge(dataset,pd.get_dummies(dataset[categorical_cols_list],columns=categorical_cols_list), left_index=True, right_index=True)
  dataset=dataset.drop(categorical_cols_list,axis=1)
  return dataset



