# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 15:27:31 2020

@author: ShajiJamesSelvakumar
"""

import geopy.distance
import numpy as np
import pandas as pd


class ParameterError(Exception):
  """
  User Defined Exception
  Thrown because the argumrnt is incorrect
  """
  pass


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
  Description: Checks if all the values in the series are numerical
  Returns: Boolean [True/False]
  """
  if len(col_series.unique())/col_series.shape[0]<=categorical_threshold:
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
  Usage: [arg1]:[pandas dataframe],[arg1]:[list of columns to be encoded]
  Description: Transformation for categorical features by getting dummies
  Returns: Dataframe [with separate column for each categorical values]
  """
  dataset=pd.merge(dataset,pd.get_dummies(dataset[categorical_cols_list],columns=categorical_cols_list), left_index=True, right_index=True)
  dataset=dataset.drop(categorical_cols_list,axis=1)
  return dataset


