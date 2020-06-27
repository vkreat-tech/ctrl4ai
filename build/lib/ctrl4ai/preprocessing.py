# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 17:54:12 2020

@author: ShajiJamesSelvakumar
"""

import scipy
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


from . import helper


def get_distance(dataset,start_latitude,start_longitude,end_latitude,end_longitude):
  """
  Usage: [arg1]:[Pandas DataFrame],[arg2]:[column-start_latitude],[arg3]:[column-start_longitude],[arg4]:[column-end_latitude],[arg5]:[column-end_longitude]
  Returns: DataFrame with additional column [Distance in kilometers]
  """
  dataset['kms_'+start_latitude+'_'+end_latitude]=dataset.apply(lambda row: helper.distance_calculator(row[start_latitude], row[start_longitude],row[end_latitude],row[end_longitude]),axis=1)
  return dataset


def get_timediff(dataset,start_time,end_time):
  """
  Usage: [arg1]:[Pandas DataFrame],[arg2]:[column-start_time],[arg3]:[column-end_time]
  Returns: DataFrame with additional column [Duration in seconds]
  """
  dataset['secs_diff_'+start_time+'_'+end_time]=(dataset[end_time]-dataset[start_time]).dt.total_seconds()
  return dataset


def derive_from_datetime(dataset):
  """
  Usage: [arg1]:[pandas dataframe]
  Prerequisite: Type for datetime columns to be defined correctly
  Description: Derives the hour, weekday, year and month from a datetime column
  Returns: Dataframe [with new columns derived from datetime columns]
  """
  for column,dtype in dataset.dtypes.items():
    if 'datetime' in str(dtype):
      dataset['hour_of_'+column]=dataset[column].apply(lambda x: x.hour)
      dataset['weekday_of_'+column]=dataset[column].apply(lambda x: x.weekday())
      dataset['year_of_'+column]=dataset[column].apply(lambda x: x.year)
      dataset['month_of_'+column]=dataset[column].apply(lambda x: x.month)
  return dataset


def drop_null_fields(dataset,dropna_threshold=0.7):
  """
  Usage: [arg1]:[pandas dataframe],[dropna_threshold(default=0.7)]:[What percentage of nulls should account for the column top be removed]
  Description: Drop columns that has more null values
  Returns: Dataframe [with null dominated columns removed]
  """
  no_of_records=dataset.shape[0]
  select_cols=[]
  for index,val in dataset.isnull().sum().items():
    if val/no_of_records<dropna_threshold:
      select_cols.append(index)
    else:
      print('Dropping '+index)
  return(dataset[select_cols])
  

def drop_single_valued_cols(dataset):
  """
  Usage: [arg1]:[pandas dataframe]
  Description: Drop columns that has only one value in it
  Returns: Dataframe [without single valued columns]
  """
  single_valued_cols=[]
  for col in dataset.columns:
    if helper.single_valued_col(dataset[col]):
      single_valued_cols.append(col)
  print('Dropping '+','.join(single_valued_cols))
  if len(single_valued_cols)>0:
    dataset=dataset.drop(single_valued_cols,axis=1)
  return dataset


def get_ohe_df(dataset,target_variable=None,ignore_cols=[],categorical_threshold=0.3):
  """
  Usage: [arg1]:[pandas dataframe],[target_variable(default=None)]:[Dependent variablr for Regression/Classification],[ignore_cols]:[categorical columns where one hot encoding need not be done],[categorical_threshold(default=0.3)]:[Threshold for determing categorical column based on the percentage of unique values(optional)]
  Description: Auto identifies categorical features in the dataframe and does one hot encoding
  Note: Consumes more system mermory if the size of the dataset is huge
  Returns: Dataframe [with separate column for each categorical values]
  """
  for col in dataset.columns:
    if helper.check_categorical_col(dataset[col],categorical_threshold=categorical_threshold) and col!=target_variable and col not in ignore_cols:
      dataset=helper.one_hot_encoding(dataset,[col])
  return dataset



def drop_non_numeric(dataset):
  """
  Usage: [arg1]:[pandas dataframe]
  Description: Drop columns that are not numeric
  Returns: Dataframe [only numeric features]
  """
  drop_cols=[]
  for col in dataset.columns:
    if helper.check_numeric_col(dataset[col])==False:
      drop_cols.append(col)
  print("Dropping "+','.join(drop_cols))
  if len(drop_cols)>0:
    dataset=dataset.drop(drop_cols,axis=1)
  return dataset


def impute_nulls(dataset,method='central_tendency'):
  """
  Usage: [arg1]:[pandas dataframe],[method(default=central_tendency)]:[Choose either central_tendency or KNN]
  Description: Auto identifies the type of distribution in the column and imputes null values
  Note: KNN consumes more system mermory if the size of the dataset is huge
  Returns: Dataframe [with separate column for each categorical values]
  """
  if str.lower(method)=='knn':
    k_knn=int(np.ceil(np.sqrt(dataset.shape[0])))
    if k_knn%2==0:
      k_knn+=1
    imputer = KNNImputer(n_neighbors=k_knn)
    knn_imputed_array = imputer.fit_transform(dataset)
    dataset=pd.DataFrame(knn_imputed_array,columns=dataset.columns)
    return dataset
  elif method=='central_tendency':
    for col,value in dataset.isnull().sum().items():
      if value>0:
        if helper.check_numeric_col(dataset[col]):
          if scipy.stats.skew(dataset[col])>1:
            print("Replaced nulls in "+col+" with median")
            dataset[col]=dataset[col].fillna(dataset[col].median())
          else:
            print("Replaced nulls in "+col+" with mean")
            dataset[col]=dataset[col].fillna(dataset[col].mean())
        elif helper.check_categorical_col(dataset[col]):
          print("Replaced nulls in "+col+" with mode")
          dataset[col]=dataset[col].fillna(dataset[col].mode()[0])
    return dataset
  else:
    print('Method should be either central_tendency or knn')
    raise helper.ParameterError
    
    
def label_encode(dataset,col):
  """
  Usage: [arg1]:[pandas dataframe],[arg1]:[column to be encoded]
  Description: Labelling categorical features with numbers from 0 to n categories
  Returns: Label Dict , Dataframe
  """
  dataset[col]=dataset[col].fillna(dataset[col].mode()[0])
  label_dict=dict(zip(dataset[col].unique(),np.arange(dataset[col].unique().shape[0])))
  dataset=dataset.replace({col:label_dict})
  dataset[col]=dataset[col].astype('int')
  dataset[col]=dataset[col].astype('category')
  return label_dict,dataset

def get_correlated_features(dataset,target_variable):
  """
  Usage: [arg1]:[pandas dataframe],[arg2]:[target/dependent variable]
  Description: Only for supervised learning to select independent variables that has some correlation with target/dependent variable
  Returns: List of columns that have considerable correlation
  """
  selected_features=[]
  for col,coeff in dataset.corr()[target_variable].items():
    if coeff>np.abs(2/np.sqrt(dataset.shape[0])):
      selected_features.append(col)
  print("Selected Features - "+','.join(selected_features))
  return selected_features
