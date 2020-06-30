# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:54:12 2020

@author: Shaji,Charu,Selva
"""

import scipy
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
pd.set_option('mode.chained_assignment', None)


from . import helper
from . import exceptions


def get_distance(dataset,
                 start_latitude,
                 start_longitude,
                 end_latitude,
                 end_longitude):
  """
  Usage: [arg1]:[Pandas DataFrame],[arg2]:[column-start_latitude],[arg3]:[column-start_longitude],[arg4]:[column-end_latitude],[arg5]:[column-end_longitude]
  Returns: DataFrame with additional column [Distance in kilometers]
  """
  dataset['kms_'+start_latitude+'_'+end_latitude]=dataset.apply(lambda row: helper.distance_calculator(row[start_latitude], row[start_longitude],row[end_latitude],row[end_longitude]),axis=1)
  return dataset


def get_timediff(dataset,
                 start_time,
                 end_time):
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


def log_transform(dataset,method='yeojohnson',categorical_threshold=0.3):
  """
  Usage: [arg1]:[pandas dataframe],[method]=['yeojohnson'/'added_constant']
  Description: Checks if the a continuous column is skewed and does log transformation
  Returns: Dataframe [with all skewed columns normalized using appropriate approach]
  """
  for col in dataset.columns:
    if helper.check_categorical_col(dataset[col],categorical_threshold=categorical_threshold)==False and helper.check_numeric_col(dataset[col]) and np.abs(scipy.stats.skew(dataset[col]))>1:
      print('Log Normalization('+method+') applied for '+col)
      if method=='yeojohnson':
        dataset[col]=dataset[col].apply(lambda x: helper.yeojohnsonlog(x))
      elif method=='added_constant':
        dataset=helper.added_constant_log(dataset,col)
  return dataset


def drop_null_fields(dataset,
                     dropna_threshold=0.7):
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
      print('Dropping null dominated column(s) '+index)
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
  if len(single_valued_cols)>0:
    print('Dropping single valued column(s) '+','.join(single_valued_cols))
    dataset=dataset.drop(single_valued_cols,axis=1)
  return dataset


def get_ohe_df(dataset,
               target_variable=None,
               ignore_cols=[],
               categorical_threshold=0.3):
  """
  Usage: [arg1]:[pandas dataframe],[target_variable(default=None)]:[Dependent variable for Regression/Classification],[ignore_cols]:[categorical columns where one hot encoding need not be done],[categorical_threshold(default=0.3)]:[Threshold for determining categorical column based on the percentage of unique values(optional)]
  Description: Auto identifies categorical features in the dataframe and does one hot encoding
  Note: Consumes more system mermory if the size of the dataset is huge
  Returns: Dataframe [with separate column for each categorical values]
  """
  for col in dataset.columns:
    if helper.check_categorical_col(dataset[col],categorical_threshold=categorical_threshold) and col!=target_variable and col not in ignore_cols:
      print('One hot encoding '+col)
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
  if len(drop_cols)>0:
    print("Dropping non categorical/continuous column(s):"+','.join(drop_cols))
    dataset=dataset.drop(drop_cols,axis=1)
  return dataset


def impute_nulls(dataset,
                 method='central_tendency'):
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
        if helper.check_categorical_col(dataset[col]):
          print("Replaced nulls in "+col+" with mode")
          mode_val=dataset[col].mode()[0]
          dataset[col]=dataset[col].fillna(mode_val)
        elif helper.check_numeric_col(dataset[col]):
          if np.abs(scipy.stats.skew(dataset[col]))>1:
            print("Replaced nulls in "+col+" with median")
            median_val=dataset[col].median()
            dataset[col]=dataset[col].fillna(median_val)
          else:
            print("Replaced nulls in "+col+" with mean")
            mean_val=dataset[col].mean()
            dataset[col]=dataset[col].fillna(mean_val)
    return dataset
  else:
    print('Method should be either central_tendency or knn')
    raise exceptions.ParameterError
    
    
def label_encode(dataset,
                 col):
  """
  Usage: [arg1]:[pandas dataframe],[arg1]:[column to be encoded]
  Description: Labelling categorical features with numbers from 0 to n categories
  Returns: Label Dict , Dataframe
  """
  mode_val=dataset[col].mode()[0]
  dataset[col]=dataset[col].apply(lambda x:str(x).strip()).astype(str).fillna(mode_val)
  label_dict=dict(zip(dataset[col].unique(),np.arange(dataset[col].unique().shape[0])))
  dataset=dataset.replace({col:label_dict})
  dataset[col]=dataset[col].astype('int')
  dataset[col]=dataset[col].astype('category')
  return label_dict,dataset


def remove_outlier_df(dataset,
                      cols):
  """
  Usage: [arg1]:[pandas dataframe],[arg2]:[list of columns to check and remove outliers]
  Description: The column needs to be continuous
  Returns: DataFrame with outliers removed for the specific columns
  """
  for col in cols:
    outlier_temp_dataset=pd.DataFrame(dataset[col])
    outlier_temp_dataset=impute_nulls(outlier_temp_dataset)
    Q1 = outlier_temp_dataset.quantile(0.25)
    Q3 = outlier_temp_dataset.quantile(0.75)
    IQR = Q3 - Q1
    outlier_bool_dataset=((outlier_temp_dataset > (Q1 - 1.5 * IQR)) & (outlier_temp_dataset < (Q3 + 1.5 * IQR)))
    select_index=outlier_bool_dataset.index[outlier_bool_dataset[col] == True]
    print('No. of outlier rows removed based on '+col+' is '+str(outlier_temp_dataset.shape[0]-len(select_index)))
    dataset=dataset.iloc[select_index].reset_index(drop=True)
  return dataset


def auto_remove_outliers(dataset,
                         ignore_cols=[],
                         categorical_threshold=0.3):
  """
  Usage: [arg1]:[pandas dataframe],[ignore_cols]:[list of columns to be ignored],[categorical_threshold(default=0.3)]:[Threshold for determining categorical column based on the percentage of unique values(optional)]
  Description: Checks if the column is continuous and removes outliers
  Returns: DataFrame with outliers removed
  """
  continuous_columns=[]
  for col in dataset.columns:
    if helper.check_categorical_col(dataset[col],categorical_threshold=categorical_threshold)==False and helper.check_numeric_col(dataset[col])==True:
      continuous_columns.append(col)
  dataset=remove_outlier_df(dataset,continuous_columns)
  return dataset


def get_label_encoded_df(dataset,
                         categorical_threshold=0.3):
  """
  Usage: [arg1]:[pandas dataframe],[categorical_threshold(default=0.3)]:[Threshold for determining categorical column based on the percentage of unique values(optional)]
  Description: Auto identifies categorical features in the dataframe and does label encoding
  Returns: Dataframe [with separate column for each categorical values]
  """
  column_labels=dict()
  for col in dataset.columns:
    if helper.check_numeric_col(dataset[col]):
        pass
    elif helper.check_categorical_col(dataset[col],categorical_threshold=categorical_threshold):
      labels,dataset=label_encode(dataset,col)
      print('Labels for '+col+': '+str(labels))
      column_labels[col]=labels
  return column_labels,dataset


def cramersv_corr(x, y):
  """
  Usage: [arg1]:[categorical series],[arg2]:[categorical series]
  Description: Cramer's V Correlation is a measure of association between two categorical variables
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


def kendalltau_corr(x, y):
  """
  Usage: [arg1]:[continuous series],[arg2]:[categorical series]
  Description: Kendall Tau Correlation is a measure of association between a continuous variable and a categorical variable
  Returns: A value between -1 and +1
  """
  x_arr=np.array(impute_nulls(pd.DataFrame(x)))
  y_arr=np.array(impute_nulls(pd.DataFrame(y)))
  corr,_=scipy.stats.kendalltau(x_arr,y_arr)
  return corr


def pearson_corr(x, y):
  """
  Usage: [arg1]:[continuous series],[arg2]:[continuous series]
  Description: Pearson Correlation is a measure of association between two continuous variables
  Returns: A value between -1 and +1
  """
  x=pd.to_numeric(x)
  y=pd.to_numeric(y)
  return np.corrcoef(x,y)[0,1]


def get_correlated_features(dataset,
                            target_col,
                            target_type,
                            categorical_threshold=0.3):
  """
  Usage: [arg1]:[pandas dataframe],[arg2]:[target/dependent variable],[arg3]:['continuous'/'categorical'],,[categorical_threshold(default=0.3)]:[Threshold for determining categorical column based on the percentage of unique values(optional)]
  Description: Only for supervised learning to select independent variables that has some correlation with target/dependent variable (Uses Pearson correlation between two continuous variables, CramersV correlation between two categorical variables, Kendalls Tau correlation between a categorical and a continuos variable)
  Returns: Dictionary of correlation coefficients, List of columns that have considerable correlation
  """
  categorical_cols=[]
  continuous_cols=[]
  col_corr=dict()
  for col in dataset:
    if col!=target_col:
      if helper.check_categorical_col(dataset[col],categorical_threshold=categorical_threshold):
        categorical_cols.append(col)
      elif helper.check_numeric_col(dataset[col]):
        continuous_cols.append(col)
  if target_type=='continuous':
    for col in continuous_cols:
      coeff=pearson_corr(dataset[col],dataset[target_col])
      col_corr[col]=coeff
    for col in categorical_cols:
      coeff=kendalltau_corr(dataset[col],dataset[target_col])
      col_corr[col]=coeff
  if target_type=='categorical':
    for col in continuous_cols:
      coeff=kendalltau_corr(dataset[col],dataset[target_col])
      col_corr[col]=coeff
    for col in categorical_cols:
      coeff=cramersv_corr(dataset[col],dataset[target_col])
      col_corr[col]=coeff
  selected_features=[]
  for col in col_corr.keys():
    if float(col_corr[col])>np.abs(2/np.sqrt(dataset.shape[0])):
      selected_features.append(col)
  return col_corr,selected_features


