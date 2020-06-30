# -*- coding: utf-8 -*-
"""
Created on Tue May 19 19:41:40 2020

@author: Shaji,Charu,Selva
"""

import pandas as pd
pd.set_option('mode.chained_assignment', None)

from . import helper

def titanic():
  """
  Type: Supervised - Classification
  Name: Titanic
  Target variable: Survived
  Description: Sample Dataset
  Returns: Pandas DataFrame
  """
  dataset=pd.read_csv(helper.__file__[:-len('helper')-3]+"sample_datasets/titanic.csv.gz",compression='gzip')
  dataset=dataset.drop(['Unnamed: 0'],axis=1)
  return dataset


def trip_fare():
  """
  Type: Supervised - Regression
  Name: Trip Fare
  Target variable: fare_amount
  Description: Sample Dataset
  Returns: Pandas DataFrame
  """
  dataset = pd.read_csv(helper.__file__[:-len('helper')-3]+"sample_datasets/trip_fare.csv.gz",compression='gzip',parse_dates=['pickup_datetime','dropoff_datetime'])
  dataset=dataset.drop(['Unnamed: 0'],axis=1)
  return dataset

