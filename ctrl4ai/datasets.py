# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:41:40 2020

@author: ShajiJamesSelvakumar
"""

import pandas as pd

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
  return dataset

