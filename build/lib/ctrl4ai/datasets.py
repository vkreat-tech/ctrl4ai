# -*- coding: utf-8 -*-
"""
Created on Tue May 19 19:41:40 2020

@author: Shaji,Charu,Selva
"""

import pandas as pd
import requests
pd.set_option('mode.chained_assignment', None)


def titanic():
  """
  Type: Supervised - Classification
  Name: Titanic
  Target variable: Survived
  Description: Sample Dataset
  Returns: Pandas DataFrame
  """
  titanic_request = requests.get("https://github.com/vkreat-tech/ctrl4ai/raw/master/ctrl4ai/sample_datasets/titanic.csv.gz", allow_redirects=True)
  open('titanic.csv', 'wb').write(titanic_request.content)
  dataset=pd.read_csv('titanic.csv',compression='gzip')
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
  trip_fare_request = requests.get("https://github.com/vkreat-tech/ctrl4ai/raw/master/ctrl4ai/sample_datasets/trip_fare.csv.gz", allow_redirects=True)
  open('trip_fare.csv', 'wb').write(trip_fare_request.content)
  dataset = pd.read_csv('trip_fare.csv',compression='gzip',parse_dates=['pickup_datetime','dropoff_datetime'])
  dataset=dataset.drop(['Unnamed: 0'],axis=1)
  return dataset

