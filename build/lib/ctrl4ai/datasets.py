# -*- coding: utf-8 -*-
"""
Created on Tue May 19 19:41:40 2020

@author: Shaji,Charu,Selva
"""

import pandas as pd
import requests
import os

pd.set_option('mode.chained_assignment', None)


def titanic(refresh=False):
    """
    Usage: [arg1]:[refresh=False(default)/True - True if the file should be downloaded and refreshed again from intenet]
    Type: Supervised - Classification
    Name: Titanic
    Target variable: Survived
    Description: Sample Dataset
    Returns: Pandas DataFrame
    """
    if (not os.path.exists('titanic.csv')) or (refresh == True):
        print('Downloading titanic.csv ......')
        titanic_request = requests.get(
            "https://github.com/vkreat-tech/ctrl4ai/raw/master/sample_datasets/titanic.csv.gz", allow_redirects=True)
        open('titanic.csv', 'wb').write(titanic_request.content)
    dataset = pd.read_csv('titanic.csv', compression='gzip')
    dataset = dataset.drop(['Unnamed: 0'], axis=1)
    return dataset


def trip_fare(refresh=False):
    """
    Usage: [arg1]:[refresh=False(default)/True - True if the file should be downloaded and refreshed again from intenet]
    Type: Supervised - Regression
    Name: Trip Fare
    Target variable: fare_amount
    Description: Sample Dataset
    Returns: Pandas DataFrame
    """
    if (not os.path.exists('trip_fare.csv')) or (refresh == True):
        print('Downloading trip_fare.csv ......')
        trip_fare_request = requests.get(
            "https://github.com/vkreat-tech/ctrl4ai/raw/master/sample_datasets/trip_fare.csv.gz", allow_redirects=True)
        open('trip_fare.csv', 'wb').write(trip_fare_request.content)
    dataset = pd.read_csv('trip_fare.csv', compression='gzip', parse_dates=['pickup_datetime', 'dropoff_datetime'])
    dataset = dataset.drop(['Unnamed: 0'], axis=1)
    return dataset
