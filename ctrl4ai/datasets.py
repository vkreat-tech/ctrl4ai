# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:41:40 2020

@author: ShajiJamesSelvakumar
"""

import pandas as pd

def titanic():
    dataset=pd.read_csv("/usr/local/lib/python3.6/dist-packages/ctrl4ai/titanic.csv.gz",compression='gzip')
    return dataset

def trip_fare():
    dataset = pd.read_csv("/usr/local/lib/python3.6/dist-packages/ctrl4ai/trip_fare.csv.gz",compression='gzip')
    return dataset
