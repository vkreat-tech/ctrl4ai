# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:54:12 2020

@author: Shaji,Charu,Selva
"""

likert_scales = open(r'C:\Users\ShajiJamesSelvakumar\Documents\Ctrl4AI\ctrl4ai\dictionary\likert_scales.txt').readlines()

for line in likert_scales:
    line_dict = eval(line)
    print(type(line_dict))



