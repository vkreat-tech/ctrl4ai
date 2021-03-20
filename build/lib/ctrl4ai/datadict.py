# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 13:15:40 2020

@author: ShajiJamesSelvakumar
"""

likert_scales=open(r'C:\Users\ShajiJamesSelvakumar\Documents\Ctrl4AI\ctrl4ai\dictionary\likert_scales.txt').readlines()

for line in likert_scales:
    line_dict=eval(line)
    print(type(line_dict))



