# -*- coding: utf-8 -*-
"""
About Ctrl4AI
    Open Source Project Developed by VKreaT (www.vkreat.com)
    Ctrl4AI (www.ctrl4ai.com) has automated methods to automate the entire flow of pre-processing

About ctrl4ai.datadict
    The module has supporting data that supports other functions in the package

Last Updated On: 14 Feb 2023
"""

likert_scales = open(r'C:\Users\ShajiJamesSelvakumar\Documents\Ctrl4AI\ctrl4ai\dictionary\likert_scales.txt').readlines()

for line in likert_scales:
    line_dict = eval(line)
    print(type(line_dict))
