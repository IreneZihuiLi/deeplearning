# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 21:43:15 2016

@author: troy
"""

import os
import pandas as pd

pwd = os.getcwd()
path = pwd + '/Data/modelData.csv'
data = pd.read_csv(path, header = None)

def data_clean(picies):
    gap = int((data[5].max() - data[5].min()) / picies)
    classify_list = []
    classify_list.append(0)
    for i in range(picies):
        classify_list.append(classify_list[i] + gap)
    classify_list[len(classify_list) - 1] = data[5].max()
    temp_map = data.copy()
    del temp_map[5]
    category_list = list(map(list, temp_map.values))
    index = 0
    for time in data[5]:
        to_classify_list = [0] * picies
        to_classify_list[classify(classify_list, time)] = 1
        category_list[index].extend(to_classify_list)
        index += 1
    category_df = pd.DataFrame(category_list)
    save_as_csv(category_df)
    
    
def save_as_csv(df):
    df.to_csv('/Data/modelDataBinary.csv', sep = ',', header = False, index = False)
        
def classify(_list, _value):
    for i in range(len(_list) - 1):
        if ((_value > _list[i]) & (_value <= _list[i + 1])):
            return i
    
data_clean(10)
    