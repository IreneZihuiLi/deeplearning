# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 21:43:15 2016

@author: troy
"""

import os
import pandas as pd
import numpy as np
import random
from sklearn import *

def save_as_csv(df):
    df.to_csv('/Users/mac/Documents/UCDstudy/Python/Projects/MnistTF/ReadCSV/NeuralNet/Data/s10_s23_train_datab.csv', sep = ',', header = False, index = False)
    
def save_as_csv1(df):
    df.to_csv('/Users/mac/Documents/UCDstudy/Python/Projects/MnistTF/ReadCSV/NeuralNet/Data/s10_s23_test_datab.csv', sep = ',', header = False, index = False)

pwd = os.getcwd()
path = pwd + '/Data/s10_s23.csv'
data = pd.read_csv(path, header = None)
split = float(input("input your split(0-1) example:0.1 represents 10% for test and 90% for training plz:\n"))
train_data,test_data = cross_validation.train_test_split(data,test_size=split,random_state=random.randint(1, 500))

save_as_csv(train_data)
save_as_csv1(test_data)
    
    


    