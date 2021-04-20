#!/usr/bin/env
'''
Author: Allanxu
Date: 2021-04-07 14:17:11
LastEditors: Allanxu
LastEditTime: 2021-04-07 14:48:05
Description: ---
'''
import numpy as np
from scipy.io import loadmat

def SetsSplit(raw_data,raw_label,ind_train,ind_test):
    ind_train-=1 # index 0-164 for np array
    ind_test-=1
 
    return raw_data[ind_train],raw_label[ind_train],raw_data[ind_test],raw_label[ind_test]


def Load_RawData(filepath):
    raw = loadmat(filepath)
    return raw['fea'], raw['gnd'].reshape(-1,1)