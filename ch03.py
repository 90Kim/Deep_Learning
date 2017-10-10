# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 16:55:23 2017

@author: GooYoung
"""

#######################################################
################ chapter 03 신경망 ####################
#######################################################

import numpy as np

##### 3.2 활성화 함수

### 3.2.2 계단함수 구현하기 

def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

step_function(1)
step_function(-3)
step_function(1.0)
step_function(np.array([1.0, 2.0]))
step_function([1.0, 2.0])
 
x = np.array([-1.0, 1.0, 2.0])
x
y = x > 0
y
y = y.astype(np.int)
y

### 3.2.3 계단함수 그래프

import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    return np.array(x>0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)

### 3.2.4 시그모이드 함수 구현하기

def sigmoid(x):
    return 1 / (1+np.exp(-x))

x = np.array([-1.0, 1.0, 2.0])
sigmoid(x)

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)

#######################################################
###################### finish##########################
#######################################################