# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 16:55:23 2017

@author: GooYoung
"""

#######################################################
########## chapter 07 합성곱 신경망 (CNN) ##############
#######################################################

import numpy as np

##### 7.4 합성곱/풀링 계층 구현하기

### 7.4.1 4차원 배열

x = np.random.rand(10, 1, 28, 28)
x[0].shape
x[0, 0]

### 7.4.3 합성곱 계층 구현하기

import sys, os
sys.path.append(os.pardir)
from common.util import im2col

x1 = np.random.rand(1,3,7,7)
col1 = im2col(x1, 5, 5, stride=1, pad=0)

x2 = np.random.rand(10,3,7,7)
col2 = im2col(x2, 5, 5, stride=1, pad=0)

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int( 1 + (H + 2*self.pad - FH) / self.stride )
        out_w = int( 1 + (W + 2*self.pad - FH) / self.stride )
        
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_w = self.W.reshape(FN, -1).T
        out = np.dot(col, col_w) + self.b
        
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        
        return out
    



#######################################################
###################### finish ##########################
#######################################################