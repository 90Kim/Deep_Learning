# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 16:55:23 2017

@author: GooYoung
"""

#######################################################
############## chapter 02 퍼셉트론 ####################
#######################################################


##### 2.3 퍼셉트론 구현하기

### 2.3.1 간단한 구현부터
# AND 회로 구현
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

AND(0,0) # print "0"
AND(1,0) # print "0"
AND(0,1) # print "0"
AND(1,1) # print "1"

### 2.3.2 가중치와 편향 도입

import numpy as np
x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7
w*x
np.sum(w*x)
np.sum(w*x) + b

### 2.3.3 가중치와 편향 구현하기

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7  # 2.3.1에서 구현한 AND의 theta 가 -b 로 치환되었음
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

AND(0,0) # print "0"
AND(1,0) # print "0"
AND(0,1) # print "0"
AND(1,1) # print "1"

# NAND 게이트와 OR 게이트
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])  # AND와는 가중치 (w,b) 만 다름
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
NAND(0,0) # print "1"
NAND(1,0) # print "1"
NAND(0,1) # print "1"
NAND(1,1) # print "0"

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.3
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

OR(0,0) # print "0"
OR(1,0) # print "1"
OR(0,1) # print "1"
OR(1,1) # print "1"

##### 2.4 퍼셉트론의 한계

##### 2.5 다층 퍼셉트론의 등장

### 2.5.2 XOR 게이트 구현하기

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

XOR(0, 0)  # print "0"
XOR(1, 0)  # print "1"
XOR(0, 1)  # print "1"
XOR(1, 1)  # print "0"

#######################################################
################ finish chapter 2 #####################
#######################################################