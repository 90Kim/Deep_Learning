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

### 3.2.7 ReLU 함수

def relu(x):
    return np.maximum(0, x)

relu(5)
relu(-3)

##### 3.3 다차원 배열 계산

import numpy as np
A = np.array([1, 2, 3, 4])
print(A)
np.ndim(A)
A.shape
A.shape[0]
A.shape[1]
type(A.shape)

B = np.array([[1,2], [3,4], [5,6]])
print(B)
np.ndim(B)
B.shape

C = np.array([[[1,2], [3,4], [5,6]],[[10,20], [30,40], [50,60]]])
print(C)
np.ndim(C)
C.shape

D = np.array([[[[1,2], [3,4], [5,6]],[[10,20], [30,40], [50,60]]], [[[2,1], [4,3], [6,5]],[[100,200], [300,400], [500,600]]]])
print(D)
np.ndim(D)
D.shape

X = np.array([1,2])
W = np.array([[1,3,5],[2,4,6]])
X.shape
W.shape
print(X)
print(W)
Y = np.dot(X, W)
print(Y)

##### 3.4 3층 신경망 구현하기

### 3.4.2 신호 전달 구현

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(X)
print(W1)
print(B1)

print(X.shape)
print(W1.shape)
print(B1.shape)

A1 = np.dot(X, W1) + B1
print(A1)

def sigmoid(x):
    return 1 / (1+np.exp(-x))

Z1 = sigmoid(A1)

print(A1)
print(Z1)

W2 = np.array([[0.1, 0.4],[0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(W2)
print(B2)

print(W2.shape)
print(B2.shape)
print(Z1.shape)

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

def identity_function(x):
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)
print(Y)

### 3.4.3 구현정리

def identity_function(x):
    return x

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot (x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot (z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot (z2, W3) + b3
    y = identity_function(a3)
    
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)

##### 3.5 출력층 설계하기

a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a)
print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)

def softmax(x):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

### 소프트맥스 함수 구현 시 주의점 : overflow

a = np.array([1010, 1000, 990])
np.exp(a) / np.sum(np.exp(a))
c = np.max(a)
a-c
np.exp(a-c) / np.sum(np.exp(a-c))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
np.sum(y)

##### 3.6 손글씨 숫자 인식 (실전 예제)

import sys, os
import numpy as np
os.getcwd()
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(img)
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)

### 3.6.2 신경망의 추론 처리

import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle
from dataset.mnist import load_mnist

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()
print(network)
type(network)
set(network)

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1
        
print("Accuracy = " + str(float(accuracy_cnt) / len(x)))

### 3.6.3 배치 처리

x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
    
print("Accuracy = " + str(float(accuracy_cnt) / len(x)))

list(range(0, 10))
list(range(0, 10, 2))

x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
y = np.argmax(x, axis = 1)
print(y)

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
mean_squared_error(np.array(y), np.array(t))
mean_squared_error(y, t)

(np.array(y)).shape

#######################################################
###################### finish #########################
#######################################################