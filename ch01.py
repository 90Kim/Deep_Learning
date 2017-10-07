# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 14:15:02 2017

@author: GooYoung
"""

import os
os.getcwd()
os.chdir("C:/Users/GooYoung/Documents/Python")

############################################################
############ chapter 01 (p.32 1.3.7 if문 부터) #############
############################################################

############################################################
##### 1.3 파이썬 인터프리터 #####

hungry = True
if hungry:
    print("I'm hungry")

hungry = False
if hungry:
    print("I'm hungry")
else:
    print("I'm not hungry")
    print("I'm sleepy")
    
for i in [1,2,3]:
    print(i)
for i in [1,3,2]:
    print(i)
    
a = [1,10]
for i in a:
    print(i)

def hello():
    print("Hello World")
hello()

def hello(object):
    print("Hello " + object + "!")
hello("cat")
hello(i)
hello(a)

def compute(object):
    print(1 + object)
compute("cat")
compute(i)
compute(a)

############################################################
##### 1.4 파이썬 스크립트 파일 #####

print("I'm hungry")

class Man:
	def __init__(self, name):
		self.name = name
		print("Initialized!")
	def hello(self):
		print("Hello " + self.name + "!")
	def goodbye(self):
		print("Good-bye " + self.name + "!")

m = Man("David")
m.hello()
m.goodbye()

############################################################
##### 1.5 넘파이 #####
# 넘파이 라이브러리를 사용하는 이유
# 넘파이 배열 클래스에 편리한 메소드가 있어, 딥러닝을 구현할 때 편함

import numpy as np
x = np.array([1.0, 2.0, 3.0])
print(x)
type(x)

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
x + y
print(x+y)
x - y
x * y
x / y

# 넘파이 배열은 원소별 계산뿐 아니라 넘파이 배열과 수치 하나의 조합으로 된 산술연산도 가능
# 이 기능을 브로드캐스트 라 부름.
x+1
x+y+1
x/2

A = np.array([[1,2],[3,4]])
A
A.shape

A1 = np.array([[1,2,3],[3,4,5]])
A1
A1.shape

A2 = np.array([[1,2,3],[3,4,5],1])
A2
A2.shape

A = np.array([[1,2],[3,4]])
B = np.array([[3,0],[0,6]])
A+B
A*B
print(A)
A*10

A = np.array([[1,2],[3,4]])
B = np.array([10,20])
A*B     # R에서 재활용 법칙? 그거랑 똑같음 ㅇㅇ

X = np.array([[51,55], [14,19], [0,4]])
print(X)
X[0]
X[0][1]

for row in X:
    print(row)

X1 = X.flatten()    # X를 1차원 배열로 변환(평탄화)
X1
X1[np.array([0, 2, 4])]
X1[0,2,4]
X1[list([0, 2, 4])]

# 리스트와 배열의 차이는 ?
X1_list = list([51, 55, 14, 19, 0, 4])
X1_array = np.array([51, 55, 14, 19, 0, 4])

X1 > 15
X1[X1>15]

############################################################
##### 1.6 matplotlib #####

### 1.6.1 단순한 그래프 그리기
import numpy as np
import matplotlib.pyplot as plt

# 데이터 준비
x = np.arange(0, 6, 0.1)
y = np.sin(x)

# 그래프 그리기
plt.plot(x, y)
plt.show()

### 1.6.2 pyplot의 기능
import numpy as np
import matplotlib.pyplot as plt

# 데이터 준비
x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

# 그래프 그리기
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--", label="cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin & cos')
plt.legend()
plt.show()

### 1.6.3 이미지 표시하기
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('C:/Users/GooYoung/Pictures/Ricki_Hall.png')

plt.imshow(img)

os.getcwd()
