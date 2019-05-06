# -*- coding: utf-8 -*-
"""
Created on Sat May  4 11:11:57 2019

@author: norhafiz.yaacob
"""
import numpy as np
from sklearn.externals import joblib

model = joblib.load('iris_job.pkl')

print('Input number: ')

swidth = float(input('Insert sepal witdh: '))
print(swidth)
slength = float(input('Insert sepal length: '))
print(slength)
pwidth = float(input('Insert petal witdh: '))
print(pwidth)
plength =float(input('Insert petal length: '))
print(plength)

features = np.array([swidth,slength,pwidth,plength])

result_iris = model.predict(features.reshape(1,-1))
print(result_iris)

if result_iris == 0:
    print('This Iris-setosa')
elif result_iris == 1:
    print('This Iris-versicolor')
elif result_iris == 2:
    print('This Iris-virginica')
else:
    print('Not applicable')
    
