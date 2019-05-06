# -*- coding: utf-8 -*-
"""
Created on Sat May  4 10:51:56 2019

@author: norhafiz.yaacob
"""

import pandas as pd
import numpy as np

df = pd.read_csv('iris.csv', delimiter = ',', header=None)

iris_input = df.iloc[:,0:4] #[r,c]
iris_target = df.iloc[:,4]

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(iris_target)
encoded_iris_target = encoder.transform(iris_target)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_input, 
                                                    encoded_iris_target, 
                                                    test_size=0.30, random_state=42)

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(10, ),activation='relu',solver='adam',
                      verbose=0, random_state=1,max_iter=1000)
model.fit(X_train, y_train) 

##
y_pred = model.predict(X_test)
y_true = y_test

## mse
from sklearn.metrics import mean_squared_error
result_mse = mean_squared_error(y_true, y_pred)
print('MSE: ', + result_mse)

## acuracy
from sklearn.metrics import accuracy_score
result_acc = accuracy_score(y_true, y_pred)
print('ACC: ', + result_acc)

## confusion matrix
from sklearn.metrics import confusion_matrix
result_confMatrix = confusion_matrix(y_true, y_pred)
print('Conf Mat: ', + result_confMatrix)

#save Model
from sklearn.externals import joblib
joblib.dump(model, 'iris_job.pkl') 
