# Naive Bayes

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customer_label.csv')
X = dataset.iloc[:, [4, 5]].values
y = dataset.iloc[:, 6].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

    #Compute accuracy on the training set
train_accuracy = classifier.score(X_train, y_train)

    #Compute accuracy on the testing set
test_accuracy = classifier.score(X_test, y_test)
#print accuracy
print('Train accuracy: ', round(train_accuracy,2))
print('Test accuracy: ', test_accuracy)
print(cm)