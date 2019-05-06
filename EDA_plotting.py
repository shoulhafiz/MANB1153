# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 08:48:48 2019

@author: norhafiz.yaacob
"""

import pandas as pd
import matplotlib.pyplot as plt


# Create data
dataset = pd.read_csv('Mall_Customers.csv')
dataset.info()
dataset.describe()
dataset.describe(include=['object', 'bool'])

dataset['Spending Score (1-100)'].value_counts(normalize=True)

x = dataset.iloc[:,3].values
y = dataset.iloc[:,4].values

#Scatter plot
plt.style.use('dark_background')
plt.scatter(x, y, c='dodgerblue', alpha=0.5)
plt.title('Scatter plot of Annual Income Vs Spending Score')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()

#piechart
gender=dataset['Genre']
colors=['#ff9999','#66b3ff']
gender.value_counts().plot(kind='pie', colors=colors, startangle=90, autopct='%1.1f%%',
                   explode=(0, 0.15), shadow=True, title ='Gender Composition of Customer')

#histogram
age = dataset['Age']
plt.hist(age,color='#66b3ff')
plt.xlabel('Age')
plt.ylabel('Number of Customers')
plt.title('Distribution of Customer Age')



