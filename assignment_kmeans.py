# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:04:30 2019

@author: norhafiz.yaacob
"""
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

#using elbow method to determine optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter=300, 
                    n_init =10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

#plotting wcss outcome
plt.plot(range(1,11), wcss)
plt.title('Optimal K value (Elbow Method)')
plt.xlabel('Number of Clusters')    
plt.ylabel('wcss')
plt.xticks
plt.show()

#Applying Kmeans to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter=300, 
                n_init =10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, 
            c = 'red', label = 'Careful')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, 
            c = 'blue', label = 'Standard')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, 
            c = 'green', label = 'Target')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, 
            c = 'cyan', label = 'Careless')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, 
            c = 'magenta', label = 'Sensible')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

#To csv
dataset_new = dataset.copy()
dataset_new['target']=y_kmeans
dataset_new.to_csv(path_or_buf='Mall_Customer_label.csv', sep=',')

    

    

