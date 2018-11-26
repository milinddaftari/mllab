from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture 
from sklearn.cluster import KMeans
# Importing the dataset
data = pd.read_csv('ex.csv') 
print("Input Data and Shape") 
print(data.shape)
data.head()
print(data.head())
# Getting the values and plotting it 
f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))
print('Graph for whole dataset')
plt.scatter(f1, f2, c='black', s=100)
plt.show()
##########################################
kmeans = KMeans(2, random_state=0)
labels = kmeans.fit(X).predict(X)
centroids = kmeans.cluster_centers_
plt.scatter(X[:, 0], X[:, 1], c=labels, s=40);
print('Graph using Kmeans Algorithm')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
plt.show()
#gmm demo
gmm = GaussianMixture(n_components=2).fit(X)
labels = gmm.predict(X)
probs = gmm.predict_proba(X)
size = 10 * probs.max(1) ** 3
print('Graph using EM Algorithm')
plt.scatter(X[:, 0], X[:, 1], c=labels, s=size, cmap='viridis')
plt.show()
