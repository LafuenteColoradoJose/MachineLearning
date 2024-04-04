#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:00:42 2024

@author: pp
"""


from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

num_samples_total = 1000

cluster_centers = np.array([(3,3), (7,7)])

num_classes = len(cluster_centers)

epsilon = .3
min_samples = 13

X, y = make_blobs(n_samples=num_samples_total, centers=cluster_centers, n_features=num_classes, center_box=(0, 1), cluster_std=0.5)

np.save('./clusters.npy', X)

X = np.load('./clusters.npy')

db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)

labels = db.labels_

no_clusters = (len(np.unique(labels)))

no_noise = np.sum(np.array(labels) == -1, axis=0)

print('Estimated no. of clusters: %d' % no_clusters)
print('Estimated no. of noise points: %d' % no_noise)

range_max = len(X)
X = np.array([X[i] for i in range(0 , range_max) if labels[i] != -1])
labels = np.array([labels[i] for i in range(0 , range_max) if labels[i] != -1])

colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', labels))
plt.scatter(X[:,0], X[:,1], c=colors, marker="o", picker=True)
plt.title(f'Ruido quitado')
plt.xlabel('Eje X[0]')
plt.ylabel('Eje X[1]')
plt.show()


