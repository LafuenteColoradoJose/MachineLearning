#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:32:54 2024

@author: pp
"""

from sklearn.datasets import make_blobs
from sklearn.cluster import OPTICS
import numpy as np
import matplotlib.pyplot as plt

# Opciones de configuración
num_samples_total = 1000
clusters_centers = [(3,3), (7,7)]
num_classes = len(clusters_centers)
epsilon = 2.0
min_samples = 22
cluster_method = 'xi'
metric = 'minkowski'

# Generar datos
X, y = make_blobs(num_samples_total, centers=clusters_centers, n_features = num_classes, center_box=(0,1) ,cluster_std=0.5)

#Computar Optics
db = OPTICS(eps=epsilon, min_samples=min_samples, cluster_method=cluster_method, metric=metric).fit(X)
labels = db.labels_
no_clusters = len(np.unique(labels))
no_noise = np.sum(np.array(labels) == -1, axis=0)

# Generar el scatter plot para data training
colors = list(map(lambda x: '#3b4cc0' if x == 0 else '#b40426', labels))
plt.scatter(X[:,0], X[:,1], c=colors, marker='o', picker=True)
plt.title(f'OPTICS clustering')
plt.xlabel('Axis X[0]')
plt.ylabel('Axis X[1]')
plt.show()

# Generar gráfica de accesibilidad
reachability = db.reachability_[db.ordering_]
plt.plot(reachability)
plt.title('Reachability plot')
plt.xlabel('Data points')
plt.ylabel('Reachability')
plt.show()
