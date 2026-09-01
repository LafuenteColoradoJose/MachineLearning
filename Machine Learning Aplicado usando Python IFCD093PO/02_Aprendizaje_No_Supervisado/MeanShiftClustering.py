#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:22:22 2024

@author: pp
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

# Opciones de configuración
num_samples_total = 10000
clusters_centers = [(5,5), (3,3), (1,1)]
num_classes = len(clusters_centers)

# Generar datos
X, targets = make_blobs(num_samples_total, centers=clusters_centers, n_features = num_classes, center_box=(0,1) ,cluster_std=0.30)
np.save('./clusters.npy', X)
X = np.load('./clusters.npy')

#Bandwidth estimado
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

# Entrenar modelo de clustering
meanshift = MeanShift(bandwidth=bandwidth).fit(X)
labels = meanshift.labels_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

# Predecir el cluster para todas las muestras
P = meanshift.predict(X)

# Generar el scatter plot para data training
colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426' if x == 2 else '#67c614', P))
plt.scatter(X[:,0], X[:,1], c=colors, marker='o', picker=True)
plt.title(f'Número estimado de clusters = {n_clusters_}')
plt.xlabel('Temperatura ayer')
plt.ylabel('Temperatura hoy')
plt.show()



