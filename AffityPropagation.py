#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:08:26 2024

@author: pp
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.datasets import make_blobs

# Opciones de configuración
num_samples_total = 50
cluster_centers = [(20,20), (4,4)]
num_classes = len(cluster_centers)

# Generar datos
X, targets = make_blobs(num_samples_total, centers=cluster_centers, n_features = num_classes, center_box=(0,1) ,cluster_std=1) 
np.save('./clusters.npy', X)
X = np.load('./clusters.npy')

# Entrenar modelo de clustering
afprop = AffinityPropagation(preference=250).fit(X)
cluster_centers_indices = afprop.cluster_centers_indices_
n_clusters_ = len(cluster_centers_indices)

# Predecir el cluster para todas las muestras
P = afprop.predict(X)

# Generar el scatter plot para data training
colors = list(map(lambda x: '#3b4cc0' if x == 0 else '#b40426', P))
plt.scatter(X[:,0], X[:,1], c=colors, marker='o', picker=True)
plt.title(f'Número estimado de clusters = {n_clusters_}')
plt.xlabel('Temperatura ayer')
plt.ylabel('Temperatura hoy')
plt.show()

