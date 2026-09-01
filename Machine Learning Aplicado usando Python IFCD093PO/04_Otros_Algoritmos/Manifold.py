#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:20:22 2024

@author: pp
"""

from collections import OrderedDict
from functools import partial
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets

Axes3D
n_points = 1000
X, color = datasets.make_s_curve(n_points, random_state=0)
n_neighbors = 10
n_components = 2

#Create figure
fig = plt.figure(figsize=(15, 8))
plt.suptitle("Manifold Learning with %i points, %i neighbors"
             % (1000, n_neighbors), fontsize=14)

# 3D scatter plot
ax = fig.add_subplot(251, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.view_init(4, -72)

#Configurar métodos de reducción de dimensionalidad
LLE = partial(manifold.LocallyLinearEmbedding,
              n_neighbors, n_components, eigen_solver='auto')
metgods = OrderedDict()
metgods['LLE'] = LLE(method='standard')
metgods['LTSA'] = LLE(method='ltsa')
metgods['Hessian LLE'] = LLE(method='hessian')
metgods['Modified LLE'] = LLE(method='modified')
metgods['Isomap'] = manifold.Isomap(n_neighbors, n_components)
metgods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=1)
metgods['SE'] = manifold.SpectralEmbedding(n_components=n_components,
                                           n_neighbors=n_neighbors)
metgods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca',
                                    random_state=0)

#Dibujar Resusltados
for i, (label, method) in enumerate(metgods.items()):
    t0 = time()
    Y = method.fit_transform(X)
    t1 = time()
    print("%s: %.2g sec" % (label, t1 - t0))
    ax = fig.add_subplot(2, 5, 2 + i + (i > 3))
    plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    plt.title("%s (%.2g sec)" % (label, t1 - t0))
    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')
    
plt.show()
    