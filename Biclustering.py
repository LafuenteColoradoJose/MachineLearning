#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:55:06 2024

@author: pp
"""

import numpy as np
nb_users = 100
nb_products = 100

items = [i for i in range(nb_products)]
transactions = []
ratigns = np.zeros(shape=(nb_users, nb_products), dtype=np.int)

for i in range(nb_users):
    n_items = np.random.randint(2, 60)
    transaction = tuple(np.random.choice(items, replace=False, size=n_items))
    transactions.append(list(map(lambda x: "p{}".format(x+1), transaction)))
    for t in transaction:
        rating = np.random.randint(1, 11)
        ratigns[i,t] = rating
        
#Visualizamos mapa de calor
import seaborn as sns
sns.heatmap(ratigns, center=0)

# implementamos el biclsutering espectral con la libreria sklearn
from sklearn.cluster import SpectralBiclustering
sbc = SpectralBiclustering(n_clusters=10, n_best=5, svd_method='arpack', random_state=1000)
sbc.fit(ratigns)

rc = np.outer(np.sort(sbc.row_labels_) + 1, np.sort(sbc.column_labels_) + 1)

#Visualizamos la matriz de datos ordenada en un mapa de calor
sns.heatmap(rc)

print("Usuarios: {}".format(np.where(sbc.rows_[8,:]== True)))
print("Productos: {}".format(np.where(sbc.columns_[8,:]== True)))
