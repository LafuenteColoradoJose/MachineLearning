#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 09:17:13 2024

@author: pp
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set()

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.50)

xfit = np.linspace(-1,3, 5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='summer')

for m,b,d in [(1, 0.64, 0.33), (0.5,1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=0.4)
    plt.xlim(-1,3.5);
    
    
from sklearn.svm import SVC

model = SVC(kernel="linear", C=1E10)
model.fit(X, y)

def decision_function(model, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
x = np.linspace(xlim[0], xlim[30],30)





