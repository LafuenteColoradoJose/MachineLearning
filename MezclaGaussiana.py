#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:20:25 2024

@author: pp
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
import numpy as np

sns.set_context("talk", font_scale=1.5)

X, y = make_blobs(n_samples=500, centers=4, cluster_std=2, random_state=2021)

data = pd.DataFrame(X)

data.columns = ['X1', 'X2']
data['cluster'] = y
data.head()

gmm = GaussianMixture(3, covariance_type='full', random_state=0).fit(data[['X1', 'X2']])

gmm.means_

np.array([[-2.16398445, 4.84860401],
    [9.97980069, -7.42299498],
    [-728420067, -386530606]])

labels = gmm.predict(data[['X1', 'X2']])

data["predicted_cluster"] = labels

plt.figure(figsize=(9, 7))

sns.scatterplot(data=data, x='X1', y='X2', hue='predicted_cluster', palette=["red", "blue", "green"])
plt.savefig('fittin_Gaussian_Misture_Models_with_3_components_scikit_learn_Python.png', format='png', dpi=150)

np_components = np.arange(1, 21)

models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(data[['X1', 'X2']]) for n in np_components]

models[0:5]

[GaussianMixture(random_state=0),
 GaussianMixture(n_components=2, random_state=0),
 GaussianMixture(n_components=3, random_state=0),
 GaussianMixture(n_components=4, random_state=0),
 GaussianMixture(n_components=5, random_state=0)]

models[0].bic(X)
models[0].aic(X)

gmm_model_comparisons = pd.DataFrame({'n_components': np_components,
                                        'BIC': [model.bic(X) for model in models],
                                        'AIC': [model.aic(X) for model in models]})
gmm_model_comparisons.head()

plt.figure(figsize=(8, 6))

sns.lineplot(data=gmm_model_comparisons[['BIC', 'AIC']])
plt.xlabel('Number of clústeres')
plt.ylabel('Puntuación')
plt.savefig("GMM_model_copariso_with_AIC_BIC_Scores_Python.png", format='png', dpi=150)

n=4
gmm = GaussianMixture(n, covariance_type='full', random_state=0).fit(data[['X1', 'X2']])
labels = gmm.predict(data[['X1', 'X2']])
data["predicted_cluster"] = labels

plt.figure(figsize=(9, 7))
sns.scatterplot(data=data, x='X1', y='X2', hue='predicted_cluster', palette=["red", "blue", "green", "purple"])
plt.savefig('fittin_Gaussian_Misture_Models_with_4_components_scikit_learn_Python.png', format='png', dpi=150)



