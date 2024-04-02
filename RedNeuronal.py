#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:46:02 2024

@author: pp
"""

#importamos librerias

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset  = pd.read_csv('/home/pp/Escritorio/Proyectos/MachineLearning/REC017260.csv')

X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()

X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2 = LabelEncoder()

X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer

onehotencoder = OneHotEncoder(categories = [1])

X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:]



print(X)

#Divivir el dataset en Training set y Test Set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train = train_test_split(X,Y, test_size=0.2)

#ajustamos los datos escalandolos
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

