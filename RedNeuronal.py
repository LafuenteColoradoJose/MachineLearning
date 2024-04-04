#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:46:02 2024

@author: pp
"""
# Implementando una red profunda - page 75 PDF
# importamos librerias

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

columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(columnTransformer.fit_transform(X), dtype=object)

X = X[:,1:]

print(X)

#Divivir el dataset en Training set y Test Set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)

#ajustamos los datos escalandolos

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Parte 2 - Construir la Red Neuronal
import keras
from keras.models import Sequential
from keras.layers import Dense

# Inicializar la Red Neuronal
classifier = Sequential()

# Añadir las capas de entrada y primera capa oculta
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Añadir la segunda capa oculta
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Añadir la capa de salida
Geography = 'Spain'
CreditScore = 500
Age = 40
Tenure = 3
Balance = 50000
NumberOfProducts = 2
HasCreditCard = 'Yes'
IsActiveMember = 'Yes'

# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Aplanar y_pred y convertirlo a entero
y_pred = y_pred.flatten().astype(int)

# Convertir y_test a binario
y_test_bin = (y_test > 0.5).astype(int)

#predecir a partir de una observacion
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0,500, 1, 40, 3, 50000, 2, 1, 1, 40000]])))
new_prediction = (new_prediction > 0.5)

# Parte 3 - Evaluar el modelo y calcular predicciones finales
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)





