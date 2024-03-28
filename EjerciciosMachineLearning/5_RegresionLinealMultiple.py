#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:03:53 2024

@author: pp
"""

##### LIBRERIAS #####

# importamos
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

##### PREPARAMOS LOS DATOS #####

# importamos los datos de la misma libreria de scikit-learn
california = datasets.fetch_california_housing()
print(california)
print()

##### ENTENDIENDO LOS DATOS  #####

#Verifico la informacion contenidad en el dataset
print('Informacion del dataset')
print(california.keys())
print()

#Verifico las caracteristicas del dataset
print('Caracteristicvas del dataset')
print(california.DESCR)

#Verifico la cantidad de datos del dataset
print('Cantidad de datos:')
print(california.data.shape)
print()

#Verifico la informacion de las columnas
print('Nombre se las columnas')
print(california.feature_names)
# print(california.target_names)

##### PREPARAR LA DATA REGRESION LINEAL MULTIPLE  #####

#Seleccionamos las columnas 2, 3, 4
X_multiple = california.data[:, 2:5]
print(X_multiple)

#Defino los datos correspondientes a las etiquetas
y_multiple = california.target

##### IMPLEMENTACION REGRESION LINEAL MULTIPLE  #####

from sklearn.model_selection import train_test_split

#Separo los datos de "train" en entrenamiento (80%) y prueba (20%) para probar algoritmos
X_train, X_test, y_train, y_test = train_test_split(X_multiple, y_multiple, test_size=0.2)

#Defino el alogitmo a utilizar
lr_multiple = linear_model.LinearRegression()

#Entreno el modelo
lr_multiple.fit(X_train, y_train)

#Realizo una prediccion
Y_pred_multiple = lr_multiple.predict(X_test)


print()
print('DATOS DEL MODELO REGRESION LINEAL MULTIPLE')
print()

print('Valor de las pendientes o coeficientes a')
print(lr_multiple.coef_)

print('Valor de interseccion o coeficiente b')
print(lr_multiple.intercept_)
print()

print('Presicion del modelo')
print(lr_multiple.score(X_train, y_train))

