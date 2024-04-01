#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 09:52:43 2024

@author: pp
"""



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

##### PREPARAR LA DATA ARBOLES ALEATORIOS REGRESION   #####

#Seleccionamos solo la columna 2 'AveRooms' variable independiente
X_bar = california.data[:, np.newaxis, 2]

#Defino los datos correspondientes a las etiquetas
y_bar = california.target

#Graficamos los datos
plt.scatter(X_bar, y_bar)
plt.show()

##### IMPLEMENTACION REGRESION ARBOLES ALEATORIOS  #####

from sklearn.model_selection import train_test_split

#Separo los datos de "train" en entrenamiento (80%) y prueba (20%) para probar algoritmos
X_train, X_test, y_train, y_test = train_test_split(X_bar, y_bar, test_size=0.2)

from sklearn.ensemble import RandomForestRegressor

#Defino el alogitmo a utilizar
bar = RandomForestRegressor(n_estimators= 300, max_depth=12)

#Entreno el modelo
bar.fit(X_train, y_train)

#Realizo una prediccion
Y_pred = bar.predict(X_test)

#Graficamos los datos junto con el modelo
X_grid = np.arange(min(X_test), max(X_test), 0.1 )
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test)
plt.plot(X_grid, bar.predict(X_grid), color='red', linewidth=3)
plt.title('Regresion arboles aleatorios')
plt.xlabel("Numero de habitaciones")
plt.ylabel('Valor medio')
plt.show()


#Calculamos la presicion del algoritmo
print()
print('Presicion del modelo')
print(bar.score(X_train, y_train))




