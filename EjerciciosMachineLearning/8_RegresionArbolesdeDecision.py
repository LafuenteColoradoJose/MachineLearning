#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:04:03 2024

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

##### PREPARAR LA DATA ARBOLES DE DESICION REGRESION   #####

#Seleccionamos solo la columna 2 'AveRooms' variable independiente
X_adr = california.data[:, np.newaxis, 2]

#Defino los datos correspondientes a las etiquetas
y_adr = california.target

#Graficamos los datos
plt.scatter(X_adr, y_adr)
plt.show()

##### IMPLEMENTACION REGRESION ARBOLES DE DESICION  #####

from sklearn.model_selection import train_test_split

#Separo los datos de "train" en entrenamiento (80%) y prueba (20%) para probar algoritmos
X_train, X_test, y_train, y_test = train_test_split(X_adr, y_adr, test_size=0.2)

from sklearn.tree import DecisionTreeRegressor

#Defino el alogitmo a utilizar
adr = DecisionTreeRegressor(max_depth=5)


#Entreno el modelo
adr.fit(X_train, y_train)

#Realizo una prediccion
Y_pred = adr.predict(X_test)

#Graficamos los datos junto con el modelo
X_grid = np.arange(min(X_test), max(X_test), 0.1 )
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test)
plt.plot(X_grid, adr.predict(X_grid), color='red', linewidth=3)
plt.title('Regresion arboles de desiciones')
plt.xlabel("Numero de habitaciones")
plt.ylabel('Valor medio')
plt.show()

# print()
# print('DATOS DEL MODELO REGRESION ARBOLES DE DESICION')
# print()


#Calculamos la presicion del algoritmo
print()
print('Presicion del modelo')
print(adr.score(X_train, y_train))