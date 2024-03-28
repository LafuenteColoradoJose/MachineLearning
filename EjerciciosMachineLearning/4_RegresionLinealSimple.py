#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:03:53 2024

@author: pp
"""


# y = ax + b
# y es la variable dependiente
# x es la variable independiente
# a es la pendiente o coeficiente
# b es la cosntante o interseccion

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

##### PREPARAR LA DATA REGRESION LINEAL SIMPLE  #####

#Seleccionamos solo la columna 2 'AveRooms' variable independiente
X = california.data[:, np.newaxis, 2]

#Defino los datos correspondientes a las etiquetas
y = california.target

#Graficamos los datos correspondientes en una dispersion
plt.scatter(X, y)
plt.xlabel('Numero de habitaciones')
plt.ylabel('Valor medio')
plt.show()


##### IMPLEMENTACION REGRESION LINEAL SIMPLE  #####

from sklearn.model_selection import train_test_split

#Separo los datos de "train" en entrenamiento (80%) y prueba (20%) para probar algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Defino el alogitmo a utilizar
lr = linear_model.LinearRegression()

#Entreno el modelo
lr.fit(X_train, y_train)

#Realizo una prediccion
Y_pred = lr.predict(X_test)

#Graficamos los datos junto con el modelo
plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred, color='red', linewidth=3)
plt.title('Regesion lineal simple')
plt.xlabel("Numero de habitaciones")
plt.ylabel('Valor medio')
plt.show()

print()
print('DATOS DEL MODELO REGRESION LINEAL SIMPLE')
print()
print('Valor de la pendiente o coeficiente "a" :')
print(lr.coef_)
print('Valor de interseccion o coeficiente b')
print(lr.intercept_)
print()
print('La ecuacion del modelo es igual a:')
print('y= ', lr.coef_, 'x ', lr.intercept_)

#Calculamos la presicion del algoritmo
print()
print('Presicion del modelo')
print(lr.score(X_train, y_train))

