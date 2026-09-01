#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:23:32 2024

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

##### PREPARAR LA DATA VECTORES DE SOPORTE REGRESION   #####

#Seleccionamos solo la columna 2 'AveRooms' variable independiente
X_svr = california.data[:, np.newaxis, 2]

#Defino los datos correspondientes a las etiquetas
y_svr = california.target

#Graficamos los datos
plt.scatter(X_svr, y_svr)
plt.show()

##### IMPLEMENTACION REGRESION VECTORES DE SOPORTE  #####

from sklearn.model_selection import train_test_split

#Separo los datos de "train" en entrenamiento (80%) y prueba (20%) para probar algoritmos
X_train, X_test, y_train, y_test = train_test_split(X_svr, y_svr, test_size=0.2)

from sklearn.svm import SVR

#Defino el alogitmo a utilizar
svr = SVR(kernel='linear', C=1.0, epsilon=0.2)


#Entreno el modelo
svr.fit(X_train, y_train)

#Realizo una prediccion
Y_pred = svr.predict(X_test)

#Graficamos los datos junto con el modelo
plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred, color='red', linewidth=3)
plt.title('Regresion vectores')
plt.xlabel("Numero de habitaciones")
plt.ylabel('Valor medio')
plt.show()

print()
print('DATOS DEL MODELO REGRESION LINEAL SIMPLE')
print()
print('Valor de la pendiente o coeficiente "a" :')
print(svr.coef_)
print('Valor de interseccion o coeficiente b')
print(svr.intercept_)
print()
print('La ecuacion del modelo es igual a:')
print('y= ', svr.coef_, 'x ', svr.intercept_)

#Calculamos la presicion del algoritmo
print()
print('Presicion del modelo')
print(svr.score(X_train, y_train))

