#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:45:55 2024

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

##### PREPARAR LA DATA REGRESION  POLINOMINAL  #####

#Seleccionamos solo la columna 2 'AveRooms' variable independiente
X_p = california.data[:, np.newaxis, 2]

#Defino los datos correspondientes a las etiquetas
y_p = california.target

#Graficamos los datos
plt.scatter(X_p, y_p)
plt.show()

##### IMPLEMENTACION REGRESION POLINOMINAL  #####

from sklearn.model_selection import train_test_split

#Separo los datos de "train" en entrenamiento (80%) y prueba (20%) para probar algoritmos
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_p, y_p, test_size=0.2)

from sklearn.preprocessing import PolynomialFeatures

#Se define el grado del polinomio
poli_reg = PolynomialFeatures(degree=2)


#Se transforma las caracteristicas existentes en caracteristicas de mayor grado
X_train_poli = poli_reg.fit_transform(X_train_p)
X_test_poli = poli_reg.fit_transform(X_test_p)

#Defino el alogitmo a utilizar
pr = linear_model.LinearRegression()

#Entreno el modelo
pr.fit(X_train_poli, y_train_p)

#Realizo una prediccion
Y_pred_p = pr.predict(X_test_poli)

#Graficamos los datos junto con el modelo
plt.scatter(X_test_p, y_test_p)
plt.plot(X_test_p, Y_pred_p, color='red', linewidth=3)
plt.title('Regesion polinominal')
plt.xlabel("Numero de habitaciones")
plt.ylabel('Valor medio')
plt.show()

print()
print('DATOS DEL MODELO REGRESION POLINOMINAL')
print()
print('Valor de la pendiente o coeficiente "a" :')
print(pr.coef_)
print('Valor de interseccion o coeficiente b')
print(pr.intercept_)
print()
print('La ecuacion del modelo es igual a:')
print('y= ', pr.coef_, 'x ', pr.intercept_)

#Calculamos la presicion del algoritmo
print()
print('Presicion del modelo')
print(pr.score(X_train_poli, y_train_p))