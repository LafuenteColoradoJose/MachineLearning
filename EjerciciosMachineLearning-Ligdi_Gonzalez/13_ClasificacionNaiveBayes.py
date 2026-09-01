#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:35:57 2024

@author: pp
"""

# importamos
from sklearn import datasets

##### PREPARAMOS LOS DATOS #####

# importamos los datos de la misma libreria de scikit-learn
dataset = datasets.load_breast_cancer()
print(dataset)
print()

##### ENTENDIENDO LOS DATOS  #####

#Verifico la informacion contenidad en el dataset
print('Informacion del dataset')
print(dataset.keys())
print()

#Verifico las caracteristicas del dataset
print('Caracteristicvas del dataset')
print(dataset.DESCR)

#Verifico la cantidad de datos del dataset
print('Cantidad de datos:')
print(dataset.data.shape)
print()

#Verifico la informacion de las columnas
print('Nombre se las columnas')
print(dataset.feature_names)

##### PREPARAR LA DATA NAIVE BAYES   #####

#Seleccionamos todas las columnas
X = dataset.data

#Defino los datos correspondientes a las etiquetas
y = dataset.target

##### IMPLEMENTACION NAIVE BAYES  #####

from sklearn.model_selection import train_test_split

#Separo los datos de "train" en entrenamiento (80%) y prueba (20%) para probar algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#Defino el alogitmo a utilizar
from sklearn.naive_bayes import GaussianNB

algoritmo = GaussianNB()

#Entreno el modelo
algoritmo.fit(X_train, y_train)

#Realizo una prediccion
y_pred = algoritmo.predict(X_test)


#Verifico la matriz de confusion
from sklearn.metrics import confusion_matrix

matriz = confusion_matrix(y_test, y_pred)
print("Matriz de confucsion")
print(matriz)

#Calculo de presicion del modelo
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred)
print("Precision del modelo")
print(precision)

#Calculo de exactitud del modelo
from sklearn.metrics import accuracy_score

exactitud = accuracy_score(y_test, y_test)
print("Exactitud del modelo")
print(exactitud)

#Calculo la sensibilidad del modelo
from sklearn.metrics import recall_score

sensibilidad = recall_score(y_test, y_pred)
print("Sensiblidad del modelo")
print(sensibilidad)

#Calculo el puntaje F1
from sklearn.metrics import f1_score

puntajeF1 = f1_score(y_test, y_pred)
print("Puntaje F1 del modelo")
print(puntajeF1)

#Calculo la curva ROC - AUC del modelo
from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_test, y_pred)
print("Curva ROC-AUC del modelo")
print(roc_auc)
