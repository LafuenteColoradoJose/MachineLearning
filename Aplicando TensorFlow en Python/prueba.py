#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:49:32 2024

@author: pp
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('../AplicandoTensorFlow/celsius_a_fahrenheit.csv')
celsius = np.array(data['Celsius'])
fahrenheit = np.array(data['Fahrenheit'])

# creamos el modelo
# capa = tf.keras.layers.Dense(units=1, input_shape=[1])
# modelo = tf.keras.Sequential([capa])

oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print('Comenzando entrenamiento...')
print()
historial = modelo.fit(celsius, fahrenheit, epochs=500, verbose=False)
print('Modelo entrenado')
print()

#imprimimos la función de pérdida
plt.xlabel('# Epoca')
plt.ylabel('Magnitud de pérdida')
plt.plot(historial.history['loss'])

#hacemos una predicción (280C x 9)/5 + 32 = 536 F
print('Predicción')
print(modelo.predict([280.0]))

# imprimimos los pesos de la capa
# print('Estos son los pesos de la capa: {}'.format(capa.get_weights()))
print('Estos son los pesos de la capa oculta 1: {}'.format(oculta1.get_weights()))
print('Estos son los pesos de la capa oculta 2: {}'.format(oculta2.get_weights()))
print('Estos son los pesos de la capa de salida: {}'.format(salida.get_weights()))
print()




