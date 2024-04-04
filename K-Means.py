#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:38:13 2024

@author: pp
"""
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Datos de entrenamiento
celsius = np.array([-40, -10, 0, 8, 15, 22, 38]).reshape(-1, 1)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100]).reshape(-1, 1)

# Crear y entrenar el modelo KMeans
kmeans = KMeans(n_clusters=1)
kmeans.fit(celsius)

# Predecir la temperatura en grados Fahrenheit
celsius_to_convert = np.array([25]).reshape(-1, 1)
predicted_cluster = kmeans.predict(celsius_to_convert)
predicted_fahrenheit = fahrenheit[predicted_cluster]

print(f"La temperatura en grados Fahrenheit es: {predicted_fahrenheit[0]}")


# Crear una gráfica
plt.scatter(celsius, fahrenheit, color='blue')  # puntos de entrenamiento
plt.scatter(celsius_to_convert, predicted_fahrenheit, color='red')  # punto de predicción
plt.xlabel('Grados Celsius')
plt.ylabel('Grados Fahrenheit')
plt.show()