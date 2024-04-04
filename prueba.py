import numpy as np
from sklearn.linear_model import LinearRegression

# Datos de entrenamiento
celsius = np.array([-40, -10, 0, 8, 15, 22, 38]).reshape(-1, 1)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100]).reshape(-1, 1)

# Crear y entrenar el modelo de regresión lineal
reg = LinearRegression()
reg.fit(celsius, fahrenheit)

# Predecir la temperatura en grados Fahrenheit
celsius_to_convert = np.array([25]).reshape(-1, 1)
predicted_fahrenheit = reg.predict(celsius_to_convert)

print(f"La temperatura en grados Fahrenheit es: {predicted_fahrenheit[0][0]}")