# 1 - Para instalar TensorFlow
# ejecutamos: pip install tensorflow

# 2 - Importamos las librerías necesarias
import tensorflow as tf
import keras as kr
import pandas as pd
from sklearn.metrics import mean_squared_error

# 3 - Obtenemos los datos del CSV
data = pd.read_csv('./AplicandoTensorFlow/celsius_a_fahrenheit.csv')
# print(data)

# 4 - Compilación del modelo


# Definir la arquitectura del modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# Compilar el modelo
model.compile(loss='mean_squared_error', optimizer='adam')

# Entrenar el modelo
model.fit(data['Celsius'], data['Fahrenheit'], epochs=10, verbose=False)


# Evaluar el modelo en los datos de entrenamiento
loss = model.evaluate(data['Celsius'], data['Fahrenheit'], verbose=0)

print()
print(f'Pérdida del modelo en los datos de entrenamiento: {loss}')

# También puedes hacer predicciones y compararlas con los valores reales
predictions = model.predict(data['Celsius'])
mse = mean_squared_error(data['Fahrenheit'], predictions)

print()
print(f'Error cuadrático medio de las predicciones: {mse}')