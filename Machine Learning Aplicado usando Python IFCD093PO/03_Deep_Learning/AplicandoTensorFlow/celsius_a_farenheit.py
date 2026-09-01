# 1 - Para instalar TensorFlow
# ejecutamos: pip install tensorflow

# 2 - Importamos las librerías necesarias
import tensorflow as tf
import keras as kr
import pandas as pd

# 3 - Obtenemos los datos del CSV
data = pd.read_csv('../Aplicando TensorFlow en Python/celsius_a_fahrenheit.csv')
print(data)

# 4 - Compilación del modelo


# Definir la arquitectura del modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# Compilar el modelo
model.compile(loss='mean_squared_error', optimizer='adam')

# Entrenar el modelo
model.fit(data['Celsius'], data['Fahrenheit'], epochs=10)