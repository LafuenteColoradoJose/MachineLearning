import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

metros = np.array([1,5,10,20,50,100,200], dtype=float)
pies = np.array([3.28084,16.4042,32.8084,65.6168,164.042,328.084,656.168], dtype=float)

# Crear el modelo secuencial
modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)), # Entrada: un solo valor (la cantidad de metros)
    tf.keras.layers.Dense(units=1) # Capa densa con 1 neurona (salida: cantidad de pies)
])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento");
historial = modelo.fit(metros, pies, epochs=100, verbose=False)
print("Modelo entrenado")

print("Hagamos una prediccion")
resultado = modelo.predict(np.array([15]).reshape(-1, 1))
print("El resultado es " + str(resultado) + " pies")

plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
plt.show()