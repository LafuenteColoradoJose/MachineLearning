"""
SCRIPT DE ESTUDIO: MACHINE LEARNING BÁSICO (REGRESIÓN LINEAL)
=============================================================
Objetivo: Entrenar a una IA sencilla (Regresión Lineal) para que 
aprenda a qué ritmo sube la temperatura media de Córdoba.
Luego le pediremos que prediga qué temperatura hará en 2030, 2040 y 2050.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime

print("1. Cargando el dataset de entrenamiento...")
df = pd.read_csv("../../data/historico_cordoba_limpio.csv")
df = df.dropna(subset=['tmed'])

# Preparamos los datos tal como hicimos en la visualización
temp_anual = df.groupby('año')['tmed'].mean().reset_index()
año_actual = datetime.now().year
temp_anual = temp_anual[temp_anual['año'] < año_actual]

print("2. Separando datos para la Inteligencia Artificial...")
# X (Mayúscula): Las características de entrada (Features). En este caso, el Año.
# Y (Minúscula): Lo que queremos predecir (Target). En este caso, la Temperatura.
# Scikit-Learn requiere que la X tenga formato de matriz de 2 dimensiones, por eso usamos reshape(-1, 1).
X = temp_anual['año'].values.reshape(-1, 1)
y = temp_anual['tmed'].values

print("3. Entrenando el modelo (Regresión Lineal)...")
modelo = LinearRegression()
modelo.fit(X, y)

# Para saber cuánto aprende, podemos ver la pendiente (coeficiente):
subida_por_ano = modelo.coef_[0]
print(f"-> El modelo ha descubierto que la temperatura sube {subida_por_ano:.4f} ºC cada año.")

print("\n4. Prediciendo el futuro...")
# Creamos una lista de años futuros que queremos adivinar
años_futuros = np.array([2030, 2040, 2050, 2060]).reshape(-1, 1)
predicciones = modelo.predict(años_futuros)

print("Resultados de la IA para Córdoba:")
for anio, temp in zip(años_futuros.flatten(), predicciones):
    print(f"- Año {anio}: Temperatura media anual de {temp:.2f} ºC")

print("\n5. Dibujando el resultado final...")
plt.figure(figsize=(10, 5))
plt.style.use('ggplot')

# Dibujamos los datos históricos reales en azul
plt.scatter(X, y, color='dodgerblue', label='Datos Históricos (Reales)')

# Dibujamos la línea que ha aprendido nuestro modelo desde 1960 hasta 2060
todos_los_anos = np.arange(1960, 2061).reshape(-1, 1)
todas_predicciones = modelo.predict(todos_los_anos)
plt.plot(todos_los_anos, todas_predicciones, color='red', linewidth=3, label='Predicción del Modelo (IA)')

plt.title("Predicción de Temperatura Media Anual en Córdoba (Machine Learning)", fontweight='bold')
plt.xlabel("Año")
plt.ylabel("Temperatura Media Anual (ºC)")
plt.legend()

ruta_grafico = "6_prediccion_temperatura.png"
plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
print(f"Gráfico guardado en: {ruta_grafico}")

plt.show()
