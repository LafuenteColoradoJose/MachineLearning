"""
SCRIPT DE ESTUDIO: VISUALIZACIÓN DE DATOS (MATPLOTLIB & SEABORN)
================================================================
Objetivo: Comprobar visualmente la evolución de la temperatura
máxima media del mes de Agosto en Córdoba desde 1960.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar los datos limpios
print("Cargando el dataset limpio...")
df = pd.read_csv("../data/historico_cordoba_limpio.csv")

# 2. Filtrar los datos: ¡Solo queremos el mes de Agosto!
# Seleccionamos las filas donde la columna 'mes' es igual a 8
df_agosto = df[df['mes'] == 8]

# 3. Agrupar datos (Groupby)
# Queremos calcular la temperatura máxima media por cada año.
# Le decimos a Pandas: Agrupa por 'año', coge la columna 'tmax' y calcula la media (.mean())
evolucion_agosto = df_agosto.groupby('año')['tmax'].mean().reset_index()

# 4. Configurar el estilo del gráfico con Seaborn
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6)) # Tamaño de la figura (ancho, alto)

# 5. Crear el gráfico de dispersión con línea de tendencia (regresión)
# sns.regplot dibuja los puntos y, además, calcula y dibuja la línea de tendencia automáticamente
ax = sns.regplot(
    data=evolucion_agosto, 
    x='año', 
    y='tmax', 
    scatter_kws={'color': 'orange', 'alpha': 0.7, 's': 50}, # Estilo de los puntos
    line_kws={'color': 'red', 'linewidth': 2}               # Estilo de la línea de tendencia
)

# 6. Personalizar etiquetas y título
plt.title("Evolución de la Temperatura Máxima Media en Agosto (Córdoba 1960-Actualidad)\nEvidencia de cambio de tendencia local", fontsize=14, fontweight='bold')
plt.xlabel("Año", fontsize=12)
plt.ylabel("Temperatura Máxima Media (ºC)", fontsize=12)

# Añadir un grid más sutil
plt.grid(True, linestyle='--', alpha=0.6)

# 7. Guardar el gráfico como imagen PNG
ruta_grafico = "../2_Visualizacion/2_evolucion_agosto_cordoba.png"
plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
print(f"¡Gráfico guardado espectacularmente en: {ruta_grafico}!")

# 8. Mostrar el gráfico en pantalla
print("Abriendo la ventana del gráfico...")
plt.show()
