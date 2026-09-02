"""
SCRIPT DE ESTUDIO: TENDENCIA DE TEMPERATURA MEDIA ANUAL
=======================================================
Objetivo: Ver la evolución histórica de la temperatura media
de todo el año (no solo de un mes) para comprobar el calentamiento global.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("Cargando el dataset limpio...")
df = pd.read_csv("data/historico_cordoba_limpio.csv")

# 1. Eliminar datos nulos de temperatura media si los hubiera
df = df.dropna(subset=['tmed'])

# 2. Calcular la temperatura media agrupada por año
# En lugar de sumar (como con la lluvia), calculamos la media (.mean())
temp_anual = df.groupby('año')['tmed'].mean().reset_index()

# 3. Eliminar el año actual en curso (2026) porque está incompleto
# Si solo hemos pasado invierno/primavera, la media será más baja de lo real.
año_actual = datetime.now().year
temp_anual = temp_anual[temp_anual['año'] < año_actual]

# 4. Configurar el estilo del gráfico con Seaborn
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

# 5. Crear el gráfico de dispersión con línea de tendencia (regresión)
ax = sns.regplot(
    data=temp_anual, 
    x='año', 
    y='tmed', 
    scatter_kws={'color': 'darkorange', 'alpha': 0.8, 's': 50}, 
    line_kws={'color': 'firebrick', 'linewidth': 3}             
)

# Línea de fondo para ver las oscilaciones anuales
plt.plot(temp_anual['año'], temp_anual['tmed'], color='darkorange', alpha=0.3, linewidth=1.5)

# 6. Personalizar etiquetas y título
plt.title("Evolución de la Temperatura Media Anual (Córdoba Histórico)\nEvidencia clara de calentamiento global a nivel local", fontsize=14, fontweight='bold')
plt.xlabel("Año", fontsize=12)
plt.ylabel("Temperatura Media Anual (ºC)", fontsize=12)

# 7. Guardar el gráfico como imagen PNG numerada
ruta_grafico = "5_tendencia_temperatura_anual.png"
plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
print(f"¡Gráfico guardado en: {ruta_grafico}!")

# 8. Mostrar el gráfico
print("Abriendo la ventana del gráfico...")
plt.show()
