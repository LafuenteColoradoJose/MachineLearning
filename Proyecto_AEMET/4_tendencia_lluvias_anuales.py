"""
SCRIPT DE ESTUDIO: TENDENCIA DE PRECIPITACIONES POR AÑO
=======================================================
Objetivo: Comprobar visualmente si llueve cada vez menos en Córdoba.
Calcularemos la lluvia total acumulada en cada año y trazaremos
una línea de tendencia histórica.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("Cargando el dataset limpio para tendencia anual de lluvias...")
df = pd.read_csv("data/historico_cordoba_limpio.csv")

# Aseguramos que no hay nulos en precipitaciones
df['prec'] = df['prec'].fillna(0)

# 1. Calcular la lluvia TOTAL acumulada por cada año
# Sumamos todos los litros caídos agrupando únicamente por 'año'
lluvia_anual = df.groupby('año')['prec'].sum().reset_index()

# 2. Eliminar el año actual en curso (2026) del gráfico
# Esto es CRÍTICO en Machine Learning: si el año en curso no ha terminado, 
# la lluvia sumada será muy baja (porque faltan meses por llover), 
# lo que engañaría a nuestra línea de tendencia tirándola hacia abajo artificialmente.
año_actual = datetime.now().year
lluvia_anual = lluvia_anual[lluvia_anual['año'] < año_actual]

# 3. Configurar el gráfico con Seaborn
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

# Usamos regplot como hicimos con la temperatura, para que nos calcule la tendencia
ax = sns.regplot(
    data=lluvia_anual, 
    x='año', 
    y='prec', 
    scatter_kws={'color': 'teal', 'alpha': 0.8, 's': 40}, # Puntos verdes/azulados
    line_kws={'color': 'red', 'linewidth': 2.5}           # Línea de tendencia en rojo
)

# 4. Añadir un gráfico de línea normal (plot) de fondo para unir los puntos 
# y ver los "picos" de años muy lluviosos y las grandes "sequías".
plt.plot(lluvia_anual['año'], lluvia_anual['prec'], color='teal', alpha=0.3, linewidth=1.5)

# 5. Personalizar etiquetas y título
plt.title("Evolución de la Lluvia Total Anual en Córdoba (Histórico)\n¿Línea roja descendente? Evidencia de menor cantidad de lluvias", fontsize=14, fontweight='bold')
plt.xlabel("Año", fontsize=12)
plt.ylabel("Lluvia Total Acumulada Anual (Litros/m²)", fontsize=12)

# Guardar y mostrar
ruta_grafico = "4_tendencia_lluvias_cordoba.png"
plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
print(f"¡Gráfico guardado en: {ruta_grafico}!")

print("Abriendo la ventana del gráfico...")
plt.show()
