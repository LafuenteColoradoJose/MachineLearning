"""
SCRIPT DE ESTUDIO: VISUALIZACIÓN DE PRECIPITACIONES POR MES
===========================================================
Objetivo: Entender la estacionalidad de las lluvias en Córdoba.
Calcularemos la lluvia media histórica acumulada para cada mes del año.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

# 1. Cargar los datos limpios
print("Cargando el dataset limpio para analizar lluvias...")
df = pd.read_csv("../data/historico_cordoba_limpio.csv")

# Nos aseguramos de que no haya nulos en precipitaciones rellenando con 0
df['prec'] = df['prec'].fillna(0)

# 2. Calcular la lluvia total por mes y por año
# Primero sumamos toda la lluvia de cada mes de cada año particular
lluvia_mensual = df.groupby(['año', 'mes'])['prec'].sum().reset_index()

# 3. Calcular la media histórica de cada mes
# Ahora calculamos la media de todos los eneros, todos los febreros, etc.
lluvia_promedio = lluvia_mensual.groupby('mes')['prec'].mean().reset_index()

# Para que quede más profesional en el gráfico, cambiamos los números (1, 2, 3...)
# por los nombres abreviados de los meses en español.
nombres_meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
lluvia_promedio['mes_nombre'] = nombres_meses

# 4. Configurar el gráfico con Seaborn
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

# Usamos un gráfico de barras (barplot) ideal para cantidades acumuladas por categoría
ax = sns.barplot(
    data=lluvia_promedio, 
    x='mes_nombre', 
    y='prec', 
    palette="Blues_d" # Una paleta de tonos azules, ideal para lluvia
)

# 5. Personalizar etiquetas y título
plt.title("Precipitación Media Mensual en Córdoba (Histórico 1960-Actualidad)", fontsize=15, fontweight='bold')
plt.xlabel("Mes del Año", fontsize=12)
plt.ylabel("Precipitación Acumulada Media (Litros/m²)", fontsize=12)

# Añadir el valor numérico exacto encima de cada barra
for i in ax.containers:
    ax.bar_label(i, fmt='%.1f', padding=3, color='black', fontsize=10)

# 6. Guardar y mostrar
ruta_grafico = "../2_Visualizacion/3_estacionalidad_lluvias_cordoba.png"
plt.savefig(ruta_grafico, dpi=300, bbox_inches='tight')
print(f"¡Gráfico de precipitaciones guardado en: {ruta_grafico}!")

print("Abriendo la ventana del gráfico...")
plt.show()
