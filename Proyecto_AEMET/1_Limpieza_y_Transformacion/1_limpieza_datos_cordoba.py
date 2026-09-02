"""
SCRIPT DE ESTUDIO: LIMPIEZA DE DATOS (DATA WRANGLING)
=====================================================
Objetivo: Preparar el dataset "sucio" de AEMET para que los modelos
de Machine Learning puedan procesarlo sin errores matemáticos.
"""

import pandas as pd
import numpy as np

print("--- 1. CARGA DE DATOS ---")
# Leemos el archivo original. 
# Importante: Como es un CSV, le decimos a Pandas cómo se llama el archivo.
ruta_archivo = "../data/historico_cordoba_aeropuerto_completo.csv"
df = pd.read_csv(ruta_archivo)

# Vemos qué aspecto tienen los datos iniciales
print("Primeras filas originales:")
print(df.head())
print("\nTipos de datos originales:")
print(df.dtypes)
print("\n-------------------------------------------------\n")


print("--- 2. EL 'PROBLEMA ESPAÑOL' DE LAS COMAS DECIMALES ---")
# Si miras df.dtypes, verás que la temperatura máxima ('tmax') es de tipo 'object' (texto).
# Esto pasa porque en España escribimos 35,4 en lugar de 35.4. Python necesita puntos.

# Lista de columnas que deberían ser números decimales (temperaturas, precipitaciones...)
columnas_numericas = ['tmed', 'prec', 'tmin', 'tmax', 'velmedia', 'racha']

for col in columnas_numericas:
    # Verificamos si la columna existe en el archivo
    if col in df.columns:
        # 1º: Reemplazamos la coma por un punto. 
        # (El parámetro regex=True es necesario en las nuevas versiones de Pandas)
        df[col] = df[col].astype(str).str.replace(',', '.', regex=True)
        
        # 2º: Reemplazamos palabras extrañas de AEMET. Por ejemplo, a veces ponen 'Ip' 
        # (Inapreciable) cuando llueven un par de gotas. Lo cambiaremos a un 0.
        df[col] = df[col].replace('Ip', '0.0')
        
        # 3º: Convertimos el texto a número decimal real (float). 
        # errors='coerce' significa que si se encuentra algo rarísimo, lo ponga como NaN (Nulo).
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("Fíjate ahora cómo las temperaturas han pasado a float64 (números reales):")
print(df[columnas_numericas].dtypes)
print("\n-------------------------------------------------\n")


print("--- 3. TRATAMIENTO DEL TIEMPO (DATETIME) ---")
# La columna 'fecha' es texto ('2023-01-01'). Si la convertimos a formato 'datetime' oficial,
# luego podremos preguntarle a Pandas: "Dime la media de temperatura del mes de julio de 1990".

if 'fecha' in df.columns:
    df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')
    print("La columna 'fecha' ahora es de tipo:", df['fecha'].dtype)
    
    # Truco pro: Extraer el año y el mes en columnas separadas nos ayudará 
    # muchísimo a hacer predicciones más adelante.
    df['año'] = df['fecha'].dt.year
    df['mes'] = df['fecha'].dt.month
print("\n-------------------------------------------------\n")


print("--- 4. GESTIÓN DE VALORES NULOS (NaN) ---")
# En Machine Learning, un dato vacío (NaN) puede hacer que tu modelo explote y dé error.
# Vamos a ver cuántos datos nos faltan.
print("Valores perdidos por columna:")
print(df[columnas_numericas].isnull().sum())

# Hay muchas formas de tratar nulos. Para este estudio de temperatura:
# Si falta un dato de temperatura media ('tmed'), rellenaremos ese hueco
# usando el dato del día anterior (esto se llama 'forward fill' o ffill).
# Es una aproximación válida porque la temperatura de ayer suele parecerse a la de hoy.
if 'tmed' in df.columns:
    df['tmed'] = df['tmed'].ffill()

# Otra opción drástica: eliminar la fila entera si falta la temperatura máxima.
# df = df.dropna(subset=['tmax'])

print("\n-------------------------------------------------\n")


print("--- 5. GUARDAR LOS DATOS LIMPIOS ---")
# Ahora que tenemos fechas reales y números matemáticos puros,
# guardamos nuestro dataset final preparado para IA.
ruta_salida = "../data/historico_cordoba_limpio.csv"
df.to_csv(ruta_salida, index=False)

print(f"¡Éxito! Tu dataset limpio está guardado en: {ruta_salida}")
print("\nAquí tienes un vistazo de tu tabla lista para Machine Learning:")
print(df[['fecha', 'tmax', 'tmin', 'prec']].head())
