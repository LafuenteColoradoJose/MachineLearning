# Proyecto AEMET: Análisis Climático Histórico ☀️🌧️

Este proyecto es un laboratorio de **Data Science y Machine Learning** centrado en la extracción, limpieza y visualización de datos climáticos históricos reales proporcionados por la Agencia Estatal de Meteorología (AEMET) en España.

Actualmente, el proyecto está configurado para extraer y analizar más de 60 años de observaciones meteorológicas diarias de la estación de **Córdoba Aeropuerto**.

## 🛠️ Configuración Inicial

Para poder descargar los datos, necesitas una clave (API Key) gratuita de la AEMET:

1. Solicita tu API Key en el [Centro de Descargas de AEMET OpenData](https://opendata.aemet.es/centrodedescargas/altaUsuario).
2. En la raíz de este proyecto (`Proyecto_AEMET/`), crea un archivo llamado `.env` (o edita el que ya existe).
3. Añade tu clave de esta forma:
   ```env
   AEMET_API_KEY=tu_clave_larga_aqui
   ```
   *(Nota: El archivo `.env` está ignorado en Git por motivos de seguridad).*

## 📦 Entorno Virtual

Se recomienda usar un entorno virtual de Python con las siguientes librerías instaladas:
- `requests` (para consultar la API)
- `pandas` (para limpieza y manipulación de datos)
- `python-dotenv` (para leer la API Key con seguridad)
- `matplotlib` y `seaborn` (para visualización y gráficos)

## 🗂️ Estructura de Scripts

El proyecto está diseñado como un flujo de trabajo paso a paso para el estudio de datos:

### Fase 1: Extracción de Datos
* **`descargar_datos.py`**: Script base para entender cómo funciona la API de AEMET (ejemplo de descarga de un solo año de Madrid).
* **`descargar_historico_cordoba.py`**: Script robusto que sortea las limitaciones de la API (descargando en bloques de 6 meses) para conseguir el histórico completo desde 1960 hasta la actualidad.

### Fase 2: Limpieza (Data Wrangling)
* **`1_limpieza_datos_cordoba.py`**: Transforma los datos en bruto de AEMET (p. ej., convirtiendo las comas decimales españolas a puntos, arreglando fechas y rellenando valores nulos) y genera el archivo maestro `historico_cordoba_limpio.csv`.

### Fase 3: Visualización (EDA)
* **`2_visualizacion_agosto_cordoba.py`**: Gráfico de dispersión con regresión para estudiar el aumento de las temperaturas máximas durante los meses de agosto.
* **`3_visualizacion_lluvias_meses.py`**: Gráfico de barras que muestra la media de precipitaciones por cada mes (estacionalidad).
* **`4_tendencia_lluvias_anuales.py`**: Analiza si la cantidad total de lluvia anual en Córdoba está descendiendo con las décadas.
* **`5_tendencia_temperatura_anual.py`**: Evolución global de la temperatura media anual para buscar evidencias de calentamiento global a nivel local.

## 📊 Gráficos Generados
Los scripts de visualización generan automáticamente imágenes `.png` numeradas en esta misma carpeta, listas para ser incluidas en presentaciones o informes.

---
*Proyecto creado como entorno de estudio para especialización en Inteligencia Artificial y Big Data.*
