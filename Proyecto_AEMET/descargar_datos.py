import os
import requests
import pandas as pd
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()
API_KEY = os.getenv('AEMET_API_KEY')

def obtener_datos_estacion(id_estacion, fecha_ini, fecha_fin):
    """
    Descarga datos climatológicos diarios de una estación específica.
    Formatos de fecha esperados: 'YYYY-MM-DDTHH:MM:SSUTC'
    Ejemplo: '2023-01-01T00:00:00UTC'
    """
    # 1. Hacemos la primera petición para que AEMET nos dé la URL de descarga
    url_base = f"https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/{fecha_ini}/fechafin/{fecha_fin}/estacion/{id_estacion}"
    
    headers = {
        'cache-control': "no-cache"
    }
    querystring = {"api_key": API_KEY}
    
    print("Solicitando acceso a los datos...")
    respuesta = requests.get(url_base, headers=headers, params=querystring)
    
    if respuesta.status_code == 200:
        datos_json = respuesta.json()
        
        # 2. AEMET devuelve una URL temporal en 'datos' de donde descargar la información real
        if 'datos' in datos_json:
            url_datos = datos_json['datos']
            print("Acceso concedido. Descargando datos finales...")
            
            respuesta_datos = requests.get(url_datos)
            if respuesta_datos.status_code == 200:
                # Convertimos el JSON final directamente a un DataFrame de Pandas
                df = pd.DataFrame(respuesta_datos.json())
                return df
            else:
                print("Error al descargar los datos finales.")
        else:
            print("No se encontraron datos para los parámetros indicados:", datos_json.get('descripcion'))
    else:
        print(f"Error de conexión: {respuesta.status_code}")
        print(respuesta.text)
        
    return None

if __name__ == "__main__":
    # Ejemplo: Estación meteorológica del Retiro en Madrid (ID: 5402)
    # Rango de fechas: Todo el año 2023
    ESTACION_RETIRO = "5402"
    INICIO = "2023-01-01T00:00:00UTC"
    FIN = "2023-06-30T23:59:59UTC"
    
    if API_KEY == "pon_aqui_tu_api_key_larga" or API_KEY is None:
        print("¡ATENCIÓN! Debes poner tu API Key real en el archivo .env antes de ejecutar esto.")
    else:
        df_clima = obtener_datos_estacion(ESTACION_RETIRO, INICIO, FIN)
        
        if df_clima is not None:
            print("\n¡Datos descargados con éxito!\n")
            print(df_clima.head())
            
            # Guardamos a CSV para no tener que volver a llamar a la API
            archivo_salida = "data/clima_cordoba_aeropuerto_2023.csv"
            df_clima.to_csv(archivo_salida, index=False)
            print(f"\nDatos guardados en {archivo_salida}")
