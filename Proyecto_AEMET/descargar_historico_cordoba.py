import os
import requests
import pandas as pd
from dotenv import load_dotenv
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta

load_dotenv()
API_KEY = os.getenv('AEMET_API_KEY')

def descargar_chunk(id_estacion, fecha_ini, fecha_fin):
    url_base = f"https://opendata.aemet.es/opendata/api/valores/climatologicos/diarios/datos/fechaini/{fecha_ini}T00:00:00UTC/fechafin/{fecha_fin}T23:59:59UTC/estacion/{id_estacion}"
    
    headers = {'cache-control': "no-cache"}
    querystring = {"api_key": API_KEY}
    
    respuesta = requests.get(url_base, headers=headers, params=querystring)
    
    if respuesta.status_code == 200:
        datos_json = respuesta.json()
        if 'datos' in datos_json:
            url_datos = datos_json['datos']
            res_datos = requests.get(url_datos)
            if res_datos.status_code == 200:
                try:
                    return pd.DataFrame(res_datos.json())
                except:
                    return pd.DataFrame()
    elif respuesta.status_code == 429:
        print("Límite de peticiones alcanzado. Esperando 10 segundos...")
        time.sleep(10)
        return descargar_chunk(id_estacion, fecha_ini, fecha_fin)
    
    return pd.DataFrame()

if __name__ == "__main__":
    ESTACION = "5402" # Córdoba Aeropuerto
    
    # Configuramos para empezar en 1970 (fecha segura) hasta el día de hoy
    fecha_inicio_global = datetime(1960, 1, 1)
    fecha_fin_global = datetime.now()
    
    df_historico = pd.DataFrame()
    
    fecha_actual = fecha_inicio_global
    print(f"Iniciando descarga histórica para la estación {ESTACION}...")
    
    while fecha_actual < fecha_fin_global:
        # Añadir 6 meses menos 1 día para el chunk (límite de AEMET)
        siguiente_fecha = fecha_actual + relativedelta(months=6) - relativedelta(days=1)
        if siguiente_fecha > fecha_fin_global:
            siguiente_fecha = fecha_fin_global
            
        f_ini_str = fecha_actual.strftime("%Y-%m-%d")
        f_fin_str = siguiente_fecha.strftime("%Y-%m-%d")
        
        print(f"Descargando periodo: {f_ini_str} hasta {f_fin_str}...")
        df_chunk = descargar_chunk(ESTACION, f_ini_str, f_fin_str)
        
        if not df_chunk.empty:
            df_historico = pd.concat([df_historico, df_chunk], ignore_index=True)
            
        fecha_actual = siguiente_fecha + relativedelta(days=1)
        
        # Pausa para no saturar la API de AEMET (Límite típico: 50/minuto)
        time.sleep(1.5)
        
    if not df_historico.empty:
        archivo_salida = "data/historico_cordoba_aeropuerto_completo.csv"
        df_historico.to_csv(archivo_salida, index=False)
        print(f"\n¡Proceso finalizado! Se han descargado {len(df_historico)} registros.")
        print(f"Datos guardados en {archivo_salida}")
    else:
        print("\nNo se pudieron obtener datos.")
