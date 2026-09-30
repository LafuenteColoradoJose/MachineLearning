"""Módulo de la API para el análisis de sentimientos.

Expone endpoints RESTful para analizar el sentimiento de textos
utilizando FastAPI.
"""

from fastapi import FastAPI
from .analyzer import analyze_sentiment

app = FastAPI(
    title="Análisis de Sentimientos API",
    description="API para determinar si un texto tiene sentimiento positivo o negativo",
    version="0.1.0"
)

@app.get("/")
def read_root():
    """Endpoint raíz que da la bienvenida a la API.

    Returns
    -------
    dict[str, str]
        Un diccionario con el mensaje de bienvenida.
    """
    return {"message": "Bienvenido a la API de Análisis de Sentimientos"}

@app.get("/analyze/{text}")
def analyze(text: str):
    """Endpoint para analizar el sentimiento de un texto proporcionado.

    Parameters
    ----------
    text : str
        El texto que se desea analizar, pasado como parámetro de la URL.

    Returns
    -------
    dict[str, str | dict]
        Un diccionario que contiene el texto original y el resultado
        del análisis de sentimientos (sentimiento y puntuación).
    """
    result = analyze_sentiment(text)
    return {"texto": text, "resultado": result}

@app.get("/health")
def health_check():
    """Verifica el estado de salud de la API.

    Returns
    -------
    dict[str, str]
        Un diccionario con el estado ("OK") y el nombre del servicio.
    """
    return {"status": "OK", "servicio": "sentiment-analysis-api"}
