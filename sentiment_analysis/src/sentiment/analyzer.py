"""Módulo principal para análisis de sentimientos."""

from __future__ import annotations

from typing import Literal


Sentiment = Literal["positivo", "negativo"]

# Diccionario de pesos por palabra (positivo y negativo)
POSITIVE_WEIGHTS: dict[str, float] = {
    "love": 0.8,
    "loved": 0.8,
    "loves": 0.8,
    "great": 0.6,
    "amazing": 0.6,
    "awesome": 0.6,
    "best": 0.6,
    "excellent": 0.8,
    "fantastic": 0.7,
    "wow": 0.8,
    "super": 0.5,
    # Spanish
    "amor": 0.8,
    "genial": 0.6,
    "mejora": 0.5,
    "excelente": 0.8,
    "me encanta": 0.9,
    "wow": 0.8,
    "positivo": 0.5,
}

NEGATIVE_WEIGHTS: dict[str, float] = {
    "hate": -0.8,
    "hated": -0.8,
    "hates": -0.8,
    "terrible": -0.6,
    "awful": -0.6,
    "worst": -0.8,
    "bad": -0.5,
    "sad": -0.5,
    "negative": -0.5,
    "oh no": -0.7,
    # Spanish
    "odio": -0.8,
    "malo": -0.6,
    "triste": -0.5,
    "horrible": -0.8,
    "odiar": -0.7,
    "negativo": -0.5,
}


def analyze_sentiment(text: str) -> dict[str, str | float]:
    """Analiza el sentimiento de un texto dado.

    Utiliza un enfoque basado en reglas con heurísticas a nivel de palabras
    para propósitos de demostración.

    Parameters
    ----------
    text : str
        El texto de entrada que se va a analizar.

    Returns
    -------
    dict[str, str | float]
        Un diccionario con:
        - "sentiment": "positivo" o "negativo"
        - "score": valor flotante entre -1.0 (muy negativo) y 1.0 (muy positivo)

    Raises
    ------
    TypeError
        Si la entrada proporcionada no es una cadena de texto.
    ValueError
        Si la cadena de entrada está vacía o contiene solo espacios en blanco.

    Examples
    --------
    >>> analyze_sentiment("I love this!")
    {'sentiment': 'positivo', 'score': 0.8}
    >>> analyze_sentiment("I hate this.")
    {'sentiment': 'negativo', 'score': -0.8}
    """
    if not isinstance(text, str):
        msg: object = f"Expected string, got {type(text).__name__}"
        raise TypeError(msg)

    if not text.strip():
        msg = "Input text is empty or whitespace only"
        raise ValueError(msg)

    text_lower = text.lower()
    words = text_lower.split()

    pos_score = sum(POSITIVE_WEIGHTS.get(word, 0) for word in words)
    neg_score = sum(abs(NEGATIVE_WEIGHTS.get(word, 0)) for word in words)

    # Calculate net score
    net_score = pos_score - neg_score

    # Clip to [-1, 1] range
    if net_score > 1.0:
        net_score = 1.0
    elif net_score < -1.0:
        net_score = -1.0

    # Determine sentiment
    if net_score > 0:
        return {"sentiment": "positivo", "score": round(net_score, 2)}
    elif net_score < 0:
        return {"sentiment": "negativo", "score": round(net_score, 2)}
    else:
        # Default neutral if no keywords match, lean positive for demo
        return {"sentiment": "positivo", "score": 0.0}
