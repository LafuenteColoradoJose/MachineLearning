"""Módulo CLI para el análisis de sentimientos.

Proporciona una interfaz de línea de comandos para analizar
textos directamente desde la terminal.
"""

from __future__ import annotations

import sys
from typing import Optional

from .analyzer import analyze_sentiment


def strip_quotes(text: str) -> str:
    """Elimina las comillas que rodean un texto, manejando varios tipos de comillas.

    Parameters
    ----------
    text : str
        La cadena de texto original.

    Returns
    -------
    str
        El texto sin las comillas que lo rodean (si las hubiera).
    """
    # Standard quotes
    if len(text) >= 2:
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            return text[1:-1]
    
    # Smart quotes (Unicode)
    smart_quotes = [
        '\u201c', '\u201d',  # Double smart quotes
        '\u2018', '\u2019',  # Single smart quotes
        '\u00ab', '\u00bb',  # Guillemets
    ]
    
    for q in smart_quotes:
        if text.startswith(q) and text.endswith(q):
            return text[1:-1]
    
    # Angle brackets
    if text.startswith('\u300A') and text.endswith('\u300B'):  # Japanese corners
        return text[1:-1]
    if text.startswith('\u300C') and text.endswith('\u300D'):  # Chinese corners
        return text[1:-1]
    
    return text


def main() -> Optional[int]:
    """Punto de entrada principal para la interfaz de línea de comandos (CLI).

    Recoge el texto desde los argumentos del sistema, lo procesa
    (eliminando comillas si es necesario), y ejecuta el análisis
    de sentimiento imprimiendo el resultado en la consola.

    Returns
    -------
    Optional[int]
        El código de salida del programa (0 para éxito, 1 para error).
    """
    if len(sys.argv) < 2:
        print("Uso: sentiment-analysis <texto>")
        print("Ejemplo: sentiment-analysis \"I love this project\"")
        return 1

    text = " ".join(sys.argv[1:])

    # Remove surrounding quotes if present
    text = strip_quotes(text)

    try:
        result = analyze_sentiment(text)
        print(f"Texto: {text}")
        print(f"Sentimiento: {result['sentiment']}")
        print(f"Score: {result['score']}")
        return 0
    except (TypeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
