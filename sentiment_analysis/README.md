# Análisis de Sentimientos con Python - Proyecto Completo

## 📋 Descripción General

Este proyecto implementa una aplicación de análisis de sentimientos en Python siguiendo principios de **Spec Driven Development (Desarrollo Dirigido por Especificaciones)** y buenas prácticas de ingeniería de software. La aplicación determina si un texto tiene sentimiento positivo o negativo.

## 🏗️ Metodología: Spec Driven Development

Este proyecto fue desarrollado utilizando **Spec Driven Development (Desarrollo Dirigido por Especificaciones)**, una metodología donde:

1. **Se definen las especificaciones primero** - Los tests (specifications) definen el comportamiento esperado antes de escribir cualquier código de implementación
2. **Cycle Rojo-Verde-Refactor**:
   - **Rojo**: Escribir un test que falle (define la especificación)
   - **Verde**: Implementar el código mínimo para que el test pase
   - **Refactor**: Mejorar la estructura del código manteniendo los tests verdes
3. **Retroalimentación continua** - Los tests sirven como especificaciones vivas del sistema
4. **Calidad integrada** - Las herramientas de calidad (flake8, black, coverage) forman parte del workflow diario

### Workflow Utilizado

```mermaid
flowchart TD
    A[Planificar alcance] --> B[Inicializar proyecto]
    B --> C[Escribir test FIRST (Rojo)]
    C --> D[Implementar código mínimo (Verde)]
    D --> E[Refactor con flake8/black]
    E --> F{¿Tests passing?}
    F -- Sí --> G[¿Más features?]
    G -- Sí --> C
    G -- No --> H[Terminar]
```

### Herramientas en el Workflow

| Fase | Herramienta | Propósito |
|------|-------------|-----------|
| **Planificación** | `question` | Definir alcance con usuario |
| **Calidad** | `flake8` | Linter estático |
| **Formateo** | `black` | Formato consistente |
| **Testing** | `pytest` | Ejecutar tests |
| **Cobertura** | `coverage` | Medir cobertura |
| **API** | `fastapi.testclient` | Tests de API |

---

## 📂 Estructura del Proyecto

```
/home/pp/Escritorio/Proyectos/MachineLearning/sentiment_analysis/
├── .venv/                   # Entorno virtual Python 3.12
├── .flake8                  # Configuración linter
├── pyproject.toml           # Metadatos + dependencias
├── setup.py                 # Instalación en modo editable
├── src/
│   └── sentiment/
│       ├── __init__.py
│       ├── analyzer.py      # Lógica principal (100% cobertura)
│       ├── api.py           # FastAPI REST API
│       └── cli.py           # Interfaz línea de comandos
└── tests/
    ├── test_analyzer.py     # 17 tests unitarios
    ├── test_api.py         # 6 tests de API
    └── conftest.py         # Configuración pytest
```

---

## 📚 Qué Se hizo en Cada Fase

### **Fase 1: Inicialización del Proyecto**
- ✅ Configurar entorno virtual `.venv`
- ✅ Instalar herramientas: `black`, `flake8`, `isort`, `pytest`, `coverage`
- ✅ Definir estructura del proyecto
- ✅ **Spec Driven**: Escribir tests antes del implementación
- ✅ Implementar `analyzer.py` con tests primeros
- ✅ Alcance: análisis positivo/negativo de texto

**Tests escritos primero (Spec Driven):**
- `test_basic_sentiments` - 5 casos parametrizados
- `test_empty_string` - Manejo de error
- `test_whitespace_only` - Manejo de error
- `test_none_input` - TypeError
- `test_non_string_input` - TypeError
- `test_case_insensitivity` - Minúsculas/mayúsculas

### **Fase 2: CLI (Interfaz de Línea de Comandos)**
- ✅ Agregar `cli.py` con comando `sentiment-analysis`
- ✅ **Spec Driven**: Tests de borde para CLI
- ✅ Manejo de comillas y caracteres especiales
- ✅ Casos testeados:
  - `sentiment-analysis "i love python"` → positivo 0.8
  - `sentiment-analysis "i hate rain"` → negativo -0.8
  - `sentiment-analysis ""` → Error manejado

### **Fase 3: FastAPI REST API**
- ✅ Instalar `fastapi` y `uvicorn`
- ✅ Crear `api.py` con 3 endpoints
- ✅ **Spec Driven**: Tests de API con `TestClient`
- ✅ Endpoints implementados:
  - `GET /` - Bienvenida
  - `GET /analyze/{text}` - Análisis de sentimiento
  - `GET /health` - Verificación de salud
- ✅ Documentación automática en `/docs` (Swagger UI) y `/redoc` (ReDoc)

**Tests de API escritos primeros:**
- `test_read_root` - Endpoint raíz
- `test_analyze_positive` - Texto positivo
- `test_analyze_negative` - Texto negativo
- `test_analyze_with_clipping` - Score clipping en 1.0
- `test_analyze_very_negative` - Score clipping en -1.0
- `test_health_check` - Health endpoint

### **Fase 4: Tests y Cobertura**
- ✅ Agregar tests de-edge cases al analyzer
- ✅ Tests que cubren clipping de scores (líneas 105, 107)
- ✅ Tests con palabras mixtas, neutrales, single words
- ✅ Lograr **100% cobertura** en analyzer.py
- ✅ Mejorar cobertura total del proyecto

**Nuevos tests de borde añadidos:**
- `test_very_positive_text` - Múltiples "love" → score 1.0
- `test_very_negative_text` - Múltiples "hate" → score -1.0
- `test_mixed_positive_negative` - Palabras equilibradas → score 0.0
- `test_single_positive_word` - Palabra simple positiva
- `test_single_negative_word` - Palabra simple negativa
- `test_only_neutral_text` - Texto sin sentimientos

---

## 🚀 Cómo Ejecutar

### 1. Activar el entorno virtual
```bash
cd /home/pp/Escritorio/Proyectos/MachineLearning/sentiment_analysis
source .venv/bin/activate
```

### 2. Probar el analizador directamente
```bash
# Desde la línea de comando
sentiment-analysis "i love python"
# Output: Texto: i love python, Sentimiento: positivo, Score: 0.8

# O usar Python directamente
python -c "from sentiment.analyzer import analyze_sentiment; print(analyze_sentiment('i hate rain'))"
# Output: {'sentiment': 'negativo', 'score': -0.8}
```

### 3. Ejecutar la API REST
```bash
cd sentiment_analysis
.venv/bin/uvicorn src.sentiment.api:app --reload
# Servidor en http://127.0.0.1:8000

# Probar endpoints
curl http://127.0.0.1:8000/health
curl "http://127.0.0.1:8000/analyze/i%20love%20python"
# Documentación: http://127.0.0.1:8000/docs
```

### 4. Ejecutar las pruebas (workflow completo)
```bash
# Todas las pruebas
.venv/bin/pytest -v

# Con cobertura
.venv/bin/pytest --cov=src/sentiment --cov-report=term-missing tests/

# Ver calidad de código
.venv/bin/flake8 src/sentiment tests/
.venv/bin/black src/sentiment tests/ --check
```

---

## 📊 Resultados Finales

| Métrica | Logrado |
|---------|---------|
| **Metodología** | Spec Driven Development ✅ |
| **Tests unitarios** | 23 tests (17 analyzer + 6 API) ✅ |
| **Cobertura analyzer** | 100% ✅ |
| **Cobertura total** | 35% (en aumento) ✅ |
| **Linter (flake8)** | 0 errores ✅ |
| **Formateador (black)** | Aplicado ✅ |
| **API REST** | 3 endpoints funcionales ✅ |
| **CLI** | Comando `sentiment-analysis` ✅ |
| **Documentación** | `/docs` automática ✅ |

---

## 🛠️ Tecnologías Utilizadas

| Categoría | Herramientas Específicas |
|-----------|-------------------------|
| **Metodología** | Spec Driven Development, TDD |
| **Linting** | flake8, pycodestyle, pyflakes |
| **Formateo** | black |
| **Testing** | pytest, fastapi.testclient, httpx |
| **Cobertura** | coverage |
| **API** | FastAPI, uvicorn |
| **Entorno** | Python 3.12, venv, pip |

---

## 🎯 Objetivos Cumplidos

- [x] Metodología Spec Driven Development documentada
- [x] Aplicación completa de análisis de sentimientos
- [x] 100% cobertura en módulo principal (analyzer)
- [x] Workflow de tests rojo-verde-refactor
- [x] Linter y formatter sin errores
- [x] CLI funcional con comando personalizado
- [x] API REST con FastAPI y documentación automática
- [x] Soporte inglés y español
- [x] Manejo de excepciones y bordes
- [x] Tests de calidad integrados en workflow

---

**Desarrollado con metodología Spec Driven Development y buenas prácticas Python.**

---

## 🤓 Curiosidades de la Sesión de Desarrollo

Datos estadísticos de la sesión de IA en la que se generó y refinó este proyecto:

- **Sesión**: Desarrollo TDD para analisis de sentimiento en Python
- **Proveedor**: OpenCode Zen
- **Modelo**: Nemotron 3.5 Lightning Free
- **Límite de contexto**: 262.144
- **Mensajes totales**: 264 (Usuario: 22 / Asistente: 215)
- **Tokens totales**: 114.894
  - **Tokens de entrada**: 1358
  - **Tokens de salida**: 269
  - **Tokens de razonamiento**: 115
  - **Tokens de caché (lectura/escritura)**: 113.152 / 0
- **Uso**: 44%
- **Costo total**: 0,00 US$
- **Sesión creada**: 30 sept 2026, 10:01
- **Última actividad**: 30 sept 2026, 12:50
