"""Tests for the Sentiment Analysis API."""

from fastapi.testclient import TestClient
from sentiment.api import app

# Crear cliente de prueba
client = TestClient(app)


class TestAPIRoot:
    """Tests for the root endpoint."""

    def test_read_root(self) -> None:
        """Test the root endpoint returns welcome message."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Bienvenido a la API de Análisis de Sentimientos"


class TestAPIAnalyze:
    """Tests for the analyze endpoint."""

    def test_analyze_positive(self) -> None:
        """Test analyzing positive text."""
        response = client.get("/analyze/i%20love%20python")
        assert response.status_code == 200
        data = response.json()
        assert data["texto"] == "i love python"
        assert data["resultado"]["sentiment"] == "positivo"
        assert data["resultado"]["score"] == 0.8

    def test_analyze_negative(self) -> None:
        """Test analyzing negative text."""
        response = client.get("/analyze/i%20hate%20rain")
        assert response.status_code == 200
        data = response.json()
        assert data["texto"] == "i hate rain"
        assert data["resultado"]["sentiment"] == "negativo"
        assert data["resultado"]["score"] == -0.8

    def test_analyze_with_clipping(self) -> None:
        """Test analyzing text that triggers score clipping."""
        # 5 "love" words should clip score to 1.0
        response = client.get("/analyze/love%20love%20love%20love%20love")
        assert response.status_code == 200
        data = response.json()
        assert data["resultado"]["sentiment"] == "positivo"
        assert data["resultado"]["score"] == 1.0

    def test_analyze_very_negative(self) -> None:
        """Test very negative text."""
        response = client.get("/analyze/hate%20hate%20hate%20hate%20hate")
        assert response.status_code == 200
        data = response.json()
        assert data["resultado"]["sentiment"] == "negativo"
        assert data["resultado"]["score"] == -1.0


class TestAPIHealth:
    """Tests for the health endpoint."""

    def test_health_check(self) -> None:
        """Test the health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "OK"
        assert data["servicio"] == "sentiment-analysis-api"
