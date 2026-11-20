"""Unit tests for the /health endpoint."""

from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_health_check_all_healthy():
    """Verify /health returns 200 when both DB and Storage checks succeed."""
    with patch("app.main.check_db_connectivity", return_value=True), \
         patch("app.main.get_storage_service") as mock_get_storage:

        mock_storage = MagicMock()
        mock_storage.check_storage_connectivity.return_value = True
        mock_get_storage.return_value = mock_storage

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"]["status"] == "connected"
        assert data["storage"]["status"] == "connected"
        assert "handouts" in data["storage"]["buckets"]
        assert "pages" in data["storage"]["buckets"]
        assert "crops" in data["storage"]["buckets"]


def test_health_check_db_unhealthy():
    """Verify /health returns 503 and mentions project pausing when DB connectivity fails."""
    with patch("app.main.check_db_connectivity", side_effect=RuntimeError("Connection timed out")), \
         patch("app.main.get_storage_service") as mock_get_storage:

        mock_storage = MagicMock()
        mock_storage.check_storage_connectivity.return_value = True
        mock_get_storage.return_value = mock_storage

        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["database"]["status"] == "disconnected"
        assert "paused" in data["database"]["error"].lower()


def test_health_check_storage_unhealthy():
    """Verify /health returns 503 when Supabase Storage is unreachable."""
    with patch("app.main.check_db_connectivity", return_value=True), \
         patch("app.main.get_storage_service") as mock_get_storage:

        mock_storage = MagicMock()
        mock_storage.check_storage_connectivity.return_value = False
        mock_get_storage.return_value = mock_storage

        response = client.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["storage"]["status"] == "unreachable"
