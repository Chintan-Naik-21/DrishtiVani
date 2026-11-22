"""
Integration tests for Module 4: api.py - FastAPI Microservice & JSON Schema Compliance
"""

import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from diagram_engine.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "diagram-engine"


def test_process_diagram_endpoint_schema():
    # Generate a synthetic schematic image (300x300 white canvas with dark wires)
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    # Draw simple horizontal wire
    cv2.line(img, (50, 150), (250, 150), (20, 20, 20), 3)

    _, encoded = cv2.imencode(".png", img)
    img_bytes = encoded.tobytes()

    response = client.post(
        "/process-diagram",
        files={"file": ("test_schematic.png", io.BytesIO(img_bytes), "image/png")},
        data={"diagram_id": "test-diag-101"},
    )

    assert response.status_code == 200
    data = response.json()

    # Verify exact schema structure required by user specification:
    assert data["diagram_id"] == "test-diag-101"
    assert "confidence_score" in data
    assert isinstance(data["confidence_score"], float)
    assert "needs_teacher_review" in data
    assert isinstance(data["needs_teacher_review"], bool)
    assert "summary" in data
    assert isinstance(data["summary"], str)
    assert "spice_netlist" in data
    assert isinstance(data["spice_netlist"], str)
    assert "nodes" in data
    assert isinstance(data["nodes"], list)

    for node in data["nodes"]:
        assert "node_id" in node
        assert "description" in node
        assert "connected_components" in node
        for conn in node["connected_components"]:
            assert "component_id" in conn
            assert "class" in conn
            assert "terminal" in conn
