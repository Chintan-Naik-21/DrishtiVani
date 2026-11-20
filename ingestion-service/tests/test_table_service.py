"""Unit tests for TableService."""

from unittest.mock import MagicMock
from PIL import Image

from app.schemas.layout import TableGridSchema
from app.services.table_service import TableService


def test_heuristic_grid_segmentation():
    """Verify heuristic grid segmentation returns valid rows and columns."""
    service = TableService()
    img = Image.new("RGB", (300, 150), color="white")

    grid = service._heuristic_grid_segmentation(img)
    assert len(grid) >= 2
    assert len(grid[0]) >= 2
    # Verify first row marked as header
    assert grid[0][0]["is_header"] is True
    assert grid[1][0]["is_header"] is False


def test_parse_table_coordinates_and_schema():
    """Verify parse_table returns valid TableGridSchema with page coordinate offsets."""
    mock_ocr = MagicMock()
    mock_ocr.recognize_text.return_value = ("10 kOhm", 0.95)

    service = TableService(ocr_service=mock_ocr)
    img = Image.new("RGB", (200, 100), color="white")

    # Table placed at [50, 100, 250, 200] on page
    grid_schema, text = service.parse_table(img, [50.0, 100.0, 250.0, 200.0])

    assert isinstance(grid_schema, TableGridSchema)
    assert len(grid_schema.rows) > 0
    first_cell = grid_schema.rows[0].cells[0]
    assert first_cell.row == 0
    assert first_cell.col == 0
    # Coordinates must be offset by table position (x0 >= 50, y0 >= 100)
    assert first_cell.bbox[0] >= 50.0
    assert first_cell.bbox[1] >= 100.0
    assert first_cell.text == "10 kOhm"
    assert "10 kOhm" in text
