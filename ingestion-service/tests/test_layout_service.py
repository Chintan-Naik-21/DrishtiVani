"""Unit tests for LayoutService reading order and heading hierarchy inference."""

from PIL import Image

from app.db.models import BlockType
from app.services.layout_service import LayoutService


def test_sort_reading_order_single_column():
    """Verify top-to-bottom sorting in a single column layout."""
    service = LayoutService()
    boxes = [
        {"bbox": [10.0, 300.0, 100.0, 350.0], "label": "Text", "confidence": 0.9},
        {"bbox": [10.0, 50.0, 100.0, 100.0], "label": "Text", "confidence": 0.9},
        {"bbox": [10.0, 150.0, 100.0, 200.0], "label": "Text", "confidence": 0.9},
    ]

    sorted_boxes = service._sort_reading_order(boxes, 500, 800)
    assert sorted_boxes[0]["bbox"][1] == 50.0
    assert sorted_boxes[1]["bbox"][1] == 150.0
    assert sorted_boxes[2]["bbox"][1] == 300.0


def test_classify_and_infer_heading_tiers():
    """Verify heading level deduction (H1-H3) based on relative font/box height."""
    service = LayoutService()
    boxes = [
        # Large header at top (H1)
        {"bbox": [20.0, 40.0, 300.0, 100.0], "label": "SectionHeader", "confidence": 0.95},
        # Medium header (H2)
        {"bbox": [20.0, 150.0, 250.0, 185.0], "label": "SectionHeader", "confidence": 0.92},
        # Small header (H3)
        {"bbox": [20.0, 250.0, 200.0, 270.0], "label": "SectionHeader", "confidence": 0.90},
        # Body text
        {"bbox": [20.0, 300.0, 400.0, 320.0], "label": "Text", "confidence": 0.88},
    ]

    blocks = service._classify_and_infer_headings(boxes, 1000)
    assert len(blocks) == 4
    assert blocks[0].block_type == BlockType.HEADING_H1.value
    assert blocks[1].block_type == BlockType.HEADING_H2.value
    assert blocks[2].block_type == BlockType.HEADING_H3.value
    assert blocks[3].block_type == BlockType.TEXT.value


def test_detect_layout_heuristic_fallback():
    """Verify detect_layout operates via heuristic fallback when ML models are absent."""
    service = LayoutService()
    img = Image.new("RGB", (600, 800), color="white")

    blocks = service.detect_layout(img, 600, 800)
    assert len(blocks) > 0
    for b in blocks:
        assert b.bbox[0] >= 0
        assert b.bbox[1] >= 0
        assert b.bbox[2] <= 600
        assert b.bbox[3] <= 800
