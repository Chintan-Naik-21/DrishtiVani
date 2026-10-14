"""Unit tests for OCRService."""

import pytest
from PIL import Image

from app.db.models import BlockType
from app.services.ocr_service import OCRService


def test_ocr_service_type_classification():
    """Verify visual crop vs text recognition target classification."""
    service = OCRService()

    assert service.is_visual_crop(BlockType.EQUATION.value) is True
    assert service.is_visual_crop(BlockType.DIAGRAM.value) is True
    assert service.is_visual_crop(BlockType.TEXT.value) is False
    assert service.is_visual_crop(BlockType.HEADING_H1.value) is False

    assert service.is_ocr_target(BlockType.TEXT.value) is True
    assert service.is_ocr_target(BlockType.HEADING_H1.value) is True
    assert service.is_ocr_target(BlockType.HEADING_H2.value) is True
    assert service.is_ocr_target(BlockType.LIST_ITEM.value) is True
    assert service.is_ocr_target(BlockType.FIGURE_CAPTION.value) is True
    assert service.is_ocr_target(BlockType.EQUATION.value) is False
    assert service.is_ocr_target(BlockType.DIAGRAM.value) is False


def test_crop_region_bounds():
    """Verify crop_region correctly extracts image region and handles padding."""
    service = OCRService()
    img = Image.new("RGB", (200, 200), color="white")

    crop = service.crop_region(img, [10, 20, 50, 80], padding=2)
    # [10-2, 20-2, 50+2, 80+2] -> 44 x 64
    assert crop.size == (44, 64)

    # Degenerate bounds check
    crop_edge = service.crop_region(img, [-10, -10, 300, 300], padding=0)
    assert crop_edge.size == (200, 200)


def test_image_to_bytes():
    """Verify image_to_bytes returns valid PNG byte stream."""
    service = OCRService()
    img = Image.new("RGB", (50, 50), color="red")
    b = service.image_to_bytes(img, "PNG")

    assert b.startswith(b"\x89PNG")


def test_recognize_text_fallback():
    """Verify recognize_text fallback returns string and confidence."""
    service = OCRService()
    img = Image.new("RGB", (50, 50), color="white")
    text, conf = service.recognize_text(img)

    assert isinstance(text, str)
    assert 0.0 <= conf <= 1.0
