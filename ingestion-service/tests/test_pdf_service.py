"""Unit tests for PDFService."""

import pytest
import pymupdf
from app.services.pdf_service import PDFService


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Create a minimal 2-page in-memory PDF document."""
    doc = pymupdf.open()
    # Page 1
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text((50, 100), "Kirchhoff's Current Law")
    # Page 2
    p2 = doc.new_page(width=595, height=842)
    p2.insert_text((50, 100), "Mesh Current Analysis")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_get_page_count(sample_pdf_bytes):
    """Verify get_page_count accurately counts pages."""
    service = PDFService()
    assert service.get_page_count(sample_pdf_bytes) == 2


def test_get_page_count_invalid_bytes():
    """Verify get_page_count raises ValueError on invalid PDF data."""
    service = PDFService()
    with pytest.raises(ValueError):
        service.get_page_count(b"not a valid pdf")


def test_render_page_png(sample_pdf_bytes):
    """Verify rendering a single page as PNG produces valid dimensions and bytes."""
    service = PDFService()
    img_bytes, width, height = service.render_page(sample_pdf_bytes, page_number=1, dpi=150, image_format="PNG")

    # A4 at 150 DPI is approximately 1240 x 1755 px
    assert width == 1240
    assert height == 1755
    assert len(img_bytes) > 0
    # PNG signature check
    assert img_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_render_page_jpeg(sample_pdf_bytes):
    """Verify rendering a single page as JPEG produces valid dimensions and bytes."""
    service = PDFService()
    img_bytes, width, height = service.render_page(sample_pdf_bytes, page_number=2, dpi=150, image_format="JPEG")

    assert width == 1240
    assert height == 1755
    assert len(img_bytes) > 0
    # JPEG SOI marker check
    assert img_bytes.startswith(b"\xff\xd8")


def test_render_page_out_of_bounds(sample_pdf_bytes):
    """Verify render_page raises IndexError for invalid page numbers."""
    service = PDFService()
    with pytest.raises(IndexError):
        service.render_page(sample_pdf_bytes, page_number=99, dpi=150)


def test_iter_pages(sample_pdf_bytes):
    """Verify iter_pages yields all pages in sequence with correct metadata."""
    service = PDFService()
    pages = list(service.iter_pages(sample_pdf_bytes, dpi=150, image_format="PNG"))

    assert len(pages) == 2
    p1_num, p1_bytes, p1_w, p1_h = pages[0]
    assert p1_num == 1
    assert p1_w == 1240
    assert p1_h == 1755
    assert p1_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    p2_num, p2_bytes, p2_w, p2_h = pages[1]
    assert p2_num == 2
    assert p2_w == 1240
    assert p2_h == 1755
    assert p2_bytes.startswith(b"\x89PNG\r\n\x1a\n")
