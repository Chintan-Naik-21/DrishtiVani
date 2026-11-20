"""Integration tests for IngestionPipeline layout parsing, OCR, and DB persistence."""

from unittest.mock import MagicMock, patch
import uuid
import pytest
import pymupdf
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, Block, BlockType, Document, DocumentStatus, Page
from app.services.layout_service import DetectedBlock, LayoutService
from app.services.ocr_service import OCRService
from app.services.pipeline import IngestionPipeline
from app.services.table_service import TableService


@pytest.fixture
def test_db():
    """Create an isolated in-memory SQLite database for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return TestingSessionLocal


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Generate a sample 1-page PDF for test."""
    doc = pymupdf.open()
    p = doc.new_page(width=600, height=800)
    p.insert_text((50, 80), "Superposition Theorem")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_pipeline_with_layout_and_crops(test_db, sample_pdf_bytes):
    """Verify IngestionPipeline creates blocks, crops visual assets, and runs OCR."""
    db = test_db()
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="Superposition Handout",
        original_filename="superposition.pdf",
        storage_path=f"{doc_id}/superposition.pdf",
        page_count=0,
        status=DocumentStatus.PROCESSING.value,
    )
    db.add(doc)
    db.commit()

    mock_storage = MagicMock()
    mock_layout = MagicMock()
    mock_ocr = MagicMock()
    mock_table = MagicMock()

    # Configure layout service mock to return 1 heading, 1 equation, 1 text block
    mock_layout.detect_layout.return_value = [
        DetectedBlock(
            block_type=BlockType.HEADING_H1.value,
            reading_order=0,
            bbox=[40.0, 50.0, 400.0, 100.0],
            layout_confidence=0.95,
            needs_review=False,
            extra={"heading_level": 1},
        ),
        DetectedBlock(
            block_type=BlockType.EQUATION.value,
            reading_order=1,
            bbox=[50.0, 120.0, 300.0, 200.0],
            layout_confidence=0.92,
            needs_review=False,
        ),
        DetectedBlock(
            block_type=BlockType.TEXT.value,
            reading_order=2,
            bbox=[40.0, 220.0, 500.0, 280.0],
            layout_confidence=0.88,
            needs_review=False,
        ),
    ]

    mock_ocr.is_visual_crop.side_effect = lambda t: t in (BlockType.EQUATION.value, BlockType.DIAGRAM.value)
    mock_ocr.is_ocr_target.side_effect = lambda t: t in (BlockType.TEXT.value, BlockType.HEADING_H1.value)
    mock_ocr.recognize_text.return_value = ("Superposition Theorem Statement", 0.94)
    mock_ocr.crop_region.return_value = MagicMock()
    mock_ocr.image_to_bytes.return_value = b"\x89PNGfakecrop"

    pipeline = IngestionPipeline(
        storage_service=mock_storage,
        layout_service=mock_layout,
        ocr_service=mock_ocr,
        table_service=mock_table,
    )

    with patch("app.services.pipeline.get_session_factory", return_value=test_db):
        pipeline.process_document(doc_id, sample_pdf_bytes)

    # Verify Document status updated to parsed
    db.refresh(doc)
    assert doc.status == "parsed"
    assert doc.page_count == 1

    # Verify Page created
    page = db.query(Page).filter(Page.document_id == doc_id).first()
    assert page is not None
    assert page.page_number == 1

    # Verify Blocks persisted
    blocks = db.query(Block).filter(Block.page_id == page.id).order_by(Block.reading_order).all()
    assert len(blocks) == 3

    # Heading block
    assert blocks[0].block_type == "heading_h1"
    assert blocks[0].text == "Superposition Theorem Statement"
    assert blocks[0].reading_order == 0

    # Equation block (visual crop, text must be None, crop_path set)
    assert blocks[1].block_type == "equation"
    assert blocks[1].text is None
    assert blocks[1].crop_path is not None
    assert f"{doc_id}/crops/" in blocks[1].crop_path

    # Text block
    assert blocks[2].block_type == "text"
    assert blocks[2].text == "Superposition Theorem Statement"

    # Verify storage was called for page upload and crop upload
    upload_buckets = [call[1]["bucket"] for call in mock_storage.upload_file.call_args_list]
    assert "pages" in upload_buckets
    assert "crops" in upload_buckets

    db.close()
