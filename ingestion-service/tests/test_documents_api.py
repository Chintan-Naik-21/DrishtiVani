"""Integration and API tests for document ingestion and status endpoints."""

from unittest.mock import MagicMock, patch
import uuid
import pytest
import pymupdf
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, Document, DocumentStatus, Page
from app.db.session import get_db
from app.main import app
from app.services.pipeline import IngestionPipeline


@pytest.fixture
def test_db_session():
    """Create an isolated in-memory SQLite database for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestingSessionLocal
    app.dependency_overrides.clear()


@pytest.fixture
def sample_pdf_content() -> bytes:
    """Generate a sample valid 2-page PDF."""
    doc = pymupdf.open()
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text((50, 100), "Kirchhoff's Voltage Law")
    p2 = doc.new_page(width=595, height=842)
    p2.insert_text((50, 100), "Nodal Analysis")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_upload_non_pdf_rejected():
    """Verify non-PDF file extensions are rejected with 400 Bad Request."""
    client = TestClient(app)
    response = client.post(
        "/documents",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )
    assert response.status_code == 400
    assert "Only PDF documents are supported" in response.json()["detail"]


def test_upload_invalid_pdf_magic_bytes():
    """Verify files without %PDF- magic bytes are rejected with 400."""
    client = TestClient(app)
    response = client.post(
        "/documents",
        files={"file": ("fake.pdf", b"fake binary content", "application/pdf")},
    )
    assert response.status_code == 400
    assert "PDF signature" in response.json()["detail"]


def test_upload_valid_pdf_success(test_db_session, sample_pdf_content):
    """Verify successful upload stores handout and enqueues background processing."""
    client = TestClient(app)
    mock_storage = MagicMock()

    with patch("app.routers.documents.get_storage_service", return_value=mock_storage), \
         patch("app.routers.documents.get_pipeline") as mock_get_pipeline:

        mock_pipeline = MagicMock()
        mock_get_pipeline.return_value = mock_pipeline

        response = client.post(
            "/documents",
            files={"file": ("circuits_lecture.pdf", sample_pdf_content, "application/pdf")},
            data={"title": "Circuits Lecture 1"},
        )

        assert response.status_code == 202
        data = response.json()
        doc_id = data["document_id"]
        assert doc_id is not None
        assert data["title"] == "Circuits Lecture 1"
        assert data["status"] == "processing"

        # Verify handout uploaded to storage
        mock_storage.upload_file.assert_called_once()
        bucket_arg = mock_storage.upload_file.call_args[1]["bucket"]
        assert bucket_arg == "handouts"

        # Verify background task dispatched
        mock_pipeline.process_document.assert_called_once()


def test_get_document_status_found_and_not_found(test_db_session):
    """Verify GET /documents/{id} returns metadata or 404 if absent."""
    client = TestClient(app)
    session = test_db_session()

    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="Ohm's Law Handout",
        original_filename="ohms_law.pdf",
        storage_path=f"{doc_id}/ohms_law.pdf",
        page_count=3,
        status=DocumentStatus.PARSED.value,
    )
    session.add(doc)
    session.commit()
    session.close()

    # Found
    res = client.get(f"/documents/{doc_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["document_id"] == str(doc_id)
    assert data["status"] == "parsed"
    assert data["page_count"] == 3

    # Not found
    non_existent = uuid.uuid4()
    res_404 = client.get(f"/documents/{non_existent}")
    assert res_404.status_code == 404


def test_pipeline_process_document_end_to_end(test_db_session, sample_pdf_content):
    """Verify IngestionPipeline renders pages, uploads to 'pages' bucket, and creates Page records."""
    session = test_db_session()

    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="Thévenin Equivalent",
        original_filename="thevenin.pdf",
        storage_path=f"{doc_id}/thevenin.pdf",
        page_count=0,
        status=DocumentStatus.PROCESSING.value,
    )
    session.add(doc)
    session.commit()

    mock_storage = MagicMock()
    pipeline = IngestionPipeline(storage_service=mock_storage)

    with patch("app.services.pipeline.get_session_factory", return_value=test_db_session):
        pipeline.process_document(doc_id, sample_pdf_content)

    # Verify document was updated to PARSED and page_count set to 2
    session.refresh(doc)
    assert doc.status == "parsed"
    assert doc.page_count == 2

    # Verify Page records created in DB
    pages = session.query(Page).filter(Page.document_id == doc_id).order_by(Page.page_number).all()
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[0].width_px == 1240
    assert pages[0].height_px == 1755
    assert pages[0].image_path == f"{doc_id}/page_1.png"

    assert pages[1].page_number == 2
    assert pages[1].width_px == 1240
    assert pages[1].height_px == 1755
    assert pages[1].image_path == f"{doc_id}/page_2.png"

    # Verify storage upload called twice for pages bucket
    assert mock_storage.upload_file.call_count == 2
    for call in mock_storage.upload_file.call_args_list:
        assert call[1]["bucket"] == "pages"
        assert call[1]["content_type"] == "image/png"

    session.close()
