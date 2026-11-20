"""Integration tests for Layout Retrieval and Block Verification API endpoints."""

import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, Block, BlockType, Document, DocumentStatus, Page, VerificationStatus
from app.db.session import get_db
from app.main import app


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

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestingSessionLocal
    app.dependency_overrides.clear()


def test_get_document_layout_success(test_db):
    """Verify GET /documents/{id}/layout returns full hierarchical JSON structure."""
    client = TestClient(app)
    db = test_db()

    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="Kirchhoff Laws Handout",
        original_filename="kvl.pdf",
        storage_path=f"{doc_id}/kvl.pdf",
        page_count=1,
        status=DocumentStatus.PARSED.value,
    )
    page_id = uuid.uuid4()
    page = Page(
        id=page_id,
        document_id=doc_id,
        page_number=1,
        width_px=1200,
        height_px=1600,
        dpi=150,
        image_path=f"{doc_id}/page_1.png",
    )
    block_h1 = Block(
        id=uuid.uuid4(),
        page_id=page_id,
        block_type=BlockType.HEADING_H1.value,
        reading_order=0,
        bbox_x0=50,
        bbox_y0=60,
        bbox_x1=600,
        bbox_y1=120,
        text="Chapter 1: Network Theorems",
        layout_confidence=0.98,
        ocr_confidence=0.96,
        needs_review=False,
        verification_status=VerificationStatus.PENDING.value,
    )
    block_eq = Block(
        id=uuid.uuid4(),
        page_id=page_id,
        block_type=BlockType.EQUATION.value,
        reading_order=1,
        bbox_x0=80,
        bbox_y0=150,
        bbox_x1=500,
        bbox_y1=250,
        text=None,
        crop_path=f"{doc_id}/crops/{uuid.uuid4()}.png",
        layout_confidence=0.92,
        ocr_confidence=None,
        needs_review=False,
        verification_status=VerificationStatus.PENDING.value,
    )
    db.add_all([doc, page, block_h1, block_eq])
    db.commit()
    db.close()

    res = client.get(f"/documents/{doc_id}/layout")
    assert res.status_code == 200
    data = res.json()

    assert data["document_id"] == str(doc_id)
    assert data["title"] == "Kirchhoff Laws Handout"
    assert len(data["pages"]) == 1

    p1 = data["pages"][0]
    assert p1["page_number"] == 1
    assert p1["width_px"] == 1200
    assert len(p1["blocks"]) == 2

    # Check block fields
    b0 = p1["blocks"][0]
    assert b0["type"] == "heading_h1"
    assert b0["text"] == "Chapter 1: Network Theorems"
    assert b0["reading_order"] == 0

    b1 = p1["blocks"][1]
    assert b1["type"] == "equation"
    assert b1["text"] is None
    assert b1["reading_order"] == 1


def test_get_document_layout_not_found(test_db):
    """Verify GET /documents/{id}/layout returns 404 for missing document."""
    client = TestClient(app)
    missing_id = uuid.uuid4()
    res = client.get(f"/documents/{missing_id}/layout")
    assert res.status_code == 404


def test_get_page_blocks(test_db):
    """Verify GET /documents/{id}/pages/{n}/blocks returns blocks for page n."""
    client = TestClient(app)
    db = test_db()

    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="Page Blocks Test",
        original_filename="test.pdf",
        storage_path=f"{doc_id}/test.pdf",
        page_count=2,
        status=DocumentStatus.PARSED.value,
    )
    page_id = uuid.uuid4()
    page = Page(
        id=page_id,
        document_id=doc_id,
        page_number=2,
        width_px=1000,
        height_px=1400,
        dpi=150,
        image_path=f"{doc_id}/page_2.png",
    )
    block = Block(
        id=uuid.uuid4(),
        page_id=page_id,
        block_type=BlockType.TEXT.value,
        reading_order=0,
        bbox_x0=10,
        bbox_y0=20,
        bbox_x1=300,
        bbox_y1=100,
        text="Sample block text",
        layout_confidence=0.90,
        ocr_confidence=0.88,
        needs_review=False,
    )
    db.add_all([doc, page, block])
    db.commit()
    db.close()

    res = client.get(f"/documents/{doc_id}/pages/2/blocks")
    assert res.status_code == 200
    blocks = res.json()
    assert len(blocks) == 1
    assert blocks[0]["text"] == "Sample block text"

    # Missing page
    res_missing_page = client.get(f"/documents/{doc_id}/pages/99/blocks")
    assert res_missing_page.status_code == 404


def test_update_block_verification(test_db):
    """Verify PATCH /blocks/{id}/verification updates status and clears needs_review on approval."""
    client = TestClient(app)
    db = test_db()

    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        title="Verification Test",
        original_filename="v.pdf",
        storage_path=f"{doc_id}/v.pdf",
        page_count=1,
        status=DocumentStatus.PARSED.value,
    )
    page_id = uuid.uuid4()
    page = Page(
        id=page_id,
        document_id=doc_id,
        page_number=1,
        width_px=1000,
        height_px=1400,
        dpi=150,
        image_path=f"{doc_id}/page_1.png",
    )
    block_id = uuid.uuid4()
    block = Block(
        id=block_id,
        page_id=page_id,
        block_type=BlockType.DIAGRAM.value,
        reading_order=0,
        bbox_x0=10,
        bbox_y0=20,
        bbox_x1=400,
        bbox_y1=300,
        layout_confidence=0.65,
        needs_review=True,
        verification_status=VerificationStatus.PENDING.value,
    )
    db.add_all([doc, page, block])
    db.commit()
    db.close()

    # 1. Approve block
    patch_payload = {
        "verification_status": "approved",
        "verified_by": "prof_sharma",
    }
    res = client.patch(f"/blocks/{block_id}/verification", json=patch_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == str(block_id)
    assert data["verification_status"] == "approved"
    assert data["verified_by"] == "prof_sharma"
    assert data["needs_review"] is False
    assert data["updated_at"] is not None

    # 2. Reject block
    patch_reject = {
        "verification_status": "rejected",
        "verified_by": "teaching_assistant_1",
    }
    res_reject = client.patch(f"/blocks/{block_id}/verification", json=patch_reject)
    assert res_reject.status_code == 200
    assert res_reject.json()["needs_review"] is True

    # 3. Missing block 404
    missing_id = uuid.uuid4()
    res_404 = client.patch(f"/blocks/{missing_id}/verification", json=patch_payload)
    assert res_404.status_code == 404
