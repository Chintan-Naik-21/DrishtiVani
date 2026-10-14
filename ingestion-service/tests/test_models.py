"""Unit tests for SQLAlchemy models and relationships."""

import uuid
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from app.db.models import Base, Block, BlockType, Document, DocumentStatus, Page, VerificationStatus


def test_models_ddl_and_relationships():
    """Test schema creation, cascade deletion, and relationships on Document, Page, Block."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # Create a document
        doc = Document(
            title="Basic Electrical Circuits - Lecture 1",
            original_filename="circuits_lec1.pdf",
            storage_path="documents/circuits_lec1.pdf",
            page_count=2,
            status=DocumentStatus.UPLOADED.value,
        )
        session.add(doc)
        session.flush()

        assert doc.id is not None
        assert isinstance(doc.id, uuid.UUID)
        assert doc.status == "uploaded"

        # Create a page
        page = Page(
            document_id=doc.id,
            page_number=1,
            width_px=1240,
            height_px=1754,
            dpi=150,
            image_path=f"pages/{doc.id}/page_1.png",
        )
        session.add(page)
        session.flush()

        assert page.id is not None
        assert page.document.id == doc.id

        # Create layout blocks
        block1 = Block(
            page_id=page.id,
            block_type=BlockType.HEADING_H1.value,
            reading_order=0,
            bbox_x0=50.0,
            bbox_y0=100.0,
            bbox_x1=500.0,
            bbox_y1=160.0,
            text="Kirchhoff's Current Law (KCL)",
            layout_confidence=0.98,
            ocr_confidence=0.99,
            needs_review=False,
            verification_status=VerificationStatus.PENDING.value,
            extra={"heading_level": 1},
        )

        block2 = Block(
            page_id=page.id,
            block_type=BlockType.EQUATION.value,
            reading_order=1,
            bbox_x0=60.0,
            bbox_y0=180.0,
            bbox_x1=400.0,
            bbox_y1=250.0,
            crop_path=f"crops/{page.id}/block_1.png",
            layout_confidence=0.92,
            ocr_confidence=None,
            needs_review=False,
            verification_status=VerificationStatus.PENDING.value,
        )

        session.add_all([block1, block2])
        session.commit()

        # Query back and verify
        loaded_doc = session.query(Document).filter_by(id=doc.id).one()
        assert len(loaded_doc.pages) == 1
        assert len(loaded_doc.pages[0].blocks) == 2
        assert loaded_doc.pages[0].blocks[0].block_type == "heading_h1"
        assert loaded_doc.pages[0].blocks[0].extra == {"heading_level": 1}
        assert loaded_doc.pages[0].blocks[1].crop_path is not None

        # Verify cascade delete
        session.delete(loaded_doc)
        session.commit()

        assert session.query(Document).count() == 0
        assert session.query(Page).count() == 0
        assert session.query(Block).count() == 0
