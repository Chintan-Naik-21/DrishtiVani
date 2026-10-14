"""Database models for Accessible Engineering Notes.

Schema definitions for documents, pages, and layout blocks using SQLAlchemy 2.0.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON, UUID


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy declarative models."""
    pass


class DocumentStatus(str, Enum):
    """Lifecycle status of a document in the ingestion pipeline."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PARSED = "parsed"
    FAILED = "failed"


class BlockType(str, Enum):
    """Allowed semantic block types detected on a page."""
    HEADING_H1 = "heading_h1"
    HEADING_H2 = "heading_h2"
    HEADING_H3 = "heading_h3"
    TEXT = "text"
    EQUATION = "equation"
    TABLE = "table"
    DIAGRAM = "diagram"
    FIGURE_CAPTION = "figure_caption"
    LIST_ITEM = "list_item"
    OTHER = "other"


class VerificationStatus(str, Enum):
    """Teacher verification state for accessibility quality control."""
    PENDING = "pending"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"


class Document(Base):
    """Represents an uploaded PDF handout document."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=DocumentStatus.UPLOADED.value,
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=text("CURRENT_TIMESTAMP"),
    )

    # Relationship to child pages
    pages: Mapped[List["Page"]] = relationship(
        "Page",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="Page.page_number",
    )

    __table_args__ = (
        Index("ix_documents_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} title='{self.title}' status='{self.status}'>"


class Page(Base):
    """Represents an individual page rendered from a document."""

    __tablename__ = "pages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    width_px: Mapped[int] = mapped_column(Integer, nullable=False)
    height_px: Mapped[int] = mapped_column(Integer, nullable=False)
    dpi: Mapped[int] = mapped_column(Integer, nullable=False)
    image_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    # Relationships
    document: Mapped["Document"] = relationship("Document", back_populates="pages")
    blocks: Mapped[List["Block"]] = relationship(
        "Block",
        back_populates="page",
        cascade="all, delete-orphan",
        order_by="Block.reading_order",
    )

    __table_args__ = (
        UniqueConstraint("document_id", "page_number", name="uq_pages_document_page_number"),
    )

    def __repr__(self) -> str:
        return f"<Page id={self.id} doc={self.document_id} page_number={self.page_number}>"


class Block(Base):
    """Represents a classified region/block on a document page.

    Provides complete original-scan traceability with pixel coordinates
    referencing the stored page image.
    """

    __tablename__ = "blocks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )
    page_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("pages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    block_type: Mapped[str] = mapped_column(String(50), nullable=False)
    reading_order: Mapped[int] = mapped_column(Integer, nullable=False)

    # Bounding box coordinates in PIXEL space of the stored page image: [x0, y0, x1, y1]
    bbox_x0: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y0: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)

    # Extracted text (populated for text, headings, lists, captions, and table cells)
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Storage path in Supabase Storage 'crops' bucket (for equation / diagram crops)
    crop_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)

    # Confidence scores
    layout_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    ocr_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Review & Teacher Verification fields
    needs_review: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    verification_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=VerificationStatus.PENDING.value,
    )
    verified_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    verified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # Extra JSON payload (e.g. heading level, table grid structure, model metadata)
    extra: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON().with_variant(postgresql.JSONB, "postgresql"),
        nullable=True,
    )

    # Relationship to parent page
    page: Mapped["Page"] = relationship("Page", back_populates="blocks")

    __table_args__ = (
        Index("ix_blocks_page_id_reading_order", "page_id", "reading_order"),
        Index("ix_blocks_needs_review", "needs_review"),
    )

    def __repr__(self) -> str:
        return f"<Block id={self.id} type='{self.block_type}' order={self.reading_order}>"
