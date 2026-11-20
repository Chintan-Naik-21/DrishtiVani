"""Initial schema for documents, pages, and blocks with Row Level Security.

Revision ID: 0001_initial_schema
Revises: 
Create Date: 2026-10-14 12:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Documents Table
    op.create_table(
        'documents',
        sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('original_filename', sa.String(length=255), nullable=False),
        sa.Column('storage_path', sa.String(length=1024), nullable=False),
        sa.Column('page_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='uploaded'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.PrimaryKeyConstraint('id', name='pk_documents')
    )
    op.create_index('ix_documents_status', 'documents', ['status'], unique=False)

    # 2. Pages Table
    op.create_table(
        'pages',
        sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('document_id', sa.UUID(), nullable=False),
        sa.Column('page_number', sa.Integer(), nullable=False),
        sa.Column('width_px', sa.Integer(), nullable=False),
        sa.Column('height_px', sa.Integer(), nullable=False),
        sa.Column('dpi', sa.Integer(), nullable=False),
        sa.Column('image_path', sa.String(length=1024), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name='fk_pages_document_id_documents', ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name='pk_pages'),
        sa.UniqueConstraint('document_id', 'page_number', name='uq_pages_document_page_number')
    )
    op.create_index('ix_pages_document_id', 'pages', ['document_id'], unique=False)

    # 3. Blocks Table
    op.create_table(
        'blocks',
        sa.Column('id', sa.UUID(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('page_id', sa.UUID(), nullable=False),
        sa.Column('block_type', sa.String(length=50), nullable=False),
        sa.Column('reading_order', sa.Integer(), nullable=False),
        sa.Column('bbox_x0', sa.Float(), nullable=False),
        sa.Column('bbox_y0', sa.Float(), nullable=False),
        sa.Column('bbox_x1', sa.Float(), nullable=False),
        sa.Column('bbox_y1', sa.Float(), nullable=False),
        sa.Column('text', sa.Text(), nullable=True),
        sa.Column('crop_path', sa.String(length=1024), nullable=True),
        sa.Column('layout_confidence', sa.Float(), nullable=False),
        sa.Column('ocr_confidence', sa.Float(), nullable=True),
        sa.Column('needs_review', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('verification_status', sa.String(length=50), nullable=False, server_default='pending'),
        sa.Column('verified_by', sa.String(length=255), nullable=True),
        sa.Column('verified_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('extra', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['page_id'], ['pages.id'], name='fk_blocks_page_id_pages', ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id', name='pk_blocks')
    )
    op.create_index('ix_blocks_page_id', 'blocks', ['page_id'], unique=False)
    op.create_index('ix_blocks_page_id_reading_order', 'blocks', ['page_id', 'reading_order'], unique=False)
    op.create_index('ix_blocks_needs_review', 'blocks', ['needs_review'], unique=False)

    # 4. Enable Row Level Security (RLS) on all three tables
    # RLS is enabled with no public policies for now: only the backend service
    # (connecting with direct DB credentials or service role) accesses them.
    # Dedicated policies for teacher/student roles will be added later.
    op.execute("ALTER TABLE documents ENABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE pages ENABLE ROW LEVEL SECURITY;")
    op.execute("ALTER TABLE blocks ENABLE ROW LEVEL SECURITY;")

    op.execute("""
        COMMENT ON TABLE documents IS 'Accessible Engineering Notes documents. RLS enabled with no public policies for now; accessed via backend service role. Teacher/student policies to be added later.';
    """)
    op.execute("""
        COMMENT ON TABLE pages IS 'Document pages. RLS enabled with no public policies for now; accessed via backend service role. Teacher/student policies to be added later.';
    """)
    op.execute("""
        COMMENT ON TABLE blocks IS 'Extracted layout blocks. RLS enabled with no public policies for now; accessed via backend service role. Teacher/student policies to be added later.';
    """)


def downgrade() -> None:
    op.drop_index('ix_blocks_needs_review', table_name='blocks')
    op.drop_index('ix_blocks_page_id_reading_order', table_name='blocks')
    op.drop_index('ix_blocks_page_id', table_name='blocks')
    op.drop_table('blocks')

    op.drop_index('ix_pages_document_id', table_name='pages')
    op.drop_table('pages')

    op.drop_index('ix_documents_status', table_name='documents')
    op.drop_table('documents')
