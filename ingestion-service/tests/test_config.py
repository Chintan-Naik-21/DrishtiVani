"""Unit tests for application configuration."""

import pytest
from app.config import Settings


def test_default_settings():
    """Verify default configuration parameters."""
    settings = Settings(
        DATABASE_URL="postgresql://user:pass@host:6543/postgres",
        SUPABASE_URL="https://test.supabase.co",
        SUPABASE_SERVICE_ROLE_KEY="test-key",
    )
    assert settings.RENDER_DPI == 150
    assert settings.PAGE_IMAGE_FORMAT == "PNG"
    assert settings.SIGNED_URL_EXPIRATION_SECONDS == 3600
    assert settings.CONFIDENCE_THRESHOLD == 0.80
    assert settings.USE_LAYOUTLMV3_FALLBACK is False
    assert settings.SUPABASE_BUCKET_HANDOUTS == "handouts"
    assert settings.SUPABASE_BUCKET_PAGES == "pages"
    assert settings.SUPABASE_BUCKET_CROPS == "crops"


def test_database_url_normalization_postgresql():
    """Verify postgresql:// is translated to postgresql+psycopg://."""
    settings = Settings(
        DATABASE_URL="postgresql://postgres:secret@db.xyz.supabase.co:6543/postgres",
        SUPABASE_URL="https://test.supabase.co",
        SUPABASE_SERVICE_ROLE_KEY="test-key",
    )
    assert settings.get_sqlalchemy_database_url() == "postgresql+psycopg://postgres:secret@db.xyz.supabase.co:6543/postgres"


def test_database_url_normalization_postgres():
    """Verify postgres:// is translated to postgresql+psycopg://."""
    settings = Settings(
        DATABASE_URL="postgres://postgres:secret@db.xyz.supabase.co:6543/postgres",
        SUPABASE_URL="https://test.supabase.co",
        SUPABASE_SERVICE_ROLE_KEY="test-key",
    )
    assert settings.get_sqlalchemy_database_url() == "postgresql+psycopg://postgres:secret@db.xyz.supabase.co:6543/postgres"


def test_custom_overrides():
    """Verify custom settings overrides."""
    settings = Settings(
        RENDER_DPI=300,
        CONFIDENCE_THRESHOLD=0.90,
        USE_LAYOUTLMV3_FALLBACK=True,
        SIGNED_URL_EXPIRATION_SECONDS=7200,
    )
    assert settings.RENDER_DPI == 300
    assert settings.CONFIDENCE_THRESHOLD == 0.90
    assert settings.USE_LAYOUTLMV3_FALLBACK is True
    assert settings.SIGNED_URL_EXPIRATION_SECONDS == 7200
