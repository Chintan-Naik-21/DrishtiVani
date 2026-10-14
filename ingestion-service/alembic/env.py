"""Alembic environment configuration for Supabase PostgreSQL migrations.

Integrates with app.config.get_settings() to load the Supabase connection-pooler
URI and maps the SQLAlchemy 2.0 metadata from app.db.models.Base.
"""

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Ensure app directory is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import get_settings
from app.db.models import Base

# Alembic Config object
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Model metadata for autogenerate support
target_metadata = Base.metadata


def get_url() -> str:
    """Fetch database connection URL from settings."""
    settings = get_settings()
    url = settings.get_sqlalchemy_database_url()
    if not url:
        # Fallback to alembic.ini or default placeholder for offline inspection
        return config.get_main_option("sqlalchemy.url") or "postgresql+psycopg://localhost/postgres"
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Configures the context with just a URL and not an Engine.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Creates an Engine connected to Supabase Postgres (via connection pooler)
    with prepared statements disabled for PgBouncer compatibility.
    """
    url = get_url()
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = url

    connect_args = {}
    if "psycopg" in url:
        connect_args["prepare_threshold"] = None

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
