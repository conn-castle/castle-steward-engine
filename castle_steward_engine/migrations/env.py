"""Alembic online migration environment for castle_steward_engine."""

from __future__ import annotations

from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import Connection, engine_from_config, pool

from castle_steward_engine.store import METADATA, include_name

config = context.config

if config.config_file_name is not None and Path(config.config_file_name).is_file():
    fileConfig(config.config_file_name)


def _run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=METADATA,
        include_name=include_name,
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations using an injected connection or an engine from config."""
    connection = config.attributes.get("connection")
    if connection is not None:
        _run_migrations(connection)
        return

    section = dict(config.get_section(config.config_ini_section, {}))
    x_url = context.get_x_argument(as_dictionary=True).get("sqlalchemy.url")
    if x_url:
        section["sqlalchemy.url"] = x_url
    if not section.get("sqlalchemy.url"):
        raise RuntimeError("Provide a database URL via `-x sqlalchemy.url=...`")
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    try:
        with connectable.connect() as conn:
            _run_migrations(conn)
    finally:
        connectable.dispose()


run_migrations_online()
