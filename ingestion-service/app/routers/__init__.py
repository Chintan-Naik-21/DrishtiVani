"""Routers package."""

from app.routers.documents import router as documents_router
from app.routers.blocks import router as blocks_router

__all__ = ["documents_router", "blocks_router"]
