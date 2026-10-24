"""App factory and entry point for production."""

from __future__ import annotations

from fastapi import FastAPI

from .middleware import install_middleware
from .routes import router
from .settings import Settings, load_settings_from_env
from .state import lifespan


def create_app(settings: Settings) -> FastAPI:
    """Makes a complete and independent app."""
    app = FastAPI(lifespan=lifespan)
    app.state.settings = settings
    # install_middleware(app) 
    app.include_router(router)
    return app


def create_app_from_env() -> FastAPI:
    """Entry point for production"""
    return create_app(load_settings_from_env())
