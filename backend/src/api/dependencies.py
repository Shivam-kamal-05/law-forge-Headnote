"""FastAPI dependency helpers — inject app-level singletons per request."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from src.core.config import Settings
from src.core.exceptions import UnauthorizedError
from src.core.security import verify_token
from src.ingestion.llm_client import LLMClient


def _get_settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def _get_llm_client(request: Request) -> LLMClient:
    return request.app.state.llm_client  # type: ignore[no-any-return]


def require_auth(request: Request) -> None:
    """Gate a route behind the demo bearer token.

    Expects `Authorization: Bearer <token>` where the token was issued by
    POST /api/v1/login. Raises 401 otherwise.
    """
    settings: Settings = request.app.state.settings
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not verify_token(settings.auth, token.strip()):
        raise UnauthorizedError("Authentication required. Please sign in.")


SettingsDep = Annotated[Settings, Depends(_get_settings)]
LLMClientDep = Annotated[LLMClient, Depends(_get_llm_client)]
RequireAuth = Depends(require_auth)
