"""Login endpoint for the demo gate.

POST /api/v1/login  { username, password }  ->  { token, username }

On success returns the stateless bearer token the client must send as
`Authorization: Bearer <token>` on protected routes. On failure returns 401.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.api.dependencies import SettingsDep
from src.core.exceptions import UnauthorizedError
from src.core.logging import get_logger
from src.core.security import derive_token, verify_credentials

log = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(..., max_length=200)
    password: str = Field(..., max_length=400)


class LoginResponse(BaseModel):
    token: str
    username: str


@router.post("/login", response_model=LoginResponse, summary="Authenticate the operator")
async def login(body: LoginRequest, settings: SettingsDep) -> LoginResponse:
    if not verify_credentials(settings.auth, body.username, body.password):
        log.warning("auth.login_failed", username=body.username)
        raise UnauthorizedError("Invalid username or password.")

    log.info("auth.login_ok", username=settings.auth.username)
    return LoginResponse(token=derive_token(settings.auth), username=settings.auth.username)
