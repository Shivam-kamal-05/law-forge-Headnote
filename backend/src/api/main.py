"""FastAPI application factory — headnote extraction demo server."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse

from src.api.routes import auth, extract, health
from src.core.config import get_settings
from src.core.exceptions import LawLensError
from src.core.logging import configure_logging, get_logger
from src.ingestion.llm_client import LLMClient

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# Resolve the static dir relative to this file so the app can be launched
# from any working directory. backend/src/api/main.py → backend/static/
_STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings)
    log.info("app.starting", env=settings.app.env.value, model=settings.anthropic.model)

    llm_client = LLMClient(settings.anthropic)

    app.state.settings = settings
    app.state.llm_client = llm_client

    log.info("app.ready", static_dir=str(_STATIC_DIR))
    try:
        yield
    finally:
        log.info("app.stopping")
        await llm_client.aclose()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Law Lens — Headnote Engine",
        description=(
            "Upload Indian court judgment PDFs and receive structured, "
            "AI-generated headnotes in the SCC/AIR tradition. "
            "All headnotes are generated from the court's own text only."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # open for client demo — lock down in production
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(LawLensError)
    async def _law_lens_error(request: Request, exc: LawLensError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("app.unhandled_error", error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "An unexpected error occurred."}},
        )

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(extract.router)

    # Serve the frontend from the root.
    @app.get("/", include_in_schema=False)
    async def serve_ui() -> FileResponse:
        index = _STATIC_DIR / "index.html"
        if not index.exists():
            return JSONResponse(
                status_code=503,
                content={"error": "Frontend not built. Expected at backend/static/index.html"},
            )
        return FileResponse(index)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug,
        log_config=None,
    )
