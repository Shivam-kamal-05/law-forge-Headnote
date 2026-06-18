"""Structured JSON logging configuration.

Uses structlog over stdlib logging so every record is a single JSON line
with consistent fields. Dev mode renders colourised key=value output;
production renders machine-readable JSON.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from src.core.config import AppEnv, Settings


def _add_service_context(settings: Settings) -> Processor:
    def processor(_logger: Any, _method_name: str, event_dict: EventDict) -> EventDict:
        event_dict.setdefault("service", settings.app.name)
        event_dict.setdefault("env", settings.app.env.value)
        return event_dict

    return processor


def configure_logging(settings: Settings) -> None:
    """Initialise structlog + stdlib logging. Call once at process startup."""
    level = getattr(logging, settings.app.log_level)

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.stdlib.add_logger_name,
        _add_service_context(settings),
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    renderer: Processor
    if settings.app.env == AppEnv.DEVELOPMENT:
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer(serializer=_json_serialize)

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    for noisy in ("uvicorn.access", "httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))


def _json_serialize(obj: Any, **_: Any) -> str:
    import orjson

    return orjson.dumps(
        obj,
        default=_default_encoder,
        option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY,
    ).decode()


def _default_encoder(obj: Any) -> Any:
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "__str__"):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound logger. Prefer `log = get_logger(__name__)` at module level."""
    return structlog.get_logger(name)
