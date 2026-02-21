from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse

from xtrc.api import build_router
from xtrc.config import Settings
from xtrc.core.daemon import AinavDaemon
from xtrc.core.errors import AinavError
from xtrc.logging import setup_logging
from xtrc.schemas import ErrorPayload, ErrorResponse

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    setup_logging()
    app_settings = settings or Settings.from_env()
    daemon = AinavDaemon(app_settings)

    app = FastAPI(
        title="xtrc",
        version="0.1.0",
        default_response_class=ORJSONResponse,
    )
    app.include_router(build_router(daemon))

    @app.exception_handler(AinavError)
    async def ainav_exception_handler(_: Request, exc: AinavError) -> ORJSONResponse:
        payload = ErrorResponse(
            error=ErrorPayload(code=exc.code, message=exc.message, details=exc.details or None)
        )
        return ORJSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception) -> ORJSONResponse:
        logger.exception("Unhandled server exception", exc_info=exc)
        payload = ErrorResponse(
            error=ErrorPayload(code="INTERNAL_ERROR", message="Unexpected server error")
        )
        return ORJSONResponse(status_code=500, content=payload.model_dump())

    return app


app = create_app()
