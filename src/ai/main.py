"""FastAPI application entrypoint."""

from uuid import uuid4

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import RequestResponseEndpoint
from starlette.middleware.cors import CORSMiddleware

from ai.api.routes.calendar import router as calendar_router
from ai.api.routes.chat import router as chat_router
from ai.api.routes.health import router as health_router
from ai.api.routes.ingestion import router as admin_router
from ai.api.routes.knowledge import router as knowledge_router
from ai.api.routes.sessions import router as sessions_router
from ai.api.routes.sources import router as sources_router
from ai.config import settings
from ai.core.errors import AppError, ErrorDetail, ErrorEnvelope
from ai.core.logging import configure_logging

configure_logging()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Prospectus-grounded IPO education service.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "X-Internal-Key",
        "X-Admin-Key",
        "X-Request-Id",
        "X-User-Id",
        "X-IPO-Id",
    ],
)


@app.middleware("http")
async def add_request_id(
    request: Request, call_next: RequestResponseEndpoint
) -> Response:
    request_id = request.headers.get("X-Request-Id")
    if not request_id or len(request_id) > 128:
        request_id = str(uuid4())

    request.state.request_id = request_id
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        return response
    finally:
        structlog.contextvars.clear_contextvars()


@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    error = ErrorEnvelope(
        error=ErrorDetail(
            code=exc.code,
            message=exc.message,
            request_id=getattr(request.state, "request_id", ""),
            details=exc.details,
        )
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error.model_dump(mode="json"),
        headers={"X-Request-Id": error.error.request_id},
    )


@app.exception_handler(HTTPException)
async def handle_http_error(request: Request, exc: HTTPException) -> JSONResponse:
    error = ErrorEnvelope(
        error=ErrorDetail(
            code="NOT_FOUND" if exc.status_code == 404 else "HTTP_ERROR",
            message=str(exc.detail),
            request_id=getattr(request.state, "request_id", ""),
        )
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error.model_dump(mode="json"),
        headers={"X-Request-Id": error.error.request_id},
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    error = ErrorEnvelope(
        error=ErrorDetail(
            code="VALIDATION_ERROR",
            message="The request is invalid.",
            request_id=getattr(request.state, "request_id", ""),
            details={
                "fields": [
                    {
                        "field": ".".join(str(part) for part in item["loc"]),
                        "message": item["msg"],
                    }
                    for item in exc.errors()
                ]
            },
        )
    )
    return JSONResponse(
        status_code=400,
        content=error.model_dump(mode="json"),
        headers={"X-Request-Id": error.error.request_id},
    )


app.include_router(health_router)
app.include_router(chat_router)
app.include_router(sessions_router)
app.include_router(sources_router)
app.include_router(knowledge_router)
app.include_router(calendar_router)
app.include_router(admin_router)
