"""アプリケーションの組み立て、DB の生存期間、共通エラー処理。"""

from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException

from jp_doc_agent.api.errors import (
    APIError,
    handle_api_error,
    handle_database_error,
    handle_http_error,
    handle_unexpected_error,
    handle_validation_error,
)
from jp_doc_agent.api.routes import router
from jp_doc_agent.config import Settings
from jp_doc_agent.database import create_database_engine


def create_app(*, engine: Engine | None = None, report_dir: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            database = engine if engine is not None else create_database_engine(Settings())
        except ValidationError:
            raise RuntimeError("サーバーの .env のデータベース設定を確認してください。") from None
        app.state.engine = database
        app.state.report_dir = report_dir if report_dir is not None else Path("data/reports")
        try:
            yield
        finally:
            if engine is None:
                database.dispose()

    app = FastAPI(
        title="Doc-Agent API",
        description="日本語 PDF の文書参照、出典付き問答、LangGraph による追加調査。",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def identify_request(request: Request, call_next):
        request.state.request_id = uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    app.add_exception_handler(APIError, handle_api_error)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(SQLAlchemyError, handle_database_error)
    app.add_exception_handler(HTTPException, handle_http_error)
    app.add_exception_handler(Exception, handle_unexpected_error)
    app.include_router(router)
    return app


app = create_app()
