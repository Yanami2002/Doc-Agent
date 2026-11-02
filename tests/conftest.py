"""PostgreSQL の一時スキーマと共通 fixture で業務データを変更せず検証する。"""

import hashlib
import json
from datetime import UTC, datetime
from io import BytesIO
from uuid import uuid4

import httpx
import pytest
from alembic import command
from alembic.config import Config
from openai import OpenAI
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from sqlalchemy import create_engine, insert, text

from jp_doc_agent.chunking.service import chunk_document
from jp_doc_agent.chunking.splitter import ChunkingConfig
from jp_doc_agent.config import ANSWER_MODEL, EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, Settings
from jp_doc_agent.database import create_database_engine
from jp_doc_agent.embedding.encoder import OpenAIEncoder
from jp_doc_agent.embedding.service import embed_document
from jp_doc_agent.models import Document, DocumentPage


@pytest.fixture
def engine():
    settings = Settings()
    admin = create_database_engine(settings)
    schema = f"test_import_{uuid4().hex}"
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(
        settings.database_url,
        connect_args={"options": f"-c search_path={schema},public -c statement_timeout=5000"},
    )
    try:
        with engine.begin() as connection:
            config = Config("infra/alembic.ini")
            config.attributes.update(connection=connection, version_table_schema=schema)
            command.upgrade(config, "head")
        yield engine
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


@pytest.fixture
def pdf_bytes():
    """小さな PDF で境界条件を検証する。実 PDF の確認は別に行う。"""

    def build(pages):
        writer = PdfWriter()
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        for content in pages:
            page = writer.add_blank_page(width=300, height=300)
            page[NameObject("/Resources")] = DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
            )
            if content:
                stream = DecodedStreamObject()
                stream.set_data(f"BT /F1 12 Tf 20 250 Td ({content}) Tj ET".encode("ascii"))
                page[NameObject("/Contents")] = writer._add_object(stream)
        buffer = BytesIO()
        writer.write(buffer)
        return buffer.getvalue()

    return build


@pytest.fixture
def text_document(engine):
    def build(pages):
        with engine.begin() as connection:
            identifier = connection.execute(
                insert(Document)
                .values(
                    sha256=hashlib.sha256(json.dumps(pages).encode()).hexdigest(),
                    title="日本語資料",
                    source_url="https://example.com/document.pdf",
                    resolved_url="https://example.com/document.pdf",
                    dataset="test",
                    file_path="/test/document.pdf",
                    page_count=len(pages),
                    acquired_at=datetime.now(UTC),
                    parser_version="test-parser",
                )
                .returning(Document.id)
            ).scalar_one()
            connection.execute(
                insert(DocumentPage),
                [
                    {"document_id": identifier, "page_number": index, "text": text}
                    for index, text in enumerate(pages, start=1)
                ],
            )
        return identifier

    return build


@pytest.fixture
def api_response():
    def embedding_response(texts: list[str]) -> dict:
        data = []
        for index, _ in enumerate(texts):
            vector = [0.0] * EMBEDDING_DIMENSIONS
            vector[index] = 1.0
            data.append({"object": "embedding", "index": index, "embedding": vector})
        return {
            "object": "list",
            "model": EMBEDDING_MODEL,
            "data": data,
            "usage": {"prompt_tokens": len(texts), "total_tokens": len(texts)},
        }

    return embedding_response


@pytest.fixture
def encoder_factory():
    clients = []

    def build(handler, *, retries=0):
        client = OpenAI(
            api_key="test-key",
            max_retries=retries,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        clients.append(client)
        return OpenAIEncoder(client)

    yield build
    for client in clients:
        client.close()


@pytest.fixture
def ready_document(engine, text_document, encoder_factory, api_response):
    def build(pages, *, vector=(1.0, 0.0)):
        identifier = text_document(pages)
        chunk_document(engine, identifier, ChunkingConfig())

        def handler(request):
            response = api_response(json.loads(request.content)["input"])
            for item in response["data"]:
                item["embedding"] = list(vector) + [0.0] * (1536 - len(vector))
            return httpx.Response(200, json=response)

        result = embed_document(engine, encoder_factory(handler), identifier)
        assert result["status"] == "embedded"
        return identifier

    return build


@pytest.fixture
def query_encoder(encoder_factory, api_response):
    state = {"calls": [], "during_api": None}

    def handler(request):
        body = json.loads(request.content)
        state["calls"].append(body)
        if state["during_api"]:
            state["during_api"]()
        return httpx.Response(200, json=api_response(body["input"]))

    return encoder_factory(handler), state


@pytest.fixture
def answer_response():
    def build(draft, *, model=ANSWER_MODEL):
        return {
            "id": "resp_test",
            "object": "response",
            "created_at": 0.0,
            "model": model,
            "status": "completed",
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
            "output": [
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "annotations": [],
                            "text": json.dumps(draft, ensure_ascii=False),
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": 50,
                "output_tokens": 20,
                "total_tokens": 70,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }

    return build


@pytest.fixture
def rag_encoder(encoder_factory, api_response, answer_response):
    state = {"calls": [], "draft": None, "during_answer": None}

    def handler(request):
        body = json.loads(request.content)
        state["calls"].append((request.url.path, body))
        if request.url.path == "/v1/embeddings":
            return httpx.Response(200, json=api_response(body["input"]))
        assert request.url.path == "/v1/responses"
        if state["during_answer"]:
            state["during_answer"]()
        if body["text"]["format"]["name"] == "research_step":
            return httpx.Response(
                200,
                json=answer_response(
                    {
                        "action": "finish",
                        "query": None,
                        "document_id": None,
                        "page_number": None,
                        "reason": "追加の根拠資料が必要です。",
                    }
                ),
            )
        evidence = json.loads(body["input"][1]["content"])["evidence"][0]
        draft = state["draft"] or {
            "status": "answered",
            "statements": [
                {
                    "text": "資料に対象条件が記載されています。",
                    "citations": [{"chunk_id": evidence["chunk_id"], "quote": evidence["text"]}],
                }
            ],
            "missing_information": [],
        }
        return httpx.Response(200, json=answer_response(draft))

    return encoder_factory(handler), state
