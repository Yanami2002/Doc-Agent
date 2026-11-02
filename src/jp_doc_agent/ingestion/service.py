"""Transactional document import; shared by CLI and future HTTP endpoints."""

import json
from pathlib import Path

import httpx
import pypdf
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, TypeAdapter
from sqlalchemy import Engine, delete, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from jp_doc_agent.ingestion.download import PdfFile, download_pdf
from jp_doc_agent.ingestion.pdf import PARSER_VERSION, extract_pages, has_unmapped_characters
from jp_doc_agent.models import Document, DocumentChunk, DocumentPage


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    url: HttpUrl
    dataset: str = Field(min_length=1, max_length=100)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


def import_pdf(engine: Engine, pdf: PdfFile, *, title: str, source_url: str, dataset: str) -> dict:
    with engine.connect() as connection:
        existing = (
            connection.execute(
                select(Document.id, Document.page_count, Document.parser_version).where(
                    Document.sha256 == pdf.sha256
                )
            )
            .mappings()
            .one_or_none()
        )
    if existing and existing["parser_version"] == PARSER_VERSION:
        return {
            "status": "duplicate",
            "document_id": existing["id"],
            "pages": existing["page_count"],
        }

    pages = extract_pages(pdf.path)
    # Metadata and all pages commit together. The unique hash also handles concurrent imports.
    status = "imported"
    with engine.begin() as connection:
        document_id = connection.execute(
            pg_insert(Document)
            .values(
                sha256=pdf.sha256,
                title=title,
                source_url=source_url,
                resolved_url=pdf.resolved_url,
                dataset=dataset,
                file_path=str(pdf.path.resolve()),
                page_count=len(pages),
                acquired_at=pdf.acquired_at,
                parser_version=PARSER_VERSION,
            )
            .on_conflict_do_nothing(index_elements=[Document.sha256])
            .returning(Document.id)
        ).scalar_one_or_none()
        if document_id is None:
            current = (
                connection.execute(
                    select(Document.id, Document.parser_version)
                    .where(Document.sha256 == pdf.sha256)
                    .with_for_update()
                )
                .mappings()
                .one()
            )
            document_id = current["id"]
            if current["parser_version"] == PARSER_VERSION:
                return {"status": "duplicate", "document_id": document_id, "pages": len(pages)}
            # Reparse after an extractor upgrade, preserving the document identity.
            connection.execute(
                delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
            )
            connection.execute(delete(DocumentPage).where(DocumentPage.document_id == document_id))
            connection.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(
                    parser_version=PARSER_VERSION,
                    page_count=len(pages),
                    file_path=str(pdf.path.resolve()),
                    chunking_signature=None,
                    chunking_config=None,
                )
            )
            status = "updated"
        connection.execute(
            insert(DocumentPage),
            [
                {"document_id": document_id, "page_number": i, "text": text}
                for i, text in enumerate(pages, start=1)
            ],
        )
    return {
        "status": status,
        "document_id": document_id,
        "pages": len(pages),
        "empty_pages": [i for i, text in enumerate(pages, start=1) if not text],
        "unmapped_character_pages": [
            i for i, text in enumerate(pages, start=1) if has_unmapped_characters(text)
        ],
    }


def import_manifest(engine: Engine, manifest: Path, data_dir: Path) -> list[dict]:
    sources = TypeAdapter(list[Source]).validate_json(manifest.read_text(encoding="utf-8"))
    results = []
    with httpx.Client(timeout=30, headers={"User-Agent": "Doc-Agent/0.1"}) as client:
        for source in sources:
            result = {"title": source.title, "source_url": str(source.url)}
            try:
                pdf = download_pdf(client, str(source.url), data_dir / "pdfs")
                if source.sha256 and source.sha256 != pdf.sha256:
                    raise ValueError(
                        "PDF の SHA-256 が取得元一覧と一致しません。"
                        "配布元で更新されていないか確認してください。"
                    )
                result.update(
                    import_pdf(
                        engine,
                        pdf,
                        title=source.title,
                        source_url=str(source.url),
                        dataset=source.dataset,
                    )
                )
                result["sha256"] = pdf.sha256
            except (httpx.HTTPError, OSError, ValueError, pypdf.errors.PyPdfError) as error:
                result.update(status="failed", reason=str(error))
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    return results


def list_documents(engine: Engine) -> list[dict]:
    with engine.connect() as connection:
        rows = connection.execute(
            select(
                Document.id, Document.title, Document.dataset, Document.page_count, Document.sha256
            ).order_by(Document.id)
        ).mappings()
        return [dict(row) for row in rows]


def read_page(engine: Engine, document_id: int, page_number: int) -> dict:
    with engine.connect() as connection:
        row = (
            connection.execute(
                select(
                    Document.title, Document.source_url, DocumentPage.page_number, DocumentPage.text
                )
                .join(DocumentPage, Document.id == DocumentPage.document_id)
                .where(Document.id == document_id, DocumentPage.page_number == page_number)
            )
            .mappings()
            .one_or_none()
        )
    if row is None:
        raise ValueError(
            "文書またはページが見つかりません。documents コマンドで登録済み文書を確認してください。"
        )
    return {
        "document_id": document_id,
        **dict(row),
        "has_unmapped_characters": has_unmapped_characters(row["text"]),
    }
