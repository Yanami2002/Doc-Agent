"""ローカル開発用 CLI の引数・実行・JSON 出力を管理する。"""

import argparse
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import ValidationError
from pypdf.errors import PyPdfError
from sqlalchemy.exc import SQLAlchemyError

from jp_doc_agent.agent.schema import AgentLimits
from jp_doc_agent.agent.service import agent_ask
from jp_doc_agent.answering.generator import OpenAIAnswerGenerator
from jp_doc_agent.benchmark import fetch_benchmark
from jp_doc_agent.chunking.service import chunk_documents, list_chunks
from jp_doc_agent.chunking.splitter import ChunkingConfig
from jp_doc_agent.config import EMBEDDING_MODEL, AnsweringSettings, OpenAISettings, Settings
from jp_doc_agent.database import check_database, create_database_engine
from jp_doc_agent.embedding.encoder import OpenAIEncoder
from jp_doc_agent.embedding.service import embed_documents, embedding_status
from jp_doc_agent.evaluation.service import evaluate, load_cases
from jp_doc_agent.ingestion.download import copy_local_pdf
from jp_doc_agent.ingestion.service import import_manifest, import_pdf, list_documents, read_page
from jp_doc_agent.reports import write_report
from jp_doc_agent.retrieval.service import search


@contextmanager
def _openai_encoder() -> Iterator[OpenAIEncoder]:
    try:
        settings = OpenAISettings()
    except ValidationError:
        raise ValueError("プロジェクトの .env に OPENAI_API_KEY を設定してください。") from None
    encoder = OpenAIEncoder.from_settings(settings)
    try:
        yield encoder
    finally:
        encoder.client.close()


def build_parser() -> argparse.ArgumentParser:
    resources = Path(__file__).resolve().parent / "resources"
    parser = argparse.ArgumentParser(description="Doc-Agent 開発用 CLI")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check-db", help="PostgreSQL と pgvector の接続を確認")
    commands.add_parser("documents", help="登録済み文書の一覧を表示")

    batch = commands.add_parser(
        "import-documents", help="取得元一覧に従って PDF をダウンロード・登録"
    )
    batch.add_argument("--manifest", type=Path, default=resources / "sources/fujitsu.json")
    batch.add_argument("--data-dir", type=Path, default=Path("data"))

    benchmark = commands.add_parser("fetch-benchmark", help="評価用の正解データと利用規約を取得")
    benchmark.add_argument("--data-dir", type=Path, default=Path("data"))

    local = commands.add_parser("import-pdf", help="ローカル PDF を登録")
    local.add_argument("path", type=Path)
    local.add_argument("--title")
    local.add_argument("--source-url", help="取得元 URL（省略時はローカルファイルの URI）")
    local.add_argument("--dataset", default="local")
    local.add_argument("--data-dir", type=Path, default=Path("data"))

    page = commands.add_parser("page", help="文書の指定ページを表示")
    page.add_argument("document_id", type=int)
    page.add_argument("page_number", type=int, help="1 始まりの PDF 物理ページ番号")

    chunking = commands.add_parser(
        "chunk-documents", help="登録済みの全文を分割し、ページをまたぐ出典も保存"
    )
    chunking.add_argument("--document-id", type=int, help="対象文書 ID（省略時は全件）")
    defaults = ChunkingConfig()
    chunking.add_argument(
        "--chunk-size", type=int, default=defaults.chunk_size, help="最大 Token 数（既定値: 300）"
    )
    chunking.add_argument(
        "--chunk-overlap",
        type=int,
        default=defaults.chunk_overlap,
        help="重複 Token 数（既定値: 30）",
    )

    chunks = commands.add_parser("chunks", help="文書のチャンク・ページ番号・原文位置を確認")
    chunks.add_argument("document_id", type=int)
    chunks.add_argument("--page", type=int, dest="page_number", help="物理ページ番号で絞り込み")
    chunks.add_argument("--limit", type=int, default=20, help="取得件数（1〜100、既定値: 20）")
    chunks.add_argument("--offset", type=int, default=0, help="先頭からスキップする件数")

    embedding = commands.add_parser(
        "embed-chunks", help="OpenAI API でチャンクをベクトル化して保存"
    )
    embedding.add_argument("--document-id", type=int, help="対象文書 ID（省略時は全件）")
    embedding.add_argument(
        "--batch-size", type=int, default=32, help="API の 1 回の入力件数（1〜32）"
    )
    status = commands.add_parser("embedding-status", help="ベクトル化済み件数と未処理件数を確認")
    status.add_argument("--document-id", type=int, help="対象文書 ID（省略時は全件）")
    retrieval = commands.add_parser("search", help="質問に関連するチャンクを出典付きで検索")
    retrieval.add_argument("query", help="日本語の質問")
    retrieval.add_argument("--top-k", type=int, default=5, help="取得件数（1〜100、既定値: 5）")
    retrieval.add_argument("--document-id", type=int, help="対象文書 ID（省略時は全件）")
    agent = commands.add_parser("ask", help="証拠を判断し、必要に応じて追加調査して日本語で回答")
    agent.add_argument("query", help="日本語の質問")
    agent.add_argument("--top-k", type=int, default=5, help="検索件数（1〜20、既定値: 5）")
    agent.add_argument("--document-id", type=int, help="対象文書 ID（省略時は全件）")
    agent.add_argument("--max-searches", type=int, default=3, help="検索上限（1〜3）")
    agent.add_argument("--max-tool-calls", type=int, default=6, help="道具の呼び出し上限（1〜8）")
    agent.add_argument("--output", type=Path, help="実行記録の保存先（省略時は data/reports）")
    evaluation = commands.add_parser("evaluate-rag", help="固定ケースで検索・回答・引用を検証")
    evaluation.add_argument(
        "--cases", type=Path, default=resources / "evaluation/questions.json"
    )
    evaluation.add_argument(
        "--output", type=Path, help="結果 JSON の保存先（省略時は data/reports）"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "fetch-benchmark":
        try:
            result = fetch_benchmark(args.data_dir)
        except (httpx.HTTPError, ValueError, OSError) as error:
            print(str(error), file=sys.stderr)
            return 1
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    try:
        settings = Settings()
    except ValidationError:
        print(
            "設定が不正です。プロジェクト直下の .env のデータベース設定を確認してください。",
            file=sys.stderr,
        )
        return 1

    engine = create_database_engine(settings)
    try:
        if args.command == "embed-chunks":
            if not 1 <= args.batch_size <= 32:
                raise ValueError("batch-size は 1〜32 にしてください。")
            with _openai_encoder() as encoder:
                results = embed_documents(
                    engine, encoder, document_id=args.document_id, batch_size=args.batch_size
                )
            result = {
                "model": EMBEDDING_MODEL,
                "embedded": sum(item["embedded"] for item in results),
                "skipped": sum(item["skipped"] for item in results),
                "failed_documents": sum(item["status"] == "failed" for item in results),
                "api_tokens": sum(item["api_tokens"] for item in results),
                "results": results,
            }
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1 if result["failed_documents"] else 0
        elif args.command in ("ask", "evaluate-rag"):
            try:
                answer_settings = AnsweringSettings()
            except ValidationError:
                raise ValueError(".env の回答モデルと出力 Token 上限を確認してください。") from None
            limits = (
                AgentLimits(max_searches=args.max_searches, max_tool_calls=args.max_tool_calls)
                if args.command == "ask"
                else None
            )
            with _openai_encoder() as encoder:
                generator = OpenAIAnswerGenerator(
                    encoder.client,
                    model=answer_settings.answer_model,
                    max_output_tokens=answer_settings.answer_max_output_tokens,
                )
                if args.command == "ask":
                    result = agent_ask(
                        engine,
                        encoder,
                        generator,
                        args.query,
                        top_k=args.top_k,
                        document_id=args.document_id,
                        limits=limits,
                    )
                    output = args.output or Path(
                        f"data/reports/agent-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.json"
                    )
                    write_report(result, output)
                    print(
                        json.dumps({**result, "report": str(output)}, ensure_ascii=False, indent=2)
                    )
                    return 1 if result["status"] == "error" else 0
                else:
                    cases = load_cases(args.cases)
                    report = evaluate(engine, encoder, generator, cases)
                    output = args.output or Path(
                        f"data/reports/rag-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.json"
                    )
                    write_report(report, output)
                    print(
                        json.dumps(
                            {**report["summary"], "report": str(output)},
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
                    return 1 if report["summary"]["errors"] else 0
        elif args.command == "search":
            with _openai_encoder() as encoder:
                result = search(
                    engine, encoder, args.query, top_k=args.top_k, document_id=args.document_id
                )
        elif args.command == "embedding-status":
            result = embedding_status(engine, document_id=args.document_id)
        elif args.command == "check-db":
            result = {"status": "ok", **check_database(engine)}
        elif args.command == "import-documents":
            results = import_manifest(engine, args.manifest, args.data_dir)
            report_dir = args.data_dir / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            report = report_dir / f"import-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.json"
            report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            counts = {
                status: sum(item["status"] == status for item in results)
                for status in ("imported", "updated", "duplicate", "failed")
            }
            print(json.dumps({**counts, "report": str(report)}, ensure_ascii=False, indent=2))
            return 1 if counts["failed"] else 0
        elif args.command == "import-pdf":
            if not 1 <= len(args.dataset) <= 100:
                raise ValueError("dataset は 1〜100 文字で指定してください。")
            pdf = copy_local_pdf(args.path, args.data_dir / "pdfs")
            result = import_pdf(
                engine,
                pdf,
                title=args.title or args.path.name,
                source_url=args.source_url or pdf.resolved_url,
                dataset=args.dataset,
            )
        elif args.command == "documents":
            result = list_documents(engine)
        elif args.command == "chunk-documents":
            results = chunk_documents(
                engine,
                ChunkingConfig(args.chunk_size, args.chunk_overlap),
                document_id=args.document_id,
            )
            result = {
                "documents": len(results),
                "chunked": sum(item["status"] == "chunked" for item in results),
                "duplicate": sum(item["status"] == "duplicate" for item in results),
                "chunks": sum(item["chunks"] for item in results),
                "results": results,
            }
        elif args.command == "chunks":
            result = list_chunks(
                engine,
                args.document_id,
                page_number=args.page_number,
                limit=args.limit,
                offset=args.offset,
            )
        else:
            result = read_page(engine, args.document_id, args.page_number)
    except SQLAlchemyError:
        print(
            "データベース操作に失敗しました。接続設定と "
            "uv run alembic -c infra/alembic.ini upgrade head の実行状況を確認してください。",
            file=sys.stderr,
        )
        return 1
    except (RuntimeError, ValueError, OSError, PyPdfError) as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        engine.dispose()

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
