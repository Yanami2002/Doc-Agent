"""評価の期待値をモデルに送らず、証拠ページ・引用・回答を個別に確認する。"""

import hashlib
import json
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator
from sqlalchemy import select
from sqlalchemy.engine import Engine

from jp_doc_agent.agent.service import agent_ask
from jp_doc_agent.answering.generator import OpenAIAnswerGenerator
from jp_doc_agent.embedding.encoder import EmbeddingError, OpenAIEncoder, profile_id, validate_query
from jp_doc_agent.llm import ModelError
from jp_doc_agent.models import Document


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_number: int = Field(ge=1)


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    scope_document_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    top_k: int = Field(default=5, ge=1, le=20)
    expected_status: Literal["answered", "insufficient_evidence"]
    expected_evidence: list[EvidenceReference] = Field(default_factory=list)
    answer_contains: list[str] = Field(default_factory=list)
    source_question_id: str | None = None
    category: str = "development"
    review_points: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_case(self) -> "EvaluationCase":
        if not self.id.strip() or not self.question.strip():
            raise ValueError("ケース ID と質問に空白以外の文字が必要です。")
        if self.expected_status == "answered" and not self.expected_evidence:
            raise ValueError("回答可能なケースには許容する証拠ページが必要です。")
        if any(not word.strip() for word in self.answer_contains):
            raise ValueError("照合するキーワードを空白だけにしないでください。")
        if not self.category.strip() or any(not point.strip() for point in self.review_points):
            raise ValueError("評価分類と確認観点に空白以外の文字が必要です。")
        return self


def load_cases(path: Path) -> list[EvaluationCase]:
    try:
        cases = TypeAdapter(list[EvaluationCase]).validate_json(path.read_text(), strict=True)
    except ValidationError:
        raise ValueError("評価ケースの JSON 形式または内容が不正です。") from None
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("評価ケースは 1 件以上必要で、ID は重複できません。")
    return cases


def _normalized(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).replace(",", "").split())


def _check_result(case: EvaluationCase, result: dict, document_ids: dict[str, int]) -> dict:
    expected = {
        (document_ids[reference.document_sha256], reference.page_number)
        for reference in case.expected_evidence
    }
    retrieved = {
        (hit["document_id"], page)
        for hit in result["retrieval"]["results"]
        for page in hit["page_numbers"]
    }
    cited = {
        (citation["document_id"], page)
        for citation in result["citations"]
        for page in citation["page_numbers"]
    }
    checks = {
        "status_match": result["status"] == case.expected_status,
        "retrieval_evidence_hit": bool(expected & retrieved) if expected else None,
        "citation_evidence_hit": bool(expected & cited) if expected else None,
        "answer_keywords_match": all(
            _normalized(word) in _normalized(result["answer"]) for word in case.answer_contains
        ),
    }
    failures = []
    if not checks["status_match"]:
        failures.append("status_mismatch")
    if checks["retrieval_evidence_hit"] is False:
        failures.append("retrieval_miss")
    if checks["citation_evidence_hit"] is False:
        failures.append("citation_miss")
    if not checks["answer_keywords_match"]:
        failures.append("answer_keyword_mismatch")
    return {
        "passed": not failures,
        **checks,
        "retrieval_evidence_recall": len(expected & retrieved) / len(expected)
        if expected
        else None,
        "citation_evidence_recall": len(expected & cited) / len(expected) if expected else None,
        "failure_categories": failures,
        "semantic_review_required": case.expected_status == "answered",
    }


def evaluate(
    engine: Engine,
    encoder: OpenAIEncoder,
    generator: OpenAIAnswerGenerator,
    cases: list[EvaluationCase],
) -> dict:
    evaluated_at = datetime.now(UTC).isoformat()
    if any(case.top_k > 20 for case in cases):
        raise ValueError("Agent の評価では top-k を 1〜20 にしてください。")
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("評価ケースは 1 件以上必要で、ID は重複できません。")
    for case in cases:
        validate_query(case.question)
    required = {
        reference.document_sha256 for case in cases for reference in case.expected_evidence
    } | {case.scope_document_sha256 for case in cases if case.scope_document_sha256 is not None}
    # 全ケースの文書を API 呼び出し前に照合。DB の連番に依存しない。
    with engine.connect() as connection:
        corpus = [
            dict(row)
            for row in connection.execute(
                select(
                    Document.sha256,
                    Document.title,
                    Document.page_count,
                    Document.parser_version,
                    Document.chunking_signature,
                    Document.chunking_config,
                ).order_by(Document.sha256)
            ).mappings()
        ]
        documents = {
            row.sha256: row
            for row in connection.execute(
                select(Document.sha256, Document.id, Document.page_count).where(
                    Document.sha256.in_(required)
                )
            )
        }
    if required - documents.keys():
        raise ValueError("評価に必要な文書がありません。import-documents を実行してください。")
    if any(
        reference.page_number > documents[reference.document_sha256].page_count
        for case in cases
        for reference in case.expected_evidence
    ):
        raise ValueError("評価の証拠ページが文書のページ数を超えています。")
    document_ids = {sha256: row.id for sha256, row in documents.items()}
    results = []
    for case in cases:
        try:
            # 期待ステータス・証拠ページ・キーワードは調査処理に渡さない。
            result = agent_ask(
                engine,
                encoder,
                generator,
                case.question,
                top_k=case.top_k,
                document_id=document_ids.get(case.scope_document_sha256),
            )
            if result["status"] == "error":
                results.append(
                    {
                        "case": case.model_dump(),
                        "status": "error",
                        "reason": result["error"],
                        "result": result,
                    }
                )
                continue
            results.append(
                {
                    "case": case.model_dump(),
                    "status": "completed",
                    "checks": _check_result(case, result, document_ids),
                    "result": result,
                }
            )
        except (ModelError, EmbeddingError, ValueError) as error:
            results.append({"case": case.model_dump(), "status": "error", "reason": str(error)})
    completed = [item for item in results if item["status"] == "completed"]
    evidence_cases = [
        item for item in completed if item["checks"]["retrieval_evidence_hit"] is not None
    ]
    passed = sum(item["checks"]["passed"] for item in completed)
    recorded = [item["result"] for item in results if "result" in item]
    answerable_count = sum(case.expected_status == "answered" for case in cases)
    insufficient_count = len(cases) - answerable_count
    insufficient_correct = sum(
        item["case"]["expected_status"] == "insufficient_evidence"
        and item["checks"]["status_match"]
        for item in completed
    )
    evidence_count = sum(bool(case.expected_evidence) for case in cases)
    case_json = json.dumps(
        [case.model_dump() for case in cases],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "workflow": "agent",
        "evaluation_version": "page-fact-v2",
        "evaluated_at": evaluated_at,
        "case_set_sha256": hashlib.sha256(case_json.encode()).hexdigest(),
        "configuration": {
            "requested_model": generator.model,
            "max_output_tokens": generator.max_output_tokens,
            "embedding_profile_id": profile_id(),
        },
        "corpus": corpus,
        "summary": {
            "cases": len(cases),
            "completed": len(completed),
            "answerable_cases": answerable_count,
            "insufficient_cases": insufficient_count,
            "passed": passed,
            "automated_pass_rate": passed / len(cases),
            "failed": len(completed) - passed,
            "errors": len(cases) - len(completed),
            # エラーを含む全対象ケースが分母。成功した実行だけで成績を高く見せない。
            "retrieval_evidence_hit_rate_all": (
                sum(item["checks"]["retrieval_evidence_hit"] for item in evidence_cases)
                / evidence_count
                if evidence_count
                else None
            ),
            "citation_evidence_hit_rate_all": (
                sum(item["checks"]["citation_evidence_hit"] for item in evidence_cases)
                / evidence_count
                if evidence_count
                else None
            ),
            "insufficient_correct_rate": (
                insufficient_correct / insufficient_count if insufficient_count else None
            ),
            "failure_categories": {
                category: sum(
                    category in item["checks"]["failure_categories"] for item in completed
                )
                for category in (
                    "status_mismatch",
                    "retrieval_miss",
                    "citation_miss",
                    "answer_keyword_mismatch",
                )
            },
            "retrieval_evidence_hit_rate": (
                sum(item["checks"]["retrieval_evidence_hit"] for item in evidence_cases)
                / len(evidence_cases)
                if evidence_cases
                else None
            ),
            "usage": {
                key: sum(result["usage"][key] for result in recorded)
                for key in (
                    "embedding_tokens",
                    "answer_input_tokens",
                    "answer_output_tokens",
                    "answer_total_tokens",
                )
            },
            "elapsed_seconds": round(sum(result["elapsed_seconds"] for result in recorded), 3),
            "counts": {
                key: sum(result["counts"][key] for result in recorded)
                for key in ("searches", "tool_calls", "model_calls")
            },
        },
        "results": results,
    }
