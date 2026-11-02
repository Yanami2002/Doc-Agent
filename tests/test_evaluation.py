"""期待値の隔離、安定した文書指定、判定と失敗記録を検証する。"""

import json

import httpx
import pytest
from sqlalchemy import select

from jp_doc_agent import cli
from jp_doc_agent.answering.generator import OpenAIAnswerGenerator
from jp_doc_agent.evaluation.service import EvaluationCase, _check_result, evaluate, load_cases
from jp_doc_agent.models import Document
from jp_doc_agent.reports import write_report
from jp_doc_agent.retrieval.service import search


def case_for(engine, identifier, **overrides):
    with engine.connect() as connection:
        sha256 = connection.scalar(select(Document.sha256).where(Document.id == identifier))
    return EvaluationCase.model_validate(
        {
            "id": "test-case",
            "question": "対象者は？",
            "scope_document_sha256": sha256,
            "expected_status": "answered",
            "expected_evidence": [{"document_sha256": sha256, "page_number": 1}],
            **overrides,
        }
    )


def test_evaluation_checks_sources_and_keeps_expectations_out_of_prompt(
    engine, ready_document, rag_encoder, tmp_path
):
    identifier = ready_document(["対象者は学生です。"])
    encoder, state = rag_encoder
    case = case_for(
        engine,
        identifier,
        answer_contains=["対象条件"],
        source_question_id="evaluation-only-id",
        review_points=["evaluation-only-criterion"],
    )
    report = evaluate(engine, encoder, OpenAIAnswerGenerator(encoder.client), [case])
    assert report["summary"]["passed"] == 1
    assert report["summary"]["retrieval_evidence_hit_rate"] == 1.0
    for _, body in state["calls"]:
        text = json.dumps(body, ensure_ascii=False)
        assert "expected_status" not in text
        assert "expected_evidence" not in text
        assert "answer_contains" not in text
        assert "evaluation-only-id" not in text
        assert "evaluation-only-criterion" not in text
        assert case.scope_document_sha256 not in text
    output = tmp_path / "reports" / "result.json"
    write_report(report, output)
    assert json.loads(output.read_text()) == report


def test_keyword_mismatch_is_a_failure_instead_of_operational_error(
    engine, ready_document, rag_encoder
):
    identifier = ready_document(["対象者は学生です。"])
    encoder, _ = rag_encoder
    case = case_for(engine, identifier, answer_contains=["答えにない文字列"])
    report = evaluate(engine, encoder, OpenAIAnswerGenerator(encoder.client), [case])
    assert report["summary"]["failed"] == 1
    assert report["summary"]["errors"] == 0
    assert report["results"][0]["checks"]["answer_keywords_match"] is False


def test_numeric_keywords_ignore_commas_and_fullwidth_characters(
    engine, ready_document, rag_encoder
):
    ready_document(["売上収益は８，６６６億円です。"])
    encoder, state = rag_encoder

    hit = search(engine, encoder, "売上は？")["results"][0]
    state["draft"] = {
        "status": "answered",
        "missing_information": [],
        "statements": [
            {
                "text": hit["text"],
                "citations": [
                    {
                        "chunk_id": hit["chunk_id"],
                        "quote": hit["text"],
                    }
                ],
            }
        ],
    }
    case = case_for(engine, hit["document_id"], answer_contains=["8666", "億円"])
    assert (
        evaluate(engine, encoder, OpenAIAnswerGenerator(encoder.client), [case])["summary"][
            "passed"
        ]
        == 1
    )


def test_insufficient_response_retains_retrieval_failure_as_separate_check(
    engine, ready_document, rag_encoder
):
    ready_document(["2024年度の説明です。"])
    identifier = ready_document(["2021年度の組織説明です。"])
    encoder, state = rag_encoder
    state["draft"] = {
        "status": "insufficient_evidence",
        "statements": [],
        "missing_information": ["対象年度の本文。"],
    }
    case = case_for(engine, identifier, scope_document_sha256=None, top_k=1)
    report = evaluate(engine, encoder, OpenAIAnswerGenerator(encoder.client), [case])
    checks = report["results"][0]["checks"]
    assert checks["status_match"] is False
    assert checks["retrieval_evidence_hit"] is False
    assert checks["citation_evidence_hit"] is False
    assert report["summary"]["failed"] == 1


def test_api_failure_is_recorded_and_next_case_runs(
    engine, ready_document, encoder_factory, api_response, answer_response
):
    identifier = ready_document(["対象者は学生です。"])
    answers = []

    def handler(request):
        body = json.loads(request.content)
        if request.url.path.endswith("embeddings"):
            return httpx.Response(200, json=api_response(body["input"]))
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
        answers.append(body)
        if len(answers) == 1:
            return httpx.Response(500, json={"error": {"message": "test-key secret-value"}})
        return httpx.Response(
            200,
            json=answer_response(
                {
                    "status": "insufficient_evidence",
                    "statements": [],
                    "missing_information": ["対象年度の本文。"],
                }
            ),
        )

    cases = [
        case_for(engine, identifier),
        EvaluationCase(
            id="second",
            question="2028年度の売上は？",
            expected_status="insufficient_evidence",
        ),
    ]
    encoder = encoder_factory(handler)
    report = evaluate(engine, encoder, OpenAIAnswerGenerator(encoder.client), cases)
    assert report["summary"]["errors"] == report["summary"]["passed"] == 1
    assert report["summary"]["automated_pass_rate"] == 0.5
    assert report["summary"]["retrieval_evidence_hit_rate_all"] == 0.0
    assert report["summary"]["citation_evidence_hit_rate_all"] == 0.0
    assert report["summary"]["insufficient_correct_rate"] == 1.0
    assert len(answers) == 2
    assert "test-key" not in report["results"][0]["reason"]
    assert "secret-value" not in report["results"][0]["reason"]


def test_evidence_coverage_distinguishes_documents_and_partial_page_matches():
    case = EvaluationCase(
        id="multi-page",
        question="両ページの情報は？",
        expected_status="answered",
        expected_evidence=[
            {"document_sha256": "a" * 64, "page_number": 1},
            {"document_sha256": "a" * 64, "page_number": 2},
        ],
    )
    result = {
        "status": "answered",
        "answer": "説明。",
        "retrieval": {
            "results": [
                {"document_id": 20, "page_numbers": [1]},
                {"document_id": 10, "page_numbers": [2]},
            ]
        },
        "citations": [{"document_id": 20, "page_numbers": [1]}],
    }
    checks = _check_result(case, result, {"a" * 64: 10})
    assert checks["retrieval_evidence_hit"] is True
    assert checks["retrieval_evidence_recall"] == 0.5
    assert checks["citation_evidence_recall"] == 0.0
    assert checks["failure_categories"] == ["citation_miss"]
    assert checks["semantic_review_required"] is True
    assert checks["passed"] is False


def test_missing_documents_never_call_api(engine, rag_encoder):
    encoder, state = rag_encoder
    case = EvaluationCase(
        id="missing",
        question="対象は？",
        expected_status="insufficient_evidence",
        scope_document_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="文書"):
        evaluate(engine, encoder, OpenAIAnswerGenerator(encoder.client), [case])
    assert not state["calls"]


def test_out_of_range_evidence_page_never_calls_api(engine, ready_document, rag_encoder):
    identifier = ready_document(["対象者は学生です。"])
    case = case_for(engine, identifier)
    case.expected_evidence[0].page_number = 2
    encoder, state = rag_encoder
    with pytest.raises(ValueError, match="ページ数"):
        evaluate(engine, encoder, OpenAIAnswerGenerator(encoder.client), [case])
    assert not state["calls"]


@pytest.mark.parametrize(
    "payload",
    [
        "invalid JSON",
        "[]",
        json.dumps([{"id": "test", "question": "?", "expected_status": "unknown"}]),
        json.dumps([{"id": "test", "question": "?", "expected_status": "answered"}]),
        json.dumps(
            [{"id": "test", "question": "?", "expected_status": "insufficient_evidence"}] * 2
        ),
    ],
)
def test_invalid_cases_are_rejected(tmp_path, payload):
    path = tmp_path / "cases.json"
    path.write_text(payload)
    with pytest.raises(ValueError):
        load_cases(path)


def test_cli_evaluation_writes_report(
    engine, ready_document, rag_encoder, monkeypatch, tmp_path, capsys
):
    identifier = ready_document(["対象者は学生です。"])
    encoder, _ = rag_encoder
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps([case_for(engine, identifier).model_dump()]))
    output = tmp_path / "report.json"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(cli, "create_database_engine", lambda _: engine)
    monkeypatch.setattr(cli.OpenAIEncoder, "from_settings", lambda _: encoder)
    monkeypatch.setattr(
        "sys.argv",
        [
            "jp-doc-agent",
            "evaluate-rag",
            "--cases",
            str(cases_path),
            "--output",
            str(output),
        ],
    )
    assert cli.main() == 0
    summary = json.loads(capsys.readouterr().out)
    report = json.loads(output.read_text())
    assert summary["passed"] == 1
    assert report["workflow"] == "agent"
    assert report["results"][0]["result"]["citations"]
    assert report["results"][0]["result"]["trace"]
    assert encoder.client.is_closed()


def test_agent_evaluation_retains_error_trace_and_usage(engine, ready_document, rag_encoder):
    identifier = ready_document(["対象者は学生です。"])
    encoder, state = rag_encoder
    state["draft"] = {
        "status": "answered",
        "missing_information": [],
        "statements": [
            {"text": "未検証の結論。", "citations": [{"chunk_id": -1, "quote": "原文"}]}
        ],
    }
    report = evaluate(
        engine,
        encoder,
        OpenAIAnswerGenerator(encoder.client),
        [case_for(engine, identifier)],
    )
    assert report["summary"]["errors"] == 1
    assert report["summary"]["usage"]["answer_total_tokens"] == 70
    assert report["summary"]["counts"]["model_calls"] == 1
    assert report["results"][0]["result"]["trace"][-1]["status"] == "error"
    assert "未検証の結論" not in json.dumps(report, ensure_ascii=False)
