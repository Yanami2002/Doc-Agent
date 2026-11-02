"""対照条件の分離、採点分母、ランキング数式と再開時の整合性を検証する。"""

import json
import math

import httpx
import pytest
from sqlalchemy import select

from jp_doc_agent.answering.generator import OpenAIAnswerGenerator
from jp_doc_agent.evaluation.comparison import (
    AnswerGrade,
    ReferenceFact,
    _closed_book,
    grade_metrics,
    public_report,
    ranking_metrics,
    run_comparison,
    summarize,
)
from jp_doc_agent.evaluation.service import EvaluationCase
from jp_doc_agent.llm import StructuredModel
from jp_doc_agent.models import Document


def test_ranking_metrics_use_rank_and_complete_relevant_set():
    result = ranking_metrics([9, 1, 2, 4], {1, 2, 3}, 3)
    assert result["hit_rate"] == 1.0
    assert result["mrr"] == 0.5
    assert result["precision"] == result["recall"] == pytest.approx(2 / 3)
    assert result["ndcg"] == pytest.approx((1 / math.log2(3) + 0.5) / (1 + 1 / math.log2(3) + 0.5))
    assert all(value == 0 for value in ranking_metrics([9, 1, 2], {1, 2, 3}, 1).values())


def test_fact_scores_separate_correctness_completeness_and_grounding():
    grade = AnswerGrade.model_validate(
        {
            "fact_coverage": [
                {"fact_id": "a", "covered": True},
                {"fact_id": "b", "covered": False},
            ],
            "claims": [
                {
                    "statement_index": 0,
                    "claim": "A",
                    "reference_supported": True,
                    "context_supported": True,
                    "citation_supported": True,
                    "relevant": True,
                    "rationale": "",
                },
                {
                    "statement_index": 0,
                    "claim": "B",
                    "reference_supported": True,
                    "context_supported": True,
                    "citation_supported": False,
                    "relevant": True,
                    "rationale": "",
                },
                {
                    "statement_index": 1,
                    "claim": "C",
                    "reference_supported": False,
                    "context_supported": False,
                    "citation_supported": False,
                    "relevant": False,
                    "rationale": "",
                },
            ],
        }
    )
    facts = [ReferenceFact(id="a", criterion="A"), ReferenceFact(id="b", criterion="B")]
    result = grade_metrics(grade, facts, 2, True)
    assert result["fact_precision"] == pytest.approx(2 / 3)
    assert result["fact_recall"] == 0.5
    assert result["fact_f1"] == pytest.approx(4 / 7)
    assert result["citation_support"] == pytest.approx(1 / 3)
    assert result["task_success"] is False
    grade.claims[2].citation_supported = True
    inconsistent = grade_metrics(grade, facts, 2, True)
    assert inconsistent["fact_f1"] == result["fact_f1"]
    assert inconsistent["grounding_consistent"] is False
    assert inconsistent["faithfulness"] is inconsistent["citation_support"] is None
    with pytest.raises(ValueError, match="事実 ID"):
        grade_metrics(grade, facts[:1], 2, True)
    with pytest.raises(ValueError, match="結論"):
        grade_metrics(grade, facts, 3, True)
    with pytest.raises(ValueError, match="適用範囲"):
        grade_metrics(grade, facts, 2, False)


def test_closed_book_never_receives_documents_or_reference_facts_and_repairs_are_bounded(
    encoder_factory, answer_response
):
    calls = []

    def handler(request):
        body = json.loads(request.content)
        payload = json.loads(body["input"][1]["content"])
        calls.append(payload)
        assert set(payload) <= {"question", "validation_feedback"}
        draft = {
            "status": "answered",
            "statements": ["対象者は学生です。"],
            "missing_information": [],
        }
        if len(calls) == 1:
            draft["missing_information"] = ["不明な項目。"]
        return httpx.Response(200, json=answer_response(draft))

    model = StructuredModel(encoder_factory(handler).client)
    result = _closed_book(model, "対象者は？")
    assert result["status"] == "answered"
    assert result["counts"]["searches"] == result["counts"]["tool_calls"] == 0
    assert result["counts"]["model_calls"] == 2
    assert result["usage"]["answer_total_tokens"] == 140
    assert len(calls) == 2 and "validation_feedback" in calls[1]


def test_comparison_is_paired_blind_to_rubric_and_resumes_without_api_calls(
    engine, ready_document, encoder_factory, api_response, answer_response, tmp_path
):
    identifier = ready_document(["private-source-text 対象者は学生です。"])
    with engine.connect() as conn:
        sha = conn.scalar(select(Document.sha256).where(Document.id == identifier))
    cases = [
        EvaluationCase(
            id="positive",
            question="対象者は？",
            expected_status="answered",
            expected_evidence=[{"document_sha256": sha, "page_number": 1}],
        ),
        EvaluationCase(
            id="negative", question="2035年度の確定額は？", expected_status="insufficient_evidence"
        ),
    ]
    rubric = {
        "positive": [ReferenceFact(id="target", criterion="private-rubric-sentinel")],
        "negative": [],
    }
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if request.url.path.endswith("embeddings"):
            return httpx.Response(200, json=api_response(body["input"]))
        payload = json.loads(body["input"][1]["content"])
        kind = body["text"]["format"]["name"]
        if kind != "answer_evaluation":
            assert "private-rubric-sentinel" not in json.dumps(body)
            assert "reference_facts" not in payload
        if kind == "closed_book_answer":
            assert "private-source-text" not in json.dumps(body)
        if kind == "answer_evaluation":
            assert "mode" not in payload
            grounded = payload["grounding_applicable"]
            draft = {
                "fact_coverage": [{"fact_id": "target", "covered": True}],
                "claims": [
                    {
                        "statement_index": 0,
                        "claim": "学生が対象。",
                        "reference_supported": True,
                        "context_supported": True if grounded else None,
                        "citation_supported": True if grounded else None,
                        "relevant": True,
                        "rationale": "基準と一致。",
                    }
                ],
            }
        elif kind == "research_step":
            draft = {
                "action": "finish",
                "query": None,
                "document_id": None,
                "page_number": None,
                "reason": "対象年度の情報がない。",
            }
        elif payload["question"].startswith("2035"):
            draft = {
                "status": "insufficient_evidence",
                "statements": [],
                "missing_information": ["対象年度の情報。"],
            }
        elif kind == "closed_book_answer":
            draft = {
                "status": "answered",
                "statements": ["対象者は学生です。"],
                "missing_information": [],
            }
        else:
            hit = payload["evidence"][0]
            draft = {
                "status": "answered",
                "statements": [
                    {
                        "text": "対象者は学生です。",
                        "citations": [
                            {"chunk_id": hit["chunk_id"], "quote": next(iter(hit["quote_options"]))}
                        ],
                    }
                ],
                "missing_information": [],
            }
        return httpx.Response(200, json=answer_response(draft))

    encoder = encoder_factory(handler)
    generator = OpenAIAnswerGenerator(encoder.client)
    output = tmp_path / "comparison.json"
    report = run_comparison(engine, encoder, generator, cases, rubric, output, repeats=1)
    assert len(report["trials"]) == 6
    for summary in report["summary"]["modes"].values():
        assert summary["trials"] == 2
        assert summary["fact_f1"] == summary["task_success_rate"] == 1
        assert summary["judge_errors"] == summary["generation_errors"] == 0
    assert report["summary"]["modes"]["no_rag"]["faithfulness"] is None
    assert len(report["retrieval_curves"]) == 1
    assert len(report["retrieval_curves"][0]["points"]) == 10
    published = json.dumps(public_report(report), ensure_ascii=False)
    assert "private-source-text" not in published and "private-rubric-sentinel" not in published
    assert "rationale" not in published and "対象者は学生です" not in published
    calls.clear()
    resumed = run_comparison(
        engine, encoder, generator, cases, rubric, output, repeats=1, resume=True
    )
    assert resumed["summary"] == report["summary"]
    assert not calls
    with pytest.raises(ValueError, match="一致しません"):
        run_comparison(engine, encoder, generator, cases, rubric, output, repeats=2, resume=True)
    assert not calls


def test_summary_keeps_judge_errors_in_denominator_and_handles_zero_baseline():
    trials = []
    for mode in ("no_rag", "single_rag", "agent"):
        for case in ("a", "b"):
            value = float(mode != "no_rag")
            error = mode == "single_rag" and case == "b"
            evaluation = {"tokens": 0}
            if error:
                evaluation["error"] = "judge failed"
            else:
                evaluation["metrics"] = {
                    "fact_precision": value,
                    "fact_recall": value,
                    "fact_f1": value,
                    "response_relevance": value,
                    "faithfulness": None,
                    "citation_support": None,
                    "task_success": bool(value),
                }
            trials.append(
                {
                    "mode": mode,
                    "case_id": case,
                    "repeat": 0,
                    "expected_status": "answered",
                    "evaluation": evaluation,
                    "result": {
                        "status": "answered",
                        "elapsed_seconds": 1,
                        "usage": {"answer_total_tokens": 70, "embedding_tokens": 0},
                    },
                }
            )
    summary = summarize({"trials": trials, "configuration": {"seed": 1}})
    assert summary["modes"]["single_rag"]["fact_f1"] == 0.5
    assert summary["modes"]["single_rag"]["judge_errors"] == 1
    assert summary["deltas_vs_no_rag"]["single_rag"]["fact_f1_delta"] == 0.5
    assert summary["deltas_vs_no_rag"]["single_rag"]["relative_change"] is None
