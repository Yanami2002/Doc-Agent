"""実 DB と模擬 API で、追加調査・引用・停止条件・失敗記録を検証する。"""

import json
from collections import deque

import httpx
import pytest
from sqlalchemy import delete, select

from jp_doc_agent import cli
from jp_doc_agent.agent.schema import AgentLimits
from jp_doc_agent.agent.service import agent_ask
from jp_doc_agent.answering.generator import OpenAIAnswerGenerator
from jp_doc_agent.chunking.service import chunk_document
from jp_doc_agent.chunking.splitter import ChunkingConfig
from jp_doc_agent.embedding.service import embed_document
from jp_doc_agent.models import ChunkEmbedding, DocumentChunk
from jp_doc_agent.retrieval.service import read_page_evidence

INSUFFICIENT = {
    "status": "insufficient_evidence",
    "statements": [],
    "missing_information": ["質問の条件を確認できる本文。"],
}


def decision(action, *, query=None, document_id=None, page_number=None):
    return {
        "action": action,
        "query": query,
        "document_id": document_id,
        "page_number": page_number,
        "reason": "不足している本文を確認します。",
    }


def grounded(payload, *, quote=None):
    hit = next(hit for hit in payload["evidence"] if quote is None or quote in hit["text"])
    return {
        "status": "answered",
        "statements": [
            {
                "text": "資料に対象条件が記載されています。",
                "citations": [{"chunk_id": hit["chunk_id"], "quote": quote or hit["text"]}],
            }
        ],
        "missing_information": [],
    }


@pytest.fixture
def scripted_agent(encoder_factory, api_response, answer_response):
    def build(*, steps=(), answers=()):
        state = {"calls": [], "steps": deque(steps), "answers": deque(answers), "during_api": None}

        def handler(request):
            body = json.loads(request.content)
            state["calls"].append((request.url.path, body))
            if request.url.path == "/v1/embeddings":
                if state["during_api"]:
                    state["during_api"]()
                return httpx.Response(200, json=api_response(body["input"]))
            payload = json.loads(body["input"][1]["content"])
            if body["text"]["format"]["name"] == "research_step":
                response = state["steps"].popleft() if state["steps"] else decision("finish")
            else:
                response = state["answers"].popleft() if state["answers"] else INSUFFICIENT
            if callable(response):
                response = response(payload)
            if isinstance(response, httpx.Response):
                return response
            return httpx.Response(200, json=answer_response(response, model="actual-test-model"))

        encoder = encoder_factory(handler)
        return encoder, OpenAIAnswerGenerator(encoder.client), state

    return build


def test_sufficient_evidence_finishes_after_first_search(engine, ready_document, scripted_agent):
    ready_document(["対象者は学生です。"])
    encoder, generator, _ = scripted_agent(answers=[grounded])
    result = agent_ask(engine, encoder, generator, "対象者は？")
    assert result["status"] == result["stop_reason"] == "answered"
    assert result["counts"] == {"searches": 1, "page_reads": 0, "tool_calls": 1, "model_calls": 1}
    assert result["model"] == "actual-test-model"
    assert [event["name"] for event in result["trace"]] == ["search", "assess_and_answer"]


def test_catalog_scoped_search_and_page_read_recover_missing_evidence(
    engine, ready_document, scripted_agent
):
    ready_document(["別の年度の説明です。"])
    identifier = ready_document(["制度の説明です。" * 40, "対象者は学生です。"], vector=(0.0, 1.0))

    def select_document(payload):
        assert any(item["id"] == identifier for item in payload["documents"])
        assert all("sha256" not in item for item in payload["documents"])
        return decision("search", query="制度の対象条件", document_id=identifier)

    def answer_page(payload):
        assert payload["pages"][0]["text"] == "対象者は学生です。"
        return grounded(payload, quote="対象者は学生です。")

    encoder, generator, state = scripted_agent(
        steps=[
            decision("documents"),
            select_document,
            decision("page", document_id=identifier, page_number=2),
        ],
        answers=[INSUFFICIENT, INSUFFICIENT, answer_page],
    )
    result = agent_ask(engine, encoder, generator, "対象者は？", top_k=1)
    assert result["status"] == "answered"
    assert result["counts"] == {"searches": 2, "page_reads": 1, "tool_calls": 4, "model_calls": 6}
    assert result["usage"] == {
        "embedding_tokens": 2,
        "answer_input_tokens": 300,
        "answer_output_tokens": 120,
        "answer_total_tokens": 420,
    }
    assert result["citations"][0]["document_id"] == identifier
    assert result["citations"][0]["page_numbers"] == [2]
    assert result["citations"][0]["quote"] == "対象者は学生です。"
    assert not state["steps"] and not state["answers"]


def test_duplicate_tools_consume_budget_without_extra_embedding_calls(
    engine, ready_document, scripted_agent
):
    ready_document(["対象年度の情報はありません。"])
    encoder, generator, _ = scripted_agent(steps=[decision("search", query="対象者は？")] * 5)
    result = agent_ask(engine, encoder, generator, "対象者は？")
    assert result["status"] == "insufficient_evidence"
    assert result["stop_reason"] == "tool_limit"
    assert result["counts"]["searches"] == 1
    assert result["counts"]["tool_calls"] == 6
    assert result["counts"]["model_calls"] == 6
    assert sum(event["status"] == "rejected" for event in result["trace"]) == 5
    assert result["citations"] == result["statements"] == []


def test_search_limit_is_enforced_even_when_model_keeps_requesting_search(
    engine, ready_document, scripted_agent
):
    ready_document(["対象年度の情報はありません。"])
    encoder, generator, _ = scripted_agent(
        steps=[decision("search", query=f"追加検索 {index}") for index in range(5)]
    )
    result = agent_ask(engine, encoder, generator, "対象者は？")
    assert result["counts"]["searches"] == result["usage"]["embedding_tokens"] == 3
    assert result["counts"]["tool_calls"] == 6
    assert result["stop_reason"] == "tool_limit"
    assert sum(event["status"] == "rejected" for event in result["trace"]) == 3


def test_scope_cannot_be_expanded_by_model(engine, ready_document, scripted_agent):
    scope = ready_document(["対象文書の説明です。"])
    other = ready_document(["指定範囲外の文書です。"])
    encoder, generator, state = scripted_agent(
        steps=[decision("page", document_id=other, page_number=1), decision("documents")]
    )
    result = agent_ask(engine, encoder, generator, "対象者は？", document_id=scope)
    rejected = [event for event in result["trace"] if event["status"] == "rejected"]
    assert len(rejected) == 1 and "範囲外" in rejected[0]["error"]
    catalogs = [event["result"] for event in result["trace"] if event["name"] == "documents"]
    assert [item["id"] for item in catalogs[0]["documents"]] == [scope]
    assert result["counts"]["page_reads"] == 0
    for path, body in state["calls"]:
        if path.endswith("responses"):
            assert "指定範囲外の文書です。" not in json.dumps(body, ensure_ascii=False)


def test_page_limit_rejects_third_read(engine, ready_document, scripted_agent):
    identifier = ready_document(["原文です。" * 40, "続きです。" * 40, "最後です。" * 40])
    encoder, generator, _ = scripted_agent(
        steps=[decision("page", document_id=identifier, page_number=page) for page in (1, 2, 3)]
    )
    result = agent_ask(engine, encoder, generator, "対象者は？", top_k=1)
    assert result["counts"]["page_reads"] == 2
    assert any("原文参照回数" in event.get("error", "") for event in result["trace"])
    assert result["status"] == "insufficient_evidence"


def test_context_limit_rejects_full_page_and_retains_original_evidence(
    engine, ready_document, scripted_agent
):
    identifier = ready_document(["長い説明です。" * 100])
    encoder, generator, _ = scripted_agent(
        steps=[decision("page", document_id=identifier, page_number=1)]
    )
    result = agent_ask(
        engine,
        encoder,
        generator,
        "対象者は？",
        top_k=1,
        limits=AgentLimits(max_evidence_tokens=300),
    )
    assert any("Token 上限" in event.get("error", "") for event in result["trace"])
    assert len(result["retrieval"]["results"]) == 1
    assert result["status"] == "insufficient_evidence"


def test_new_chunk_version_replaces_previous_evidence(
    engine, ready_document, scripted_agent, encoder_factory, api_response
):
    identifier = ready_document(["対象者は学生です。" * 25])
    with engine.connect() as connection:
        old_ids = set(connection.scalars(select(DocumentChunk.id)))
    encoder, generator, state = scripted_agent(
        steps=[decision("search", query="申請対象者", document_id=identifier)],
        answers=[INSUFFICIENT, grounded],
    )
    embeddings = 0

    def rebuild():
        nonlocal embeddings
        embeddings += 1
        if embeddings != 2:
            return
        chunk_document(engine, identifier, ChunkingConfig(40, 5))
        rebuild_encoder = encoder_factory(
            lambda request: httpx.Response(
                200, json=api_response(json.loads(request.content)["input"])
            )
        )
        embed_document(engine, rebuild_encoder, identifier)

    state["during_api"] = rebuild
    result = agent_ask(engine, encoder, generator, "対象者は？", document_id=identifier)
    assert result["status"] == "answered"
    assert all(hit["chunk_id"] not in old_ids for hit in result["retrieval"]["results"])
    assert result["citations"][0]["chunk_id"] not in old_ids


@pytest.mark.parametrize("failure", ["invalid_decision", "api_error", "invalid_citation"])
def test_failures_keep_safe_trace_and_never_return_unverified_claims(
    engine, ready_document, scripted_agent, failure
):
    ready_document(["対象者は学生です。"])
    steps, answers = [], []
    if failure == "invalid_decision":
        steps = [decision("page")]
    elif failure == "api_error":
        answers = [httpx.Response(500, json={"error": {"message": "test-key secret-value"}})]
    else:
        answers = [
            lambda _: {
                "status": "answered",
                "statements": [
                    {"text": "未検証の結論。", "citations": [{"chunk_id": -1, "quote": "原文"}]}
                ],
                "missing_information": [],
            }
        ]
    encoder, generator, _ = scripted_agent(steps=steps, answers=answers)
    result = agent_ask(engine, encoder, generator, "対象者は？")
    assert result["status"] == "error"
    assert result["trace"][-1]["status"] == "error"
    assert "answer" not in result and "citations" not in result
    serialized = json.dumps(result, ensure_ascii=False)
    assert "test-key" not in serialized and "secret-value" not in serialized
    assert "未検証の結論" not in serialized


def test_cli_failure_persists_trace_and_closes_client(
    engine, ready_document, scripted_agent, monkeypatch, tmp_path, capsys
):
    ready_document(["対象者は学生です。"])
    encoder, _, _ = scripted_agent(steps=[decision("page")])
    output = tmp_path / "agent.json"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(cli, "create_database_engine", lambda _: engine)
    monkeypatch.setattr(cli.OpenAIEncoder, "from_settings", lambda _: encoder)
    monkeypatch.setattr("sys.argv", ["jp-doc-agent", "ask", "対象者は？", "--output", str(output)])
    assert cli.main() == 1
    result = json.loads(output.read_text())
    assert result["status"] == "error" and result["trace"][-1]["status"] == "error"
    assert json.loads(capsys.readouterr().out)["report"] == str(output)
    assert encoder.client.is_closed()


def test_page_read_can_cite_real_cross_page_chunks_without_embeddings(engine, ready_document):
    identifier = ready_document(["対象者は、申請時点で", "日本国内に居住する学生です。"])
    with engine.begin() as connection:
        connection.execute(delete(ChunkEmbedding))
    result = read_page_evidence(engine, identifier, 2)
    assert result["text"] == "日本国内に居住する学生です。"
    assert result["results"][0]["page_numbers"] == [1, 2]
    assert result["results"][0]["chunk_id"] > 0
    for source in result["results"][0]["sources"]:
        assert source["chunk_end_char"] > source["chunk_start_char"]
    with pytest.raises(ValueError, match="物理ページ"):
        read_page_evidence(engine, identifier, 3)


def test_rebuild_during_answer_uses_retrieved_snapshot(engine, ready_document, rag_encoder):
    identifier = ready_document(["対象者は学生です。", "期限は三月です。"])
    encoder, state = rag_encoder
    state["during_answer"] = lambda: chunk_document(engine, identifier, ChunkingConfig(15, 3))
    result = agent_ask(engine, encoder, OpenAIAnswerGenerator(encoder.client), "条件は？")
    assert result["citations"][0]["page_numbers"] == [1, 2]
    with engine.connect() as connection:
        current_ids = connection.scalars(select(DocumentChunk.id)).all()
    assert result["citations"][0]["chunk_id"] not in current_ids


def test_cli_ask_uses_configured_model_and_closes_client(
    engine, ready_document, rag_encoder, monkeypatch, tmp_path, capsys
):
    ready_document(["対象者は学生です。"])
    encoder, state = rag_encoder
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_ANSWER_MODEL", "test-answer-model")
    monkeypatch.setattr(cli, "create_database_engine", lambda _: engine)
    monkeypatch.setattr(cli.OpenAIEncoder, "from_settings", lambda _: encoder)
    monkeypatch.setattr(
        "sys.argv", ["jp-doc-agent", "ask", "対象は？", "--output", str(tmp_path / "answer.json")]
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "answered"
    assert state["calls"][1][1]["model"] == "test-answer-model"
    assert encoder.client.is_closed()


def test_cli_invalid_config_never_calls_api(engine, rag_encoder, monkeypatch, capsys):
    encoder, state = rag_encoder
    monkeypatch.setenv("OPENAI_ANSWER_MAX_OUTPUT_TOKENS", "0")
    monkeypatch.setattr(cli, "create_database_engine", lambda _: engine)
    monkeypatch.setattr(cli.OpenAIEncoder, "from_settings", lambda _: encoder)
    monkeypatch.setattr("sys.argv", ["jp-doc-agent", "ask", "対象は？"])
    assert cli.main() == 1
    assert "出力 Token" in capsys.readouterr().err
    assert not state["calls"]


def test_cli_bad_citation_never_displays_unverified_answer(
    engine, ready_document, rag_encoder, monkeypatch, tmp_path, capsys
):
    ready_document(["対象者は学生です。"])
    encoder, state = rag_encoder
    state["draft"] = {
        "status": "answered",
        "missing_information": [],
        "statements": [
            {
                "text": "表示してはいけない結論。",
                "citations": [
                    {
                        "chunk_id": -1,
                        "quote": "対象者は学生です。",
                    }
                ],
            }
        ],
    }
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(cli, "create_database_engine", lambda _: engine)
    monkeypatch.setattr(cli.OpenAIEncoder, "from_settings", lambda _: encoder)
    monkeypatch.setattr(
        "sys.argv", ["jp-doc-agent", "ask", "対象は？", "--output", str(tmp_path / "answer.json")]
    )
    assert cli.main() == 1
    output = capsys.readouterr()
    assert json.loads(output.out)["status"] == "error"
    assert "引用先" in output.out
    assert "表示してはいけない" not in output.out
    assert output.err == ""
    assert encoder.client.is_closed()


def whitespace_error(payload):
    draft = grounded(payload)
    draft["statements"][0]["text"] = "表示してはいけない草案。"
    draft["statements"][0]["citations"][0]["quote"] = payload["evidence"][0]["text"].replace(
        "\n", " "
    )
    return draft


def test_validation_repair_reuses_evidence_and_only_returns_exact_original_quote(
    engine, ready_document, scripted_agent
):
    ready_document(["対象者は学生です。\n期限は三月です。"])

    def corrected(payload):
        assert payload["validation_feedback"]["details"]["match_type"] == "whitespace_changed"
        return grounded(payload)

    encoder, generator, state = scripted_agent(answers=[whitespace_error, corrected])
    result = agent_ask(engine, encoder, generator, "対象者は？")
    assert result["status"] == "answered"
    assert result["counts"]["searches"] == 1
    assert result["counts"]["model_calls"] == 2
    assert result["usage"]["answer_total_tokens"] == 140
    assert result["citations"][0]["quote"] == "対象者は学生です。\n期限は三月です。"
    assert result["trace"][1]["status"] == "retrying"
    assert result["trace"][2]["validation_repair"] is True
    assert "表示してはいけない" not in json.dumps(result, ensure_ascii=False)
    requests = [json.loads(body["input"][1]["content"]) for _, body in state["calls"][1:]]
    assert requests[0]["evidence"] == requests[1]["evidence"]


def test_failed_repair_is_bounded_and_never_returns_unverified_answer(
    engine, ready_document, scripted_agent
):
    ready_document(["対象者は学生です。\n期限は三月です。"])
    encoder, generator, _ = scripted_agent(answers=[whitespace_error] * 5)
    result = agent_ask(engine, encoder, generator, "対象者は？")
    assert result["status"] == "error"
    assert result["error_code"] == "quote_mismatch"
    assert result["counts"]["model_calls"] == 2
    assert result["usage"]["answer_total_tokens"] == 140
    assert "answer" not in result
    assert "表示してはいけない" not in json.dumps(result, ensure_ascii=False)


@pytest.mark.parametrize("fail_later", [False, True])
def test_partial_answer_repair_preserves_insufficiency_and_run_wide_repair_budget(
    engine, ready_document, scripted_agent, fail_later
):
    identifier = ready_document(["対象者は学生です。\n期限は三月です。"])

    def conflicting(payload):
        draft = grounded(payload)
        draft["missing_information"] = ["未確認の条件。"]
        return draft

    def corrected(payload):
        assert payload["validation_feedback"]["details"]["violations"][0]["type"] == (
            "answered_with_missing_information"
        )
        return INSUFFICIENT

    encoder, generator, _ = scripted_agent(
        answers=[conflicting, corrected, whitespace_error if fail_later else grounded, grounded],
        steps=[decision("page", document_id=identifier, page_number=1)],
    )
    result = agent_ask(engine, encoder, generator, "対象者は？")
    assert result["trace"][2]["answer_status"] == "insufficient_evidence"
    assert result["counts"]["tool_calls"] == 2
    assert result["counts"]["model_calls"] == 4
    assert result["usage"]["answer_total_tokens"] == 280
    assert sum(event.get("validation_repair", False) for event in result["trace"]) == 1
    assert result["status"] == ("error" if fail_later else "answered")
