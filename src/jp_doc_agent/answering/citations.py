"""回答草案の引用を原文と照合し、全出典と引用番号を補完する。"""

import re

from jp_doc_agent.answering.schema import AnswerDraft
from jp_doc_agent.llm import ModelError


def _quote_diagnostics(hit: dict, quote: str, hits: list[dict]) -> dict:
    def compact(value):
        return "".join(value.split())

    target = compact(quote)
    match_type = "not_in_evidence"
    if target and target in compact(hit["text"]):
        match_type = "whitespace_changed"
    elif any(quote and quote in other["text"] for other in hits):
        match_type = "other_chunk"
    elif any(target and target in compact(other["text"]) for other in hits):
        match_type = "other_chunk_whitespace_changed"
    else:
        # 実際に連続・重複している検索スナップショットだけを診断用に結合する。
        # 欠けた区間を補完したり、この結合本文を引用として受理したりはしない。
        related = sorted(
            [
                other
                for other in hits
                if other["document_id"] == hit["document_id"] and "start_char" in other
            ],
            key=lambda other: other["start_char"],
        )
        combined = ""
        end = -1
        for other in related:
            if other["start_char"] > end:
                combined = other["text"]
            else:
                overlap = end - other["start_char"]
                combined += other["text"][overlap:]
            end = max(end, other["end_char"])
            if target and target in compact(combined):
                match_type = "spans_chunks"
                break
    return {
        "chunk_id": hit["chunk_id"],
        "quote_characters": len(quote),
        "chunk_characters": len(hit["text"]),
        "match_type": match_type,
    }


def _resolve_quote(hit: dict, quote: str, hits: list[dict]) -> dict:
    start = hit["text"].find(quote)
    if not quote.strip() or start < 0:
        raise ModelError(
            "引用原文が検索した本文に存在しません。回答を中止しました。",
            code="quote_mismatch",
            details=_quote_diagnostics(hit, quote, hits),
        )
    end = start + len(quote)
    sources = []
    for source in hit["sources"]:
        left = max(start, source["chunk_start_char"])
        right = min(end, source["chunk_end_char"])
        if left < right:
            sources.append(
                {
                    "page_number": source["page_number"],
                    "start_char": source["start_char"] + left - source["chunk_start_char"],
                    "end_char": source["start_char"] + right - source["chunk_start_char"],
                    "chunk_start_char": left,
                    "chunk_end_char": right,
                }
            )
    if not sources:
        raise ModelError("引用原文の出典を特定できません。回答を中止しました。")
    return {
        "chunk_id": hit["chunk_id"],
        "document_id": hit["document_id"],
        "title": hit["title"],
        "source_url": hit["source_url"],
        "resolved_url": hit["resolved_url"],
        "quote": quote,
        "quote_start_char": start,
        "quote_end_char": end,
        "page_numbers": [source["page_number"] for source in sources],
        "sources": sources,
        "has_unmapped_characters": hit["has_unmapped_characters"],
    }


def resolve_answer(draft: AnswerDraft, hits: list[dict]) -> dict:
    """引用番号と出典は検証済みの検索スナップショットから生成する。"""
    hit_by_id = {hit["chunk_id"]: hit for hit in hits}
    citations = []
    citation_ids = {}
    statements = []
    for statement in draft.statements:
        if re.search(r"\[\d+\]|https?://", statement.text):
            raise ModelError("結論に生成済みの引用番号または URL があります。回答を中止しました。")
        identifiers = []
        for citation in statement.citations:
            if citation.chunk_id not in hit_by_id:
                raise ModelError(
                    "引用先が今回の検索結果に存在しません。回答を中止しました。",
                    code="unknown_chunk",
                )
            key = (citation.chunk_id, citation.quote)
            if key not in citation_ids:
                identifier = len(citations) + 1
                citations.append(
                    {
                        "id": identifier,
                        **_resolve_quote(hit_by_id[citation.chunk_id], citation.quote, hits),
                    }
                )
                citation_ids[key] = identifier
            if citation_ids[key] not in identifiers:
                identifiers.append(citation_ids[key])
        statements.append({"text": statement.text.strip(), "citation_ids": identifiers})
    if draft.status == "answered":
        answer = "\n".join(
            item["text"] + " " + "".join(f"[{identifier}]" for identifier in item["citation_ids"])
            for item in statements
        )
    else:
        answer = "取得した資料だけでは回答できません。\n不足情報: " + "／".join(
            draft.missing_information
        )
    return {
        "status": draft.status,
        "answer": answer,
        "statements": statements,
        "citations": citations,
        "missing_information": draft.missing_information,
    }
