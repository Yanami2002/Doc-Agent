"""モデルが選べる道具と、コード側で強制する実行上限。"""

from dataclasses import dataclass
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jp_doc_agent.answering.generator import GeneratedAnswer


@dataclass(frozen=True)
class AgentLimits:
    max_searches: int = 3
    max_tool_calls: int = 6
    max_page_reads: int = 2
    max_evidence_tokens: int = 16000
    max_answer_repairs: int = 1

    def __post_init__(self):
        if not 1 <= self.max_searches <= 3 or not 1 <= self.max_tool_calls <= 8:
            raise ValueError("検索上限は 1〜3、道具の呼び出し上限は 1〜8 にしてください。")
        if not 0 <= self.max_page_reads <= 2 or not 300 <= self.max_evidence_tokens <= 16000:
            raise ValueError("原文参照上限は 0〜2、証拠本文の上限は 300〜16000 Token です。")
        if not 0 <= self.max_answer_repairs <= 1:
            raise ValueError("回答の検証修正は実行全体で最大 1 回です。")


class ToolDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["search", "documents", "page", "finish"]
    query: str | None
    document_id: int | None = Field(ge=1)
    page_number: int | None = Field(ge=1)
    reason: str = Field(description="不足情報と、この道具を選ぶ目的を短く説明する")

    @model_validator(mode="after")
    def validate_arguments(self) -> "ToolDecision":
        if not self.reason.strip():
            raise ValueError("道具を選ぶ目的が必要です。")
        if self.action == "search":
            if self.query is None or not self.query.strip() or self.page_number is not None:
                raise ValueError("検索には質問が必要で、ページ番号は指定できません。")
        elif self.action == "page":
            if self.document_id is None or self.page_number is None or self.query is not None:
                raise ValueError("原文参照には文書 ID とページ番号が必要です。")
        elif any(item is not None for item in (self.query, self.document_id, self.page_number)):
            raise ValueError("一覧参照と終了には引数を指定しません。")
        return self


class AgentState(TypedDict):
    question: str
    decision: ToolDecision
    hits: list[dict]
    pages: list[dict]
    documents: list[dict]
    generated: GeneratedAnswer | None
    next_node: str
    stop_reason: str
