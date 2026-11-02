"""LLM が生成できる項目を定義する。URL とページ情報はプログラムが補完する。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError


class CitationDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    chunk_id: int
    quote: str = Field(description="指定 chunk の本文から変更せず抜き出した連続する原文")


class StatementDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str = Field(description="引用番号を含まない、根拠付きの短い日本語の結論")
    citations: list[CitationDraft]


class AnswerDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["answered", "insufficient_evidence"]
    statements: list[StatementDraft]
    missing_information: list[str]

    @model_validator(mode="after")
    def validate_status(self) -> "AnswerDraft":
        if self.status == "answered":
            if not self.statements:
                raise PydanticCustomError("answered_without_statements", "回答には結論が必要です。")
            if self.missing_information:
                raise PydanticCustomError(
                    "answered_with_missing_information", "回答時の不足情報は空配列にしてください。"
                )
            for statement in self.statements:
                if not statement.text.strip() or not statement.citations:
                    raise PydanticCustomError(
                        "statement_without_evidence", "全ての結論に本文と引用が必要です。"
                    )
                if any(not citation.quote.strip() for citation in statement.citations):
                    raise PydanticCustomError("blank_quote", "引用原文に文字が必要です。")
        elif (
            self.statements
            or not self.missing_information
            or any(not item.strip() for item in self.missing_information)
        ):
            raise PydanticCustomError(
                "invalid_insufficient_evidence",
                "根拠不足では結論を生成せず、不足情報を示してください。",
            )
        return self
