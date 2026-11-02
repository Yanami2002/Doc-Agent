"""回答と Agent の計画で共用する、構造化出力のモデルクライアント。"""

import json
from dataclasses import dataclass

from openai import APIConnectionError, APIError, APIStatusError, OpenAI
from pydantic import BaseModel, ConfigDict, ValidationError

from jp_doc_agent.config import ANSWER_MODEL


@dataclass(frozen=True)
class ModelUsage:
    model: str
    input_tokens: int
    output_tokens: int


class ModelError(RuntimeError):
    """固定コードと診断値だけを保持し、API 本文は公開しない。"""

    def __init__(
        self,
        message: str,
        *,
        code: str = "model_error",
        details: dict | None = None,
        usage: ModelUsage | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.details = details or {}
        self.usage = usage


def _validation_details(error: ValidationError, schema: dict) -> dict:
    # extra_forbidden の loc などにモデル生成の任意文字列が入るため、既知フィールドだけ残す。
    fields = set()

    def collect(node):
        if isinstance(node, dict):
            fields.update(node.get("properties", {}))
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    collect(schema)
    return {
        "violations": [
            {
                "type": item["type"],
                "path": [
                    part if isinstance(part, int) or part in fields else "<unknown_field>"
                    for part in item["loc"]
                ],
            }
            for item in error.errors(include_input=False, include_context=False, include_url=False)[
                :10
            ]
        ]
    }


@dataclass(frozen=True)
class ModelResult[T: BaseModel]:
    value: T
    model: str
    input_tokens: int
    output_tokens: int


class _OutputContent(BaseModel):
    model_config = ConfigDict(strict=True)
    type: str
    text: str | None = None


class _OutputItem(BaseModel):
    model_config = ConfigDict(strict=True)
    type: str
    content: list[_OutputContent] | None = None


class _Usage(BaseModel):
    model_config = ConfigDict(strict=True)
    input_tokens: int
    output_tokens: int
    total_tokens: int


class _ResponseEnvelope(BaseModel):
    # 使用する項目だけを検証し、SDK の未使用フィールドの追加に依存しない。
    model_config = ConfigDict(strict=True)
    model: str
    status: str
    output: list[_OutputItem]
    usage: _Usage | None = None


class StructuredModel:
    def __init__(self, client: OpenAI, *, model: str = ANSWER_MODEL, max_output_tokens: int = 2048):
        if not model.strip() or not 256 <= max_output_tokens <= 8192:
            raise ValueError("回答モデルと出力 Token 上限（256〜8192）を確認してください。")
        self.client = client
        self.model = model
        self.max_output_tokens = max_output_tokens

    def request[T: BaseModel](
        self,
        payload: dict,
        *,
        instructions: str,
        output_type: type[T],
        schema_name: str,
        output_schema: dict | None = None,
    ) -> ModelResult[T]:
        try:
            response = self.client.responses.create(
                model=self.model,
                store=False,
                max_output_tokens=self.max_output_tokens,
                input=[
                    {"role": "developer", "content": instructions},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": True,
                        "schema": output_schema or output_type.model_json_schema(),
                    }
                },
            )
        except APIStatusError as error:
            raise ModelError(
                f"モデル API が HTTP {error.status_code} を返しました。"
                "モデル・API キー・利用枠を確認してください。"
            ) from None
        except APIConnectionError:
            raise ModelError("モデル API に接続できません。接続状況を確認してください。") from None
        except APIError:
            raise ModelError("モデル API の応答を読み取れません。再実行してください。") from None
        try:
            response = _ResponseEnvelope.model_validate(
                response.model_dump(warnings=False), strict=True
            )
        except ValidationError:
            raise ModelError("モデル API の応答形式が不正です。回答を中止しました。") from None
        if response.status != "completed":
            raise ModelError("モデル API の応答が完了していません。回答を中止しました。")
        usage = response.usage
        if (
            not response.model.strip()
            or usage is None
            or min(usage.input_tokens, usage.output_tokens) < 0
            or usage.total_tokens != usage.input_tokens + usage.output_tokens
        ):
            raise ModelError("モデル API のモデルまたは使用量が不正です。回答を中止しました。")
        verified_usage = ModelUsage(response.model, usage.input_tokens, usage.output_tokens)
        contents = [
            content
            for item in response.output
            if item.type == "message" and item.content is not None
            for content in item.content
        ]
        if any(content.type == "refusal" for content in contents):
            raise ModelError("モデルが生成を拒否しました。回答を中止しました。")
        texts = [content.text for content in contents if content.type == "output_text"]
        if len(texts) != 1 or texts[0] is None:
            raise ModelError("モデル API の本文件数が不正です。回答を中止しました。")
        try:
            value = output_type.model_validate_json(texts[0], strict=True)
        except ValidationError as error:
            raise ModelError(
                "モデル出力の構造が不正です。回答を中止しました。",
                code="schema_validation",
                details=_validation_details(error, output_type.model_json_schema()),
                usage=verified_usage,
            ) from None
        return ModelResult(value, response.model, usage.input_tokens, usage.output_tokens)
