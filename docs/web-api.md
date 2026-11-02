# Web API

FastAPI で文書参照と日本語問答を HTTP から実行します。問答は CLI・評価と共通の Agent フローを使い、RAG による検索・必要な追加調査・引用検証を実行します。PDF 取り込みとベクトル生成は CLI で実行してください。

## 起動

リポジトリのルートで実行します。既存の `.env` は上書きしないでください。

```bash
uv sync --locked
docker compose --project-directory . -f infra/compose.yaml up -d --wait
uv run alembic -c infra/alembic.ini upgrade head
uv run uvicorn jp_doc_agent.api.app:app --host 127.0.0.1 --port 8000 --reload
```

- [Swagger UI](http://127.0.0.1:8000/docs): 日本語の概要、入出力の型、リクエスト例を確認して API を試せます。
- [OpenAPI JSON](http://127.0.0.1:8000/openapi.json): インターフェース定義を取得できます。

DB 接続プールは [lifespan](https://fastapi.tiangolo.com/advanced/events/) で作成・終了します。健康チェックと文書・ページの参照はモデル API キーを必要としません。問答時だけモデル設定を読み、リクエストごとに API クライアントを作成して終了します。API キーはサーバーの `.env` に設定し、HTTP の入力には含めません。

既存の DB・SDK は同期処理なので、各 API は通常の `def` で定義しています。[FastAPI のスレッドプール](https://fastapi.tiangolo.com/async/#path-operation-functions) で処理し、モデルの待機でイベントループをブロックしません。質問の実行中に健康チェックが応答することをテストしています。

## インターフェース

| メソッド・パス | 動作 |
| --- | --- |
| `GET /health` | PostgreSQL・pgvector・実ベクトル演算を確認。DB 障害は 503 |
| `GET /documents` | 登録文書の ID・名称・データセット・ページ数・SHA-256 を取得 |
| `GET /documents/{id}/pages/{page}` | 1 始まりの PDF 物理ページ本文、文書名・取得元 URL を取得 |
| `POST /ask` | 日本語の回答・原文引用・出典・調査履歴・使用量を取得 |

文書・ページが存在しない場合は 404、参照できる白紙ページは空文字の本文で 200 です。PDF の保存先や DB 接続情報は返しません。

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/documents

# 1 は文書 ID の例です。documents の出力で確認してください。
curl http://127.0.0.1:8000/documents/1/pages/1

curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"2024年度第2四半期の売上収益はいくらですか？","top_k":5}'
```

## 質問と回答

| 入力 | 型・制約・既定値 |
| --- | --- |
| `question` | 必須の文字列。空白だけは不可、最大 8,191 Token・32,768 文字。原文の空白を維持 |
| `document_id` | 正の整数または `null`。省略時は全文書、追加調査も指定範囲を維持 |
| `top_k` | 整数 1〜20、既定値 5。1 回の検索で取得する件数 |

JSON の数値文字列や未定義の入力項目は拒否します。不正入力・不存在の指定文書は、モデル接続や課金 API を呼ぶ前に判定します。

全ての質問を [Agent](agent.md) に渡します。初回の証拠で回答できれば終了し、足りなければ追加調査します。検索最大 3 回、原文参照最大 2 回、道具全体最大 6 回、証拠本文最大 16,000 Token で停止します。この API では Agent の上限をリクエストで変更しません。

成功時は `request_id`、回答、検証済みの引用・文書名・URL・物理ページ・文字範囲を返します。`usage`・`counts`・`elapsed_seconds`・モデル・指示版も返します。`trace` は調査過程と短い目的、`limits` は実行上限、`stop_reason` は停止理由です。引用の完全一致と出典の検証は既存サービスが担当します。

根拠不足は 200・`status: insufficient_evidence` で、不足情報と空の結論・引用を返します。別年度の数値を流用して成功扱いにしません。

API の 1 回の接続タイムアウトは 30 秒、SDK 再試行は最大 2 回で、HTTP リクエスト全体の 30 秒制限ではありません。Agent は複数回のモデル呼び出しを行います。`usage` は検証できた応答の Token 数で、再試行・応答形式の検証失敗・課金総額は含みません。ストリーミング、履歴参照 API、Web 画面は未実装です。

## 実行記録とエラー

各 HTTP リクエストにサーバー側で UUID を割り当て、`X-Request-ID` ヘッダーで返します。問答を実行した場合は、成功・根拠不足・モデルや引用検証の失敗を `data/reports/api-<request_id>.json` に保存します。失敗時も実行済みの履歴と取得済みの使用量を保存します。入力検証や設定の失敗、文書参照は問答の実行記録を作りません。質問・回答を DB に保存せず、既存のページ・チャンク・ベクトルを変更しません。

エラー本文は統一した構造です。[カスタム例外処理](https://fastapi.tiangolo.com/tutorial/handling-errors/) で入力本文・SDK 応答・DB の SQL を返さず、安全な日本語の説明を表示します。

```json
{
  "request_id": "サーバーが生成したリクエスト ID",
  "error": {"code": "document_not_found", "message": "指定された文書が見つかりません。"},
  "trace": []
}
```

| HTTP | 主なエラーコード |
| --- | --- |
| 404 | `document_not_found`・`page_not_found`・`not_found` |
| 405 | `method_not_allowed` |
| 409 | `document_state_error`: 文書の状態変更などにより調査を開始できない |
| 422 | `invalid_request`: 入力の型・範囲・未定義項目が不正 |
| 502 | `model_error`: API 接続・タイムアウト・出力や引用の検証が失敗 |
| 503 | `database_unavailable`・`model_configuration_error`・`report_unavailable` |
| 500 | `agent_execution_error`・`internal_error` |

Agent の実行中に DB 障害が起きた場合も 503 で履歴を返します。実行記録の保存失敗は 503 です。その時点でモデルの処理が完了している場合があります。サーバー側で問答全体を自動再実行しません。

## 構成と確認

`api/app.py` は初期化・終了、`dependencies.py` は接続の取得、`schema.py` は HTTP の型、`routes.py` は既存サービスへの接続、`errors.py` は HTTP エラーを担当します。`reports.py` の JSON 保存処理は CLI・評価と共用します。Agent はエラー種別を返し、API はメッセージ文字列を解析せず HTTP ステータスに対応付けます。

```bash
uv run pytest -q tests/test_api.py
uv run ruff check .
uv run ruff format --check .
uv run alembic -c infra/alembic.ini check
```

実 DB の一時スキーマと模擬 API で、回答・根拠不足、ページをまたぐ引用、入力・設定・文書の検証、タイムアウト、引用不一致、実行記録・保存失敗、接続の終了、DB 障害、長い問答と健康チェックの並行応答を確認しています。旧 `mode` 項目は未定義入力として 422 を返し、モデル API を呼びません。

統一後は FastAPI TestClient・既存 PostgreSQL・実 OpenAI API で決算概要の問答を確認しました。HTTP 200 で 8,666 億円と物理ページ 31 の引用を返し、引用のページ内文字範囲を原文と照合しました。検索・道具・モデルの呼び出しは各 1 回で、保存 JSON は HTTP 応答と一致しました。5 文書・170 ページ・1,037 チャンク・1,235 出典範囲・1,037 ベクトルとページ本文のハッシュは実行前後で変わっていません。

Agent に統一する前の実 HTTP・OpenAI API 検証では、Agent が決算概要に 200 で回答し、8,666 億円・物理ページ 31 の引用と保存 JSON を照合しました。組織構成では一覧と原文ページを取得した一度の実行で引用原文の不一致となり、502 と失敗履歴を保存しました。先行する [3 ケースの開発記録](evaluation.md) と別の実行であり、この失敗も残しています。原文一致だけで結論の意味的な正しさを保証できず、モデル出力は実行ごとに変わる場合があります。
