# チャンクのベクトル化

登録済みのチャンク本文を OpenAI の `text-embedding-3-small` API に送り、1,536 次元のベクトルを PostgreSQL / pgvector に保存します。ローカルで Embedding モデルを推論しません。分割設定は最大 300 Token、重複最大 30 Token のままです。

## 準備と実行

[OpenAI の API キー画面](https://platform.openai.com/api-keys) で作成したキーを、既存の DB 設定を保持したまま `.env` に追記します。キーは Git に登録しません。

```dotenv
OPENAI_API_KEY=your_api_key
```

```bash
uv sync --locked
docker compose --project-directory . -f infra/compose.yaml up -d --wait
uv run alembic -c infra/alembic.ini upgrade head

# 全文書の未処理チャンクをベクトル化
uv run doc-agent embed-chunks

# 1 文書だけ処理（ID は documents で確認）
uv run doc-agent embed-chunks --document-id 1 --batch-size 16

# API を呼ばずに、保存済み・未処理件数を確認
uv run doc-agent embedding-status
uv run doc-agent embedding-status --document-id 1
```

`batch-size` は 1〜32、既定値は 32 です。取り込み・分割・保存件数の確認には API キーは不要です。未分割の文書は処理を中止し、`chunk-documents` の実行を案内します。

## API と入力の契約

`embedding/encoder.py` が公式 OpenAI SDK を使用します。API エンドポイントとモデルは固定し、本文だけをそのまま入力します。E5 用の `passage:` / `query:` 接頭辞や評価データは追加しません。設定と本文のハッシュは `embedding/service.py` が管理します。

- 1 件の本文は空白以外を含む 1〜300 Token。`cl100k_base` で確認します。
- モデル名、応答件数、入力番号、1,536 次元、有限値、非ゼロベクトルを保存前に検証します。
- 応答配列の順序に依存せず、`index` に従って入力との対応を復元します。
- 1 回の要求のタイムアウトは 30 秒、SDK の再試行上限は 2 回です。
- 接続障害、408・409・429・5xx は SDK の標準再試行を利用します。API エラーの応答本文やキーは CLI に表示しません。

仕様は [OpenAI の Embedding ガイド](https://developers.openai.com/api/docs/guides/embeddings)、[公式 SDK のリトライ仕様](https://github.com/openai/openai-python#retries)、[pgvector の SQLAlchemy 対応](https://github.com/pgvector/pgvector-python#sqlalchemy) を参照してください。

## 保存形式

| テーブル | 保存内容 |
| --- | --- |
| `embedding_profiles` | provider、モデル名、次元数、エンコーディング、入力方針とその設定ハッシュ |
| `chunk_embeddings` | chunk ID、設定 ID、本文ハッシュ、Token 数、`vector(1536)`、保存時間 |

`(chunk_id, profile_id)` が複合主キーです。同じ設定・本文ハッシュの行は再利用し、API を呼びません。本文ハッシュが違う場合だけ、その設定のベクトルを更新します。今のモデルの次元数はスキーマで 1,536 に固定しており、別次元のモデルを採用する際はマイグレーションが必要です。

文書・チャンクの本文は変更しません。PDF の再解析や分割設定の変更で古い chunk が削除されると、対応する `chunk_embeddings` も外部キーで削除されます。出典は既存の `chunk_sources` で引き続き管理します。

## 失敗・再開・並行更新

API の待機中は DB 接続と文書ロックを保持しません。保存直前に文書をロックし、分割署名・chunk ID・本文が取得時点と一致することを確認します。API 実行中に再解析や再分割が行われた場合、古い本文から生成したベクトルは保存しません。

保存のトランザクションはバッチ単位です。途中で失敗しても、それ以前に完了したバッチを保持します。同じコマンドを再実行すると、完了済みの本文をスキップして未処理分から再開します。API や入力のエラーは文書ごとに `failed` と理由を返し、他の文書の処理を続けます。DB 障害では処理を中止し、保存件数は `embedding-status` で確認できます。

同時実行では複合主キーと本文ハッシュで重複保存・上書きを防ぎます。ただし両方の処理が API を呼ぶ場合があり、API 呼び出しや課金の一度だけの実行は保証しません。通常は 1 プロセスで実行してください。

CLI の `embedded` は今回保存した件数、`skipped` は再利用した件数です。文書の状態は `embedded` / `duplicate` / `empty` / `failed`、失敗がある場合の終了コードは 1 です。`api_tokens` は正常に検証した API 応答の使用量、`api_requests` はその応答数で、SDK 内の再試行や課金総額を表す値ではありません。API 応答後に DB 保存が失敗した場合、再実行で API の再呼び出しが必要になることがあります。

## 確認と次の段階

```bash
uv run pytest -q tests/test_embedding.py tests/test_migrations.py
uv run ruff check .
uv run ruff format --check .
uv run alembic -c infra/alembic.ini check
```

HTTP 応答のモックと実際の PostgreSQL の一時 schema を使用して、API 入力、応答の並び、異常応答、SDK の再試行、バッチ保存、再開、本文変更、並行実行、削除の連鎖、マイグレーションを検証します。テストは実際の API に接続しません。

ローカル検証では、実 API を使って 5 文書・1,037 チャンクのベクトルを全件保存しました。各ベクトルは 1,536 次元、未処理件数は 0 です。再実行すると全件をスキップし、API 応答数と Token 使用量はともに 0 になりました。元の 170 ページと 1,235 件の出典範囲も保持されています。これはベクトル化と保存の動作確認であり、検索品質の評価ではありません。

質問のベクトル化・余弦距離での検索・全出典付きの結果返却は [ベクトル検索](retrieval.md) で実装済みです。[Agent](agent.md) が検索と追加調査を管理し、[回答と引用検証](answering.md) を実行します。開発用の [固定ケース検証](evaluation.md) を記録し、本格的な検索品質の評価は今後実施します。HNSW などの近似検索インデックスは、検索の基準と性能を測定した後に検討します。
