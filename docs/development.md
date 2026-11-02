# ローカル開発ガイド

コマンドはリポジトリのルートで実行してください。Python はホスト側、PostgreSQL は Docker コンテナで動かします。モデル API は取り込み・チャンク分割の実行には必要ありません。

## 初回セットアップ

Git、uv、起動済みの Docker Desktop を用意します。

```bash
uv python install
uv sync --locked
cp .env.example .env
```

`.env` の `POSTGRES_PASSWORD` をランダムなパスワードに変更してから起動します。既に `.env` がある環境では上書きしないでください。

```bash
docker compose up -d --wait
uv run alembic upgrade head
uv run jp-doc-agent check-db
```

接続確認に成功すると `status: ok`、PostgreSQL と pgvector のバージョン、ベクトル距離 `1.0` が返ります。`check-db` は読み取り専用で、業務テーブルは作成しません。

既定の接続先は `127.0.0.1:5433`、DB 名・ユーザー名は `jp_doc_agent` です。独立したコンテナと永続化ボリュームを使います。

## 日常の開発

```bash
# DB を起動
docker compose up -d --wait

# 依存関係とスキーマを最新のリポジトリに合わせる
uv sync --locked
uv run alembic upgrade head

# ヘルプを確認
uv run jp-doc-agent --help

# PDF と評価データを取得
uv run jp-doc-agent import-documents
uv run jp-doc-agent fetch-benchmark

# 最大 300 Token・重複最大 30 Token でチャンクを生成して文書一覧を確認
uv run jp-doc-agent chunk-documents
uv run jp-doc-agent documents

# API キー設定後にベクトル化し、保存件数を確認
uv run jp-doc-agent embed-chunks
uv run jp-doc-agent embedding-status

# 関連する本文と出典を検索
uv run jp-doc-agent search "2024年度第2四半期の売上収益はいくらですか？" --top-k 5

# 必要な追加調査を含む日本語回答・原文引用・実行記録を生成
uv run jp-doc-agent ask "2024年度第2四半期の売上収益はいくらですか？"
uv run jp-doc-agent ask "2021年1月1日時点の富士通の組織構成はどうなっていますか？"

# 正式な問答フローを固定ケースで検証
uv run jp-doc-agent evaluate-rag

# HTTP API を起動し、ブラウザーで http://127.0.0.1:8000/docs を開く
uv run uvicorn jp_doc_agent.api.app:app --host 127.0.0.1 --port 8000 --reload

# テスト・静的検査・スキーマの差分確認
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run alembic check

# コンテナの状態・ログ
docker compose ps
docker compose logs --tail 50 db

# データを保持したまま停止
docker compose stop
```

文書 ID は `documents` の出力で確認してください。ページは `page <文書ID> <ページ番号>`、チャンクは `chunks <文書ID> --page <ページ番号>` で参照できます。

## ファイル構成

```text
jp-doc-agent/
├── README.md                 # 概要、実装状況、技術選定、評価方針
├── pyproject.toml            # 依存関係、CLI、検査設定
├── uv.lock                   # 依存バージョンの固定
├── .python-version           # Python バージョン
├── .env.example              # 設定のテンプレート
├── compose.yaml              # PostgreSQL と永続化設定
├── alembic.ini
├── infra/                    # DB 初期化とスキーマ変更
│   ├── docker/init.sql        # pgvector の有効化
│   └── migrations/           # スキーマ変更履歴
├── resources/                # 公開データの取得元と評価ケース
│   ├── sources/              # PDF の取得元とハッシュ
│   └── evaluation/           # 開発用の質問・許容証拠ページ・期待値
├── tests/
│   ├── conftest.py           # 一時 DB schema と PDF・モデル API fixture
│   ├── test_ingestion.py    # PDF 取り込みの検証
│   ├── test_chunking.py     # 分割・原文位置・保存・再生成・CLI
│   ├── test_embedding.py    # API 応答・リトライ・保存・再開・競合
│   ├── test_migrations.py   # スキーマ移行で原文・出典を保持
│   ├── test_retrieval.py     # 検索順位、範囲、出典、並行更新
│   ├── test_answering.py     # 回答草案の構造、引用照合、原文位置
│   ├── test_agent.py         # 問答・CLI、追加調査、停止上限、失敗記録
│   ├── test_api.py           # HTTP の型、引用、失敗、接続、並行応答
│   └── test_evaluation.py    # 期待値の隔離、判定、失敗記録
├── docs/
│   ├── development.md        # 本ガイド
│   ├── document-import.md    # PDF 取り込み
│   ├── chunking.md           # 分割方法、保存形式、検証結果
│   ├── embedding.md          # API 接続、ベクトル保存、再開
│   ├── retrieval.md          # 検索方法、出典、未処理範囲
│   ├── answering.md          # 回答生成、構造化出力、引用の検証
│   ├── agent.md              # 道具、状態遷移、停止条件、実行記録
│   ├── web-api.md            # HTTP の起動、型、ステータス、実行記録
│   └── evaluation.md         # 固定ケース、判定、実 API の成功・失敗
└── src/jp_doc_agent/
    ├── config.py             # .env から設定を読み込む
    ├── database.py           # DB 接続と pgvector の確認
    ├── cli.py                # CLI の引数と出力
    ├── llm.py                # 回答・道具選択で共用する構造化出力 API
    ├── reports.py            # 評価・CLI・HTTP の JSON 実行記録
    ├── models.py             # データモデルと DB 制約
    ├── benchmark.py          # 正解データを本文とは別に取得
    ├── chunking/
    │   ├── splitter.py       # Token 計数、本文分割、ページへの対応付け
    │   └── service.py        # チャンクの保存、再生成、出典付き参照
    ├── embedding/
    │   ├── encoder.py        # API 接続、入力と応答の検証
    │   └── service.py        # バッチ保存、重複スキップ、保存件数の確認
    ├── retrieval/
    │   └── service.py        # 余弦距離検索、文書範囲、出典検証
    ├── answering/
    │   ├── schema.py         # 結論・引用・根拠不足の草案
    │   ├── generator.py      # 回答用の指示と草案生成
    │   └── citations.py      # 引用照合、原文位置と文書情報の補完
    ├── agent/
    │   ├── schema.py         # 状態、道具の引数、実行上限
    │   ├── tools.py          # 検索・文書一覧・原文ページの参照
    │   └── service.py        # LangGraph の追加調査、停止、実行記録
    ├── api/
    │   ├── app.py            # 初期化・終了とアプリケーションの組み立て
    │   ├── dependencies.py   # DB 接続と問答用のモデル接続
    │   ├── schema.py         # HTTP の入出力型と質問の検証
    │   ├── routes.py         # 文書参照・問答を既存サービスに接続
    │   └── errors.py         # 安全な共通エラー本文と HTTP ステータス
    ├── evaluation/
    │   └── service.py        # 期待値の隔離、証拠と回答の確認、レポート
    └── ingestion/
        ├── download.py       # ダウンロード、形式・サイズ・ハッシュの確認
        ├── pdf.py            # ページ単位の日本語本文抽出
        └── service.py        # 取り込みとページ参照
```

`.env`、`.venv/`、`data/`、キャッシュは Git の管理対象外です。Agent・評価・HTTP 問答の JSON レポートは `data/reports/` に保存します。LangGraph は Agent、FastAPI は HTTP API に使用しています。

機能ごとの処理は `ingestion/`、`chunking/`、`embedding/`、`retrieval/`、`answering/`、`agent/`、`evaluation/` に置き、設定・接続・データモデル・モデル API・CLI は共通部分としてパッケージ直下に置きます。`api/` は HTTP の型・ルーティング・エラーを担当します。CLI・Web API・固定ケース評価は同じ `agent/service.py` の問答フローを呼び出し、JSON 保存は `reports.py` で共用します。`chunking/splitter.py` は本文分割、`chunking/service.py` は永続化、`retrieval/service.py` は検索 SQL と出典検証、`answering/` は回答草案と引用検証を担当します。Agent の道具は既存の参照処理を再利用し、回答と道具選択は `llm.py` を共用します。評価の期待値はモデルに渡しません。`infra/migrations/versions/` は適用済み環境を更新するための履歴なので、古いファイルも保持します。

テストは Web API を含む 9 ファイルにまとめ、分割と保存、Embedding API と保存をそれぞれ同じ機能のファイルに統合しました。移行テストも `test_migrations.py` に統合し、重複する引数検証・既定値や固定ケース名だけの確認を削除しました。並行処理・ロールバック・失敗後の再開・原文一致の検証は保持しています。

## データベースの注意点

`.env` のパスワードを書き換えても、作成済みの DB のパスワードは変更されません。初期化 SQL は空のボリュームから初めて起動したときにだけ実行されます。既存の DB で拡張を有効にする場合は次のコマンドを使います。

```bash
docker compose exec -T db sh -c 'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < infra/docker/init.sql
```

Docker に接続できない場合は Docker Desktop の起動を確認してください。テストの DB fixture は一時 schema を作成・削除するため、その権限が必要です。

## Git の運用

機能開発は `feat/basic-rag` のようなブランチで行い、動作確認できる単位でコミットします。必要なテスト・説明を含めて GitHub に push し、機能が利用可能になったら PR を通して `main` に統合します。PR には変更点と確認結果を記載します。
