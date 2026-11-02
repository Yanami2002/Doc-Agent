# Doc-Agent｜日本語ドキュメント調査エージェント

公開されている日本語 PDF を対象に、根拠となる文書とページ番号を確認できる調査支援アプリケーションを開発しています。日本語 RAG、ツール呼び出し、評価、バックエンドのデータ管理を一つのプロジェクトで扱います。

**現在は PDF の取り込み・300 Token 分割・API によるベクトル化・出典付き検索・LangGraph による調査と日本語回答・実行記録・FastAPI Web API まで実装済みです。** 問答は一つの Agent フローで提供し、RAG を証拠の取得に使います。Web 画面は今後実装します。少数の固定ケースを検証していますが、本格的な検索・回答精度の評価は未実施です。

## 解決したい課題

企業の報告書や制度の説明資料を読む際は、複数の PDF から必要な情報を探し、数値や条件を照合し、根拠を確認する作業が必要です。本プロジェクトでは、日本語の質問から関連資料を検索し、必要に応じて追加で調査し、出典付きで回答することを目指します。

想定する質問例（データセットの問題ではありません）:

> この資料に記載されている制度の対象者と申請条件を整理し、根拠となるページを示してください。

回答には結論、根拠の本文、文書名、取得元 URL、PDF の物理ページ番号を付けます。根拠が足りない場合は不足情報を示す設計です。

## 実装状況

| 機能 | 状況 | 現在できること／今後の要件 |
| --- | --- | --- |
| 開発環境 | 実装済み | Python 3.12、uv、PostgreSQL 17、pgvector、Alembic |
| PDF 取り込み | 実装済み | 取得、ハッシュ検証、ページ単位の日本語抽出、重複排除、再解析、失敗記録 |
| チャンク分割 | 実装済み | 文書全体を対象にしたページをまたぐ分割、300 Token 上限、ページごとの原文位置の検証、設定の保存、重複排除 |
| 文書・ページ・チャンク参照 | 実装済み | CLI で取得元、ページ番号、本文、文字位置を確認 |
| ベクトル化・保存 | 実装済み | OpenAI API、1,536 次元、バッチ保存、重複スキップ、再開、旧チャンク削除時のベクトル削除 |
| ベクトル検索 | 実装済み | 質問のベクトル化、余弦距離の上位 K 件、文書範囲の指定、本文・全出典の返却、旧ベクトルの除外 |
| 質問応答・Agent | 実装済み | LangGraph で検索・証拠の判断・必要な追加調査・日本語回答を実行。全結論の原文引用、根拠不足、検索最大 3 回・道具全体既定 6 回の停止条件 |
| 引用検証・実行記録 | 実装済み | 実 chunk ID・引用原文・全ページ範囲を照合。Agent の引数・結果・停止理由・モデル・時間・Token 数を JSON 保存 |
| 固定ケース検証 | 実装済み | 開発用 3 ケースで正式な Agent フローの検索・回答・引用・根拠不足・実行コストを記録 |
| Web API | 実装済み | 健康チェック、文書一覧、原文ページ、Agent 問答、型の検証、共通エラー、実行記録 |
| 本格評価・画面 | 未実装 | 独立したテスト集合、デモ画面 |

現在のローカル検証では、Fujitsu RAG Hard Benchmark の **5 文書・170 ページから 1,037 チャンク**を生成しました（`cl100k_base` で最大 300 Token、重複は最大 30 Token）。187 チャンクが複数ページをカバーし、出典は 1,235 件のページ内文字範囲として保存しています。同じ設定で再実行すると全 5 文書が `duplicate` になり、チャンク数は変わりません。これは取り込み・分割の検証結果であり、検索や回答の品質を示す数値ではありません。

実際の OpenAI API を使用して全 1,037 チャンクのベクトルを保存し、未処理件数 0 を確認しました。再実行は全件スキップし、API を呼びません。過去の開発用 3 ケースの実 API 検証では、Agent が組織資料の検索漏れを追加調査で解消し、未収録年度は根拠不足を返しました。一方、別の Web API 検証では組織構成の引用不一致も記録しています。少数例は全体の精度を表さず、引用の意味的な正しさを保証する評価でもありません。[固定ケース検証と開発記録](docs/evaluation.md)、[Web API の確認結果](docs/web-api.md) を参照してください。

## クイックスタート

Git、uv、起動済みの Docker Desktop が必要です。以下はリポジトリのルートで実行します。

```bash
uv python install 3.12
uv sync --locked
cp .env.example .env
```

`.env` の `POSTGRES_PASSWORD` をランダムなパスワードに変更してから起動してください。

```bash
docker compose --project-directory . -f infra/compose.yaml up -d --wait
uv run alembic -c infra/alembic.ini upgrade head
uv run doc-agent check-db

# 公開ベンチマークの PDF を取得・登録
uv run doc-agent import-documents

# 評価用の正解データを本文とは別に取得
uv run doc-agent fetch-benchmark

# 登録済みページを分割
uv run doc-agent chunk-documents
uv run doc-agent documents
```

一覧に表示された文書 ID を使って参照します。次の `1` は文書 ID の例です。

```bash
uv run doc-agent page 1 1
uv run doc-agent chunks 1 --page 1 --limit 5
```

詳しい準備・確認方法は [開発ガイド](docs/development.md)、[PDF 取り込み](docs/document-import.md)、[チャンク分割](docs/chunking.md)、[ベクトル化](docs/embedding.md)、[ベクトル検索](docs/retrieval.md)、[回答と引用検証](docs/answering.md)、[Agent](docs/agent.md)、[Web API](docs/web-api.md)、[固定ケース検証](docs/evaluation.md) を参照してください。取り込みと分割に API キーは不要です。ベクトル化・検索・回答生成には `.env` の `OPENAI_API_KEY` が必要です。

```bash
uv run doc-agent embed-chunks
uv run doc-agent embedding-status

# 質問に関連する本文・出典を上位 5 件返す
uv run doc-agent search "2024年度第2四半期の売上収益はいくらですか？" --top-k 5

# 指定文書に限定（1 は ID の例）
uv run doc-agent search "対象者と申請条件は？" --document-id 1

# 検索・根拠の判断・必要な追加調査を行い、回答と実行記録を保存
uv run doc-agent ask "2024年度第2四半期の売上収益はいくらですか？" --document-id 1
uv run doc-agent ask "2021年1月1日時点の富士通の組織構成はどうなっていますか？"

# 少数の固定ケースで成功・失敗を記録
uv run doc-agent evaluate-rag

# HTTP API を起動（http://127.0.0.1:8000/docs で試す）
uv run uvicorn jp_doc_agent.api.app:app --host 127.0.0.1 --port 8000 --reload
```

## ディレクトリ構成

```text
src/        アプリケーション本体
tests/      自動テスト
docs/       機能・開発ガイド
resources/  PDF の取得元と評価ケース
infra/      DB 初期化とマイグレーション
```

ルートには起動・依存管理に必要な設定を置いています。詳細は [開発ガイド](docs/development.md#ファイル構成) を参照してください。

## 構成と技術選定

```text
実装済み:
取得元一覧 → PDF の取得・検証 → ページ本文 → チャンク → PostgreSQL
                                            ↓
                                    OpenAI Embedding API
                                            ↓
                                    pgvector に保存

CLI / FastAPI → 日本語の質問 → LangGraph
                                ↓
           RAG 検索 / 文書一覧 / 原文ページ参照
                                ↓
                 根拠の判断 → 必要な追加調査
                                ↓
                  引用照合 → 回答・根拠不足
                                ↓
                         実行記録 JSON

RAG 検索: 質問 → OpenAI Embedding API → pgvector → 本文・出典
文書参照: FastAPI → 文書一覧・原文ページ

今後実装:
デモ画面 → 独立した問題での評価
```

| 技術 | 用途・採用理由 | 状況 |
| --- | --- | --- |
| Python 3.12 + uv | 依存関係と実行環境を `uv.lock` で固定 | 使用中 |
| pypdf + pdfminer.six | ファイル・ページ数の検証と日本語本文の抽出を分離 | 使用中 |
| [langchain-text-splitters](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter) | 日本語の段落・句点を優先し、tiktoken で Token 数を計測して分割 | 使用中 |
| [tiktoken](https://github.com/openai/tiktoken) | `text-embedding-3-small` に対応する `cl100k_base` で最大 300 Token を確認 | 使用中 |
| PostgreSQL 17 + pgvector | 文書・ページ・チャンク・ベクトルを管理し、余弦距離で検索 | 使用中 |
| SQLAlchemy + Alembic | トランザクション、制約、バージョン管理されたスキーマ変更 | 使用中 |
| pytest + Ruff | データの整合性を検証し、コードの形式を統一 | 使用中 |
| [FastAPI](https://fastapi.tiangolo.com/) + Pydantic | 文書参照・質問応答の HTTP API、入出力検証、OpenAPI | 使用中 |
| [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) | 検索、証拠の判断、追加調査、停止条件を状態遷移で管理 | 使用中 |
| OpenAI Embedding API + 公式 SDK | `text-embedding-3-small` でベクトル化。設定・本文ハッシュを保存し、実行時の Token 使用量を表示 | 使用中 |
| OpenAI Responses API + GPT-4.1 mini | 構造化出力で根拠付き回答。モデル・指示版・時間・Token 数を返却 | 使用中 |
| Streamlit | 質問、引用、実行過程を確認できるデモ画面 | 計画中 |

LangGraph は Agent の実行フローを担当し、チャンク分割は独立した前処理として実装します。日本語の形態素解析は、今後キーワード検索を追加する段階で検討します。

アプリケーション固有の実装範囲は、取得元・ハッシュの管理、解析と保存の連携、原文位置の検証、分割設定の管理、トランザクションによる再生成、ベクトルのバッチ保存・再開・原文更新との整合性、検索範囲・出典・並行更新の管理、回答草案と引用の検証、追加調査と停止条件、実行記録、固定ケースの期待値の隔離、CLI とテストです。PDF 抽出と再帰的な分割アルゴリズムには既存ライブラリを利用しています。

## データと利用範囲

主なデータは [Fujitsu RAG Hard Benchmark](https://github.com/FujitsuResearch/Fujitsu-RAG-Hard-Benchmark) です。上流のコミットとファイルハッシュを固定し、元のファイル名、取得元、取得時間、解析器のバージョンを記録しています。

現在の 5 文書は、元の 100 問のうち 16 問について全証拠文書をカバーします。ただし図表を読む必要のある問題も含まれるため、16 問すべてに本文だけで回答できるとは限りません。正解データは評価用ディレクトリに保存し、本文・チャンクのテーブルには登録しません。

PDF にはソフトウェアと別の利用条件があります。PDF は本ベンチマークの評価用途に限定され、商用利用や再配布は禁止されています。このリポジトリには取得コードと一覧のみを含めます。公開 Web デモには別途利用可能な資料を用意します。固定バージョン、問題 ID、条項へのリンクは [データの取得元](resources/sources/README.md) を参照してください。

[JQaRA](https://github.com/hotchpotch/JQaRA) は、今後検索と再ランキングを比較するための独立した評価データとして検討しています。

## 評価方針

CLI・Web API と同じ Agent フローを固定ケースで検証します。今後は独立した問題を追加し、キーワード検索、ハイブリッド検索、再ランキングなどの変更前後で検索・回答の品質とコストを測ります。過去の単回 RAG との比較は開発記録として保持しています。

| 観点 | 評価する内容 |
| --- | --- |
| 検索 | 初回と追加調査で正解の証拠ページを取得できるか。JQaRA は別途 nDCG@10 を報告 |
| 回答 | 参照回答との主要事実の一致、数値・単位の照合、サンプルの人手確認 |
| 引用 | 原文にたどれるか、本文が実際に結論を裏付けているか |
| 根拠不足 | 回答できない場合に不足情報を示せるか |
| 調査の効率 | 根拠不足を追加調査で補えるか、重複・無効な呼び出しが増えないか |
| 実行コスト | 所要時間、呼び出し回数、Token 使用量、推定費用 |

文書単位で開発用・テスト用を分け、正解を検索対象や回答生成のプロンプトに含めません。本文で回答可能な部分集合の選定規則と問題 ID を公開し、部分集合の結果をデータセット全体の成績として扱いません。改善しなかったケースや退行も実行記録とともに報告します。

## 確認と制約

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run alembic -c infra/alembic.ini check
```

テストは Web API を含む機能ごとの 9 ファイルに整理しています。共通 fixture は `conftest.py`、スキーマ移行は `test_migrations.py`、回答草案と引用は `test_answering.py`、問答の一連の処理は `test_agent.py` で検証します。PDF・原文・出典・並行更新・ロールバック・再開、Agent の追加調査・停止上限・失敗履歴・固定ケース、HTTP の型・ステータス・接続の終了・並行応答を確認します。DB テストは実際の PostgreSQL の一時 schema で実行し、アプリケーションの業務テーブルを変更しません。

現在は OCR、図表の意味理解、複雑なレイアウトの復元に対応していません。解析できない字形は `(cid:番号)` として残し、参照時に警告用のフラグを返します。分割はページをまたぐことができますが、長い文は Token 数上限で分かれる場合があります。ページの読み順、見出し・脚注、表の構造は復元していません。引用照合は原文の一致を確認しますが、結論の意味・数値・条件の正しさは保証しません。検索・回答の品質は独立した問題で今後評価します。

## ロードマップ

- [x] 開発環境とデータベースの初期化
- [x] 公開 PDF の取り込み、重複排除、ページ参照
- [x] 日本語チャンク分割、原文位置と設定の保存、再生成
- [x] API によるベクトル化、pgvector 保存、重複スキップと再開
- [x] ベクトル検索、文書範囲の指定、本文と全出典の返却
- [x] 出典付き回答、原文引用の照合、根拠不足、少数の固定ケース検証
- [ ] 本格的な検索・回答評価と引用の意味・数値・条件の検証
- [x] LangGraph による問答フローの統一、追加調査、停止条件、引用検証、実行記録
- [ ] 検索の改善と再ランキング、根拠不足・複数ステップの評価
- [x] FastAPI の文書参照・問答 API、入出力検証、共通エラー、実行記録
- [ ] デモ画面、実測評価と失敗例を含む再現可能なリリース
