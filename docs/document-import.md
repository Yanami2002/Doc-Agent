# PDF の取り込み

Fujitsu RAG Hard Benchmark の 5 文書・170 ページを取り込み、ページごとの本文と取得元情報を保存します。取り込み後のチャンク生成は独立した処理です。[チャンク分割](chunking.md) を参照してください。

## 取得元

[Fujitsu RAG Hard Benchmark](https://github.com/FujitsuResearch/Fujitsu-RAG-Hard-Benchmark) の原文 PDF を使用します。URL と SHA-256 を `resources/sources/fujitsu.json` に固定し、タイトルは評価データの証拠ファイル名と照合できるように元のファイル名を保持します。

PDF の利用条件はソフトウェアと別です。原文はローカルの評価用途で使用し、GitHub や公開文書サービスには掲載しません。正解データも本文テーブルには入れません。[取得元と利用条件](../resources/sources/README.md) を参照してください。

## 実行方法

```bash
# DB とスキーマの準備
docker compose --project-directory . -f infra/compose.yaml up -d --wait
uv run alembic -c infra/alembic.ini upgrade head

# 既定の一覧にある 5 文書を取得・登録
uv run jp-doc-agent import-documents

# 正解データ・利用規約を別ディレクトリに取得
uv run jp-doc-agent fetch-benchmark

# 文書一覧を確認
uv run jp-doc-agent documents

# 文書 1 の物理ページ 1 を参照（ID は一覧に合わせて変更）
uv run jp-doc-agent page 1 1

# 登録した本文からチャンクを生成
uv run jp-doc-agent chunk-documents
```

ローカルファイルも取り込めます。

```bash
uv run jp-doc-agent import-pdf /absolute/path/document.pdf --title "文書タイトル"
```

`--source-url` で取得元 URL、`--dataset` でデータセット名を指定できます。省略時はローカルファイルの URI と `local` を保存します。別の取得元一覧を使う場合は `import-documents --manifest <一覧のパス>` を指定してください。

## 処理と整合性

```text
取得元一覧またはローカル PDF
  → 取得・コピー → 形式・サイズの確認 → SHA-256 → 原文の保存
  → 重複の確認 → ページごとの本文抽出 → 文書とページを同一トランザクションで登録
```

- **重複排除**: 同じハッシュ・解析器バージョンは `duplicate` を返します。別 URL から同じ内容を取得した場合は最初の取得元を保持します。URL の内容が変われば別ハッシュの文書になります。
- **再解析**: 解析器のバージョンが変わった場合は `updated` を返し、文書 ID を保持したままページを置き換えます。既存チャンクを明示的に削除し、関連する出典範囲も外部キーで削除します。分割設定をリセットします。その後 `chunk-documents` を実行してください。
- **ハッシュ検証**: 固定した取得元一覧とハッシュが一致しないファイルは登録しません。固定ハッシュを省略した独自の一覧では新しい内容を登録できます。
- **ページ番号**: PDF の物理的な順序を 1 から数えます。白紙ページも保存し、文書に印刷されたページ番号とは区別します。
- **トランザクション**: ページ保存が失敗した場合は、その文書の DB 更新を全てロールバックします。取得済みの PDF は調査・再試行用に残します。
- **失敗記録**: HTTP エラー、PDF 以外の形式、破損、本文抽出不能などを記録し、取得元一覧の次の文書を処理します。失敗があると CLI は非ゼロで終了します。DB の接続失敗など全体に関わるエラーは処理を終了します。

## 対応範囲

1 ファイルの上限は 50 MiB、500 ページです。pypdf で形式とページ数を確認し、pdfminer.six で日本語本文とフォントマッピングを処理します。暗号化の情報があっても開くためのパスワードが不要な PDF は扱えますが、パスワードが必要な PDF は取り込めません。

OCR や図表の意味理解には対応していません。解決できない字形は `(cid:番号)` のまま保持します。該当ページは取り込み結果に記録され、`page`・`chunks` の参照結果には `has_unmapped_characters` が含まれます。これらの文字が必要な回答は、原文確認や OCR による補完が必要です。

## 保存先

| 保存先 | 内容 |
| --- | --- |
| `data/pdfs/<sha256>.pdf` | 原文 PDF のローカルキャッシュ |
| `data/reports/import-<時刻>.json` | 文書ごとの取得・登録結果と失敗理由 |
| `data/evaluation/fujitsu/` | 元の正解 YAML、利用規約、ソフトウェアのライセンス |
| `documents` | 取得元、ハッシュ、原文パス、ページ数、日時、解析器・分割設定 |
| `document_pages` | 文書 ID、物理ページ番号、ページ本文 |
| `document_chunks` | 文書全体から分割したチャンク本文、文書内の順序、結合本文の文字範囲 |
| `chunk_sources` | 各チャンクに対応するページ番号、ページ内とチャンク内の文字範囲 |
| `embedding_profiles` | Embedding モデル・次元数・入力方針と設定ハッシュ |
| `chunk_embeddings` | チャンク本文に対応するベクトル・本文ハッシュ・保存時間 |

ベクトル化・保存は [ベクトル化](embedding.md)、出典付き検索は [ベクトル検索](retrieval.md) を参照してください。原文パスはローカルのパスなので、別の環境では一覧から改めて取り込んでください。

## 確認方法

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run alembic -c infra/alembic.ini check
```

取り込みのテストでは、重複、ページ番号、ロールバック、並行取り込み、HTTP エラー、サイズ制限、破損・暗号化 PDF、ハッシュ不一致、解析器更新を確認します。DB テストは一時 schema を使用します。小さな人工 PDF は境界条件のテスト用で、開発用の実データは公開ベンチマークから取得します。

5 文書は 16 問の全証拠文書をカバーしますが、図表を必要とする問題も含まれます。本文抽出に成功したことと、質問に正しく回答できることは別々に検証します。
