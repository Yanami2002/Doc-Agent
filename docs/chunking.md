# 日本語本文のチャンク分割

登録済みのページ本文を文書ごとに結合し、検索に使える小さなテキストの単位に分割して保存します。ページ境界を越えたチャンクにも、各ページの原文位置を付けます。分割と原文位置の検証は `src/jp_doc_agent/chunking/splitter.py`、DB への保存・参照・再生成は `src/jp_doc_agent/chunking/service.py` が担当します。保存先は `document_chunks` と `chunk_sources` です。Embedding は [ベクトル化](embedding.md) で実装済みです。出典付き検索は [ベクトル検索](retrieval.md) で実装済みです。

## 実行方法

```bash
uv sync --locked
docker compose --project-directory . -f infra/compose.yaml up -d --wait
uv run alembic -c infra/alembic.ini upgrade head

# 登録済みの全文書を分割
uv run jp-doc-agent chunk-documents

# 1 文書だけ分割（ID は documents の出力に合わせる）
uv run jp-doc-agent chunk-documents --document-id 1

# 最大 Token 数と重複の上限を変更して再生成
uv run jp-doc-agent chunk-documents --document-id 1 --chunk-size 250 --chunk-overlap 25

# 分割結果、原文位置、取得元、使用設定を確認
uv run jp-doc-agent chunks 1 --page 1 --limit 5
uv run jp-doc-agent chunks 1 --limit 20 --offset 20
```

`chunks` の `limit` は 1〜100、既定値は 20 です。`total` はページで絞り込んだ後の全件数、`offset` は先頭からスキップする件数です。新規取り込み直後の文書は `total: 0`、`chunking_config: null` で返します。

`--page 2` は、出典に物理ページ 2 を含むチャンクを返します。ページ 1〜2 をカバーするチャンクは、どちらのページからも同じ ID で参照できます。白紙ページを出典として扱うことはありませんが、元のページ記録と番号は保持します。

各チャンクの `token_count` は `cl100k_base` で数えた本文の Token 数です。`start_char` / `end_char` の単位は引き続き文字位置で、Token 位置ではありません。

## 分割の設定と理由

[LangChain の RecursiveCharacterTextSplitter](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter) を利用し、次の順で区切りを探します。

```python
["\n\n", "。", "！", "？", "\n", "、", " ", ""]
```

段落と日本語の文末を優先し、長すぎる部分にはより細かい区切りを使います。最後の空文字列は、区切りのない長文を文字単位で分割するためのものです。PDF の単一改行はレイアウト由来の場合があるため、句点より低い優先度にしています。これは文章構造を復元する処理ではなく、現在の抽出本文に適用する初期設定です。

| 設定 | 既定値 | 意味 |
| --- | --- | --- |
| `chunk_size` | 300 | 前文の重複を含めた 1 チャンクの最大 Token 数 |
| `chunk_overlap` | 30 | 前文から補う重複部分の最大 Token 数 |
| 長さの計測 | tiktoken / `cl100k_base` | 採用した API モデル `text-embedding-3-small` に対応 |
| `keep_separator` | `end` | 区切り文字を直前のテキストに残す |
| `strip_whitespace` | `False` | 原文との対応のため、空白を自動削除しない |
| 分割範囲 | 同じ文書の全文 | ページ境界を強制的な分割位置にしない |
| ページの結合 | 改行 1 文字 | 元の本文を保持し、ページ位置を別途追跡 |

300 Token は上限であり、全チャンクを同じ長さにはしません。`chunk_size` は正の整数、`chunk_overlap` は 0 以上かつ `chunk_size` 未満です。300/30 は初期設定であり、検索品質を評価して最適化した値ではありません。

まず `chunk_size - chunk_overlap`（既定値: 270 Token）を本文の分割予算として、非重複の本文を LangChain で分割します。原文との対応を文字位置で記録した後、直前の文脈を最大 30 Token 補います。合成した全文を再計数し、300 Token を超えないことを確認します。境界での再エンコードや Unicode 文字境界の調整により、重複は 30 Token より短くなる場合があります。重複を 0 にすると、本文の分割予算は 300 Token です。

LangChain の `add_start_index` は重複量を文字数として扱うため利用せず、非重複部分を原文の順に照合して文字位置を求めます。Token 列の途中から UTF-8 を復号して原文を作り直すことはせず、常に元の文字列の範囲を保存します。特殊トークン風の文字列も通常の本文として計数します。日本語の形態素解析は、後続のキーワード検索で別途検討します。LangGraph は今後の Agent フローを担当し、分割処理は独立して実行できます。

## Embedding API の採用方針

ベクトル化は OpenAI の `text-embedding-3-small` API を使用します。[Dify の公式 OpenAI プラグイン](https://github.com/langgenius/dify-official-plugins/blob/main/models/openai/models/text_embedding/text-embedding-3-small.yaml) に同モデルの定義があり、[Haystack](https://docs.haystack.deepset.ai/docs/openaidocumentembedder) も OpenAI Embedding モデルを接続できます。両プロジェクトとも複数モデルに対応しており、全てのオープンソース RAG が同じモデルを使うという意味ではありません。

[OpenAI の公式ガイド](https://developers.openai.com/api/docs/guides/embeddings) に従い、対応する `cl100k_base` で本文の長さを計測します。tiktoken は Token 計数だけを行い、ローカルで Embedding モデルを推論しません。初回はエンコーディングデータのダウンロードにネットワーク接続が必要ですが、分割に API キーは不要です。API 呼び出しとベクトル保存は実装済みで、`OPENAI_API_KEY` を用いて接続します。実行方法と再開の仕組みは [ベクトル化](embedding.md) を参照してください。検索方法は [ベクトル検索](retrieval.md) を参照してください。

## 原文への追跡

`document_chunks` には `document_id`、文書内の 0 始まりの `chunk_index`、本文、`start_char`、`end_char` を保存します。この文字範囲は、全ページを改行 1 文字で結合した本文に対する位置です。ページ番号はチャンク本体に一つだけ保存する形式から、出典テーブルで複数記録する形式に変更しました。

```python
document_text = "\n".join(page_texts)
chunk.text == document_text[chunk.start_char : chunk.end_char]
```

`chunk_sources` はチャンクとページの対応を保存します。`start_char` / `end_char` はページ内の位置、`chunk_start_char` / `chunk_end_char` はチャンク内の位置です。いずれも Python のインデックスで、開始を含み、終了は含みません。

```python
page.text[source.start_char : source.end_char] == (
    chunk.text[source.chunk_start_char : source.chunk_end_char]
)
```

例えば、ページ 5 の「申請時点で」とページ 6 の「日本国内に居住する学生です。」が同じチャンクに入った場合、出典はページ 5 と 6 のそれぞれの文字範囲になります。参照結果の `page_numbers` は対象ページの一覧、`sources` は対応する文字範囲です。結合時に挿入した改行は原文の文字として引用しません。

保存前に原文の一致、最大 Token 数、位置の進行、空白以外の本文のカバー範囲を検証します。原文位置やページとの対応を検証できなければ保存を中止します。元の本文は変更しません。出典の複合外部キーで、チャンクとページが同じ文書に属することを確認します。タイトルと取得元は文書から参照し、解決できない字形の有無は `has_unmapped_characters` で返します。

## 再実行・設定変更・再解析

文書ごとに、分割アルゴリズム、ライブラリのバージョン、設定、解析器のバージョン、全ページ本文から署名を作ります。設定は `documents.chunking_config`、署名は `documents.chunking_signature` に保存します。今回の方針は `document-token-v3` として記録し、Token 単位、対象モデル、エンコーディング、tiktoken のバージョンも含めます。旧文字数設定からの再実行は署名が変わるため自動的に再生成されます。Token 計数の導入による DB スキーマ変更は不要です。

- 同じ署名では `duplicate` を返し、既存チャンクを再利用します。ID も変わりません。
- 設定または本文が変わると、その文書のチャンクと出典範囲を同一トランザクションで置き換えます。生成するチャンクの ID は変わります。
- 文書をロックしてから処理し、同じ文書の並行分割や PDF 再解析と競合する更新を直列化します。
- チャンクまたは出典範囲の保存が失敗すると、以前のチャンク、出典範囲、設定を保持します。
- PDF 再解析でページを置き換える際は、古いチャンクを削除して設定をリセットします。改めて `chunk-documents` を実行してください。
- 一括処理のトランザクション境界は文書単位です。途中で失敗した場合、それ以前に完了した文書は保持されます。エラーを修正して再実行すれば、同じ設定で完了済みの文書はスキップされます。

## 既存環境の更新

`0003_cross_page_chunks` のマイグレーションは、既存のページ内チャンクを削除せずに出典範囲を追加し、文書全体での文字位置とチャンク順序に変換します。分割設定はリセットされるため、続けて `chunk-documents` を実行して新しい方針で生成してください。元の PDF とページ本文は変更しません。

```bash
uv run alembic -c infra/alembic.ini upgrade head
uv run jp-doc-agent chunk-documents
```

ダウングレード時は、複数ページをカバーするチャンクを旧形式で表現できないため、派生データであるチャンク・出典範囲・分割設定をクリアします。文書とページ本文は保持され、対応するコードで再生成できます。

## 検証結果と現在の制約

既定設定で 5 文書・170 ページから 1,037 チャンクを生成し、うち 187 チャンクが複数ページをカバーします。出典範囲は 1,235 件、実測の最大長は 300 Token です。同じ処理を繰り返すと 5 件とも `duplicate` になり、件数は変わりません。全 170 ページの元の本文が更新前と一致することも確認しました。

テストでは、日本語の句点、改行、反復文、Unicode、長文、300 Token 上限、英語の Token 数と文字数の違い、特殊トークン風の本文、重複上限、ページをまたぐ文、白紙ページ、原文位置、出典ページでの絞り込み、設定変更、チャンクと出典の保存失敗、並行処理、再解析、CLI、旧データのマイグレーションを確認しています。

```bash
uv run pytest -q tests/test_chunking.py tests/test_migrations.py
```

ページをまたぐことはできますが、全ての文の完全性を保証するものではありません。Token 数上限や段落境界では分割されます。見出しの階層、ヘッダー・フッター、表の行・列関係、図の内容も復元していません。後続では、必要な周辺文脈を検索結果に補う方法を評価します。検索の改善は、分割結果の見た目やチャンク数だけで判断せず、証拠ページの取得率と回答結果で確認します。
