# 出典付きベクトル検索

日本語の質問を OpenAI の `text-embedding-3-small` でベクトル化し、保存済みのチャンクを pgvector の余弦距離で検索します。`retrieval/service.py` が検索範囲と出典を管理し、`embedding/encoder.py` が API の入力・応答を検証します。[Agent](agent.md) はこの検索を道具として使い、[回答と引用検証](answering.md) を実行します。

## 実行

`.env` に DB 設定と `OPENAI_API_KEY` が必要です。対象文書の取り込み・分割・[ベクトル化](embedding.md) を先に完了してください。

```bash
uv run jp-doc-agent documents
uv run jp-doc-agent embedding-status

# 全文書を検索（既定値は上位 5 件）
uv run jp-doc-agent search "2024年度第2四半期の売上収益はいくらですか？"

# 文書を限定し、上位 3 件を返す（1 は ID の例）
uv run jp-doc-agent search "対象者と申請条件は？" --document-id 1 --top-k 3
```

`top-k` は 1〜100 です。質問は空白以外を含む 1〜8,191 Token とし、`cl100k_base` で検証します。モデルの入力上限に余裕を取った制限で、チャンクの 300 Token 上限とは独立しています。長い質問を切り詰めたり分割したりはしません。質問の本文は接頭辞なしで API に送り、質問ベクトルは DB に保存しません。

## 検索の仕組み

1. 質問・件数・対象文書・検索可能なベクトルの存在を検証します。対象がない場合は API を呼びません。
2. チャンクと同じモデル・1,536 次元で質問をベクトル化します。API 待機中は DB 接続を保持しません。
3. 現在の設定 ID と本文 SHA-256 が一致するベクトルだけを対象に、余弦距離の小さい順に K 件を取得します。文書の指定も K 件の取得前に適用します。
4. 本文・文書情報・ページごとの出典を同じ `REPEATABLE READ` スナップショットから読み取り、出典範囲と原文の一致を検証します。

現時点では近似検索インデックスを使わない厳密検索です。同じ距離の場合は文書 ID・チャンク順序・チャンク ID で並べ、順位を再現可能にします。類似度は `1 - cosine_distance` で、回答の正しさを表す確率ではありません。関連資料がなくても近い K 件が返るため、根拠不足の判断は回答処理で行います。

仕様は [OpenAI の Embedding ガイド](https://developers.openai.com/api/docs/guides/embeddings)、[pgvector の余弦距離・厳密検索](https://github.com/pgvector/pgvector#querying)、[SQLAlchemy での距離検索](https://github.com/pgvector/pgvector-python#sqlalchemy) を参照してください。

## 出力と出典

出力は JSON です。

| フィールド | 内容 |
| --- | --- |
| `query` / `query_tokens` | 質問原文とローカルで計測した Token 数 |
| `model` / `dimensions` / `profile_id` | 質問とチャンクに共通する Embedding 設定 |
| `top_k` / `document_id` | 要求件数と文書の指定。指定なしは `null` |
| `coverage` | 対象範囲のチャンク数、検索可能数、未処理数、未分割文書数 |
| `api_tokens` | 正常に検証した API 応答の使用 Token 数 |
| `results` | 順位、距離・類似度、本文、chunk ID、文書 ID・名前・取得元 URL、全ページ番号・出典範囲 |

`results[].sources` は物理ページ番号 `page_number`、ページ内の `start_char` / `end_char`、チャンク内の `chunk_start_char` / `chunk_end_char` です。文字範囲は Python 文字列の 0 始まり・終端を含まないインデックスです。結合本文の位置は結果直下の `start_char` / `end_char` です。複数ページにまたがる結果では全出典を返し、ページ結合の改行だけは原文への対応を持ちません。

`has_unmapped_characters` は `(cid:番号)` のような未解析字形が本文に含まれるかを示します。ページ番号は 1 始まりの PDF 物理ページで、紙面に印字されたページ番号とは異なる場合があります。

未処理や旧ベクトルがある場合も検索可能な部分を返します。`coverage.pending` または `coverage.unchunked_documents` が 0 以外なら検索範囲が不完全です。`coverage` は順位・出典と同じスナップショットの件数です。`embedding-status` で確認し、未分割の文書は分割後にベクトル化してください。

## エラーと確認

不正な質問、件数、文書 ID、検索可能なベクトルがない場合は終了コード 1 です。API の安全なエラー表示・再試行はベクトル化と共通です。出典の欠落や原文との不一致は検索を中止します。API 呼び出し中に再分割され、検索可能なベクトルがなくなった場合も、ベクトル化の再実行を案内します。DB 取得中の再生成では、取得開始時点の本文と出典を返します。

```bash
uv run pytest -q tests/test_retrieval.py tests/test_embedding.py
uv run ruff check .
uv run ruff format --check .
uv run alembic -c infra/alembic.ini check
```

テストは HTTP を模擬し、実際の PostgreSQL の一時 schema で、余弦距離の順位、K 件制限、文書範囲、異なる設定と旧ベクトルの除外、ページをまたぐ出典、出典の欠落・原文不一致、検索中の再生成、秘密情報を含まない失敗表示を検証します。

## 実 API での動作確認と制約

ローカルの 5 文書・1,037 ベクトルを対象に、以下の少数例を確認しました。検索品質のベンチマークではありません。

- 「2024年度第2四半期の売上収益と調整後営業利益はいくらですか？」を決算概要の文書に限定すると、3 位で関連する連結 PL の物理ページ 31 を取得しました。1 位は短い見出し断片で、上位 1 件だけでは回答の根拠が不足します。
- その連結 PL の原文を質問入力にした確認では、元の chunk が 1 位になり、余弦距離は約 0.000003 でした。
- 「2021年1月1日時点の富士通の組織構成はどうなっていますか？」の全文書検索では、対象の組織資料を上位 5 件に取得できませんでした。PDF に対象日の本文はありますが、現在の本文のみのベクトル検索では、この質問の証拠取得に失敗しています。

検索結果の文字範囲は元のページと照合し、検索実行後も元の 170 ページ・1,037 チャンク・1,235 出典範囲・1,037 ベクトルが保持されていることを確認しました。少数の動作確認を検索精度の保証として扱いません。[Agent による問答](agent.md) と [少数の固定ケース](evaluation.md) を実装しました。今後は独立した問題の評価で検索範囲・文脈付与・ハイブリッド検索・再ランキングの改善効果を測ります。
