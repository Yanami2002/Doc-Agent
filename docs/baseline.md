# 正文ベースの基線評価 v1

## 評価対象と固定条件

`src/jp_doc_agent/resources/evaluation/baseline-v1.json` の 10 ケースを実行前に固定します。本文で回答可能な 8 ケースと根拠不足を期待する 2 ケースで、全件を同じ 5 文書に対する全文書検索（`top_k=5`）で実行します。ケースごとに対象文書を指定しません。検索・回答のプロンプト、モデル、チャンク分割設定は評価結果に合わせて変更しません。

既存の `questions.json` の 3 ケースは開発用として残します。今回の問題と許容証拠ページは重複させず、回答可能ケースの根拠文書は株主総会資料と統合レポートに限定しました。ただし検索対象の 5 文書は開発時と共通であり、未知の企業・文書への汎化を測る独立した文書分割テストではありません。今回の結果を確認した後は、この v1 も回帰確認用とし、次の改善の最終判定には別の未使用ケースが必要です。

モデルは `gpt-4.1-mini-2025-04-14`、出力上限は既定の 2,048 Token、Embedding は `text-embedding-3-small` の 1,536 次元です。検索最大 3 回・ページ取得最大 2 回・道具全体最大 6 回など、通常の Agent の上限をそのまま使用します。実際に使用した設定・モデル・文書 SHA-256・解析版・分割設定は JSON レポートに保存します。

## 問題の選定

上流の固定コミット `39fc62415ae5557f51d61ae64faf05494703afe5` の正解 YAML と、現在の抽出本文を照合しました。収録 5 文書が全証拠をカバーする 16 問のうち、図表の位置関係を復元せずに本文から答えられる要約・原因説明 2 問を採用します。

| 上流問題 ID | 判断 | 理由 |
| --- | --- | --- |
| 66、77 | 採用 | 正文の段落から原因説明・要約を構成できる |
| 64、67 | 除外 | 組織・役職の図表、時点の曖昧さがある |
| 65、68、70、71、73、74 | 除外 | 表の列・年度・注記の対応や複数資料の数値比較が必要 |
| 69、75、76、78、80、81 | 除外 | 図表の対応、計算、複数ページ・文書の視覚的な照合が必要 |

除外はモデルの回答結果を見る前に決めます。本文にも数値が書かれている新規問題 6 問を追加し、日付・企業名・金額と単位・理由・比較を確認します。これは上流問題の採点を置き換えるものではなく、プロジェクト独自の小規模評価です。すべてのケースに分類と原文照合時の確認観点を付け、根拠不足を含め期待値は評価関数内だけで使用します。

| ケース | 内容 | 許容する物理 PDF ページ |
| --- | --- | --- |
| `fujitsu-66` | FY2023 の営業減益の原因 | 株主総会 3 |
| `fujitsu-77` | Vertical areas の要約 | 統合レポート 19・36・39・40・41・42 |
| `authored-uvance-orders` | FY2023 の Uvance の売上・受注と伸長率 | 株主総会 5 |
| `authored-ubiquitous-profit` | ユビキタスの調整後営業利益と増益策 | 株主総会 6 |
| `authored-capital-investment` | 設備投資総額と部門別金額 | 株主総会 9 |
| `authored-gk-acquisition` | GK Software の連結子会社化の日付と買付主体 | 株主総会 9 |
| `authored-plan-themes` | 中期計画の 3 テーマ | 統合レポート 16・18 |
| `authored-value-pricing` | SI と Uvance の価格設定の違い | 統合レポート 41 |
| `unsupported-2035-dividend` | 未収録の将来年度の確定配当 | 根拠なし |
| `unsupported-individual-salary` | 未収録の従業員個人別給与一覧 | 根拠なし |

## 再実行

プロジェクト直下で `.env` を準備し、OpenAI API キーを設定します。既存の DB を利用する場合、PDF の再取得・再解析・Embedding の再生成は不要です。API の利用料金が発生します。

```bash
uv python install 3.12
uv sync --locked
# 初回のみ: cp .env.example .env

docker compose --project-directory . -f infra/compose.yaml up -d --wait
uv run alembic -c infra/alembic.ini upgrade head
uv run doc-agent check-db
uv run doc-agent fetch-benchmark
uv run doc-agent import-documents
uv run doc-agent chunk-documents
uv run doc-agent embed-chunks
uv run doc-agent embedding-status

uv run doc-agent evaluate-rag \
  --cases src/jp_doc_agent/resources/evaluation/baseline-v1.json \
  --output data/reports/baseline-v1.json
```

`--output` の既存ファイルは上書きされるため、比較時は保存先を分けます。原文・引用を含む完全なレポートは Git 管理外の `data/reports/` に保持し、公開文書には集計と失敗分類だけを記載します。

## 指標の読み方

- `automated_pass_rate`: ステータス、許容ページへの検索・引用の一致、全指定キーワードの確認を通過した件数 / 全 10 件。エラーも分母に含めます。意味的な正答率ではありません。
- `retrieval_evidence_hit_rate_all` と `citation_evidence_hit_rate_all`: 許容する証拠ページがある全 8 件のうち、少なくとも 1 ページと交差した割合。エラーは未達として扱います。
- 各ケースの `retrieval_evidence_recall` と `citation_evidence_recall`: 許容ページ集合に対するページ一致率。代替の根拠ページも含むため、全ページの一致を必須とはしません。主張を裏付ける範囲が足りるかは別途確認します。
- `insufficient_correct_rate`: 全 2 件の根拠不足ケースのうち、根拠不足を返した割合。
- `failed` と `failure_categories`: 正常に完了したが確認を通過しなかったケース。原因は `status_mismatch`、`retrieval_miss`、`citation_miss`、`answer_keyword_mismatch` を併記します。
- `errors`: API・引用検証などで完了しなかったケース。品質上の失敗と区別します。
- `retrieval_evidence_hit_rate`: 旧レポートとの互換用で、正常完了した証拠付きケースだけが分母です。新しい比較では `_all` を使用します。

`passed` は自動チェックの通過を示すだけです。数値・単位・年度の対応、回答の網羅性、引用が結論を支持するかは `review_points` に従って原文と照合します。キーワード一致だけで意味的な正しさを保証せず、独立した人手評価がなければ人手評価済みとは表記しません。

レポートの `case_set_sha256` は既定値を補完したケース配列をキー順で JSON 化した SHA-256 です。ケースの変更と単なる書式変更を区別できます。`evaluated_at` は UTC、本文中の日付は日本時間で記録します。時間と Token 数は検証できた応答の集計であり、再試行を含む請求額ではありません。正答率の一般化や統計的な有意差を主張できる規模ではありません。
