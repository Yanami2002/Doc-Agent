# 出典付き回答と引用検証

質問応答は [LangGraph の Agent](agent.md) が検索・根拠の判断・追加調査を管理します。`answering/` はその中で使う回答生成と引用検証を担当します。

- `generator.py`: 取得済みの本文を使い、日本語の回答草案を生成します。
- `schema.py`: 結論・引用・根拠不足の構造と必須条件を定義します。
- `citations.py`: 実 chunk ID と引用原文を照合し、文書情報・ページ・文字範囲を補います。
- `llm.py`: 回答と道具選択で共用する構造化出力 API クライアントです。

## 準備と実行

[ベクトル検索](retrieval.md) が実行できる環境と `.env` の `OPENAI_API_KEY` を使います。次の設定は省略すると既定値になります。

```dotenv
OPENAI_ANSWER_MODEL=gpt-4.1-mini-2025-04-14
OPENAI_ANSWER_MAX_OUTPUT_TOKENS=2048
```

モデルのスナップショットを指定し、返却結果に実際のモデル名と要求したモデル名を記録します。別モデルには Responses API と JSON Schema の構造化出力への対応が必要です。API エラー時に別モデルへ自動的に切り替えません。仕様は [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini)、[Responses API](https://developers.openai.com/api/docs/guides/text)、[構造化出力](https://developers.openai.com/api/docs/guides/structured-outputs) を参照してください。

```bash
# 文書 ID は documents で確認（1 は例）
uv run doc-agent ask "2024年度第2四半期の売上収益はいくらですか？" --document-id 1

# 全文書を対象に調査し、実行記録を保存
uv run doc-agent ask "2021年1月1日時点の富士通の組織構成はどうなっていますか？" \
  --top-k 5 --output data/reports/organization.json
```

`top-k` は 1〜20、既定値は 5 です。出力 Token 上限は 256〜8,192、既定値は 2,048 で、1 回のモデル呼び出しに適用します。初回の証拠で回答できる場合はそのまま終了し、不足する場合は上限内で追加調査します。指定文書の範囲と質問の年月・条件は維持します。

## 回答と引用の契約

回答生成 API には質問と取得済みの chunk ID・文書名・本文を送ります。原文ページを取得した場合は、全文を補助文脈として追加します。資料内の命令は参照データとして扱い、外部知識、年度・日付・単位・条件・図表の対応関係の推測で不足を補わないよう指示します。評価の正解・証拠ページ・期待キーワードは送信しません。

厳密な JSON Schema と Pydantic を使い、次の草案を受け取ります。

- `answered`: 1 件以上の結論があり、全ての結論に chunk ID と連続した引用原文がある。不足情報は空配列。
- `insufficient_evidence`: 結論と引用は空配列。不足している情報を日本語で示す。

引用した chunk が今回の取得済み証拠に含まれること、引用原文が chunk 本文に完全一致することを確認します。改行・空白の正規化や離れた原文の連結を許可しません。同じ chunk と同じ引用原文は一つの引用番号にまとめ、各結論に番号を付けます。草案に URL や引用番号を直接生成した場合は検証を失敗させます。

URL・文書名・ページ番号・文字範囲は取得済みの証拠から補います。引用がページをまたぐ場合は、交差する全ページの原文位置を返します。一つのページに収まる場合は、そのページだけを引用します。文字範囲は 0 始まり・終端を含まない Python 文字列の位置です。ページ結合の改行は原文ページに対応しません。

ローカルの決算概要を対象にした実 API 検証では、8,666 億円という回答と物理ページ 31 の原文引用を照合しました。少数例の動作確認であり、全体の回答品質を示すものではありません。

## 出力・保存・失敗

CLI は回答・引用・不足情報と調査の実行記録を JSON で返し、`data/reports/agent-<時刻>.json` に保存します。`--output` で保存先を指定できます。明示したファイルは上書きします。[Web API](web-api.md) は `request_id` に対応する実行記録を保存します。

| 項目 | 内容 |
| --- | --- |
| `question` / `status` / `answer` | 質問、日本語回答、回答可能・根拠不足・実行失敗 |
| `statements` / `citations` | 結論と検証済み引用番号、引用原文、chunk ID、文書、URL、ページ・文字範囲 |
| `missing_information` | 取得資料では確認できない情報 |
| `retrieval` | 検索設定、対象文書、追加調査を含む取得済み chunk ID・ページ |
| `trace` / `counts` / `limits` / `stop_reason` | 調査履歴、実行回数、上限、停止理由 |
| `model` / `requested_model` / `prompt_version` | 実行モデル、要求モデル、Agent の指示版 |
| `usage` / `elapsed_seconds` | Embedding・回答判断・道具選択の Token 数と合計時間 |

質問・回答は DB に保存せず、既存の文書・ページ・チャンク・ベクトルを更新しません。API 待機中は DB 接続を保持しません。回答生成中に再分割された場合は、取得した時点の本文と出典で引用を照合します。chunk ID が後から DB で参照できなくなる場合も、保存した引用原文と文字範囲は保持します。DB による回答履歴と文書版への永続的なリンクは未実装です。

API 接続のタイムアウトは 1 回 30 秒、SDK の再試行上限は 2 回です。応答の未完了・生成拒否・不正な JSON・引用不一致では未検証の回答を表示せず、失敗理由と実行済み履歴を保存します。CLI は終了コード 1、正常な回答・根拠不足は 0 を返します。API 本文・キー・不正な草案はエラーに表示しません。

`usage` は検証できた応答の Token 数です。SDK 内の再試行・応答形式の検証失敗に伴う使用量や課金総額は含みません。Responses API の `store=False` はアカウント全体のデータ保持条件を変更するものではありません。

## 確認と制約

```bash
uv run pytest -q tests/test_answering.py tests/test_agent.py tests/test_evaluation.py
uv run ruff check .
uv run ruff format --check .
uv run alembic -c infra/alembic.ini check
```

模擬 API と一時 DB schema で、草案の構造・全結論の引用・未知 chunk の拒否・原文の完全一致・ページをまたぐ位置・重複引用・根拠不足・API 失敗・並行更新・CLI とクライアント終了を検証します。

引用検証は参照先と原文の一致を確認します。原文が結論を意味的に裏付けるか、数値の解釈・年月・単位が正しいかは、この検証だけでは保証しません。OCR 未対応、表の読み順や字形の解析不良、検索漏れも回答に影響します。[固定ケース検証](evaluation.md) で成功と失敗を記録しています。
