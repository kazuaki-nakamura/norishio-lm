# Norishio authored toy corpus v1

Issue #3向けに新規作成した、小規模な条件付き生成・意味ラベル学習用の教材。
会話履歴、Project添付ファイル、個人情報、外部辞書からの抽出ではない。
例文・参照文・意味ラベルはすべてAI作成の `authored_demo`。人の検証は未実施。
HowNet義原、既存辞書の語義ID、普遍的意味論を実装したデータではない。

## 再生成と検査

リポジトリルートから実行する。生成器はPython 3.11+の標準ライブラリのみ。

```bash
python data/issue3/toy_corpus.py --check
python data/issue3/toy_corpus.py --out codex/work_output/norishio-toy-v1
python -m pytest -q -p no:cacheprovider tests/test_issue3_data.py
```

出力先が存在すれば上書きせず失敗する。再検査は `--check`、別のexportは別出力先を指定。
`seed.json` が例文の原稿、`toy_corpus.py` が決定的生成器。
`expected-manifest.json` と生成された `manifest.json` のSHA-256を照合できる。
展開済みJSONLは再生成可能な成果物であり、Gitには原稿・生成器・検査を保存する。

| split | 件数 | 用途 |
|---|---:|---|
| train | 450 | 学習。語彙fitや辞書adapterの構築に使えるのはこのsplitのみ |
| validation | 150 | 開発・モデル選択 |
| test | 150 | 設定固定後の最終評価。選択・学習へ戻さない |
| diagnostic | 24 | 文脈・語順・曖昧性・誤字形cueの点検。主指標とは別枠 |

主データは5人物語×5時点×10意味条件×3表現=750件。25状況を15/5/5に分ける。
同一人物語・時点の状況に属する全条件・全言い換えは同じsplitに固定。
各人物語・時点はtrainに現れるが、評価用の組合せはtrainには出ない。
これは**同じテンプレート群での未見組合せ**を試すsplitであり、未見テンプレートや
実社会の対話へ汎化することを証明しない。近似表現を排除した自然言語ベンチではない。
主データの入力全文は重複なし。参照文は同一split内の言い換えで共有するがsplitを跨がない。
診断セットには、同じ発言で文脈が異なる例、同一入力で補助cueだけを変えるペアがある。

## データ境界

各行の `inputs` は `context` と `text` のみ。`model_inputs()` がallowlistで取り出す。
`targets` は損失と採点用。`id/group_id/split/metadata` は管理用でありモデル入力にしない。
主データのtargetには次が入る。

- `text`: 意味の主要部分を明示した参照文。唯一の自然な言い換えという主張はしない。
- `sense`: ローカルな3語義 `meet_person / be_present / separate_people`。
  文脈語義選択の高度な評価ではなく、L_senseの接続確認用。
- `sememes`: `MEET/STAY/SEPARATE` と `HUMAN_INTERACTION`。この教材だけの粗い意味タグ。
- `concept`: event、operators、agent、participant、time、location、repeat_marked。

`operators` は外側から内側の順。
`[NOT,WANT]` は「望んでいるわけではない」、`[WANT,NOT]` は「しないでいたい」。
`[PLAN,NOT]` はしない予定。予定・可能性・願望を、実際の出来事の成立と同一にしない。
`agent` は出来事の主体。SELF=話し手、OTHER=participant欄の人物、SELF_AND_OTHER=両者。
WANTの主体はこの主データでは常にSELF。participantは同僚等の人物語で実在個人を指さない。
`repeat_marked=false` は反復が明示されないという意味で、初対面だと断言していない。
`UNSPECIFIED` は発話に指定がない値。注釈欠損の `null` とは区別する。
全ラベルは簡略化した教材仕様。自然言語の全意味、発話意図、実際の心情を網羅しない。

診断の補助ラベル3種はnull。自由記述のinterpretationを主データの固定headへ押し込まない。
診断には未学習の事象・表現が含まれ、失敗から一般的能力の有無を断定しない。
『心生』は文脈で定義する架空語。subcharactersの金/口は**意図的な誤情報fixture**であり
字源の説明ではない。metadata.perturbationは頑健性試験時のadapter上書き指示だけに使い、
通常の入力・辞書・教師ラベルへ混入しない。cue無効時の不変性と有効時の変化を別々に測る。

## Issue #3の実装契約

まずは入力全文を与えて参照文を生成する**条件付き**課題として実装する。
無条件の日本語LMや、まだ読んでいない入力を予測する課題とは区別して報告する。
`teacher_forcing()` は固定UTF-8 byte語彙を提供する: PAD=0,BOS=1,EOS=2,SEP=3,byte+4。
語彙サイズ260。外部tokenizer不要。ソースが全文入力済みである条件を明示する。

- A: ソースprefixを読めるtiny causal LM。B/Cと同一の観測ソース・分割・学習予算で比較。
- B: ソースからのみ作るmulti-channel表現で条件付けたdecoder。
- C: 同じencoderから予測した明示conceptを経由するdecoder。
  gold conceptを与えるoracleは別実験。Cの通常評価に混ぜない。
- 全層encoderは **source_ids / model_inputsだけ** を読む。参照文を含むfull input_idsを
  encoderへ丸ごと入れない。辞書候補は観測済みソースからのみ引く。
- teacher forcingでは過去の参照トークンのみ利用。labelsは1トークンずらし、
  prefixとPADはloss=-100で無視。完全参照文を全位置に見せない。
- conceptの教師信号はソース末尾位置へ付ける。ソースを読む途中へ全文のgoldを配らない。
- Cにsurface latentの隠れたbypassを入れない。通常の自己回帰履歴は許す。
  hard/soft concept、残存経路、介入時の挙動を明記する。auxiliary lossだけで
  bottleneckの因果的利用を証明したと言わない。
- `supervised_targets(row, drop=...)` は注釈をnullにする。欠損を0/負例へ変換しない。
  全aux欠損でもLM損失のみでforward/backwardできること。
- testを見てテンプレートやラベルを都合よく変更しない。修正が必要なら版と評価を更新する。

## Codexへの引き継ぎ

データ準備ブランチのcommitを、最新masterから作るIssue #3作業ブランチへcherry-pickする。
既存作業があれば確認して再利用し、強制pushや自動mergeはしない。
データはまずこの版を利用し、恣意的に作り替えてCを勝たせない。
CPUで小さい学習を実行し、A/B/Cのtrain/validation/test損失、対象数、パラメータ数、
seed、step数、時間を記録する。最初はseed=7、比較は同条件を優先。
L_sense/L_sememe/L_conceptと各チャネルのablation、prefix因果性、gold差替え不変性を検査。
concept介入・無効化による出力変化も確認し、出力とラベルの単なる相関と区別する。
既存全スイート・両デモ・knowledge/OKF/MCPを実行してdocs/handoff.mdと関連OKFを更新する。
このデータ提供ではモデル実装、学習、masterへのmergeはしていない。

## 検証記録と限界

作成時: Linux / Python 3.13.5 / pytest 9.0.2。
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider tests/test_issue3_data.py`
は20 passed。既存repo全体のpytestは未実行。Git直接接続はDNS失敗のため、
隔離した新規データファイルだけを検査し、共有はGitHub connectorを使用する。
サブエージェントへの委譲なし。生成器の再現性、データ境界、split等の機械検査であり、
言語学的正しさの独立査読、LLMの性能、学習済みモデルの因果性は未検証。
生成と採点用ラベルの作者が同じなので、AI作成者の偏りが両方にある。
後続では別作者のデータや人手確認を加える。私的ログを無断で補充しない。
