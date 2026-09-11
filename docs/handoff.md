# Norishio-LM 引き継ぎ記録

検証日: 2026-09-11

## 最新の実装状態: Issue #7

`codex/issue-7`、実装コミット `67ebd169c1606e934fb77ed5f09abbdcf3b52aa4`。
PR #8マージ後のmaster `2e0f05a30718cd029c0a9a7e862578b756b83340` から開始。
作業ディレクトリは既存の `codex/work_output/issue-worker/issue-3` を再利用したが、
ブランチと追加範囲はIssue #7。過去の教材・結果は保持した。

### 実装と比較条件

`concept_metrics` はクラス別正答/対象数、項目accuracy/balanced accuracy、
項目balancedのmacro平均、全7項目一致、欠損/未見の別maskと分母を返す。
多数派はtrainだけで決定し、同率は最小ID。operatorsは順序付き別クラス。
`toy_evaluation` は通常C/定数Cの同初期値・同学習順対照と6介入を集約。
`toy_checkpoint` はfloat32 CPUモデル、語彙ID、byte仕様、教材版を保存・復元する。
weights-only読込、語彙metadata整合性検査、既存出力拒否、競合時の上書き防止を実装。
保存物はignored出力。SHAは整合性確認であり署名・発行者認証ではない。

固定教材v1: train450/validation150/test150/diagnostic24。今回test未評価。
seed7、60更新、batch16、CPU1thread、Adam .003、clip1、全4損失の重み1。
両モデルの初期state SHA256:
`f07d83adec88a37040886e1374800b17b56b7d2f3f49cea3f1b801c4b61b4b70`。
共通抽出順SHA256:
`0f9889521f50650b1fe152f20045b49871ddc3650a63fc13e2e71e9572b3fb51`。
定数は初期モデルのtrain-source予測平均をdetachして固定し、補助損失学習は両方で継続。

### validationの観測

| 概念採点 | micro正答/対象数 | 項目balancedのmacro平均 | 全7項目一致 |
|---|---:|---:|---:|
| train多数派 | 570/1050 | 0.319047619 | 0/150 |
| 通常C | 571/1050 | 0.320000000 | 0/150 |
| 定数C側のencoder予測 | 572/1050 | 0.320952381 | 0/150 |

定数Cの最後の行は補助学習されたencoderの採点であり、decoderへは固定ベクトルを渡す。
全条件のevent/operators/agent/time/location/repeat_marked accuracyは順に
0.6/0.3/0.8/0.2/0.8/0.9。participantは多数派30/150、通常31/150、定数側32/150。
今回は全項目に既知goldがあり、欠損/未見はともに0。mask境界は別の単体テストで確認。
operatorsは全3採点とも NOT→WANT:0/15、PLAN→NOT:0/15、PLAN:0/30、
POSSIBLE:0/15、WANT→NOT:0/30、WANT:45/45。否定作用域の識別は実証できていない。

| decoder条件 | validation LM（10005 token） |
|---|---:|
| soft | 2.088276588 |
| hard | 2.088078150 |
| zero | 2.131895389 |
| 最終モデルのtrain-source平均 | 2.088286604 |
| seed17対応置換 | 2.088275530 |
| 定数条件で再学習 | 2.088902805 |

seed17置換は固定点1、移動149、比較可能な完全既知フレーム150組中、意味が異なる148組。
項目差分ペア数はevent90/operators128/agent54/participant122/time122/location52/repeat28。
soft/hard/zero/平均/置換は通常モデルへの推論介入、定数条件だけは再学習対照。
zeroは確率の正規化も壊すため、その損失増加だけを意味利用の根拠にしない。
平均/置換/定数再学習の損失は近く、意味内容利用の有効性は未実証のまま。

全6条件の自由生成は各150件でEOS終了0、有効UTF-8 0、完全一致0、unique sequence1、
特殊token行0。全件cap128終了。source/予測概念/介入値/生成raw ID/停止理由/参照を
別フィールドで保存。参照は生成器へ渡さない。完全一致は教材参照との一致だけを測る。

両モデル登録パラメータ42423、初期/最終で値が変わった要素29552。
これは正味の変化で、途中の更新回数や一度変わって戻った要素は数えない。
学習秒は通常5.471、定数4.266（各1回、速度優劣の根拠にはしない）。
target履歴長は両方49–76、train/validation平均66.7、source-prefixなし。
共通960抽出例の平均66.8、有効損失token64128。validation有効token10005。

### 保存・復元と検証

両checkpointでvalidation150件のconcept/logits/greedy/stateが再読込前後で完全一致。
さらに別プロセスの`--restore`で学習・語彙再fitなしに採点と全生成例を再現し、
元レポートの該当条件とのJSON構造比較で完全一致を確認した。

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_evaluation --out-dir codex/work_output/issue7-seed7-v1
.\.venv\Scripts\python.exe -m norishio_lm.toy_evaluation --restore codex/work_output/issue7-seed7-v1/predicted.pt --out-dir codex/work_output/issue7-replay-predicted-v1
.\.venv\Scripts\python.exe -m norishio_lm.toy_evaluation --restore codex/work_output/issue7-seed7-v1/constant.pt --out-dir codex/work_output/issue7-replay-constant-v1
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest tests/test_toy_checkpoint.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m norishio_lm.demo
.\.venv\Scripts\python.exe -m norishio_lm.encoder_demo
.\.venv\Scripts\python.exe codex/tools/build_context.py
.\.venv\Scripts\python.exe codex/tools/validate_okf.py
.\.venv\Scripts\python.exe codex/tools/check_knowledge.py --mode full
.\.venv\Scripts\python.exe codex/okf_mcp/tests/live_check.py --server codex/okf_mcp/server.py --root okf
```

全pytest: **220 passed, 2 subtests passed**、skipなし、11.97秒。
既存の空tensor初期化warning1。Temp fixture統一後のcheckpoint9件も再実行成功。
両デモと評価/復元CLIは終了0。CLIにはNumPy未導入warningがあるがNumPy変換は使わない。
checkpointテスト初回のTemp権限失敗は成功に数えず、許可された環境で再実行した。
索引再構築51 sources、OKF 8 files/6 conceptsでerrors0/warnings0、knowledge full healthy。
MCP実プロセス検査は6 tools、ok true。source追加中のquick検査はinventory差分を報告し、
原本確認・文書更新後に再構築して解消した。

レポート `codex/work_output/issue7-seed7-v1/report.json` SHA256:
`be96e104d69dd6cdd4b03bcc1ce32dc675b12a87c7bf9a45d6755198a0097971`。
predicted.pt SHA256: `7f111ff8458a01c479ac3d7fdd0aa57e089af61b908f60ee9675b6b4657e3ab3`。
constant.pt SHA256: `5fa36ffd0578dab84fc365e3bdddb50c9b0e74cd7193e752179b3fdf84e42968`。

Luna (`gpt-5.6-luna`) 2担当に採点とcheckpointを独立委譲。親がJSON非対応、語彙metadataの
hash範囲、保存型/競合処理、検査不足を確認して修正・追加検証した。採点担当の初期cwd
誤認は指定worktreeへ直し、rootへの誤配置を除去後rootがcleanであることを確認。
今後の改善候補: 委譲開始時にcwd・Python実体・torch可用性をコマンドで必ず照合する。

残課題は入力依存の生成、否定作用域、多数派を超える概念識別、独立seedの再現性。
次候補は予算と評価基準を先に固定した入力対応学習の診断。v1教材やtestを結果に合わせて
改変しない。この作業はレビューPRまでで、自動merge/Issue close/worker再起動は行わない。

## 最新の実装状態: Issue #3

### 2026-09-11: byte崩壊・EOS未生成の切り分け

追補ブランチ `codex/byte-eos-diagnosis`。PR #6は別途`1e0a6c7`でマージ済み
（master `6e521daf9cf03f4a9a090e8cef96306381e25ea6`）。以下の診断追加はその後の
別PR対象で、PR #6のマージ内容には含まれない。

診断実装 `894f3ebb022155eb014fb6a02b766e43168d9d45`。
新しい`toy_diagnosis`コマンドで、seed7、batch16、cap128、更新回数60/600を
実行前に固定。同一初期stateと同じ抽出順のprefixを使用し、通常C/定数Cを各予算で
初期化から学習した。モデル、教材、v1/controlsハーネス、元の60step既定値は変更なし。
train/validationだけを使用し、testの再評価・調整・モデル選択はしていない。

Luna (`gpt-5.6-luna`) に生成実装の独立監査を委譲。128stepのincrementalとfull-history
比較はlogit最大差1.1920929e-07、argmax相違0との報告。EOSの1token先教師、PAD除外、
byte±4の往復にも具体的不具合なし。親は診断実装、全4条件の最初のvalidation例の
full-history/実生成列一致、EOS位置別集計とUTF-8遷移のテスト、最終実行を確認した。

| 条件 | validation LM | 正解履歴のbyte accuracy | 正解末尾のEOS accuracy | 自由生成: 有効UTF-8 / EOS終了 | 完全一致 |
|---|---:|---:|---:|---:|---:|
| 通常C 60step | 2.088276588 | 0.480974 | 0 | 0/150 / 0/150 | 0/150 |
| 定数C 60step | 2.088902805 | 0.480974 | 0 | 0/150 / 0/150 | 0/150 |
| 通常C 600step | 0.179737919 | 0.950989 | 1 | 150/150 / 150/150 | 0/150 |
| 定数C 600step | 0.190636492 | 0.949467 | 1 | 150/150 / 150/150 | 0/150 |

各条件のvalidationは9,855 byte教師と150 EOS教師。60step時は全150件で
不正UTF-8への遷移が生じ、単なるcap末尾の文字切れではなかった。先頭例は
`E7 A7 81 E3 81 E3 ...`（token ID 235,171,133,231,133,231...）で、
最初の「私」は正しいが、6番目のbyteが継続byteであるべき位置にE3を再出力する。
通常Cの正解末尾EOS確率平均は0.101001、自由生成先頭例の最大EOS確率は0.012390。
つまり正解の過去を与えてもEOSはargmaxにならず、自己出力の履歴ではさらに低い。

600stepでは通常Cの正解末尾EOS確率平均0.985960、先頭自由生成例の最大0.988273。
字形やUTF-8制約を追加せず、学習更新を増やすだけでbyte整合性と終了は回復した。
この条件では60stepの学習量不足が生成崩壊に寄与したという説明を支持する。
全般的な実装無欠陥や、任意のデータ・seedで十分な学習量を保証するものではない。

ただし600stepの両条件とも全150件に同じ
「私は、来月先輩と会うことを望んでいる。」を出力し、入力内容への対応は改善したと
言えない。各条件のunique sequenceは1、完全一致0。文字として成立し終了することと、
入力の人物・時点・作用域を正しく生成することは別の課題として残る。

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_diagnosis --out codex/work_output/issue3-byte-eos-diagnosis.json
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

診断4条件は各1回、CPU1thread。学習秒は通常60=4.431、定数60=4.449、
通常600=44.530、定数600=44.139。従前60stepの結果を再現し、600stepの反復や
複数seedは未実施。レポートSHA256
`a399d6af7fd0fb8073a579e1d60bed711c26226c417caa9c7d55a00e6c5abb8e`。
生token、参照と生成例、EOS確率・順位、最初の不正遷移はignored JSONへ保存。
全検査 **200 passed, 2 subtests passed**、skipなし、10.08秒、既知のzero-element警告1件。
NumPy未導入警告は診断実行時にも継続。今回の追加は診断で、生成モデルの修正版ではない。
既存両デモ終了0、原本45件のknowledge full healthy、OKF errors/warnings 0、
MCP live check ok=true/6tools/stale0、Git空白検査成功。ローカル検証として記録する。

次は人物・時点・作用域のどこで入力依存情報が失われるかを、encoder→concept→decoderの
各境界で測る。600stepを性能保証の既定値に昇格せず、以後の予算・seed・評価を先に固定する。

### 2026-09-11: 対照実験と自由生成のローカル追試

追加実装 `e6d037141917f55b3b8b7d4934df07d61219feb9`。
`toy_controls`は既存v1ハーネスと別のコマンド。モデル・教材・v1結果を変更せず、
trainとvalidationだけを使い、testの再評価・選択・調整は行わない。

通常Cと定数条件Cを同一初期stateのdeepcopyから学習。初期state SHA256は両方
`f07d83adec88a37040886e1374800b17b56b7d2f3f49cea3f1b801c4b61b4b70`。
全パラメータ42,423、seed7、60step、batch16、同一復元抽出順960行、Adam .003、
clip1、CPU1thread。定数は**初期モデルのtrainソース予測の平均をdetachして固定**。
gold由来ではなく、学習中に平均を更新しない。両条件とも補助損失は有効。
定数条件のLM勾配はencoderへ戻らないが、auxiliary勾配は戻る。
共通のglobal gradient clippingを含む学習処方であり、両条件の勾配ノルムを同一にはしない。

学習後、通常Cにtrain予測平均を与える対照と、validation全体のconcept対応を反転する
対照も実施。これは定数条件の再学習とは別の推論時介入。
validation150行、LM対象10,005 byte/EOS、概念項目1,050判断。

| 条件 | validation LM | 通常Cとの差 |
|---|---:|---:|
| 通常C | 2.088276588 | 0 |
| 学習後のtrain予測平均を固定 | 2.088286604 | +0.000010015 |
| validation対応を全体で反転 | 2.088276439 | -0.000000149 |
| 初期train予測平均の定数条件で再学習 | 2.088902805 | +0.000626217 |

train多数派概念基準は570/1,050=0.542857143、通常Cは571/1,050=0.543809524。
レビューの多数派・平均・反転結果をこのWindows環境でも再現した。定数で再学習しても
差は小さく、今回の設定で入力別概念内容の有効利用は示せていない。
学習時間は通常4.931秒、定数4.477秒（同一環境での観測、速度優位性の主張なし）。

自由生成は正解文を渡さず、予測conceptからGRU状態を作り、BOSの後は自身の出力のみを
入力。argmax、最大128生成token、EOSで行ごとに停止。PAD/BOS/SEPを後処理で禁止せず
出現を数える。厳密UTF-8復号に失敗した表示だけreplacement文字とし、元token列を保持。
正解参照は生成が完了してからbyte+EOS完全一致の採点にだけ使う。

4条件ともvalidation150件で、完全一致0%、EOS終了0%、有効UTF-8 0%、
各条件内の生成列は1種類、全件128token上限到達。不適切な特殊token出力は0件。
この結果は正常な文章生成の成功ではない。teacher forcing中の低損失と自由生成の失敗を
分けて報告し、得点改善のために教材・上限・採点規則を変更していない。

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_controls --seed 7 --steps 60 --max-new-tokens 128 --out codex/work_output/issue3-controls-seed7.json
.\.venv\Scripts\python.exe -m norishio_lm.toy_controls --seed 7 --steps 60 --max-new-tokens 128 --out codex/work_output/issue3-controls-seed7-repeat.json
```

時間項目を除く全metrics、初期state、trace、生成token列が2回で完全一致。
レポートSHA256 `19682dd80aa6ecbd3936281646773e8ad400bb97766dc22067ffa4a19bef125c`。
生成例600件を含むJSONはignored work_outputに保持し、重みや生成物はGitへ追加しない。
PyTorch 2.14.0+cpu、Python3.11.15。今回もNumPy未導入警告があるが変換は未使用。

Luna (`gpt-5.6-luna`) は自由生成と独立対照テストを担当。親は対照ハーネス、
指標・入力境界レビュー、UTF-8/完全一致回帰補強、統合実行を担当。
検証件数は親の最終実行結果を採用する。
`python -m pytest -q -p no:cacheprovider`は **194 passed, 2 subtests passed**、
skipなし、9.73秒（全aux欠損ケースのzero-element警告1件）。既存両デモ終了0。
`build_context.py`原本43件、`validate_okf.py` errors/warnings 0、knowledge full healthy、
MCP live check ok=true/6tools/stale0、`git diff --check`成功。
これらはローカル検証で、GitHub Actionsや人による研究内容検証ではない。

次の焦点は、まず自由生成のbyte列崩壊とEOS未生成の原因分析、その後に複数seed・
学習量を事前固定した比較、語順/作用域encoderの検討。小さい差や失敗だけで概念層の
一般的可能性を断定しない。現在のvalidationは既に診断に使ったことを明記して扱う。

### 初回実装とレビューの履歴

ブランチ `codex/issue-3`、基点 `bf0a0b94f8026b68f37b1f53833085e8ecdc04f5`。
提供教材 `e080504082946dbcc584b10f638eb066785db9b1` を
`6530153` としてcherry-pick。実装・検査のコミットは
`498035f13c553a5aaea24ba411d525a906e4a071`。以下のIssue #2節は履歴。

実装済み: train-only概念語彙、7つの名前付き概念head、語義/sememe補助head、
masked個別損失、GRU A/B/C、source-only adapter、CPU学習・比較ハーネス。
C decoderは過去targetと予測concept確率だけを受け取り、sourceやencoder latent、
語義/sememe予測の抜け道を持たない。概念はsoft確率であり離散意味の証明ではない。
詳細は [契約と限界](concept-toy.md)。教材・分割・採点規則は変更していない。

Luna (`gpt-5.6-luna`) へモデル/損失と独立検査、入力adapter/入力漏洩レビューを委譲。
親は実験ハーネス、統合レビュー、厳密C境界の修正指示、未来target/誤字形の回帰追加、
最終実行、文書とGitHubを担当。サブエージェントの成功報告だけでは完了扱いにしていない。

### 検証環境と実行

Windows、Python 3.11.15、PyTorch 2.14.0+cpu、pytest 9.1.1。
専用worktreeの`.venv`を使用。GPU/外部辞書/大規模学習/OKF教材投入はなし。
最初のvenv作成はサンドボックスのensurepipで失敗し、許可付き実行で復旧した。
途中のサブテストはtorch未導入のskipとWindows Temp権限失敗があったが、
依存導入と許可付き全検証で解消。初回失敗を成功に数えていない。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install 'torch>=2.5,<3' --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e '.[dev,model]' --disable-pip-version-check
.\.venv\Scripts\python.exe data/issue3/toy_corpus.py --check
.\.venv\Scripts\python.exe data/issue3/toy_corpus.py --out codex/work_output/norishio-toy-v1
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m norishio_lm.demo
.\.venv\Scripts\python.exe -m norishio_lm.encoder_demo
.\.venv\Scripts\python.exe -m norishio_lm.toy_experiment --seed 7 --steps 60 --out codex/work_output/issue3-seed7.json
```

環境作成の先頭`python`はWindowsAppsエイリアスではなく既存プロジェクトの実Pythonを使う。
全pytest **179 passed, 2 subtests passed**、skipなし、7.44秒。両既存デモ終了0。
全注釈欠損テストのzero-element tensor警告1件。デモ/学習時のNumPy未導入警告は
既存環境と同じで、NumPy変換は使わない。無警告と偽らない。
教材のmanifest JSON内容と4 JSONLのSHA256はexpected-manifestと一致。
manifest自体の生バイトhashはWindows改行差があるためJSON内容で比較した。

### seed=7、60 step、batch=16、CPU 1 threadの観測

Adam lr=0.003、gradient clip=1、hidden=32。各条件960行を復元抽出で観測。
全条件で同じseed・学習行順・step数。パラメータ数とsource-prefix処理量は異なる。
train=450、validation=150、test=150、診断=24。testは全条件の学習後に評価し、
選択・調整へ戻していない。LMは有効target byte+EOSあたり自然対数のcross entropy。
train 30,015 token、validation/testは各10,005 token。同じ参照を全経路で採点。

| 条件 | train LM | validation LM | test LM | test概念項目accuracy | 登録parameter数 | 学習秒 |
|---|---:|---:|---:|---:|---:|---:|
| A source-prefix GRU | 2.187229 | 2.194592 | 2.191837 | 対象headなし | 24,292 | 5.258 |
| B encoder bypass | 2.136073 | 2.135584 | 2.136616 | 0.550476 | 42,423 | 6.233 |
| C strict concept | 2.086730 | 2.088277 | 2.087212 | 0.551429 | 42,423 | 4.683 |
| C LM無効 | 5.530803 | 5.531502 | 5.530769 | 0.545714 | 42,423 | 4.689 |
| C sense無効 | 2.101513 | 2.102988 | 2.102009 | 0.544762 | 42,423 | 4.491 |
| C sememe無効 | 2.089069 | 2.090601 | 2.089551 | 0.549524 | 42,423 | 4.606 |
| C concept loss無効 | 2.089810 | 2.091323 | 2.090292 | 0.414286 | 42,423 | 4.621 |
| C LMのみ | 2.101546 | 2.103019 | 2.102020 | 0.414286 | 42,423 | 4.492 |

上記loss ablationは初期化から再学習。Cの各lossはtestでLM=2.087212、
sense=0.865284、sememe=0.419663、concept=1.044530。補助注釈は各150行、
conceptは7項目計1,050件。項目別accuracyはevent .60、operators .30、agent .80、
participant .26、time .20、location .80、repeat_marked .90。多数派の強い項目を含み、
約55%という平均だけで文の意味を理解したと結論しない。

チャネル除去は**学習済みCへのvalidation時介入**で再学習ではない。
LMはsurface除去2.088261、tokens除去2.088428、characters除去2.088268。
他7チャネルは空なので全て元と同じ2.088277。空層の有効性の反証にはならない。
tokens/charactersは同じ観測文字列の別embeddingであり形態素解析ではない。
surfaceの全文カテゴリはheld-outでUNKになる。語彙はtrainのみから452/77/77、
残り7層はPAD/UNKの2項目で構成する。

Cの概念ゼロ介入でvalidation LM=2.131895、hard argmax介入で2.088078。
診断例1件のlogit最大差はゼロ介入2.601220、hard介入0.055858。
予測concept固定・source差替えは差0、意図的誤字形fixture有効時0.009778、
subcharacters無効時0。fixtureは未知IDへの入力で、実字源を学習した証拠ではない。
診断24件のLM=2.960399、1,362 token。補助ラベルは全nullでaccuracyは採点しない。
解釈自由文や正解ラベルを通常のモデル入力に使っていない。

詳細ログ・名前付き概念dumpはignoredの`codex/work_output/issue3-seed7.json`。
SHA256 `ce939b6d0f7983a26dba8679f3b16aeec2464f17be4e6fc775c6f65a9aea77c6`。
学習済み重みは保存・公開していない。再現は原稿・コード・seed・設定から行う。
同コマンドの出力先を`issue3-seed7-repeat.json`に変えて全8条件を再実行し、
時間項目を除く全metrics・trace・概念dumpが完全一致した。同一環境内の再現検査であり、
異なるOS/PyTorch版での一致は未確認。

知識検査は以下全て終了0。index原本39件、OKF 8files/6concepts・error/warning 0、
full check healthy、MCP live check ok=true・6tools・stale/obsolete 0。
Git diffの空白検査も終了0。GitHub Actions/人による研究検証の成功とは区別する。

```powershell
.\.venv\Scripts\python.exe codex/tools/build_context.py
.\.venv\Scripts\python.exe codex/tools/validate_okf.py
.\.venv\Scripts\python.exe codex/tools/check_knowledge.py --mode full
.\.venv\Scripts\python.exe codex/okf_mcp/tests/live_check.py --server codex/okf_mcp/server.py --root okf
git diff --check
```

### 2026-09-11: 外部AIレビューの追加診断

[PR #6レビュー](https://github.com/kazuaki-nakamura/norishio-lm/pull/6#issuecomment-5621273041)
は `d46abe8c8fec73fc495612425226d839727e7a41` を対象に、実装上のマージ阻害問題なしと報告。
以下はレビュー側の実行報告であり、この追記時にCodexが再実行した結果ではない。
20ファイルのGit blob SHAを照合した復元コピーをLinux/Python 3.13.5/PyTorch 2.10.0+cpuで
検査し、上流の一部63件と独立18件、計81件成功。全リポジトリ検査とは区別する。
8条件学習も完走し、A/B/Cの3 split計9値は既報と絶対差1e-6以内だった。

追加対照はvalidationのみ。trainで決めた各概念項目の多数派を固定するとaccuracyは
0.542857143、Cは0.543809524。1,050判断中の差は1件で、約54%の平均だけでは
入力依存の概念理解を示せない。

| Cのvalidation条件付け | LM loss | 通常との差 |
|---|---:|---:|
| 入力ごとの予測concept | 2.088276638 | 0 |
| trainソースから予測したconcept確率の固定平均 | 2.088286629 | +0.000009991 |
| validation内のconcept対応を反転 | 2.088276501 | -0.000000137 |

固定平均はgoldではなくtrain予測から作ったもの。内容の対応を消しても損失はほぼ同じ。
zero介入は確率和も変えるため、zeroで出力が変わることだけでは意味内容の使い分けを
立証しない。参照履歴・共通形式への依存という仮説と整合するが、単一短期学習から
概念層の将来性を否定もしない。これは事後診断であり事前登録した性能検定ではない。

次段階の優先対照: train多数派基準、同じdecoder初期値・予算で再学習した定数条件C、
入力とconceptの対応置換、自由生成。v1教材・既報値を保持して別実験として追加し、
testを調整に使わない。外部AIレビューは人による検証やGitHub APPROVEではない。

今回の対応は資料追記のみのため親が直接実施し、Lunaへの新規委譲は行っていない。

### 未解決の研究課題

単一seed、短時間、同テンプレートの未見組合せ、作者共通の例文/採点ラベル。
Cが低損失でもconcept層の一般的優位性やLLM性能を示さない。soft確率の余剰情報、
順序を失うencoder、入力semantic層の欠損、自由生成品質は未解決。
次候補は複数seedと多数派ラベルbaseline、語順/作用域を扱うsource encoder、
hard概念の学習比較、別作者/未見テンプレート評価、greedy生成の品質点検。
testを調整に流用せず、評価設計を別版として固定して進める。

## 以前の実装状態: Issue #2

### PR #5 出典レビュー対応（2026-09-10）

層全体に出典を指定し候補には指定しない入力で、senses/sememes/conceptsの出典が
tensor化後に失われる問題を再現した。候補出典をunknownで正規化するとgetのfallbackが
使われないことが原因。`ChannelBatch.layer_provenance` に全行の層出典を保持し、
候補固有の `Feature.provenance` と分けた。候補unknownを既知出典へ昇格しない。
特徴が空の行や `.to(device)` 後も層出典を保持する。既存の語彙checkpoint形式は変更なし。

Luna (`gpt-5.6-luna`) が独立回帰テストを作成。修正前の初版検査は15 failed / 1 passed。
親が実装修正とテストレビューを担当し、省略出典の正規化、異なるsource/revision、
3チャネル、空行、層除外、device移動、元入力の変更、出力不変性を補強した。
修正後の回帰検査は20 passed。統合全スイートは **136 passed, 2 subtests passed**、skipなし。
実行は下記と同じ専用venvで `python -m pytest -q -p no:cacheprovider`（Windows Temp用許可付き）。
両デモ終了コード0。encoderのNumPy未導入警告は継続するが、NumPy変換は使用していない。
この節より下の116件・24原本は初回実装時の履歴。

`codex/multichannel-encoder`、基点 `dfddf315168e81e11dde52007e5267ba5de6413e`。
以下の Issue #1 / 初期引き継ぎ節は履歴。この節と [encoder契約](multichannel-encoder.md) を現在の実装範囲として優先する。

実装済み: 10チャネルの独立した固定語彙・tensorizer、特徴ごとの原本パスと出典・候補対応、
PAD/UNKの分離、入力文字列と予約IDの衝突防止、語彙checkpointの往復。
独立Embedding/mean pooling/projectionと入力依存のscalar gateで融合するCPU encoderを追加した。
各チャネルの無効化、欠損時のゼロ寄与、全無効時の有限ゼロ出力、勾配分離を検証する。
字形・字源から語義ラベルを生成しない。語義IDを外した比較にsememe/concept値経由で同IDを混入しない。

隔離worktreeの専用venvへCPU版PyTorchをインストールし、editable import先を確認。
Python 3.11.15 / PyTorch 2.14.0+cpu / pytest 9.1.1。

```powershell
.\.venv\Scripts\python.exe -m pip install 'torch>=2.5,<3' --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e '.[dev,model]' --disable-pip-version-check
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m norishio_lm.demo
.\.venv\Scripts\python.exe -m norishio_lm.encoder_demo
.\.venv\Scripts\python.exe codex/tools/build_context.py
.\.venv\Scripts\python.exe codex/tools/validate_okf.py
.\.venv\Scripts\python.exe codex/tools/check_knowledge.py --mode full
.\.venv\Scripts\python.exe codex/okf_mcp/tests/live_check.py --server codex/okf_mcp/server.py --root okf
```

全テスト **116 passed, 2 subtests passed**（モデル検査のskipなし）。既存のWindows Temp制限のため
全テストは許可付きで実行。CPU forward/backwardのネットワーク禁止注入検査も成功。
既存辞書デモ6例が正常終了。encoderデモはseed=7、6入力でfused `[6,32]`、gate `[6,10]`、
有限値・字形/字源無効化・全無効時ゼロを確認。ランダム初期値の配線検査であり学習成果ではない。
PyTorchの直接import/デモ時にNumPy未導入警告が出るが終了コード0。NumPy変換は使用していない。
知識索引24原本、full=healthy、OKF files=7 / concepts=5 / errors=0 / warnings=0。
MCP実プロセスの検索・原本追跡検査も成功。GitHub Actionsでの検証は未実行。

Luna (`gpt-5.6-luna`) にtensorizer/同テストとencoder回帰テスト・独立レビューを分担。
親が予約文字列衝突などをレビューして修正を統合し、encoder・デモ・文書・最終全テストを担当。
AIレビューのみで、人の研究内容検証印は追加していない。

未実装: 候補選択、順序/グラフを反映するencoder、概念ボトルネック、decoder、学習loss、比較実験。
語彙はtraining splitでfitし、モデルと対応する語彙checkpointを必ず保持する。
外部辞書・性能改善・大規模学習・GPU起動は含まない。次は本PRのレビュー・マージ後にIssue #3の
toy baseline / bypass / concept bottleneck比較へ進む。

## Issue #1 の実装履歴

辞書ベース SemanticCompiler に schema 1.0 を導入した。キー・型・同一表現内の重複語義ID・relation三つ組・JSON重複キーを検証し、位置と原因を持つ SchemaError を返す。
旧辞書の読み込み互換性、レコードJSON保存・復元、層と語義別の由来/source/revision、未選択候補、context/span、元レコードを保持する層除外を実装。
空・空白・未知語も surface を損失なく保持する。否定と願望の区別2例を追加し、元の4例は保持。心生は実験的・詩的な用例として明示。
仕様・移行経路は [スキーマ設計](semantic-schema.md) を参照。以下の初期引き継ぎ節の未実装一覧は履歴であり、現在の状態はこの節と設計書を優先する。

作業ブランチ: `codex/issue-1`。基点: `af25c414341fab91b456080543032bd377880e64`。
隔離 worktree と専用 `.venv` を使用し、import 先が worktree 内であることを確認。

実行コマンド（worktree ルート、Windows）:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]' --disable-pip-version-check
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m norishio_lm.demo
.\.venv\Scripts\python.exe codex/tools/build_context.py
.\.venv\Scripts\python.exe codex/tools/check_knowledge.py --mode full
.\.venv\Scripts\python.exe codex/tools/validate_okf.py
.\.venv\Scripts\python.exe codex/okf_mcp/tests/live_check.py --server codex/okf_mcp/server.py --root okf
```

Python 3.11.15 / pytest 9.1.1。環境構築成功。初回は JSON null を空辞書として受け入れる回帰テストが失敗（1 failed / 80 passed）。入力境界を修正し、再実行は **81 passed, 2 subtests passed**、終了コード0。
一時ファイルを使うテストは既知の Windows sandbox 制限を避けて許可付きで実行。ネットワーク禁止を注入したコンパイラ往復・ablation テストも成功。
デモ6例（既存4例と否定・願望比較2例）が正常終了。知識索引17原本、full=healthy、OKF形式検査 errors=0 / warnings=0。MCP実プロセス検証も成功。

未実装: 文脈語義選択、外部辞書連携、出典の内容確認、関係端点のオントロジー検証、テンソル化、学習・生成、比較実験。
学習成果・意味層の有効性は主張しない。大規模学習、有料GPU、外部辞書転載は行っていない。
次の候補: 本PRのレビュー後に adapter / encoder 境界を設計し、CPUベースラインと層別比較へ進む。

## 受け入れた内容と履歴

- ユーザー確認により、現在の作業フォルダを ZIP の展開済み内容として受け入れた。再展開や別コピーによる置換はしていない。
- 既存 Git 履歴を保持。引き継ぎ開始時の `master` は `e89ae540e6050e8d5c89d0a52ce5a323a830e90e`。
- AGENTS.md、README.md、docs/architecture.md、全 Python 実装、テスト、デモ辞書、pyproject.toml を確認した。
- 開始時から存在した demo.py の標準出力 UTF-8 対応を保持し、その変更を含む作業ツリーで検証した。

## 実装済み

- dataclass による `SemanticRecord` と `LexicalSense`。
- 表記、トークン、形態素、文字、字形構造、字源注記、現代語義、語義に付属する sememe / concept、関係の明示的な保持。
- JSON 辞書を読み、入力文字列との完全一致でレコードを組み立てる `SemanticCompiler`。
- 未知語では表記・トークン・文字にフォールバックし、語義を推測して追加しない。
- 「性」「性器」「心生」「ハナレナイ」の手書き辞書と表示デモ。
- 字形と語義の別フィールド保持、多義性、解剖学概念、未知語の基本テスト。

これは SemanticCompiler の初期雛形であり、LLM 本体や学習済みモデルではない。
デモ結果は辞書に記載された情報の出力であり、モデルが学習・推論した成果ではない。
「心生」は実験用の詩的複合語として記載されており、一般語としての意味の実証ではない。

## 未実装と検証上の限界

- 実用的な形態素解析、外部辞書アダプター、文脈による語義選択。
- 入力辞書の厳密なスキーマ検証、関係の三つ組の長さ検証、出典・信頼度の追跡。
- 各層のテンソル化・エンコーダー・融合・概念ボトルネック。
- causal LM、生成器、学習ループ、複合損失、学習済み重み。
- 各意味層を外す ablation harness と、比較実験による有効性評価。
- 日本語の曖昧語、誤誘導する字形、未知複合語、言い換えを網羅するベンチマーク。
- 現在の3テストは限定的な構造確認であり、意味理解の品質や層の有効性を証明しない。
- デモ辞書の参照はソースツリー配置に依存する。通常の wheel 配布での動作は未検証。

## 環境と実行結果

既存 `.venv` を再利用。Python 3.11.15、pytest 9.1.1。
README の仮想環境・editable install 方針に従い、Windows では activate の代わりに仮想環境の Python を直接指定した。

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe -m pip install -e '.[dev]' --disable-pip-version-check
.\.venv\Scripts\python.exe -m norishio_lm.demo
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

- 最初の pip install: 終了コード1。一時ディレクトリの build tracker へのアクセス権限エラー。
- 権限を付けて再実行した pip install: 終了コード0。norishio-lm 0.1.0 の editable install 成功。
- デモ: 終了コード0。4例すべての SemanticRecord を表示し、「性」の字形構造と複数の現代語義を別フィールドに保持していることを確認。
- pytest 通常実行: 終了コード0、3 passed, 1 warning。既存 pytest キャッシュへの書き込みが WinError 5 で拒否された警告。
- キャッシュ無効実行: 終了コード0、3 passed、警告なし。テストは省略していない。
- 大規模学習、有料 GPU 起動は実施していない。テストからのネットワークアクセスはない。

## GitHub の状態

- 接続先: https://github.com/kazuaki-nakamura/norishio-lm
- 認証済み個人アカウント: `kazuaki-nakamura`（表示名 `nkmrkzak`）。
- 同名リポジトリは既に Public として存在する。非公開リポジトリの新規作成要件は同名競合のため未達。
- 今回の引き継ぎではリポジトリの上書き、公開範囲変更、push、強制 push を行っていない。
- 引き継ぎ結果はローカルの `master` に記録する。記録コミットは `git log -1 --format=%H` で確認できる。
- 非公開の共同作業拠点を作る場合は、別名の採用または既存リポジトリの扱いについてユーザーの方針が必要。

## 次の実装候補

1. 辞書入力のスキーマ検証と出典情報を追加し、不正な語義・関係・欠損値のテストを整備する。
2. 曖昧語と意図的に誤誘導する字形の回帰テストを増やし、字形情報から現代語義を創作しないことを確認する。
3. 層ごとの有効・無効設定と辞書アダプター境界を設計し、出典を保持してテンソル化する。
4. CPU で動く小さなベースラインから、同一条件で各層を外した比較実験を実装する。

字形・字源は現代語義の根拠と同一視しない。各意味層の採用は比較実験の結果で判断する。

## AI 作業基盤の追加（2026-09-05）

ユーザー指定の ai-project-foundation から、開発用 OKF、読取専用 MCP、索引・整合性検査を取り込んだ。
研究モデル内部の概念表現とは分離している。元コミット、導入範囲、実行コマンド、権限エラーを含む検証結果は [AI 作業基盤](ai-foundation.md) を参照。
導入後はコンパイラと基盤を合わせて19テスト、2 subtests が成功し、デモ4例と MCP 実プロセス検証も成功した。
GitHub への push と MCP クライアント登録は未実施。
