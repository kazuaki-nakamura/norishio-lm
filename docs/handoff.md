# Norishio-LM 引き継ぎ記録

検証日: 2026-09-13

## 最新の実装状態: Issue #34 Phase 1

`codex/issue-34`、PR #33 merge `fee9b0923ebdc5034a82f7fd57c1a6b50b155b37`
から開始。学習前のbenchmark freezeは
`BENCHMARK_FREEZE_SHA = 19ce7145495a47b40a178255272050caa615dc79`。
Phase 1ではv2モデル学習を実行していない。

6 participant x 6 time x 4 event x 4 operator x 2表現の1152行を機械生成する。
train / diagnostic-validation / final-holdoutは各384行・192 frame group。
各評価splitはunseen participant-time pair 192行と、pair既知・triple未見192行を持つ。
trainは全原子値、24 pair、48 tripleを含み、各選択tripleは4 operatorを含む。
同一frameの2表現はsplitを跨がない。source入力は`context/text`だけで、完全なsource表現を
単一categorical IDにしない。split seed、語彙、template、規則、各JSONL SHA-256、
factor shuffle、strict target grammar、content digestをexpected manifestに固定した。

共通scorerは生成frameの4原子accuracy/balanced accuracy、pair/triple exact、train support別、
exact text、EOS、UTF-8、unique output、intervention localityを同じ分母契約で集計する。
中間headは生成指標と分離し、利用可能行だけの真の2x2表を出す。parse失敗は分母に残し、
空群は0でなくnull。teacher-forced byteは型・範囲・整合性を検証した別診断であり、
自由生成成果には含めない。final-holdoutは明示flagと一致するmanifest digestが必要。

Luna (`gpt-5.6-luna`) へ既存資産監査、漏洩テスト設計、共通scorer実装、generator監査、
metrics再監査を分担。親がsplit仕様、generator/parser/manifest、統合、レビュー修正、全検証、
freeze commitを担当した。generator監査で改行依存hashとcustom spec帰属を修正し、metrics監査で
非有限JSON、teacher-forced検証、2x2表記を修正。最終再監査はblocking issueなし。

```powershell
.\.venv\Scripts\python.exe data\benchmark_v2\benchmark.py --check
.\.venv\Scripts\python.exe data\benchmark_v2\benchmark.py --out codex\work_output\benchmark-v2-freeze-a
.\.venv\Scripts\python.exe data\benchmark_v2\benchmark.py --out codex\work_output\benchmark-v2-freeze-b
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp codex\work_output\pytest-v2-freeze-final
.\.venv\Scripts\python.exe -m norishio_lm.demo
.\.venv\Scripts\python.exe -m norishio_lm.encoder_demo
```

独立exportは4ファイルすべてbyte一致。最終全テストは **415 passed, 1 warning,
2 subtests passed**。両demoは終了コード0。encoder demoのNumPy未導入warningは既存で、
出力はfinite。これは構造fixtureと配線の検証であり、学習済み性能・日本語品質の主張ではない。

Phase 2設定も学習前に
`TOURNAMENT_FREEZE_SHA = 38e363e1b4e836b890f1c852f6d0a45fe3119296`へ固定した。
A_G0/A_G1/B/C/D/E、29,272〜29,848 parameters、共通byte encoder/decoder、600更新、
batch16、Adam .003、clip1、CPU1thread、seed 7/17/29、18 terminal run gateを機械検証する。
freeze時点の全テストは **423 passed, 1 warning, 2 subtests passed**。

残課題は固定仕様どおりのv2 model/checkpoint/evaluator実装、preflight parameter照合、18 run実行、
diagnostic-validation集計。全run終端前にfinalを開かず、一度のinvocationでcomplete checkpointだけを
評価する。失敗runは明示して残す。

## 最新の実装状態: Issue #32

`codex/issue-32`、PR31 merge `29533877a11850bb801f54f28296415797d10f80` から開始。
PR #33（レビュー中、未merge）: https://github.com/kazuaki-nakamura/norishio-lm/pull/33。
事前計画は [gradient-routing.md](gradient-routing.md)、実測結果は
[gradient-routing-results.md](gradient-routing-results.md) に固定した。

### 実装・観測

G0 `end_to_end` はdecoder LM lossからparticipant/time専用headへの勾配を従来どおり通し、
G1 `stop_slot_lm` はそのdecoder経路のslot確率だけをdetachした。slot logitsはCE lossに残し、
decoder/base LM、共有encoder、global gradient clippingは保持した。checkpointにはrouting modeを
記録し、旧形式は既定値G0として読み込む。専用回帰テストは7 passed。

同一初期state・同一600-step schedule・同一loss/optimizerで、seed 7/17/29を各G0/G1実行。
parameter数は全arm 42,689。validationは150行すべて未見pair、testは未評価。
G0のIssue26固定baselineへの評価と最終state digestは3 seedすべて完全一致した。

G1の16-row probeでは、update 0/100/300/600のparticipant/time LM-only head gradientが全て
厳密ゼロ、slot CE gradientは全て非ゼロ。G0ではLM/CEとも非ゼロ。通常のpredicted条件は
G0が全seed 0/150、G1がseed7=0/150・seed17=1/150・seed29=0/150（合計1/450）。
この1件とgold介入下の差は固定toy grammar内の診断に留まり、一般化改善とはしない。
teacher-forced byte-v2は4条件×6 armで実行し、各checkpointのstate・旧concept・head・logit
再読込一致も確認した。これは学習済みモデルの意味理解や自由生成品質を示す測定ではない。

raw report `codex/work_output/issue32-gradient-v2/report.json` はignoredな配下に保持し、
履歴reportを上書きしていない。

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_gradient_routing --baseline-report codex/work_output/issue26-fixed-v1/report.json --historical-report codex/work_output/issue24-seed7-v1/report.json --out-dir codex/work_output/issue32-gradient-v2
.\.venv\Scripts\python.exe -m pytest tests/test_slot_gradient_routing.py tests/test_gradient_routing_checkpoint.py -q -p no:cacheprovider
```

最終検証は **390 passed, 1 warning, 2 subtests passed**（18.18秒、既存のzero-element warning）。
両デモ、OKF 8files/6concepts、knowledge full healthy、MCP live check も終了コード0。
NumPy未導入warningはencoder demoの既存環境警告で、実験・テストの判定には使っていない。

### 残課題

この結果だけではslot headが失敗の唯一の原因とも、意味層の有効性とも言えない。global clipを
共有するため、G1でも共有encoder/base更新はG0と同一ではない。次候補は同じrouting controlを
大きめのcompositional splitへ移し、共有encoder経路のablationと自由生成評価を事前固定する。

## 最新の実装状態: Issue #30

`codex/issue-30`、PR29 merge `d844e1667475bf5926b8c503ade7e2997a4db511` から開始。
事前計画 `bb0933c`、実装SHA `596e1387142550409294fc882f07757411defaa8`。
[固定条件](compositional-slots.md)と[全結果表](compositional-slot-results.md)。

### 実装・観測

A独立head D、B flat25-wayの保存済み3seedを完全再現。CはA既存headの外積を周辺化するだけで
追加parameter0/学習更新0。各周辺は数学的にAと同じ。float32最大誤差は下表、bitwise一致とはしない。

| seed | marginal maxabs | logit maxabs | A/C通常・gold生成一致 | C通常両slot | C gold両slot |
|---|---:|---:|---|---:|---:|
| 7 | 1.78813934e-7 | 2.86102295e-6 | 完全一致 | 0/150 | 48/150 |
| 17 | 1.19209290e-7 | 3.33786011e-6 | 完全一致 | 0/150 | 46/150 |
| 29 | 1.19209290e-7 | 5.36441803e-6 | 完全一致 | 0/150 | 64/150 |

全seedでatol=rtol=1e-5のallclose成立。通常生成の改善なし。
validation未見gold pair確率平均C0.0709574764/B0.000495385041、rank平均C6.304444/B21.202222。
固定top-k coverageはC top1/3/5/10=0.004444/0.242222/0.442222/0.9、Bは全て0。
train未出現10pairへの確率和平均C0.192065561/B0.00545005129。
pair entropy平均C1.992786819/B1.855719939 nats。Cは未見pairへ確率を置くが通常両slot0は不変。
個別/marginal calibration、train診断、gold pair別top-kと全生成値は結果表参照。

factorized pair NLLはparticipant CE+time CEと値/勾配が等価。追加すればhead CE重みが倍になるだけで、
新しいpair相互作用を学習しない。別arm Dの学習は行わず、結果後の新loss発明・調整もない。
train450のみfit、validation150全未見pair、test未評価。元gate・decoder・全重みを保持。

### Byte診断の修復

Issue26で到達不能だった組立をerror branch外へ修復。明示opt-inのteacher-forced-byte-v2として、
seed7凍結Dのpredicted/participant-gold/time-gold/both-gold4条件を新規測定。
旧版既定値nullと過去reportは保持。修復前に測定済みだったとはしない。

| condition | 人物 正解byte/900 | time 正解byte/900 | 人物NLL | time NLL |
|---|---:|---:|---:|---:|
| predicted | 774 | 799 | 0.618294369 | 0.441661070 |
| participant_gold | 816 | 799 | 0.291035138 | 0.441661070 |
| time_gold | 766 | 900 | 0.645301078 | 0.069695553 |
| both_gold | 810 | 900 | 0.317568600 | 0.069695553 |

正解過去履歴の次byte診断であり自己履歴生成ではない。self L1/KL=0は自己比較のためで、
介入不感性の証明ではない。通常/both-goldの生成とLMが旧版と変わらないことを確認。

### 保存・検証・残課題

report SHA256 `51f1abade28912011c29726a38991236701204cf46e18ce689cbb5c98f82eef4`。
ignored `codex/work_output/issue30-fixed-v1/report.json`。A/B全3seedの過去評価とstate完全一致、
B再読込state/旧concept/独立head/pair/marginal/logits/greedy一致。旧Issue24/26/28 report SHA不変。
全383 pytest + 2 subtests passed（19.57秒、既存zero-element警告1）、両demo exit0。
知識索引133件healthy、OKF errors0/warnings0、MCP live check成功（6tools、stale0）。
NumPy未導入警告は既存で今回の実行に不要。Lunaは直前の利用上限停止のため委譲せず、親が全作業を実施。

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_compositional_slots --baseline-report codex/work_output/issue28-fixed-v1/report.json --out-dir codex/work_output/issue30-fixed-v1
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m norishio_lm.demo
.\.venv\Scripts\python.exe -m norishio_lm.encoder_demo
.\.venv\Scripts\python.exe codex/tools/build_context.py
.\.venv\Scripts\python.exe codex/tools/validate_okf.py
.\.venv\Scripts\python.exe codex/tools/check_knowledge.py --mode full
.\.venv\Scripts\python.exe codex/okf_mcp/tests/live_check.py --server codex/okf_mcp/server.py --root okf
git diff --check
```

通常同時保持は未解決。次Issue候補は非等価な表現/組合せ目的の事前固定比較。
単なる外積・同じCE追加を新しい構成性学習と呼ばない。n3/共通拡張seed/文法prior/容量や目的の交絡は残る。
各意味層の有効性・一般日本語理解の証明ではない。追加学習、paid GPU、worker再開、自動merge/closeなし。

## 過去の実装状態: Issue #28

`codex/issue-28`、PR27 merge `1c34f73c27a9652b657834f42fc2fef50f6e230f` から開始。
事前計画 `f02f190`、実装 `08f0077`、保存guard補強 `a429120`。
[固定条件](pair-head.md)と[全seed・calibration・75factorial表](pair-head-results.md)。

### 実装と固定境界

Aは保存済みD seed7/17/29のhead・predicted/both-gold全評価とstateを厳密再現。
Bはfused latent32から25-way pair headを追加（825増、43514parameter）。train vocab直積順を使い、
softmax25の人物/time marginalだけを既存局所gateへ投入。独立headは診断用に残しCE各1を維持。
既存4損失各1 + pair CE1、600更新/batch16/Adam .003/clip1/CPU1thread。
common初期化/sampling seed7/17/29、拡張seed20とpair seed28は全seed共通。
train450のみfit、validation150全未見pair、test未評価。未見10pairにはpair CE正例がない。

### 実数値（validation分母150）

| seed | B pair正解 | B marginal joint | B独立head joint | B通常 人物/time/両slot/全frame | B両head gold 人物/time/両slot/全frame |
|---|---:|---:|---:|---|---|
| 7 | 0 | 0 | 0 | 39 / 35 / 0 / 0 | 67 / 91 / 61 / 12 |
| 17 | 0 | 3 | 29 | 38 / 39 / 0 / 0 | 56 / 72 / 36 / 6 |
| 29 | 0 | 0 | 24 | 39 / 43 / 0 / 0 | 47 / 90 / 47 / 17 |

B train pair正解446/450・450/450・438/450。validation pair平均NLL8.14916054、Brier1.2500376、
ECE0.39609727、gold確率0.000495385041、gold rank21.2022222/25。
通常両slotはA/B全seed0。B gold両slot平均48、全frame11.667（Aは52.667/16）。
B通常LM平均0.172167103、gold0.125643320。B全4条件/全seedのEOS/UTF8は150/150、special0。
単独oracle・個別head accuracy/balanced/calibration、全factorialの値は結果表に記録。
train未見群/validation既見群は0例で率null。成功を前提に条件を調整していない。

### 検証・保存

report SHA256 `902545c14f91c68845d45b0af880432aa8c004137e5ff9cbd458ae84506f69c1`。
ignored出力 `codex/work_output/issue28-fixed-v1/`。各checkpoint/初期値/schedule SHAは結果表参照。
全3保存物でstate/旧concept/独立head/pair/marginal/logits/greedyが完全一致。
保存guard補強後も3保存物のload成功。全pytest379 passed + 2 subtests passed（18.75秒）、
既存zero-element警告1。両demo exit0、NumPy未導入警告は既存で今回の処理には不要。
知識索引127件healthy、OKF errors0/warnings0、MCP live check成功（6tools、stale0）。
最終索引作成時、権限を広げると旧テストcheckpoint13件が混入し140件になることを発見。
foundation.jsonで.pt/.pth/.safetensorsを除外し、保存物を索引から分離。権限による索引差は運用改善対象。

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_pair_head --baseline-report codex/work_output/issue26-fixed-v1/report.json --out-dir codex/work_output/issue28-fixed-v1
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m norishio_lm.demo
.\.venv\Scripts\python.exe -m norishio_lm.encoder_demo
.\.venv\Scripts\python.exe codex/tools/build_context.py
.\.venv\Scripts\python.exe codex/tools/validate_okf.py
.\.venv\Scripts\python.exe codex/tools/check_knowledge.py --mode full
.\.venv\Scripts\python.exe codex/okf_mcp/tests/live_check.py --server codex/okf_mcp/server.py --root okf
git diff --check
```

### 委譲と残課題

gpt-5.6-lunaへpair metricsとpair model/checkpoint・回帰テストを独立委譲。
モデル担当が利用上限で停止した後、親がRNG隔離・入力検証・保存guard・勾配/因果/再現テストを補完。
親が全変更を点検しharness・実験・最終検証を実行。自動merge/close、worker再開、有料GPUなし。

通常経路の同時保持は未解決。次候補は別Issueでfactorized/compositional objective、
独立soft headをdecoderに戻す固定対照。今回の容量/目的/入力分布の効果は未分離。
未見pair正例不在・n3・共通拡張seed・文法priorを一般意味理解の結論へ拡張しない。
既存Issue26 scorerのteacher_forced組立が到達不能でnullだった点を発見し、過去結果文書を訂正。
そのbyte診断は未測定。旧reportを保持し、修復後の追加測定と区別することが残課題。

## 過去の実装状態: Issue #26

`codex/issue-26`、PR25 merge `3880c85555f876d05d7ef481a5efde7ac73258d8` から開始。
事前計画 `c95375e`、実装SHA `0c0141abaee77dc594b64cc35809afaabfffd14c`。
[条件・計算式](head-following.md)と[各seed・pair・ablation全表](head-following-results.md)。

### 固定条件と実装

Issue24 Dの構造・gate・42689parameterを固定。seed7は再学習せず、過去8条件/25factorialとstateを厳密再現。
追加seed17/29はcommonモデル初期値とsamplingを変更し、拡張head/projectionのseed20は全seed共通。
各600更新/batch16/Adam .003/clip1/CPU1thread、既存4損失各1+専用head CE各1。slot byte CEなし。
train450のみfit、validation150全未見pair、test未評価。条件・閾値・weightを結果後に変更しない。

個別accuracy/balanced/Brier/NLL/10bin ECE、同一rowのjoint exact、gold pair別4分類、誤り相関を追加。
soft headのjoint logprob/entropy/marginは独立近似と明記し、各rowの確率・gold・予測を保存。
decoder追随はpredicted/single-gold/both-goldのgold pair別採点と25factorialへ分離。
factorialはtrain-seen/validation-unseen/neitherを注記し、通常reference精度とは混同しない。

### 実数値（validation分母150）

| seed | 通常 人物/time/両slot/全frame | 両head gold 人物/time/両slot/全frame | source head joint |
|---|---|---|---:|
| 7 | 57 / 43 / 0 / 0 | 48 / 99 / 48 / 16 | 0 |
| 17 | 44 / 52 / 0 / 0 | 46 / 114 / 46 / 12 | 2 |
| 29 | 27 / 43 / 0 / 0 | 64 / 103 / 64 / 20 | 0 |

通常両slot平均0、両head gold平均52.667/150、全frame平均16/150。
通常LM平均0.178836220、both-gold平均0.123238005。seed17のみEOS/UTF8は両条件144/150、他seed150/150。
source head誤り相関は-0.648136/-0.583282/-0.454944。argmax成功が同一rowで少ないが、soft情報の欠如を断定しない。

seed7 confident train subset（両head max>=0.8固定）はcorrect263/wrong0。correct群の両slot210/263・全frame56/263。
残187例は閾値外。wrong群の評価はnullで、比較不能。validationラベルでsubsetを選んでいない。

seed7推論時ablationはfull/base21 zero/event zero/operators zero×predicted/both-gold。
全base zeroは両条件parse0/150、生成1種類。既知prefix成立0/150でgateが開始しないことを保存tokenから補足照合。
event zero + goldは両slot72/150へ増えるが全frame12/150へ低下。operators zero + goldは両slot48/全frame12。
入力zero後のbiasは残し、頭の予測と重みは固定。分布外介入・文法prior依存を一般意味理解の結論へ拡張しない。

### 保存・検証

report SHA256: `964ab4d7dd7fc8b2fdddac816af51ac2ea8ab28a806a33dff32e40631250170f`。
seed17 checkpoint: `bb3dc4c5d83d0e6622bdaf0f94bfbd8f925d2cb29460091d7a27c162a3be5614`。
seed29 checkpoint: `fd6d79852acb39ea2f7c0c56571744449085449e0f19f05946866dabea56803e`。
両保存物でstate/旧concept/head/logits/greedy再読込が完全一致。seed7 stateも不変。
出力はignored `codex/work_output/issue26-fixed-v1/`、Gitに重み/reportや会話を追加しない。

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_head_following --baseline-report codex/work_output/issue24-seed7-v1/report.json --out-dir codex/work_output/issue26-fixed-v1
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m norishio_lm.demo
.\.venv\Scripts\python.exe -m norishio_lm.encoder_demo
.\.venv\Scripts\python.exe codex/tools/build_context.py
.\.venv\Scripts\python.exe codex/tools/validate_okf.py
.\.venv\Scripts\python.exe codex/tools/check_knowledge.py --mode full
.\.venv\Scripts\python.exe codex/okf_mcp/tests/live_check.py --server codex/okf_mcp/server.py --root okf
git diff --check
```

本worktreeのPython3.11.15/Torch2.14.0+cpuで実験と両demo exit0。
全pytest366 passed + 2 subtests passed（16.93秒）、既存zero-element警告1。
NumPy未導入警告は出るが今回の処理はNumPy不要。Windows Temp制限は承認済み実行で回避。
索引117件、OKF errors0/warnings0、knowledge full117 sources healthy、MCP live6 toolsでok。
ローカル結果とGitHub CIを区別する。

Luna (`gpt-5.6-luna`) にjoint/calibrationと介入/採点を委譲。親がECE集計・clamp式・空群/未知ラベルの扱いを
レビュー補修し、統合harness、固定seed学習、全検証とGitHub操作を担当。PRまで進めてレビュー待ち。
自動merge/close、停止中worker再開、有料GPU、追加architecture探索はしない。

### 未解決と次候補

通常pair保持は3seedとも未解決、oracle追随も不完全。train正解subsetでもdecoder失敗が残る。
次は別Issueで、予測headの組合せ一般化、soft/hard head情報の差、文頭失敗でgateが閉じる依存を事前固定して分離する。
現n=3・同じAI-authored教材・共通拡張seed20は一般性能の証明ではない。上流意味層の再学習ablationも未実施。

## 過去の実装状態: Issue #24

`codex/issue-24`、PR23 merge `615fc29db1dc0ccef21ab9d4e1c044005e582e16` から開始。
事前計画 `3eeebd7`、実装SHA `e86a7c5d66cec75fd11cce577d6bfda3cd613844`。
[事前固定条件](local-slot-injection.md)と[全結果表](local-slot-results.md)。

### 実装・測定結果

A（Issue20 C）/B（Issue22）は再学習せず、保存物hash/stateと過去の全評価を厳密再現。
Cは旧concept内のparticipant/time各6次元をdecoder投影から除外。有効31次元、外部transport43。
DはCと同じ重みで、h0はbaseだけ、生成済みの既知文頭からtime6byte/person6byteの注入窓を決める。
未来token・正解span・reference長・sourceによる位置選択は使わず、壊れた文頭ではgateを開かない。
対象外stepで直接寄与0でもGRU過去状態経由の作用は残る。教師強制と自己履歴生成を区別。
C/Dは各42689parameter、A/B比384減。C/D共通初期state完全一致、seed7/600更新/batch16/Adam .003/clip1/CPU1thread。
既存4損失各1+head CE各1、slot byte CEなし。train450のみfit、validation150全未見pair、seen群率null。

- C通常: LM 0.163995881、人物15/time37、両slot0/全frame0（各分母150）。
  EOS150/UTF8150、26種類、parse120/150。
  participant_gold: 人物46/time29、両slot0/150。
  time_gold: 人物2/time126、両slot2/150。
  both_gold: 人物12/time108、両slot6/150。
- D通常: LM 0.164825604、人物57/time43、両slot0/全frame0（各分母150）。
  EOS150/UTF8150、26種類、parse120/150。
  participant_gold: 人物95/time39、両slot18/150。
  time_gold: 人物20/time120、両slot20/150。
  both_gold: 人物48/time99、両slot48/150。

各arm8条件のgold/zero/permutation、cross-slot感度と25組factorialの指定pair追随を保存。
通常の両slotはC/Dとも0。Dは両head goldで48/150まで回復するが、未見5組中3組のfactorialは0。
source headの同時argmax正解は補足集計C1/D0（/150）。予測側とdecoder側の両方に失敗が残る。
Dの人物介入→先行time logitは0変化で、全150行の人物注入前tokenも不変。
人物goldによるtime採点43→39は全体parse失敗を含むため、過去byteへの逆因果作用とは呼ばない。
両新checkpointでstate/旧concept/head/logits/greedyの再読込が完全一致。
report SHA256: `eb2c1fc26c35e429d25b3815f15060e7fd481cf9ff5cca62f0e36198ae335af8`。

C.pt SHA256: `0532e7c1e859b50cc86d2de541da42f8a9e9f311a77800b7e7fe4b686ebef68e`。
D.pt SHA256: `abed886085e2a862be1a8015ca910adbc77e687c09c10de52411cbc3e8bbe143`。

重み/reportはignored `codex/work_output/issue24-seed7-v1/`。Gitに学習artifactや会話を追加しない。

### 実行コマンド・検証

Windows本worktreeのPython3.11.15 / Torch2.14.0+cpu。

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_local_slots --baseline-report codex/work_output/issue22-seed7-v1/report.json --historical-report codex/work_output/issue20-seed7-v1/report.json --out-dir codex/work_output/issue24-seed7-v1
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m norishio_lm.demo
.\.venv\Scripts\python.exe -m norishio_lm.encoder_demo
.\.venv\Scripts\python.exe codex/tools/build_context.py
.\.venv\Scripts\python.exe codex/tools/validate_okf.py
.\.venv\Scripts\python.exe codex/tools/check_knowledge.py --mode full
.\.venv\Scripts\python.exe codex/okf_mcp/tests/live_check.py --server codex/okf_mcp/server.py --root okf
git diff --check
```

全pytest 356 passed + 2 subtests passed（20.45秒）、既存zero-element警告1。両demoと実験exit0。
TorchのNumPy未導入警告は出たがNumPyを必要とせず完走。Windows Temp制限は承認済み実行で回避。
索引110件、OKF errors0/warnings0、knowledge full 110 sources healthy、MCP live6 toolsでok。
ローカル検証をGitHub CIと混同しない。

Luna (`gpt-5.6-luna`) にモデル/生成とcheckpointを委譲。親は要件/計画・初期値照合・統合harness・
因果性/直接寄与の回帰テスト・実測・全検証を担当。委譲成果のC保存不可と特殊token処理を測定前に補修した。
レビュー用PRまで作成し、自動merge/closeや停止中workerの再開はしない。

### 未実装・制約と次候補

Cは入力情報とparameter数が同時に変わる。Dは既知文法と注入時刻/h0が同時に変わる。
単一原因の確証や一般意味理解の証明とは呼ばない。seen-validation群はなく、複数seedも未実施。
次候補は別Issueで、得られた失敗に応じたhead予測/decoder追随の分離、既知grammar以外の位置推定、
複数seedの事前固定追試と意味層ablation。教材/split/testは結果後に変更せず、有料GPUは起動しない。

## 過去の実装状態: Issue #22

`codex/issue-22`、PR #21 merge `10768943bf9e52afc7929dc8557172a0024d03ca` から開始。
事前計画 `e021418`、実装SHA `7f4ede9ead6fcad2d1a5899bf5850b538fe000d2`。
[事前固定した条件](joint-slots.md)と[実数値・全表](joint-slot-results.md)を参照。

### 実装と観測

- train450の15組とvalidation150の5組は重ならず、validation全150例がunseen pair。
  5×5 support表、all/seen/unseen群のLM/head/byte/生成を保存。seen群0例は率null。
- AはIssue20 Cの保存物を厳密再現。Bは43入力の線形投影を33/5/5の3項へ分割しtanh前に加算。
  43073parameter・増分0、同初期重み、同600更新schedule、head CE各1、slot byte CEなし。
  数学的に同じ関数クラスであり、性能差を表現力増大や干渉の因果分離と呼ばない。
- 通常生成はA人物36/time25、B人物40/time25、両slot・全frameは双方0/150。
  BのLMは0.163130231、EOS/UTF8各150/150、11種類、parse120/150。
- B人物head goldは人物72/time1、time head goldは人物37/time49、両head goldは52/39。
  いずれも両slot0/150。単独修復に伴う他slot悪化が残る。
- 各arm8条件のhead-only gold/zero/permutation、cross-slot logit感度と25組のfactorial系列を測定。
  元conceptは予測値固定。factorialの指定pairへの追随は通常reference性能と別集計。
- B checkpoint再読込でstate/旧concept/専用head/logits/greedyが完全一致。
  report SHA256 `a303e74fab83ccc11b4350579988b2e88e2077ba44abc263f32549890de2be39`。
  B.pt SHA256 `207420aa7c8b4b8ba0d3aa5ad4a667edaf2c7342daaeac9705caa0792b2831f9`。
  保存物はignored `codex/work_output/issue22-seed7-v1/`、Gitに学習artifactを追加しない。

### 検証コマンド

Windowsの本worktree `.venv`（Python3.11.15 / Torch2.14.0+cpu）で実行。

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_joint_slots --baseline-report codex/work_output/issue20-seed7-v1/report.json --out-dir codex/work_output/issue22-seed7-v1
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m norishio_lm.demo
.\.venv\Scripts\python.exe -m norishio_lm.encoder_demo
.\.venv\Scripts\python.exe codex/tools/build_context.py
.\.venv\Scripts\python.exe codex/tools/validate_okf.py
.\.venv\Scripts\python.exe codex/tools/check_knowledge.py --mode full
.\.venv\Scripts\python.exe codex/okf_mcp/tests/live_check.py --server codex/okf_mcp/server.py --root okf
git diff --check
```

実験・両demoはexit0。全pytestは341 passed + 2 subtests passed（16.76秒）、既存zero-element警告1。
NumPy未導入警告はdemo/実験で出たがNumPyを使わず完走。
テスト編集と重なった初回全実行は旧期待値で1件失敗、確定後の全実行で解消した。失敗を成功扱いしない。
known Windows Temp権限制限はproject内Tempまたは承認済み実行で回避。
索引101件、OKF errors0/warnings0、knowledge fullは101 sources healthy、MCP liveは6 toolsでok。
ローカル成功をGitHub CI成功とは呼ばない。

Luna (`gpt-5.6-luna`) にモデル/保存形式とpair採点を独立委譲。
親が初期値照合・介入範囲テスト・統合harness・実測・全検証を担当し、保存形式の検査を補強した。
PRまで進めてレビュー待ちとし、自動merge/closeや停止中workerの再開はしない。

### 未実装・次候補

seen群が0例のためこの分割でseen/unseen性能差は推定できない。組合せ偏りが唯一の原因とは断定不可。
次は別Issueで数学的に異なる局所/直交注入、old concept内の人物/time重複除去ablation、複数seedを事前固定。
今回の教材/split/testは変更せず、追加weight探索・外部辞書・有料GPUは実行していない。
意味層の有効性や一般日本語理解の証明は未達。手書き辞書出力を学習成果と呼ばない。

## 過去の実装状態: Issue #20

`codex/issue-20`、PR #19 merge `429cd8ea9718579502f8ac047187389b044f6e6c` から開始。
[事前計画](explicit-slot-head.md) `257ba0e`、実装SHA
`3bae02d9fa1e8d21c21d213390b9650eccc37c3f`。

### 固定構造と学習

A/BはIssue #18の保存済み重みを再利用し、hash・state・全4条件の生成/LM/byte/head評価を
厳密再現してからCを測定。歴史結果を変える再学習は行わない。
Cはseed7共通初期モデルから開始し、既存parameterと投影の先頭33列/biasを完全コピー。
encoder fused latentから人物/timeのLinear(32,5)を各1本追加。train語彙ID1..5をhead ID0..4へ対応。
元の33次元concept（既存人物/timeも含む）に2つのsoftmax分布計10次元を連結し、
Linear(43,32)→tanhをh0と各token入力へ加算。直接source latentをdecoderへ渡さない。

追加head330+投影列320=650parameter、C43073（A/B42423）。fork_rng seed20で
人物head、time head、拡張投影の順に初期化して共通列/biasをコピー、呼出元RNGを保持。
既存4損失各1 + 人物head mean CE1 + time head mean CE1。Cにslot byte CEは加えない。
train450のみfit、validation150、test未評価。600更新/batch16/Adam .003/clip1/CPU1thread、
共通sampling schedule。新headの各loss分母16例/更新。C学習44.051秒（速度比較の実証ではない）。
Cの容量増分と追加目的の効果は混ざるため、同容量の因果比較とは呼ばない。

共通初期SHA256: `f07d83adec88a37040886e1374800b17b56b7d2f3f49cea3f1b801c4b61b4b70`。
C初期SHA256: `661d2fe58f6b0872ac0da39ad86a75f2474bdae60753dd78cb0ba58f45a298e6`。
sampling SHA256: `6c3ae94191450c6c60d8ea3375711f56b241475b5da890940855f999d7e95e66`。

### 通常生成A/B/C

| arm | LM | 人物 /150 | 時点 /150 | 全frame /150 | EOS /150 | UTF-8 /150 | 種類 | parse /150 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.168318838 | 20 | 22 | 0 | 150 | 150 | 9 | 109 |
| B | 0.172247004 | 26 | 25 | 0 | 149 | 149 | 6 | 125 |
| C | 0.164552725 | 36 | 25 | 0 | 150 | 150 | 10 | 120 |

A/B/Cとも正常経路はsource/contextのみから確率を作りBOSから自己生成する。
slotは既存target文型の機械採点。parse失敗を全row分母から除かず、条件付き分母はreportへ別保存。

### Cの専用headと元concept

専用headは各5class support30、confusionはrow=gold/column=prediction、head ID順0..4。

- participant: 57/150、balanced 0.380000、entropy 1.041703。
  class順 `["先輩", "友人", "同僚", "知人", "隣人"]`。
  confusion `[[24, 6, 0, 0, 0], [0, 0, 0, 5, 25], [0, 0, 24, 0, 6], [0, 21, 0, 9, 0], [0, 6, 0, 24, 0]]`。

- time: 61/150、balanced 0.406667、entropy 1.010194。
  class順 `["今日", "明日", "来月", "来週", "週末"]`。
  confusion `[[4, 26, 0, 0, 0], [0, 23, 0, 0, 7], [0, 0, 30, 0, 0], [0, 0, 26, 4, 0], [0, 0, 1, 29, 0]]`。

元concept headの正解数は event 150/150, operators 83/150, agent 135/150, participant 54/150, time 67/150, location 150/150, repeat_marked 135/150.

専用time head61/150は元time head67/150を超えておらず、専用化だけの優越性とはしない。
各例のsoft確率とクラス別support、元conceptの詳細もreportへ保存。

### Cのoracle・専用head介入

participant/time oracleは該当する元concept群と専用headの両方をgoldへ修復する。
full oracleは元7conceptと専用2headを修復。head_mean/permuted/goldは元conceptを通常予測のまま保持。
meanは学習後train予測の平均、permutationはseed17でvalidation行を全体置換。介入は推論時だけ。

| condition | LM | 人物 /150 | 時点 /150 | 全frame /150 | EOS /150 | UTF-8 /150 | 種類 | parse /150 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| predicted | 0.164552725 | 36 | 25 | 0 | 150 | 150 | 10 | 120 |
| participant_oracle | 0.157968680 | 86 | 0 | 0 | 150 | 150 | 14 | 120 |
| time_oracle | 0.150191686 | 5 | 72 | 0 | 150 | 150 | 13 | 120 |
| full_oracle | 0.147264605 | 63 | 48 | 0 | 147 | 147 | 18 | 111 |
| head_train_mean | 0.163561117 | 25 | 24 | 0 | 150 | 150 | 7 | 120 |
| head_permuted | 0.167234517 | 31 | 20 | 0 | 150 | 150 | 13 | 120 |
| head_gold | 0.152015500 | 59 | 31 | 0 | 150 | 150 | 10 | 120 |

全条件の完全一致文0/150、特殊token行0。headだけのgoldでも人物59/time31へ変わるが、
全frame0。人物oracleで人物86/time0、time oracleで人物5/time72となり、単独slotの回復が
他slotの保持へ結び付かない。学習分布の組合せやdecoder内の依存が候補だが原因確定ではない。

### Cのteacher-forced byte指標

正解履歴を与えた診断であり、自由生成品質とは別。各slot900byte/150例。

| condition | slot | NLL | 正解確率 | rank | argmax /900 |
|---|---|---:|---:|---:|---:|
| predicted | participant | 0.656773 | 0.777719 | 1.405556 | 740 |
| predicted | time | 0.349710 | 0.815822 | 1.165556 | 781 |
| participant_oracle | participant | 0.538434 | 0.807100 | 1.313333 | 774 |
| participant_oracle | time | 0.381343 | 0.802030 | 1.194444 | 755 |
| time_oracle | participant | 0.639540 | 0.775678 | 1.424444 | 720 |
| time_oracle | time | 0.192758 | 0.861121 | 1.076667 | 837 |
| full_oracle | participant | 0.557556 | 0.797828 | 1.356667 | 735 |
| full_oracle | time | 0.219935 | 0.845101 | 1.106667 | 813 |
| head_train_mean | participant | 0.653703 | 0.782206 | 1.384444 | 731 |
| head_train_mean | time | 0.344677 | 0.817707 | 1.170000 | 780 |
| head_permuted | participant | 0.664907 | 0.778737 | 1.381111 | 739 |
| head_permuted | time | 0.359484 | 0.816274 | 1.174444 | 775 |
| head_gold | participant | 0.578783 | 0.795651 | 1.314444 | 773 |
| head_gold | time | 0.270836 | 0.829579 | 1.157778 | 785 |

Cの通常time byte NLL .349710はA .308307/B .289612より悪い。全体LMや人物slotの
改善だけで性能全体を良いと結論しない。

### head単独置換のlogit感度

seed17で片方の専用headだけを他行と置換。元conceptと他headは同じ。正解履歴を固定して比較。
beforeはslot start-1（150byte）、insideはslot内（900byte）、各150行。KLはbase||介入。
各byteの位置・L1/KL/argmax差もreportに保存。

| 置換head | 採点slot | 領域 | logit L1 | KL | argmax変化率 |
|---|---|---|---:|---:|---:|
| participant_head_permuted | participant | before | 0.119358 | 0.002708 | 0.000000 |
| participant_head_permuted | participant | inside | 0.091218 | 0.016020 | 0.040000 |
| participant_head_permuted | time | before | 0.023611 | 0.000164 | 0.000000 |
| participant_head_permuted | time | inside | 0.060269 | 0.002399 | 0.014444 |
| time_head_permuted | participant | before | 0.049076 | 0.001527 | 0.000000 |
| time_head_permuted | participant | inside | 0.044793 | 0.003463 | 0.024444 |
| time_head_permuted | time | before | 0.018112 | 0.000057 | 0.000000 |
| time_head_permuted | time | inside | 0.036314 | 0.005825 | 0.018889 |

専用headの情報がlogit/生成へ届くことを支持するが、意味が正しく保持されることとは別。
head平均は通常よりLMがわずかに低く、すべての指標で内容利用が有益とは言えない。

### 再現・検証・残課題

A/B report SHA256: `9db04ef46231cfb79439f1dc4f90883567b3f80433a7b8fc295cb2f2fcc1a382`。
今回 `codex/work_output/issue20-seed7-v1/report.json` SHA256:
`66e8c507dc5485d5e01825882688aeb8e34f0d6cbbbba36b0ce14d4960761ad8`。
C checkpoint SHA256: `8bba5fb4915fc0435af474fae9649bf3b3b8e86737968c1f97271c7e77850956`。
C final state SHA256: `d85d4af474d46b09ae17f3bb75d2ddaf1157933b2fc295057c6dc91b4dd65629`。
A/Bの全評価とstate、C再読込のstate/元concept/専用head/logits/greedy150例は厳密一致。
新checkpointはprimitive/tensorのみ、CPU weights_only読込、metadata hash・語彙対応・次元検証、
排他的保存。hashは改変検知用であり署名ではない。保存物はignoredでGitには含めない。
再実行には上記hashの過去reportとA/B保存重みが必要。

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_explicit_slots --baseline-report codex/work_output/issue18-seed7-v1/report.json --out-dir codex/work_output/issue20-seed7-v1
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m norishio_lm.demo
.\.venv\Scripts\python.exe -m norishio_lm.encoder_demo
.\.venv\Scripts\python.exe codex/tools/build_context.py
.\.venv\Scripts\python.exe codex/tools/validate_okf.py
.\.venv\Scripts\python.exe codex/tools/check_knowledge.py --mode full
.\.venv\Scripts\python.exe codex/okf_mcp/tests/live_check.py --server codex/okf_mcp/server.py --root okf
git diff --check
```

実験v1終了0。全pytest **330 passed, 2 subtests passed**、14.70秒、終了0。
knowledge91原本full=healthy、OKF8 files/6 concepts errors0/warnings0、
MCP実プロセス6 toolsとgit diff --checkも終了0。
既存zero-element警告1、両デモ終了0。実験/encoderのNumPy未導入警告あり。
初回のcheckpoint担当テストはTemp制限で未検証だったが、親が許可付きで全検証済み。
15追加テストはhead CE→encoder/head、LM→headの非ゼロ勾配を別々に確認し、
未来input/label、他群不変、RNG/共通parameter、保存物改変/語彙不一致を検査する。

`gpt-5.6-luna`にCモデルとcheckpointを独立委譲。親が不足テストと語彙/保存境界の補完を指示し、
ハーネスと介入/感度テストを実装、全体を検証。Luna最終読取レビューはblockerなし。
人の検証印は追加しない。一般日本語、意味層の有効性、複数seedの再現は未実証。
次候補は人物/timeの同時保持と組合せ対照、元conceptと専用headの重複情報を分離する対照。
結果後の重み変更・test tuning・有料GPU・自動merge/close・定期worker再開は行わない。

## 過去の実装状態: Issue #18

`codex/issue-18`、PR #17 merge `545b81fd035182376f44cdbdc5dad50cae08a180` から開始。
[事前計画](slot-objective.md) `81ba65e`、実装SHA
`915374ace3de9c6198e0bf3099b36ca227d4a3ec`。

### 損失と固定比較

Aは従来per_step_additiveでslot追加損失なし。IssueでいうLM-onlyは新slot損失なしの意味とし、
従来のLM/concept/sense/sememeの4損失（各重み1）はA/B両方に残した。
Bは同じ4損失に `1.0 * slot CE` を追加。人物/timeのUTF-8 byte半開区間の和集合だけで
CEをbyte数加重平均する。EOS/padding/他位置は追加損失から除外し、重複spanは拒否する。
通常LMは従来どおり全target byte+EOSを含むため、slot byteは追加の重みを受ける。
各slot6byte、1例12byte、batch16の追加分母192byte。train全450例の対象は5400byte。

構造・parameter増分0、両モデル42423parameters。Bは学習済みAからでなく共通初期値から開始。
seed7/600更新/batch16/Adam .003/clip1/CPU1thread、同sampling schedule。
train450だけでtensorizer/語彙fit、validation150のみ評価、test未使用。重み探索はしない。
任意C（専用slot head）は事前に見送り、損失のみの対照とした。
共通初期state SHA256:
`f07d83adec88a37040886e1374800b17b56b7d2f3f49cea3f1b801c4b61b4b70`。
共通sampling SHA256:
`6c3ae94191450c6c60d8ea3375711f56b241475b5da890940855f999d7e95e66`。
A/Bとも正味変更parameter要素30416。学習秒A43.256/B65.334は各1回の実測であり、
バックグラウンド負荷を統制した速度比較ではない。
B最終sampled batchはbase .728921、slot .251303、total .980224、追加対象192byte。
traceは10更新ごとと初回/最終を保存。最後のbatch損失はvalidation平均ではない。

### 通常生成とoracle診断

各条件はBOSから自己生成。正解conceptは明示oracleだけへ渡す。LMは正解履歴下の10005token平均。
slotは既存target文型採点、各分母150、parseできない行も失敗として残す。

| arm / concept | LM | 生成種類 | EOS /150 | UTF-8 /150 | parse /150 | 人物 /150 | 時点 /150 | 全frame /150 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A / predicted | 0.168318838 | 9 | 150 | 150 | 109 | 20 | 22 | 0 |
| A / participant_oracle | 0.171052042 | 16 | 150 | 150 | 110 | 15 | 23 | 0 |
| A / time_oracle | 0.171685302 | 19 | 150 | 150 | 98 | 7 | 22 | 0 |
| A / full_oracle | 0.174590945 | 23 | 144 | 144 | 99 | 6 | 21 | 0 |
| B / predicted | 0.172247004 | 6 | 149 | 149 | 125 | 26 | 25 | 0 |
| B / participant_oracle | 0.174796121 | 11 | 147 | 147 | 127 | 27 | 23 | 0 |
| B / time_oracle | 0.174279768 | 11 | 145 | 145 | 125 | 24 | 9 | 0 |
| B / full_oracle | 0.178736054 | 11 | 141 | 141 | 135 | 27 | 9 | 0 |

全条件の特殊token行0、完全一致文0/150。条件付きparse分母と全項目の詳細はreportに保持。

### 正解履歴のslot byte / concept head

以下のbyte値はteacher-forcedであり、自由生成品質と同一ではない。各slotは900byte/150例。

| arm / concept | slot | NLL | 正解確率 | rank | argmax /900 |
|---|---|---:|---:|---:|---:|
| A / predicted | participant | 0.708228 | 0.749411 | 1.435556 | 691 |
| A / predicted | time | 0.308307 | 0.804247 | 1.173333 | 780 |
| A / participant_oracle | participant | 0.695442 | 0.750715 | 1.456667 | 696 |
| A / participant_oracle | time | 0.311334 | 0.803454 | 1.174444 | 779 |
| A / time_oracle | participant | 0.700678 | 0.750282 | 1.410000 | 711 |
| A / time_oracle | time | 0.308493 | 0.803817 | 1.156667 | 780 |
| A / full_oracle | participant | 0.686247 | 0.751157 | 1.430000 | 702 |
| A / full_oracle | time | 0.312590 | 0.803110 | 1.170000 | 774 |
| B / predicted | participant | 0.630132 | 0.778445 | 1.367778 | 719 |
| B / predicted | time | 0.289612 | 0.816971 | 1.170000 | 780 |
| B / participant_oracle | participant | 0.627049 | 0.778731 | 1.356667 | 729 |
| B / participant_oracle | time | 0.290176 | 0.816697 | 1.174444 | 776 |
| B / time_oracle | participant | 0.630727 | 0.778346 | 1.380000 | 708 |
| B / time_oracle | time | 0.289869 | 0.816576 | 1.190000 | 762 |
| B / full_oracle | participant | 0.630297 | 0.776873 | 1.376667 | 711 |
| B / full_oracle | time | 0.290704 | 0.816096 | 1.190000 | 762 |

concept headはdecoder byteと別採点。class support/confusion/soft確率/balanced値も保存。

| field | A正解 /150 | B正解 /150 | A balanced | B balanced |
|---|---:|---:|---:|---:|
| event | 150 | 150 | 1.000000 | 1.000000 |
| operators | 97 | 99 | 0.512963 | 0.524074 |
| agent | 150 | 150 | 1.000000 | 1.000000 |
| participant | 41 | 44 | 0.273333 | 0.293333 |
| time | 43 | 44 | 0.286667 | 0.293333 |
| location | 150 | 150 | 1.000000 | 1.000000 |
| repeat_marked | 140 | 141 | 0.666667 | 0.700000 |

### 再現・保存物

Aは過去Issue #12のper-step state、通常生成全150例・LM・slot採点と完全一致。
A/Bとも保存再読込のstate/concept/validation logits/greedy全150例が厳密一致。
既存教材・scorer・過去reportは変更していない。

- 参照report SHA256: `3b08a5982c89fd63b4c5887f03ee7c1276a9ce51b3e9f8b072e830c448b791d2`
- 今回 `codex/work_output/issue18-seed7-v1/report.json` SHA256: `9db04ef46231cfb79439f1dc4f90883567b3f80433a7b8fc295cb2f2fcc1a382`
- A checkpoint（過去と同hash）: `0bb375a82b62baac367d76b6ecf344835de013ca0cb61e1217979937eac42057`
- B checkpoint: `123fb938ba20ba4aec3c3557b2a49e704f5e62ebe5397c76092b8a0be66c7776`
- A state: `92b21ac06055b99733e1114b6ea0cd57de316d9b4ec2c19ebb565b54c76afb58`
- B state: `046ff50de199f053cc7b420fea67d1902974c08c9458056753bb141fcbd71699`

checkpointと詳細reportはignoredローカル保存物でありGitには含めない。
再実行には同hashの過去report/checkpointが必要。別出力先を使用し上書きしない。

### 解釈・未実装・次候補

slot追加損失で通常人物byte NLL .708228→.630132、時点 .308307→.289612。
自由生成人物20→26、時点22→25/150だが、全frame0のまま。全体LM .168319→.172247、
EOS/有効UTF-8は150→149、生成種類9→6。headも人物41→44、時点43→44と変化するため、
改善をdecoder単独の効果とは断定しない。parse率の変化もslot正解数に影響する。
Bで時点oracleは時点25→9/150へ悪化し、gold条件での回復も示せていない。

一部byte/slot指標の改善と全体劣化を併記し、有効な意味保持方式を確立したとはしない。
単一seed・教材共有template・one-hot oracle分布差・未見組合せの限界を残す。
追加の意味層やデモ辞書の有効性を示す比較ではない。
次候補は事前固定した専用slot head対照（未実装C）や、概念抽出とdecoderの寄与を分ける対照。
予算/重みの事後探索やtest評価は行わない。

### コマンド・検証・委譲

worktreeルートで実行:

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_slot_objective --baseline-report codex/work_output/issue12-seed7-v1/report.json --baseline-checkpoint codex/work_output/issue12-seed7-v1/per_step_additive.pt --out-dir codex/work_output/issue18-seed7-v1
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m norishio_lm.demo
.\.venv\Scripts\python.exe -m norishio_lm.encoder_demo
.\.venv\Scripts\python.exe codex/tools/build_context.py
.\.venv\Scripts\python.exe codex/tools/validate_okf.py
.\.venv\Scripts\python.exe codex/tools/check_knowledge.py --mode full
.\.venv\Scripts\python.exe codex/okf_mcp/tests/live_check.py --server codex/okf_mcp/server.py --root okf
git diff --check
```

A/B実験v1終了0。全pytest **315 passed, 2 subtests passed**、14.18秒、終了0。
knowledge84原本full=healthy、OKF8 files/6 concepts errors0/warnings0、
MCP実プロセス6 toolsとgit diff --checkも終了0。
既存zero-element警告1、両デモ終了0、encoder/実験の既存NumPy未導入警告。
Temp依存の全テストは許可付きで実行。12追加テストにはslot-onlyのGRU/lm_head非ゼロ勾配、
非slot logit勾配0、未来input/label非干渉、source境界、重み0の両mode既存学習一致を含む。

`gpt-5.6-luna`へslot loss/学習ループと境界テストを独立委譲。
親がA/Bハーネス、byte数加重の独立検査、per-step/label境界修正、統合検証を担当。
Luna最終読取レビューはblockerなし。人の検証印は追加せず、PRレビュー待ちまで進める。
定期worker再開、有料GPU、外部辞書/private logs、自動merge/closeは行わない。

## 過去の実装状態: Issue #16

`codex/issue-16`、PR #15 merge `631bf70e263218ac1531e75c05ae99e2fdffa811` から開始。
[事前計画](prefix-intervention.md)は `ba454e6` と `52f9495`、実装SHAは
`98d8cb57b36901fd3dbe5e427265b7438e8ad2b6`。測定前に境界・分母・比較条件を固定。
保存済みseed7/600 per-stepモデルを凍結し、CPU1thread・validation150のみで診断。
追加学習・語彙refit・test評価はない。

### 入力・採点境界と再現

人物/timeの各start-1、start、endの6境界 × predicted/人物gold/時点gold/全7goldの4条件。
UTF-8 byte位置であり、start-1は文字の途中にもなる。BOSを除く総token上限128は
与えたprefixも含む。生成にはBOS+reference[0:j]だけを渡し、位置j以降は自己生成tokenのみ。
正解continuationは生成APIの引数にない。採点は生成終了後に別経路で実行。
通常source-only性能とprefix oracleを混ぜない。

局所slot完全一致はauthored固定位置で採点。prefixがslot内/後なら全文slotの加点なし。
残りbyteだけを別採点し、完全に与えたslotはexpected/evaluatedとも0、NLL等はnull。
早期EOS後の未生成byteはexpected分母に残し、評価できたdecisionだけでNLL/rankを計算。
slot開始後継続は当該slotから文末EOSまでの一致であり、生成済みのslot前部分は要求しない。
全continuation一致はprefix以後すべてとEOSの一致。全row分母は各150。

Issue #14 report全体のJSON値一致、通常生成150例のempty-prefix token一致、重み不変を確認。
モデルstate SHA256: `92b21ac06055b99733e1114b6ea0cd57de316d9b4ec2c19ebb565b54c76afb58`。
checkpoint SHA256: `0bb375a82b62baac367d76b6ecf344835de013ca0cb61e1217979937eac42057`。
参照Issue #14 report SHA256: `a4d18bb7fab3d63a94d0d0e5b026b59956aae6db276241eba60b248e03621c92`。
参照Issue #12 report SHA256: `3b08a5982c89fd63b4c5887f03ee7c1276a9ce51b3e9f8b072e830c448b791d2`。
今回 `codex/work_output/issue16-seed7-v1/report.json` SHA256:
`2afc2f6c4d6c6078397bb273bc9e350799239098952583211945ef13d17df053`。
詳細例・byte値・token列はignored reportに保存。保存重み・reportはGitへ追加しないため、
再実行には同hashのローカル保存物が必要。

### 生成後だけの結果

以下のslot欄は正解数/eligible数。全rowは150、0/0は与え済みで評価対象外（成功ではない）。
通常BOS生成の同じ固定位置採点は人物25/150、時点27/150、全continuation一致0/150。
前段の文型parse限定採点20/22とは定義が違うため直接比較しない。

| prefix境界 / concept | 人物slot | 時点slot | 全continuation /150 | 人物以後 /150 | 時点以後 /150 |
|---|---:|---:|---:|---:|---:|
| participant_before/predicted | 0/150 | 0/0 | 0 | 0 | 0 |
| participant_before/participant_oracle | 0/150 | 0/0 | 0 | 0 | 0 |
| participant_before/time_oracle | 0/150 | 0/0 | 0 | 0 | 0 |
| participant_before/full_oracle | 0/150 | 0/0 | 0 | 0 | 0 |
| participant_start/predicted | 0/150 | 0/0 | 0 | 0 | 0 |
| participant_start/participant_oracle | 0/150 | 0/0 | 0 | 0 | 0 |
| participant_start/time_oracle | 0/150 | 0/0 | 0 | 0 | 0 |
| participant_start/full_oracle | 0/150 | 0/0 | 0 | 0 | 0 |
| participant_end/predicted | 0/0 | 0/0 | 44 | 0 | 0 |
| participant_end/participant_oracle | 0/0 | 0/0 | 36 | 0 | 0 |
| participant_end/time_oracle | 0/0 | 0/0 | 33 | 0 | 0 |
| participant_end/full_oracle | 0/0 | 0/0 | 48 | 0 | 0 |
| time_before/predicted | 25/150 | 27/150 | 0 | 7 | 0 |
| time_before/participant_oracle | 18/150 | 26/150 | 0 | 2 | 0 |
| time_before/time_oracle | 19/150 | 27/150 | 0 | 3 | 0 |
| time_before/full_oracle | 9/150 | 21/150 | 0 | 0 | 0 |
| time_start/predicted | 29/150 | 30/150 | 0 | 7 | 0 |
| time_start/participant_oracle | 21/150 | 29/150 | 0 | 2 | 0 |
| time_start/time_oracle | 19/150 | 30/150 | 0 | 3 | 0 |
| time_start/full_oracle | 12/150 | 24/150 | 0 | 0 | 0 |
| time_end/predicted | 0/150 | 0/0 | 0 | 0 | 0 |
| time_end/participant_oracle | 0/150 | 0/0 | 0 | 0 | 0 |
| time_end/time_oracle | 0/150 | 0/0 | 0 | 0 | 0 |
| time_end/full_oracle | 0/150 | 0/0 | 0 | 0 | 0 |

人物endの全continuation成功36〜48件は人物/timeを既に与えた後の文末生成であり、
人物・時点保持の成功には数えない。

### 切替後のbyte診断

各境界で対象slotを記載。endは対象slotを全て与えており、byte評価なし。
各start/before対象slotはexpected900byte、評価済み900byte、coverage1（各150例）。
以下は自己履歴での値であり、Issue #14の全正解履歴teacher-forced指標とは別。

| 条件 | 対象 | 正解byte /900 | NLL | 正解確率 | rank |
|---|---|---:|---:|---:|---:|
| participant_before/predicted | participant | 223 | 5.403000 | 0.242416 | 23.688889 |
| participant_before/participant_oracle | participant | 204 | 5.485039 | 0.223010 | 25.228889 |
| participant_before/time_oracle | participant | 192 | 5.765840 | 0.191414 | 27.725556 |
| participant_before/full_oracle | participant | 186 | 5.766433 | 0.195622 | 28.263333 |
| participant_start/predicted | participant | 223 | 5.403000 | 0.242416 | 23.688889 |
| participant_start/participant_oracle | participant | 204 | 5.485039 | 0.223010 | 25.227778 |
| participant_start/time_oracle | participant | 192 | 5.765840 | 0.191414 | 27.725556 |
| participant_start/full_oracle | participant | 186 | 5.766433 | 0.195622 | 28.263333 |
| time_before/predicted | time | 351 | 4.097434 | 0.339882 | 15.872222 |
| time_before/participant_oracle | time | 348 | 4.112487 | 0.336845 | 15.895556 |
| time_before/time_oracle | time | 357 | 4.056763 | 0.343061 | 15.787778 |
| time_before/full_oracle | time | 330 | 4.187900 | 0.323452 | 15.313333 |
| time_start/predicted | time | 372 | 3.722736 | 0.356497 | 13.106667 |
| time_start/participant_oracle | time | 369 | 3.738307 | 0.353357 | 13.130000 |
| time_start/time_oracle | time | 378 | 3.681650 | 0.359663 | 12.995556 |
| time_start/full_oracle | time | 351 | 3.812298 | 0.339756 | 12.520000 |

最初の生成位置における正解byte確率を既存teacher-forced reportと独立照合し、
最大差は人物4.53e-7、時点3.46e-7（CPU batch形状差による浮動小数点差）。
通常生成の全tokenおよび前段reportの値の完全一致とは区別する。

### 観測の限界と次候補

人物直前prefixで人物slotは全concept条件0/150、時点直前ではpredictedとtime-oracleが30/150。
正解履歴だけで人物/timeが回復する説明、gold概念との組合せで大幅回復する説明は支持されない。
ただしdecoderのslot表現・出力head・学習目的のどれが原因かは分離していない。
正解時点を与えた後も人物を外す傾向は、教材の未見組合せに対する対応付け失敗の候補となるが、
本診断だけで原因とは断定しない。one-hot oracleの分布差、単一seed、固定位置採点の限界を保持。

任意のcounterfactual prefixは事前計画で未実施。等byte長でも文型位置や内容が一致せず、
一部の時点前prefixは同文になるため、このrunでは長さ効果と内容効果を分離しない。
次候補は長さと文型位置を揃えたprefix対照、またはdecoderのslot出力と組合せ学習の事前固定対照。
結果を見た条件追加・境界追加・test tuningは行わない。

### コマンド・検証・委譲

worktreeルートで実行:

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_prefix_experiment --checkpoint codex/work_output/issue12-seed7-v1/per_step_additive.pt --baseline-report codex/work_output/issue12-seed7-v1/report.json --slot-report codex/work_output/issue14-seed7-v2/report.json --out-dir codex/work_output/issue16-seed7-v1
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m norishio_lm.demo
.\.venv\Scripts\python.exe -m norishio_lm.encoder_demo
.\.venv\Scripts\python.exe codex/tools/build_context.py
.\.venv\Scripts\python.exe codex/tools/validate_okf.py
.\.venv\Scripts\python.exe codex/tools/check_knowledge.py --mode full
.\.venv\Scripts\python.exe codex/okf_mcp/tests/live_check.py --server codex/okf_mcp/server.py --root okf
git diff --check
```

診断v1終了0。全pytest **303 passed, 2 subtests passed**、12.80秒、終了0。
knowledge78原本full=healthy、OKF8 files/6 concepts errors0/warnings0、
MCP実プロセス6 tools検証とgit diff --checkも終了0。
既存zero-element tensor警告1。両デモ終了0、encoderと診断は既存NumPy未導入警告。
Temp依存テストは許可付きで実行し、skipを成功件数に含めていない。

`gpt-5.6-luna`へprefix生成と局所採点を独立委譲。生成担当の初回は誤ったrootへ配置・
別venvでtorchなしskipとなり、親が発見して指定worktreeへ移動・誤配置除去を確認。
親レビューでper-step加算漏れ、slot前byte混入採点、早期EOSの添字境界を修正してから測定。
親が実教材embedding入力spy、与え済みslot除外、早期EOS、slot前誤りの回帰テストを追加。
指定環境の21追加テストを含め統合検証し、Lunaの最終読取監査もblockerなし。
人の検証印・自動merge/close・定期worker再開・有料computeは追加しない。

## 過去の実装状態: Issue #14

`codex/issue-14`、PR #13マージ `e10edb0c59a6f8228b3b7a7b61b937cd5706e60e` から開始。
事前計画 `cc76f56d954b0a13e361571b810f5515aa096dcb`、実装
`9ff310b9bde77522df3b0550a3d5c8192a884180`。[診断計画](slot-retention.md)の必須項目を実装。
Issue #12のper-step checkpointをCPUで凍結再利用し、追加学習・語彙refit・test評価はない。

### 診断の契約と再現

人物のみ・時点のみ・両方・全7項目のgold oracleと通常予測を比較。他項目の確率は変更しない。
正解文のteacher-forced診断とBOSからの自己履歴生成を別記録する。前者は正解履歴oracleである。
教材テンプレートを展開して各slotのUTF-8 byte半開区間を算出し、文字列検索は使用しない。
位置jのlogitはbyte jを予測し、+4はtoken IDだけに適用。EOSはslot採点から除外。
各byteの正解確率・NLL・rank・argmax・logit L1・KL(base||介入)を保存。
rankは厳密に大きいlogitの数+1、集約はbyte数加重。各slotは150行/900byte。

保存済み通常条件のLM・全生成例・slot採点とconcept指標が厳密一致（JSON表現で比較）。
診断前後のstate SHA256は同一:
`92b21ac06055b99733e1114b6ea0cd57de316d9b4ec2c19ebb565b54c76afb58`。
checkpoint SHA256:
`0bb375a82b62baac367d76b6ecf344835de013ca0cb61e1217979937eac42057`。
参照したIssue #12 report SHA256:
`3b08a5982c89fd63b4c5887f03ee7c1276a9ce51b3e9f8b072e830c448b791d2`。
今回のignored出力 `codex/work_output/issue14-seed7-v2/report.json` SHA256:
`a4d18bb7fab3d63a94d0d0e5b026b59956aae6db276241eba60b248e03621c92`。
checkpointと詳細reportは学習artifactとしてGitに追加しない。チェックアウトだけでは保存済み重みがないため、
再実行には上記hashのローカル保存物、またはIssue #12手順による再作成とhash/結果照合が必要。

### concept headと自由生成

headの人物は41/150、時点43/150（各5class support30、balanced accuracy .273333/.286667）。
他5項目はevent150、operators97、agent150、location150、repeat140（各分母150）。
balanced accuracyは順に1/.512963/1/1/.666667。全7項目micro771/1050。
クラス別support・confusion matrix・soft確率・entropy・確率分散はreportに保存。

| 条件 | LM (10005 token) | 生成種類 | parse /150 | 人物 /150 | 時点 /150 | 全項目 /150 |
|---|---:|---:|---:|---:|---:|---:|
| predicted | 0.168318838 | 9 | 109 | 20 | 22 | 0 |
| participant_oracle | 0.171052042 | 16 | 110 | 15 | 23 | 0 |
| time_oracle | 0.171685302 | 19 | 98 | 7 | 22 | 0 |
| both_oracle | 0.175030359 | 18 | 111 | 3 | 23 | 0 |
| full_oracle | 0.174590945 | 23 | 99 | 6 | 21 | 0 |

全7oracleのみEOS/有効UTF-8が144/150、他4条件は150/150。特殊token行は全条件0。
文型外も全対象分母に保持し、parse成功例だけの条件付き値もreportへ分離保存する。
この機械採点は教材target文型限定であり、一般日本語の意味判定ではない。

### 正解履歴下のslot byte診断

| 条件 | slot | NLL | 正解確率 | rank | argmax /900 | logit L1 | KL |
|---|---|---:|---:|---:|---:|---:|---:|
| predicted | participant | 0.708228 | 0.749411 | 1.435556 | 691 | 0.000000 | 0.000000 |
| predicted | time | 0.308307 | 0.804247 | 1.173333 | 780 | 0.000000 | 0.000000 |
| participant_oracle | participant | 0.695442 | 0.750715 | 1.456667 | 696 | 0.046269 | 0.000872 |
| participant_oracle | time | 0.311334 | 0.803454 | 1.174444 | 779 | 0.037755 | 0.000251 |
| time_oracle | participant | 0.700678 | 0.750282 | 1.410000 | 711 | 0.052457 | 0.000620 |
| time_oracle | time | 0.308493 | 0.803817 | 1.156667 | 780 | 0.027805 | 0.000232 |
| both_oracle | participant | 0.686627 | 0.751454 | 1.433333 | 696 | 0.065046 | 0.001679 |
| both_oracle | time | 0.311787 | 0.803076 | 1.165556 | 778 | 0.049432 | 0.000682 |
| full_oracle | participant | 0.686247 | 0.751157 | 1.430000 | 702 | 0.077472 | 0.002314 |
| full_oracle | time | 0.312590 | 0.803110 | 1.170000 | 774 | 0.058193 | 0.001752 |

人物oracleで人物NLLは.708228→.695442、byte正解691→696/900だが自由生成の人物は20→15/150。
時点oracleで時点byte正解は780/900のまま、自由生成時点も22/150のまま。
正解conceptの差し替えだけで生成保持が回復するという説明は、この条件では支持されない。
ただしone-hot oracleは学習時soft分布と異なり、head誤りだけ/decoderだけを唯一原因とは断定できない。
高いteacher-forced byte正解率は直前の正解byteや共有byteを利用でき、語やslot全体の正解とは異なる。
任意のgold-prefix介入は事前計画どおり未実施。自己履歴誤りの伝播は因果的に分離していない。
単一seed・教材共有テンプレート・validation限定で、一般化や各意味層の有効性は未実証。
次候補はslot直前prefix介入の事前固定比較、または学習時soft分布を保つslot条件付けの対照。

### コマンド・検証・委譲

作業worktreeで実行:

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_slot_retention --checkpoint codex/work_output/issue12-seed7-v1/per_step_additive.pt --baseline-report codex/work_output/issue12-seed7-v1/report.json --out-dir codex/work_output/issue14-seed7-v2
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m norishio_lm.demo
.\.venv\Scripts\python.exe -m norishio_lm.encoder_demo
.\.venv\Scripts\python.exe codex/tools/build_context.py
.\.venv\Scripts\python.exe codex/tools/validate_okf.py
.\.venv\Scripts\python.exe codex/tools/check_knowledge.py --mode full
.\.venv\Scripts\python.exe codex/okf_mcp/tests/live_check.py --server codex/okf_mcp/server.py --root okf
git diff --check
```

診断v2は終了0、baseline_replay_equal/model_unchangedともtrue。
初回v1はtupleとJSON配列の型差300箇所で再現照合が停止、成功扱いしない。
数値と生成の差がないことを確認し、保存JSON形式で全値を比較する修正後に別出力先v2で成功。
追加17テストを含む全pytest **282 passed, 2 subtests passed**、11.76秒、終了0。
知識索引71原本、full=healthy、OKF8 files/6 concepts・errors0/warnings0、
MCP実プロセス6 tools検証成功、git diff --check成功。いずれも終了0。
既存のzero-element tensor警告1件。両デモ終了0（encoderは既存NumPy未導入警告）。
Tempを使う初回の一部テストはWinError5でsetup errorとなり、許可付き全pytestで再検証済み。

`gpt-5.6-luna`にテンプレート位置計算とbyte採点の独立実装・テストを分担。
親が位置/ID offsetの誤りを発見して修正を指示し、実教材のラベル復号テストを追加。
親がハーネス・oracle境界・未来token非干渉・保存物照合を統合検証。
Lunaの最終読取監査はblockerなし（監査テストのTemp setup errorは親の全検証と区別）。
人の検証印、学習予算追加、自動merge/close、定期worker再開は行わない。

## 過去の実装状態: Issue #12

`codex/issue-12`、PR #11マージ `604f61774ce0bb4fdfb198d972817b24b71821ee` から開始。
事前計画コミット `79a07d0de853bb195f193eb37538c3576ba3ded0`、
実装 `4ca55115debd33cdc53412c2ada7094c124aa381`。
[比較計画](step-conditioning.md)の必須A/Bだけを実装し、gated方式は追加していない。

### 構造・容量・学習条件

A=initial_onlyは従来通りconcept投影のtanhをh0だけに使用。
B=per_step_additiveは同じh0に加えて、同じ投影ベクトルを各時点のtoken embeddingへ加算。
投影の共有により追加parameter0、hidden32、concept33、両モデル総数42423。
全parameter key/valueを同一にして学習開始し、元モデルや既定initial_onlyの初期化順も保持。
既存60step既定値は変更していない。checkpointにmodeを保存し、modeのない旧保存物は
initial_onlyとして読込。未知mode・必須config欠落を拒否する。

教材v1 train450/validation150のみ、test未評価。seed7、600更新、batch16、Adam .003、
clip1、4損失重み1、CPU1thread、同一sampling順。追加seedや結果を見た予算変更なし。
共通初期state SHA256:
`f07d83adec88a37040886e1374800b17b56b7d2f3f49cea3f1b801c4b61b4b70`。
共通sampling SHA256:
`6c3ae94191450c6c60d8ea3375711f56b241475b5da890940855f999d7e95e66`。
初期/最終の正味変更要素数はA30415、B30416、学習秒はA43.813/B43.794（各1回）。

### validationの結果

| 方式・介入 | LM (10005 token) | 生成列種類 | EOS / UTF-8有効 | 文型parse数 | 全slot一致 |
|---|---:|---:|---:|---:|---:|
| A predicted | .179737919 | 1 | 150/150 / 150/150 | 150/150 | 0/150 |
| A train-mean | .180703698 | 1 | 150/150 / 150/150 | 150/150 | 0/150 |
| A seed17 permutation | .181523176 | 1 | 150/150 / 150/150 | 150/150 | 0/150 |
| A gold oracle | .179309284 | 1 | 150/150 / 150/150 | 150/150 | 0/150 |
| B predicted | .168318838 | 9 | 150/150 / 150/150 | 109/150 | 0/150 |
| B train-mean | .234352527 | 1 | 150/150 / 150/150 | 150/150 | 0/150 |
| B seed17 permutation | .286683485 | 9 | 150/150 / 150/150 | 109/150 | 0/150 |
| B gold oracle | .174590945 | 23 | 144/150 / 144/150 | 99/150 | 0/150 |

全8条件の参照完全一致0、特殊token行0。Aの4条件は従前600診断の値を再現し、
全て「私は、来月先輩と会うことを望んでいる。」を出力する。
Bは通常9種類へ分化し、平均化/対応置換でLMが大きく悪化する。
これはこの固定条件でconceptに依存する挙動が増えた観測で、意味の正確さや汎化の証明ではない。
goldは診断専用のone-hot介入であり、通常性能に含めない。B goldでは6件がcap128終了し、
UTF-8も不正。oracleは学習時のsoft分布との差があり、数値改善を保証しない。

機械slot採点はseedの**target文型への全文一致だけ**を認める。文型が一致すれば許可済み
person/timeを抽出し、文型に対応するevent/agent/location/repeat/operatorsを取り出す。
source言い換え、部分文字列、文型外の自然な文はこの採点器では未解釈。
無効UTF-8・特殊token・非EOS・曖昧な文型もparse不可。これらは全gold観測分母に残し、
一致成功には数えない。parse可能例に限るconditional_accuracyとevaluable_countも別記。
全frameのevaluable_countはparse可能かつ全7gold既知の件数で、fullframe_countとは区別する。
「parse不可」は自然言語として誤りと判断したことを意味しない。

| 項目正答数（各分母150） | A predicted | B predicted | B permuted | B gold oracle |
|---|---:|---:|---:|---:|
| event | 90 | 109 | 50 | 99 |
| operators | 45 | 60 | 28 | 66 |
| agent | 120 | 109 | 86 | 99 |
| participant | 30 | 20 | 14 | 6 |
| time | 30 | 22 | 23 | 21 |
| location | 120 | 109 | 86 | 99 |
| repeat_marked | 135 | 94 | 96 | 84 |

A predictedの合計570/1050に対し、B predictedは523/1050で、全対象の機械slot成功数は増えない。
Bの41件は文型外。event/operatorsの一致は増えるが人物・時点の保持は改善していない。
concept head自身のmicroはA/Bとも771/1050、balanced macroはA .676667/B .677090。

### 位置別感度

各方式のpredicted自由生成の履歴を固定し、conceptだけをmean/permuted/goldへ変更。
L1はvocab260上のlogit絶対差平均、KLはbaseline||intervention、自然対数。
生成に実際に使ったdecision位置だけを計測し、終了した行は以後の分母に含めない。
cap128行の入力はBOS+先頭127生成tokenで128 decision。未生成129番目は測らない。
全位置と分母をレポートに保存。方式間では自分の基準履歴が異なり、履歴まで同一な対照ではない。

| seed17 permutation感度 | A位置0 | A位置50 | B位置0 | B位置50 |
|---|---:|---:|---:|---:|
| 対象行 | 150 | 150 | 150 | 150 |
| logit L1 | .263806993 | 7.811875e-7 | .093354513 | .263128271 |
| KL | .000734770 | 1.565548e-13 | .000759559 | .374661814 |
| argmax変化率 | 0 | 0 | 0 | .193333333 |

Aは58位置、Bは最大76位置（最後は24行）。Aの後半は差が浮動小数点誤差に近い水準へ
減衰する一方、Bでは後半にもconditioning差が残る。初期状態だけの条件が履歴に埋もれる
仮説を支持する限定的な構造対照であり、唯一の原因確定や一般的な意味理解とは呼ばない。

### 検証と成果物

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_step_experiment --out-dir codex/work_output/issue12-seed7-v1
.\.venv\Scripts\python.exe -m pytest tests/test_step_conditioning.py tests/test_toy_slots.py tests/test_toy_step_experiment.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m norishio_lm.demo
.\.venv\Scripts\python.exe -m norishio_lm.encoder_demo
.\.venv\Scripts\python.exe codex/tools/build_context.py
.\.venv\Scripts\python.exe codex/tools/validate_okf.py
.\.venv\Scripts\python.exe codex/tools/check_knowledge.py --mode full
.\.venv\Scripts\python.exe codex/okf_mcp/tests/live_check.py --server codex/okf_mcp/server.py --root okf
```

全pytest **265 passed, 2 subtests passed**、skipなし、11.99秒、既存空tensor警告1。
focused23件成功後にcap履歴テスト1件追加し全検査を実施。両デモと評価CLI終了0。
CLIのNumPy未導入warningは残るがNumPy変換は使わない。Temp利用可能環境で全検査。
索引64 sources、OKF 8 files/6 concepts errors0/warnings0、knowledge full healthy、
MCP実プロセス6 toolsでok true。
A/B両checkpointでvalidation150件のconcept/logits/greedy/stateが再読込前後で完全一致。
旧modeなしcheckpoint、未来tokenの非干渉、per-stepのgold/reference非漏洩、
slotの全文文型・曖昧性・欠損・未解釈分母、位置別感度のpadding除外も検証。

ignored出力 `codex/work_output/issue12-seed7-v1/report.json` SHA256:
`3b08a5982c89fd63b4c5887f03ee7c1276a9ce51b3e9f8b072e830c448b791d2`。
initial_only.pt: `fe69ff545559a8390fc912afa8915fa23fae3c9a9fd792981ae7416f5565b3d7`。
per_step_additive.pt: `0bb375a82b62baac367d76b6ecf344835de013ca0cb61e1217979937eac42057`。
過去の教材・split・結果・保存物は保持し、重みと全生成レポートはGitへ追加しない。

Luna (`gpt-5.6-luna`) にdecoder/保存とslot採点を独立委譲。親はtarget/source文型の誤り、
欠測/条件付き分母、config必須key検査と不足テストを修正し、統合/全検証/測定を担当。
独立監査のcap最終token指摘は「実際に生成したdecision数を測る」契約上は変更不要と判断し、
明文化と回帰テストで境界を固定した。委譲報告だけを成功根拠にしていない。

残課題: 人物・時点保持、文型外出力、gold介入の崩壊、全slot一致0、複数seedの再現。
次候補は別途予算を事前固定し、participant/time条件の保持と生成履歴中の誤り伝播を
分離する診断。今回の結果だけでper-stepを既定方式へ昇格せず、PRレビュー待ちまでとする。

## 最新の実装状態: Issue #9

ブランチ `codex/issue-9`。PR #10マージ後のmaster
`dc44de6146a389c2cd9048cb93dbbf450a377feb` から開始。
測定前の計画コミット `62a1928c0f3190d4914ad760cef7614fea3a1264`、
実装 `bd859c406a55e4a027705598e73aa9cbddcbfbd0`。
[境界診断計画](collapse-diagnosis.md)に予算・採点・限界を先に固定した。

### 固定条件と実装

教材v1とsplitはそのまま、train450/validation150のみ使用、test未評価。
通常Cをseed7、600更新、batch16、Adam .003、clip1、全4損失重み1、CPU1threadで学習。
600は診断予算であり、既存60step既定値を変更していない。
学習後モデル全体をfreezeし、source context/textだけから32次元fused latentを抽出。
probeは7個の線形headのみ（1089パラメータ）、train平均/標準偏差で正規化し、
seed7、full-batch Adam .01、300更新。validationは正規化fitや学習へ渡さない。
probe前後で元モデルのstate SHAが完全一致、latentはdetach、encoder更新なし。
source用APIはgold/target等の余分なキーを拒否する。

### source→encoder→conceptの観測

| 採点 | micro正答/1050 | 項目balancedのmacro平均 | 全7項目一致/150 |
|---|---:|---:|---:|
| train多数派 | 570 | 0.319047619 | 0 |
| 凍結encoder上の線形probe | 860 | 0.808994709 | 12 |
| 既存concept head | 771 | 0.676666667 | 1 |
| concept対応をseed17で置換 | 485 | 0.307645503 | 0 |

| 項目 | 多数派accuracy | probe accuracy | head accuracy | head entropy (nats) | 置換時の平均L1 |
|---|---:|---:|---:|---:|---:|
| event | .600000 | 1.000000 | 1.000000 | .026718 | 1.198466 |
| operators | .300000 | .953333 | .653333 | 1.107510 | 1.057973 |
| agent | .800000 | 1.000000 | 1.000000 | .150900 | .701878 |
| participant | .200000 | .400000 | .260000 | 1.600992 | .156126 |
| time | .200000 | .400000 | .293333 | 1.598172 | .165639 |
| location | .800000 | 1.000000 | 1.000000 | .012255 | .692435 |
| repeat_marked | .900000 | .980000 | .933333 | .273873 | .346966 |

全項目のsupportは150、missing/unseenはともに0。confusionはgold行×prediction列
（列0はunknown）として全7項目をJSONへ保存。各クラスsupport、確率の入力間分散も保存。
participant/timeは各5クラス×30でほぼ一様な予測分布が残る。
headのoperators正答/対象数は NOT→WANT 0/15、PLAN→NOT 10/15、PLAN 15/30、
POSSIBLE 0/15、WANT→NOT 28/30、WANT 45/45。NOT→WANTとPOSSIBLEは全件WANTへ誤分類。
probeでは順に15/15、15/15、29/30、10/15、29/30、45/45。
作用域を読む情報はencoderから線形に取り出せるが、既存headでは十分使われていない。
これは当該教材・予算での解読可能性であり、一般的な意味理解を証明しない。

### concept→decoderの観測

| 条件 | validation LM (10005 token) | 異なるconditioning数 | 異なる生成列数 |
|---|---:|---:|---:|
| predicted | .179737919 | 150 | 1 |
| train-source平均 | .180703698 | 1 | 1 |
| seed17対応置換 | .181523176 | 150 | 1 |
| authored gold oracle | .179309284 | 50 | 1 |

goldは既知7項目のone-hotを渡す診断専用のoracleで、通常のsource-only性能に混ぜない。
全4条件で150/150が有効UTF-8かつEOS終了、参照完全一致0/150、特殊token行0。
全て同じ「私は、来月先輩と会うことを望んでいる。」となった。
全例のsource/予測concept/decoder介入値/生成/参照を分離し、raw token、特殊tokenを除いた
byte列、UTF-8妥当性、停止理由を保存。参照文はloss/scoringだけに使い生成器へ渡さない。
gold/参照だけを変えても通常生成が不変で、oracleは完全既知概念以外を拒否するテストを追加。

encoderのexact unique rowsは150、L2距離min .080064 / mean 3.213024 / max 6.066077。
soft conceptも150通り、距離min .013902 / mean 1.372293 / max 2.644082。
各11175ペア中、距離1e-6以下は0。concept argmaxの7項目パターンは34通り。
近接ペア比はクラスタ数や意味同一性ではなく、尺度に依存する数値統計である。
入力表現が完全に同一化したという説明は当てはまらず、出力列になる段階で多様性が失われる。
goldでも同一生成なので、concept予測の改善だけで解決するとは言えない。
ただしgoldのone-hotは学習時のsoft分布から外れ得るため、decoder側の原因を確定したものではない。

### 再現性・検証・保存

通常モデル学習43.353秒、probe17.504秒。probeのtrain損失は更新前1.650924444、
最終更新後.315056413。各1回のみで追加seedや結果を見た予算変更は行っていない。
元の600更新診断の通常C LM .179737919を再現した。
probe前後モデルstate SHA256:
`1e5360bdbece1cc667b02b722e30eb58834763724a1b1b909e9157ec6d78bd78`。
学習順SHA256: `6c3ae94191450c6c60d8ea3375711f56b241475b5da890940855f999d7e95e66`。
checkpoint往復はvalidation150件のlogits/concepts/greedy/state全て完全一致。

```powershell
.\.venv\Scripts\python.exe -m norishio_lm.toy_collapse --out-dir codex/work_output/issue9-seed7-v1
.\.venv\Scripts\python.exe -m pytest tests/test_toy_collapse.py tests/test_toy_probe.py tests/test_collapse_metrics.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m norishio_lm.demo
.\.venv\Scripts\python.exe -m norishio_lm.encoder_demo
.\.venv\Scripts\python.exe codex/tools/build_context.py
.\.venv\Scripts\python.exe codex/tools/validate_okf.py
.\.venv\Scripts\python.exe codex/tools/check_knowledge.py --mode full
.\.venv\Scripts\python.exe codex/okf_mcp/tests/live_check.py --server codex/okf_mcp/server.py --root okf
```

全pytest **241 passed, 2 subtests passed**、skipなし、11.54秒、既存の空tensor警告1。
追加21件のfocusedテストも成功。評価CLIと両デモは終了0。
CLIのNumPy未導入warningは残るがNumPy変換は使わない。全pytestはTemp権限を確保して実行。
索引58 sources、OKF 8 files/6 conceptsでerrors0/warnings0、knowledge full healthy、
MCP実プロセス検査6 toolsでok true。

レポート `codex/work_output/issue9-seed7-v1/report.json` SHA256:
`8484f115c55dad7d9d6e59fa73ed208b2d15d706f8ff2f63355081b8f4185445`。
`normal600.pt` SHA256: `9e07b95fa7b096991f088061a3a38aa841b097ef8183eb8fcd7f0e1be0552bdb`。
いずれもignoredの新規出力で、既存の結果・教材・重みを上書きしていない。

Luna (`gpt-5.6-luna`) にprobeとcollapse統計を独立委譲。親が最終更新後の損失、
L1定義、欠測と0の区別、optional torch依存、strict C境界を確認・修正し、統合/測定/全検証。
Lunaによる親ハーネスの独立監査でもleakage・予算にblocking指摘なし。人の検証印ではない。

残課題: participant/timeの識別、headの否定作用域、decoderの条件依存生成、複数seedでの再現。
次候補は、条件を初期状態だけへ入れる現在のdecoderと各時点へ入れる対照を、予算・容量・
評価基準を事前固定して比較すること。今回は構造変更や追加学習をせず、PRレビュー待ちまで。

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

## Issue #34 benchmark v2 architecture tournament（2026-09-13）

Benchmark freeze `19ce7145495a47b40a178255272050caa615dc79` と tournament
freeze `38e363e1b4e836b890f1c852f6d0a45fe3119296` の後に、6 arm × 3 seedの
CPU実行系を実装した。共通source encoder/causal byte GRU、A_G0/A_G1/B/C/D/E、
Bのprefix-only FSM、固定schedule、checkpoint認証、terminal record、development
集計、最終holdoutの排他的な一回限りgateを含む。手書きbenchmark fixtureは
学習済みモデルの成果ではなく、この小規模tournamentから一般LLM性能を主張しない。

正式実行前の統合検証は benchmark-v2 focused **76 passed**、全体 **463 passed,
2 subtests passed**。既知のzero-element tensor warningが1件ある。NumPyは未導入で、
torch import時のoptional NumPy warningはfocused実行でのみ確認した。全18 runの
parameter preflightは29,272〜29,848でfreeze値と一致した。Lunaへモデル/FSM/runner
実装と独立監査を委譲し、親がraw-latent bypass、B projection、checkpoint認証、
report schemaを修正して統合した。

未完了は、実装コミット後の空の専用出力先で行う正式18 run、development結果の
記録、全terminal/checkpoint再検証、最終holdout一回評価、結果レビューである。
Lunaがrunner動作確認としてD/Eを各1回600 update実行したが、保存も採用もせず、
正式結果には含めない。

最初の正式実行はA_G0/A_G1の6 terminal完了後、BのFSMが各位置で全1,152候補を
再走査する性能問題を確認して中断した。6件を含む専用出力全体を削除し、結果は採用していない。
FSM規則を変えず、全候補から事前構築したprefix→next-tag表による参照へ置換し、候補走査との
等価性テストを追加した。正式18 runはこの修正commit後の新規出力先から再開する。

修正commit `565c2bf1f8b22435f21a74d5833dfeb4821d8174` からの正式development実行は
18 complete / 0 failed。checkpoint/terminal再認証とLuna独立集計が成功し、三seed平均の
順位は `B > C > E > A_G0 > A_G1 > D`。詳細値と解釈上の制約は
[development tournament result](results/benchmark-v2-development.md) に記録した。
final holdoutは未開封。自動承認レビューが一回限りの開封にはユーザーの明示承認が必要として
コマンドを拒否したため、承認後に同じ専用出力rootへ一度だけ実行する。

## AI 作業基盤の追加（2026-09-05）

ユーザー指定の ai-project-foundation から、開発用 OKF、読取専用 MCP、索引・整合性検査を取り込んだ。
研究モデル内部の概念表現とは分離している。元コミット、導入範囲、実行コマンド、権限エラーを含む検証結果は [AI 作業基盤](ai-foundation.md) を参照。
導入後はコンパイラと基盤を合わせて19テスト、2 subtests が成功し、デモ4例と MCP 実プロセス検証も成功した。
GitHub への push と MCP クライアント登録は未実施。
