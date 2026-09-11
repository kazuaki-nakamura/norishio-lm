# Joint-slot results (Issue #22)

測定日2026-09-11。事前計画 `e021418`、実装 `7f4ede9ead6fcad2d1a5899bf5850b538fe000d2`。
AはIssue20 C保存物、Bはtanh前の3線形投影加算。両者43073parameter、追加0。
関数クラスは数学的に同一。共通重み・列ブロックは完全一致、初期投影の最大絶対差は
`2.384185791015625e-07`。浮動小数演算順序と最適化経路の差を含み、構造の優越性を示す比較ではない。

seed7/600更新/batch16/Adam .003/clip1/CPU1thread。seed20拡張重みをコピー。
既存4損失各1と専用head CE各1、slot byte CEなし。B学習時間 44.084秒（速度比較ではない）。

## Pair support

行=participant `[先輩, 友人, 同僚, 知人, 隣人]`、列=time `[今日, 明日, 来月, 来週, 週末]`。
trainの15組×30例=450例、validationの5組×30例=150例。validationは全例unseen pair。
seen群0例のLM/head/byte評価と率はnull。未測定群を0点とは扱わない。分割は変更していない。

### Train450

| participant | 今日 | 明日 | 来月 | 来週 | 週末 |
|---|---|---|---|---|---|
| 先輩 | 30 | 0 | 30 | 30 | 0 |
| 友人 | 0 | 0 | 30 | 30 | 30 |
| 同僚 | 0 | 30 | 0 | 30 | 30 |
| 知人 | 30 | 30 | 30 | 0 | 0 |
| 隣人 | 30 | 30 | 0 | 0 | 30 |

### Validation150

| participant | 今日 | 明日 | 来月 | 来週 | 週末 |
|---|---|---|---|---|---|
| 先輩 | 0 | 0 | 0 | 0 | 30 |
| 友人 | 0 | 30 | 0 | 0 | 0 |
| 同僚 | 30 | 0 | 0 | 0 | 0 |
| 知人 | 0 | 0 | 0 | 30 | 0 |
| 隣人 | 0 | 0 | 30 | 0 | 0 |

## 自由生成とhead-only介入

以下はすべてall=unseen pairの150例。seen pairは0例。分母からparse失敗を除かない。
old33 conceptsは全条件で予測値固定。goldは指定専用headだけのoracle。通常生成はsource-only、BOSから自己履歴。
専用projectionのzero/permutationは対応する5次元入力のみ操作し、他headとold33は固定。
biasなしなのでzeroはその加算項だけを除去する。old33内の人物/time情報は残る。

| arm | condition | LM | 人物 | 時点 | 両方 | 全frame | EOS | UTF8 | 種類 | parse |
|---|---|---|---|---|---|---|---|---|---|---|
| A | predicted | 0.164552725 | 36 | 25 | 0 | 0 | 150 | 150 | 10 | 120 |
| A | participant_gold | 0.160055733 | 76 | 1 | 0 | 0 | 150 | 150 | 12 | 120 |
| A | participant_zero | 0.165062902 | 31 | 25 | 0 | 0 | 150 | 150 | 9 | 120 |
| A | participant_permuted | 0.166096184 | 31 | 22 | 0 | 0 | 150 | 150 | 13 | 120 |
| A | time_gold | 0.156997837 | 41 | 49 | 0 | 0 | 150 | 150 | 10 | 121 |
| A | time_zero | 0.165147438 | 39 | 24 | 0 | 0 | 150 | 150 | 8 | 120 |
| A | time_permuted | 0.164449002 | 37 | 28 | 0 | 0 | 150 | 150 | 12 | 120 |
| A | both_gold | 0.152015500 | 59 | 31 | 0 | 0 | 150 | 150 | 10 | 120 |
| B | predicted | 0.163130231 | 40 | 25 | 0 | 0 | 150 | 150 | 11 | 120 |
| B | participant_gold | 0.159805272 | 72 | 1 | 0 | 0 | 150 | 150 | 13 | 119 |
| B | participant_zero | 0.164373023 | 25 | 25 | 0 | 0 | 150 | 150 | 12 | 118 |
| B | participant_permuted | 0.165331294 | 32 | 23 | 0 | 0 | 150 | 150 | 12 | 120 |
| B | time_gold | 0.155585081 | 37 | 49 | 0 | 0 | 150 | 150 | 12 | 121 |
| B | time_zero | 0.162663038 | 35 | 24 | 0 | 0 | 150 | 150 | 8 | 120 |
| B | time_permuted | 0.163178944 | 35 | 28 | 0 | 0 | 150 | 150 | 12 | 120 |
| B | both_gold | 0.152465121 | 52 | 39 | 0 | 0 | 149 | 149 | 11 | 120 |

全条件exact sentence=0/150、special-token rows=0。Bのboth_goldだけEOS/UTF8が149/150。
人物goldでA 36→76 / time25→1、B 40→72 / time25→1。単独修復と他slot悪化が併存。
両head goldも両slot一致0。Bの人物4例増加を分離構造の効果とは呼ばない。

### Aの歴史的joint oracle

Issue20の7条件・全感度・stateを厳密再現。full_oracleはold7概念も修復し、head-only goldと区別。

| condition | LM | 人物 /150 | 時点 /150 | 両方 /150 | 全frame /150 |
|---|---|---|---|---|---|
| predicted | 0.164552725 | 36 | 25 | 0 | 0 |
| head_gold | 0.152015500 | 59 | 31 | 0 | 0 |
| full_oracle | 0.147264605 | 63 | 48 | 0 | 0 |

## 専用headとteacher-forced byte

専用headは通常source予測の評価。介入goldをheadの予測成功に加算しない。全例unseen。

| arm | head | accuracy | balanced | entropy |
|---|---|---|---|---|
| A | participant | 0.380000000 | 0.380000000 | 1.041703243 |
| A | time | 0.406666667 | 0.406666667 | 1.010194156 |
| B | participant | 0.380000000 | 0.380000000 | 1.041829217 |
| B | time | 0.406666667 | 0.406666667 | 1.010297256 |

各headは5class各30例。両armとも人物57/150、time61/150。confusionと各例確率はreport保存。
次表は固定された正解の過去履歴でのbyte診断。未来byteは渡さず、自由生成性能とは区別する。

| arm | condition | slot | NLL | mean rank | argmax /900 |
|---|---|---|---|---|---|
| A | predicted | participant | 0.656772751 | 1.405555556 | 740 |
| A | predicted | time | 0.349710443 | 1.165555556 | 781 |
| A | participant_gold | participant | 0.569257872 | 1.341111111 | 756 |
| A | participant_gold | time | 0.376657589 | 1.194444444 | 755 |
| A | participant_zero | participant | 0.639596414 | 1.324444444 | 767 |
| A | participant_zero | time | 0.354717552 | 1.173333333 | 779 |
| A | participant_permuted | participant | 0.662343941 | 1.382222222 | 740 |
| A | participant_permuted | time | 0.356352848 | 1.172222222 | 778 |
| A | time_gold | participant | 0.661763203 | 1.388888889 | 742 |
| A | time_gold | time | 0.250521959 | 1.136666667 | 804 |
| A | time_zero | participant | 0.667892703 | 1.420000000 | 732 |
| A | time_zero | time | 0.333417909 | 1.170000000 | 780 |
| A | time_permuted | participant | 0.650376874 | 1.407777778 | 733 |
| A | time_permuted | time | 0.351244124 | 1.162222222 | 784 |
| A | both_gold | participant | 0.578783193 | 1.314444444 | 773 |
| A | both_gold | time | 0.270835814 | 1.157777778 | 785 |
| B | predicted | participant | 0.650800087 | 1.391111111 | 739 |
| B | predicted | time | 0.348242131 | 1.164444444 | 782 |
| B | participant_gold | participant | 0.568432221 | 1.328888889 | 760 |
| B | participant_gold | time | 0.375522145 | 1.194444444 | 755 |
| B | participant_zero | participant | 0.643480291 | 1.323333333 | 776 |
| B | participant_zero | time | 0.354305137 | 1.171111111 | 780 |
| B | participant_permuted | participant | 0.660834019 | 1.367777778 | 743 |
| B | participant_permuted | time | 0.355190852 | 1.168888889 | 780 |
| B | time_gold | participant | 0.660590966 | 1.363333333 | 748 |
| B | time_gold | time | 0.242188845 | 1.137777778 | 804 |
| B | time_zero | participant | 0.659680503 | 1.405555556 | 731 |
| B | time_zero | time | 0.329185451 | 1.170000000 | 780 |
| B | time_permuted | participant | 0.646237338 | 1.385555556 | 735 |
| B | time_permuted | time | 0.349594961 | 1.162222222 | 784 |
| B | both_gold | participant | 0.580560938 | 1.313333333 | 774 |
| B | both_gold | time | 0.261376603 | 1.148888889 | 794 |

## Cross-slot sensitivity

同じ正解過去履歴でpredictedと比較。L1はlogit平均絶対差、KLはbase||intervention。
before=slot start-1（150位置）、inside=slot内（900 byte）。argmaxは変化率。
下表は他slotへの作用だけを掲載。同slot・全介入・各位置の詳細はreportに保存。

| arm | intervention | destination | region | L1 | KL | argmax change |
|---|---|---|---|---|---|---|
| A | participant_gold | time | before | 0.025391316 | 0.000250213 | 0.000000000 |
| A | participant_gold | time | inside | 0.071996496 | 0.005587971 | 0.028888889 |
| A | participant_zero | time | before | 0.030739813 | 0.000221264 | 0.000000000 |
| A | participant_zero | time | inside | 0.089500928 | 0.003076473 | 0.011111111 |
| A | participant_permuted | time | before | 0.023611050 | 0.000163663 | 0.000000000 |
| A | participant_permuted | time | inside | 0.060269137 | 0.002398985 | 0.014444444 |
| A | time_gold | participant | before | 0.073084443 | 0.002425944 | 0.000000000 |
| A | time_gold | participant | inside | 0.061474031 | 0.009300012 | 0.024444444 |
| A | time_zero | participant | before | 0.065133021 | 0.002165245 | 0.000000000 |
| A | time_zero | participant | inside | 0.057315344 | 0.003466369 | 0.022222222 |
| A | time_permuted | participant | before | 0.049076489 | 0.001526884 | 0.000000000 |
| A | time_permuted | participant | inside | 0.044793391 | 0.003463065 | 0.024444444 |
| B | participant_gold | time | before | 0.025912716 | 0.000270782 | 0.000000000 |
| B | participant_gold | time | inside | 0.072764675 | 0.005255990 | 0.030000000 |
| B | participant_zero | time | before | 0.031252519 | 0.000243189 | 0.000000000 |
| B | participant_zero | time | inside | 0.088088385 | 0.002516638 | 0.011111111 |
| B | participant_permuted | time | before | 0.021872671 | 0.000142093 | 0.000000000 |
| B | participant_permuted | time | inside | 0.060394941 | 0.002174616 | 0.011111111 |
| B | time_gold | participant | before | 0.077477936 | 0.002317066 | 0.000000000 |
| B | time_gold | participant | inside | 0.063653416 | 0.010466486 | 0.025555556 |
| B | time_zero | participant | before | 0.067614428 | 0.002095679 | 0.000000000 |
| B | time_zero | participant | inside | 0.058890073 | 0.003270692 | 0.027777778 |
| B | time_permuted | participant | before | 0.051429948 | 0.001475524 | 0.000000000 |
| B | time_permuted | participant | inside | 0.046119157 | 0.003611418 | 0.025555556 |

## 固定人物・固定時点のfactorial系列

全150 sourceそれぞれに専用headの25 one-hot組合せを与える。old33はsource予測のまま。
行の横比較は人物固定/time変化、列の縦比較はtime固定/人物変化。各cell分母150。
これは指定した反実仮想pairへの追随であり、元referenceとの正解率ではない。
元conceptと専用headの矛盾、one-hot分布差は残る。各例token/parseはreportに保存。

### A: 人物一致 /150

| participant | 今日 | 明日 | 来月 | 来週 | 週末 |
|---|---|---|---|---|---|
| 先輩 | 78 | 110 | 117 | 93 | 3 |
| 友人 | 119 | 118 | 102 | 125 | 119 |
| 同僚 | 26 | 4 | 0 | 4 | 0 |
| 知人 | 0 | 77 | 120 | 42 | 0 |
| 隣人 | 9 | 0 | 0 | 0 | 9 |

### A: 時点一致 /150

| participant | 今日 | 明日 | 来月 | 来週 | 週末 |
|---|---|---|---|---|---|
| 先輩 | 0 | 0 | 120 | 116 | 9 |
| 友人 | 0 | 0 | 79 | 125 | 5 |
| 同僚 | 0 | 0 | 119 | 126 | 0 |
| 知人 | 0 | 0 | 120 | 78 | 0 |
| 隣人 | 9 | 0 | 73 | 127 | 5 |

### A: 両方一致 /150

| participant | 今日 | 明日 | 来月 | 来週 | 週末 |
|---|---|---|---|---|---|
| 先輩 | 0 | 0 | 117 | 90 | 0 |
| 友人 | 0 | 0 | 61 | 125 | 5 |
| 同僚 | 0 | 0 | 0 | 4 | 0 |
| 知人 | 0 | 0 | 120 | 0 | 0 |
| 隣人 | 9 | 0 | 0 | 0 | 5 |

### B: 人物一致 /150

| participant | 今日 | 明日 | 来月 | 来週 | 週末 |
|---|---|---|---|---|---|
| 先輩 | 88 | 113 | 120 | 98 | 3 |
| 友人 | 117 | 116 | 101 | 119 | 119 |
| 同僚 | 28 | 0 | 0 | 0 | 0 |
| 知人 | 0 | 76 | 120 | 36 | 0 |
| 隣人 | 9 | 0 | 0 | 0 | 18 |

### B: 時点一致 /150

| participant | 今日 | 明日 | 来月 | 来週 | 週末 |
|---|---|---|---|---|---|
| 先輩 | 0 | 0 | 120 | 117 | 13 |
| 友人 | 2 | 0 | 81 | 120 | 5 |
| 同僚 | 1 | 0 | 119 | 123 | 4 |
| 知人 | 0 | 0 | 120 | 84 | 0 |
| 隣人 | 3 | 0 | 78 | 125 | 10 |

### B: 両方一致 /150

| participant | 今日 | 明日 | 来月 | 来週 | 週末 |
|---|---|---|---|---|---|
| 先輩 | 0 | 0 | 120 | 96 | 0 |
| 友人 | 0 | 0 | 62 | 119 | 5 |
| 同僚 | 1 | 0 | 0 | 0 | 0 |
| 知人 | 0 | 0 | 120 | 0 | 0 |
| 隣人 | 3 | 0 | 0 | 0 | 7 |

25cellのlogit比較は、time変更なら(p,0)、人物変更なら(0,t)を事前固定anchorにした。
元referenceの過去履歴を固定して各slotのbefore/insideを別保存。counterfactualの未来参照はない。

## 保存・再現・制約

- report SHA256: `a303e74fab83ccc11b4350579988b2e88e2077ba44abc263f32549890de2be39`
- B.pt SHA256: `207420aa7c8b4b8ba0d3aa5ad4a667edaf2c7342daaeac9705caa0792b2831f9`
- B final state SHA256: `f54a52b931b706951e1b2dd2bdc5f7e8e1f95335321d98ee5be94fdd9649016f`
- B initial state SHA256: `a404b1fa9d3997248442d81f7fd83a0e3c6d05ebe7ce41a6df581c93baa45e33`
- schedule SHA256: `6c3ae94191450c6c60d8ea3375711f56b241475b5da890940855f999d7e95e66`

A歴史評価の完全一致、Bのstate/old概念/専用head/logits/greedyの再読込完全一致を確認。
保存形式はCPU weights_only、版/語彙/設定/有限state/整合性digestを検査、上書き禁止。digestは署名ではない。
保存物はignored `codex/work_output/issue22-seed7-v1/`。Gitには重みやreportを入れない。
実行コマンドは[事前計画](joint-slots.md)。過去checkpoint/reportが前提。ローカル検証結果は[handoff](handoff.md)。

未解決: seen-validation群がないため、この分割ではseen/unseen性能差は推定不可。
全validation未見pairで両slot一致0を確認したが、学習pairの偏りだけを原因と断定できない。
次候補は別Issueでの数学的に異なる局所/直交注入、old concept重複のablation、事前固定した複数seed検証。
既存split/教材/testは変更せず、追加探索・大規模学習・有料GPUは実行していない。
AI-authored toy単一seedの診断。辞書デモや一般日本語意味理解の証明として扱わない。
