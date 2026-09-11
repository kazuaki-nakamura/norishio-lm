# Duplicate removal and local injection results (Issue #24)

事前計画 `3eeebd7`。構造・初期化・因果的な位置規則は[計画](local-slot-injection.md)。
実装SHAと検証コマンドは[handoff](handoff.md)。測定日2026-09-11。

A/Bは保存済み重みを使い、Issue20 Aの7条件とIssue22 A/Bの8条件・25pairの全評価を厳密再現。
Cは旧人物/time各6次元を投影から切り、Dは同じCの投影を生成済みprefixに基づいて時間制限。
外部transport43次元のうち有効31次元だけを投影する。旧bottleneckの補助学習は残す。
C/Dは42689 parameter（A/Bから384減）。同じ重みから600更新、seed7/batch16/Adam .003/clip1/CPU1thread。
既存4損失各1と専用head CE各1、slot byte CEなし。C/D間では容量は等しいがh0と注入時刻が異なる。

## 主要観測

- 通常生成の両slotはA/B/C/Dすべて0/150。Cは人物15/time37、Dは人物57/time43。
- 両head goldの両slotはA0/B0/C6/D48、全frameはA0/B0/C3/D16（各分母150）。
  局所注入Dはこのoracle条件で追随を改善したが、通常の組合せ保持は未解決。
- Dでも両head goldの未見5組のうち3組はfactorial両slot一致0。decoder側の失敗も残る。
- 既存reportのsource head確率から、人物/timeのargmaxが同時にgold一致する行を補足集計するとC1/D0（/150）。
  単独head accuracyだけで同時予測を保証しない。これは同じ保存結果の記述的集計で、追加学習や設定選択ではない。

## Supportと評価の境界

train15組×30例、validation5組×30例、全validation150例がunseen pair。seen0群の率はnull。
supportはIssue22と完全一致。[5×5表](joint-slot-results.md#pair-support)。教材/split/testは変更していない。
通常生成はsource-only/BOS自己履歴。goldは専用headだけの明示oracleで、old概念は予測値固定。
parse失敗を分母から除かず、両slotは人物と時点が同時に正しい場合だけ加点する。

## 自由生成・介入

各countの分母150。all=unseen、seen群は0例。

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
| C | predicted | 0.163995881 | 15 | 37 | 0 | 0 | 150 | 150 | 26 | 120 |
| C | participant_gold | 0.158625644 | 46 | 29 | 0 | 0 | 150 | 150 | 20 | 120 |
| C | participant_zero | 0.163048333 | 11 | 53 | 0 | 0 | 150 | 150 | 18 | 120 |
| C | participant_permuted | 0.164845420 | 5 | 46 | 0 | 0 | 150 | 150 | 33 | 120 |
| C | time_gold | 0.150619512 | 2 | 126 | 2 | 2 | 150 | 150 | 22 | 126 |
| C | time_zero | 0.168557763 | 36 | 28 | 0 | 0 | 150 | 150 | 15 | 120 |
| C | time_permuted | 0.167129652 | 34 | 22 | 0 | 0 | 150 | 150 | 29 | 120 |
| C | both_gold | 0.144315077 | 12 | 108 | 6 | 3 | 150 | 150 | 17 | 126 |
| D | predicted | 0.164825604 | 57 | 43 | 0 | 0 | 150 | 150 | 26 | 120 |
| D | participant_gold | 0.135151562 | 95 | 39 | 18 | 5 | 150 | 150 | 26 | 116 |
| D | participant_zero | 0.195121320 | 5 | 24 | 0 | 0 | 150 | 150 | 15 | 48 |
| D | participant_permuted | 0.171874401 | 28 | 43 | 1 | 0 | 150 | 150 | 36 | 120 |
| D | time_gold | 0.133766674 | 20 | 120 | 20 | 6 | 150 | 150 | 27 | 120 |
| D | time_zero | 0.287388948 | 0 | 0 | 0 | 0 | 150 | 120 | 17 | 0 |
| D | time_permuted | 0.171634226 | 55 | 21 | 0 | 0 | 150 | 150 | 44 | 120 |
| D | both_gold | 0.104085495 | 48 | 99 | 48 | 16 | 150 | 150 | 20 | 99 |

## 専用head / 正解過去履歴byte

headは通常source予測を採点し、与えたgoldをheadの成功に含めない。各class support30。

| arm | head | accuracy | balanced | entropy |
|---|---|---|---|---|
| A | participant | 0.380000000 | 0.380000000 | 1.041703243 |
| A | time | 0.406666667 | 0.406666667 | 1.010194156 |
| B | participant | 0.380000000 | 0.380000000 | 1.041829217 |
| B | time | 0.406666667 | 0.406666667 | 1.010297256 |
| C | participant | 0.380000000 | 0.380000000 | 1.041163950 |
| C | time | 0.413333333 | 0.413333333 | 1.010443665 |
| D | participant | 0.380000000 | 0.380000000 | 1.040895606 |
| D | time | 0.406666667 | 0.406666667 | 1.009975975 |

byte診断は正解の過去履歴を固定する明示oracle。自由生成の成功率ではない。各slot900byte。

| arm | condition | slot | NLL | rank | argmax /900 |
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
| C | predicted | participant | 0.716658844 | 1.477777778 | 721 |
| C | predicted | time | 0.313820631 | 1.131111111 | 795 |
| C | participant_gold | participant | 0.601144097 | 1.444444444 | 726 |
| C | participant_gold | time | 0.359458827 | 1.145555556 | 779 |
| C | participant_zero | participant | 0.694768662 | 1.473333333 | 739 |
| C | participant_zero | time | 0.316119394 | 1.134444444 | 811 |
| C | participant_permuted | participant | 0.734868762 | 1.491111111 | 734 |
| C | participant_permuted | time | 0.305424930 | 1.131111111 | 805 |
| C | time_gold | participant | 0.740163086 | 1.506666667 | 722 |
| C | time_gold | time | 0.121293394 | 1.013333333 | 894 |
| C | time_zero | participant | 0.746873892 | 1.490000000 | 726 |
| C | time_zero | time | 0.326469937 | 1.186666667 | 765 |
| C | time_permuted | participant | 0.703713118 | 1.486666667 | 724 |
| C | time_permuted | time | 0.359346997 | 1.177777778 | 764 |
| C | both_gold | participant | 0.620677300 | 1.442222222 | 750 |
| C | both_gold | time | 0.153422350 | 1.047777778 | 863 |
| D | predicted | participant | 0.618294369 | 1.373333333 | 774 |
| D | predicted | time | 0.441661070 | 1.135555556 | 799 |
| D | participant_gold | participant | 0.291035138 | 1.211111111 | 816 |
| D | participant_gold | time | 0.441661070 | 1.135555556 | 799 |
| D | participant_zero | participant | 0.943772374 | 1.415555556 | 706 |
| D | participant_zero | time | 0.441661070 | 1.135555556 | 799 |
| D | participant_permuted | participant | 0.686029674 | 1.426666667 | 745 |
| D | participant_permuted | time | 0.441661070 | 1.135555556 | 799 |
| D | time_gold | participant | 0.645301078 | 1.401111111 | 766 |
| D | time_gold | time | 0.069695553 | 1.000000000 | 900 |
| D | time_zero | participant | 0.713237352 | 1.412222222 | 736 |
| D | time_zero | time | 1.708934085 | 2.386666667 | 449 |
| D | time_permuted | participant | 0.604266994 | 1.374444444 | 772 |
| D | time_permuted | time | 0.531565244 | 1.192222222 | 758 |
| D | both_gold | participant | 0.317568600 | 1.255555556 | 810 |
| D | both_gold | time | 0.069695553 | 1.000000000 | 900 |

## Cross-slot sensitivity

predicted基準、同じ正解過去履歴、KL(base||intervention)。beforeはstart-1の150位置、insideは900byte。
対象外stepの直接加算がゼロでも、GRUの過去状態経由の影響は残り得る。

Dのparticipant介入→先行するtimeの固定履歴logitは全て変化0。これは因果的な注入順序からの構造的帰結。
逆方向のtime→participantにはGRU経由の作用が残る。Dの人物head goldでtime正答43→39となるのは、
後続部分も含む全体parseが失敗した行を分母に残す採点の影響を含む。保存済み生成の補足照合では、
人物gold/zero/permutationの全150行で、人物注入開始前までの出力tokenはpredictedと完全一致した。
したがって、このtime採点差を未来の人物headから過去time byteへの逆向き作用とは解釈しない。

| arm | intervention | destination | region | logit L1 | KL | argmax change |
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
| C | participant_gold | time | before | 0.056969972 | 0.000432036 | 0.000000000 |
| C | participant_gold | time | inside | 0.096492823 | 0.023898482 | 0.055555556 |
| C | participant_zero | time | before | 0.061932377 | 0.000540546 | 0.000000000 |
| C | participant_zero | time | inside | 0.101527733 | 0.014670179 | 0.053333333 |
| C | participant_permuted | time | before | 0.041171818 | 0.000211775 | 0.000000000 |
| C | participant_permuted | time | inside | 0.074888851 | 0.009061503 | 0.034444444 |
| C | time_gold | participant | before | 0.099009379 | 0.007326939 | 0.000000000 |
| C | time_gold | participant | inside | 0.086827662 | 0.013523551 | 0.020000000 |
| C | time_zero | participant | before | 0.083836185 | 0.002545111 | 0.000000000 |
| C | time_zero | participant | inside | 0.081197271 | 0.009130115 | 0.038888889 |
| C | time_permuted | participant | before | 0.079246275 | 0.002257771 | 0.000000000 |
| C | time_permuted | participant | inside | 0.077092347 | 0.007359493 | 0.032222222 |
| D | participant_gold | time | before | 0.000000000 | 0.000000000 | 0.000000000 |
| D | participant_gold | time | inside | 0.000000000 | 0.000000000 | 0.000000000 |
| D | participant_zero | time | before | 0.000000000 | 0.000000000 | 0.000000000 |
| D | participant_zero | time | inside | 0.000000000 | 0.000000000 | 0.000000000 |
| D | participant_permuted | time | before | 0.000000000 | 0.000000000 | 0.000000000 |
| D | participant_permuted | time | inside | 0.000000000 | 0.000000000 | 0.000000000 |
| D | time_gold | participant | before | 0.200817440 | 0.004584227 | 0.000000000 |
| D | time_gold | participant | inside | 0.088106117 | 0.010500297 | 0.013333333 |
| D | time_zero | participant | before | 0.443330261 | 0.199901926 | 0.140000000 |
| D | time_zero | participant | inside | 0.156381716 | 0.046760699 | 0.054444444 |
| D | time_permuted | participant | before | 0.164211419 | 0.004702979 | 0.000000000 |
| D | time_permuted | participant | inside | 0.038359430 | 0.006015113 | 0.010000000 |

## Factorial指定pairへの追随

各sourceに25組の専用one-hotを介入。元referenceでなく指定pairへの一致。各cell分母150。
横方向は人物固定/time変化、縦方向はtime固定/人物変化。元conceptは予測値固定。
全cellの個別token/parseと固定class0 anchorからのlogit感度はreport保存。

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

### C: 人物一致 /150

| participant | 今日 | 明日 | 来月 | 来週 | 週末 |
|---|---|---|---|---|---|
| 先輩 | 150 | 120 | 120 | 120 | 0 |
| 友人 | 0 | 30 | 120 | 120 | 120 |
| 同僚 | 30 | 0 | 0 | 90 | 46 |
| 知人 | 149 | 120 | 120 | 0 | 0 |
| 隣人 | 129 | 120 | 0 | 0 | 120 |

### C: 時点一致 /150

| participant | 今日 | 明日 | 来月 | 来週 | 週末 |
|---|---|---|---|---|---|
| 先輩 | 150 | 0 | 120 | 120 | 120 |
| 友人 | 120 | 90 | 120 | 120 | 120 |
| 同僚 | 150 | 114 | 120 | 120 | 120 |
| 知人 | 149 | 120 | 120 | 75 | 120 |
| 隣人 | 129 | 120 | 120 | 120 | 120 |

### C: 両方一致 /150

| participant | 今日 | 明日 | 来月 | 来週 | 週末 |
|---|---|---|---|---|---|
| 先輩 | 150 | 0 | 120 | 120 | 0 |
| 友人 | 0 | 0 | 120 | 120 | 120 |
| 同僚 | 30 | 0 | 0 | 90 | 46 |
| 知人 | 149 | 120 | 120 | 0 | 0 |
| 隣人 | 129 | 120 | 0 | 0 | 120 |

### D: 人物一致 /150

| participant | 今日 | 明日 | 来月 | 来週 | 週末 |
|---|---|---|---|---|---|
| 先輩 | 120 | 120 | 120 | 120 | 0 |
| 友人 | 120 | 120 | 120 | 120 | 120 |
| 同僚 | 120 | 120 | 120 | 120 | 120 |
| 知人 | 120 | 120 | 120 | 0 | 0 |
| 隣人 | 120 | 120 | 0 | 0 | 120 |

### D: 時点一致 /150

| participant | 今日 | 明日 | 来月 | 来週 | 週末 |
|---|---|---|---|---|---|
| 先輩 | 120 | 120 | 120 | 120 | 120 |
| 友人 | 120 | 120 | 120 | 120 | 120 |
| 同僚 | 120 | 120 | 120 | 120 | 120 |
| 知人 | 120 | 120 | 120 | 11 | 107 |
| 隣人 | 120 | 120 | 120 | 120 | 120 |

### D: 両方一致 /150

| participant | 今日 | 明日 | 来月 | 来週 | 週末 |
|---|---|---|---|---|---|
| 先輩 | 120 | 120 | 120 | 120 | 0 |
| 友人 | 120 | 120 | 120 | 120 | 120 |
| 同僚 | 120 | 120 | 120 | 120 | 120 |
| 知人 | 120 | 120 | 120 | 0 | 0 |
| 隣人 | 120 | 120 | 0 | 0 | 120 |

## 保存物・再現

report SHA256: `eb2c1fc26c35e429d25b3815f15060e7fd481cf9ff5cca62f0e36198ae335af8`

### C

- 初期state SHA256: `c4596c498db74b9cda10392299645bf399264b2244e11a844f9a07d50f67f344`
- checkpoint SHA256: `0532e7c1e859b50cc86d2de541da42f8a9e9f311a77800b7e7fe4b686ebef68e`
- 最終state SHA256: `56b2bb58867d2a56524e969e7d7368c077b2e137ec9a847c914915b21f52d5f6`
- 学習秒数: 44.181（速度比較の実証ではない）
- state/old concept/head/logits/greedy再読込: `{'state': True, 'old_concepts': True, 'heads': True, 'logits': True, 'greedy': True}`

### D

- 初期state SHA256: `c4596c498db74b9cda10392299645bf399264b2244e11a844f9a07d50f67f344`
- checkpoint SHA256: `abed886085e2a862be1a8015ca910adbc77e687c09c10de52411cbc3e8bbe143`
- 最終state SHA256: `ce6c5fbbe31c433251f129c3fe220bc0b7829624c968db79db3beee7e9a7ecd9`
- 学習秒数: 44.965（速度比較の実証ではない）
- state/old concept/head/logits/greedy再読込: `{'state': True, 'old_concepts': True, 'heads': True, 'logits': True, 'greedy': True}`

schedule SHA256: `6c3ae94191450c6c60d8ea3375711f56b241475b5da890940855f999d7e95e66`

保存物はignored `codex/work_output/issue24-seed7-v1/`。Gitに重み/reportを追加しない。
安全なCPU weights_only形式と版/構造/位置規則/語彙/finite state/整合性digestを検査。digestは署名ではない。

## 解釈の限界

Cは情報と384parameterを同時に除くため重複情報単独の因果効果を断定できない。
Dは既知のauthored文法を注入する。文頭が崩れるとgateが開かず、その失敗も評価に含まれる。
正解過去履歴の診断はgateを正常位置へ誘導するため、通常生成との区別が必要。
単一seed、AI-authored教材、全validation未見pairでseen対照がない。一般意味理解の証明ではない。
test未評価、分割変更・外部辞書・private data・有料GPU・結果後の探索は行わない。
