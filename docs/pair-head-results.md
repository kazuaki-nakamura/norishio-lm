# Joint pair head results (Issue #28)

測定日2026-09-12。事前計画 `f02f190`、実装 `08f0077`。
構造・loss・seed・解釈の境界は[事前計画](pair-head.md)。
Aは保存済みDの3seed、Bは25-way joint headのmarginalを局所注入する追加825parameter対照。
train450のみfit、validation150全未見pair、test未評価。全seed値を掲載し選別しない。

## 主要観測と残課題

- Bの25-way正解はtrainで446/450・450/450・438/450、validationでは全seed0/150。
  validation gold確率平均0.000495385、gold rank平均21.2022/25。未見10pairにCE正例がない
  固定25-way分類は、この条件では未見pairに確率を置けていない。
- Bの通常生成両slot/全frameは全seed0。marginalのjoint argmaxも0/3/0にとどまる。
  独立headのjointは0/29/24へ変わるが、このheadはBのdecoderに投入しておらず、
  それを入力に戻した対照は未実施。soft情報の有無はargmaxだけでは決めない。
- B both-goldの両slot61/36/47（平均48）は部分追随を示すが、Aの平均52.667より低い。
  全frame平均もA16からB11.667。通常LM平均は0.178836から0.172167へ下がっても
  通常両slot0を解決していない。単一指標だけで改善を主張しない。
- 次候補は別Issueでfactorized/compositional objectiveと独立head入力の固定対照。
  今回結果を見てB条件・weight・seed・教材・splitを変更していない。
  容量825追加、joint損失、decoder入力分布変更の寄与は未分離。n3/共通拡張seed/文法priorも限界。

測定後のcheckpoint保存guard補強は `a429120`。学習・採点の数値経路や保存データは変更せず、
3保存物を補強後loaderで再確認。全pytest379 passed + 2 subtests passed、両デモ成功。

## 通常生成とoracle

正解数の分母150。parse失敗も分母に残す。LMは正解過去履歴、生成は自己履歴。

| arm | seed | condition | LM | 人物 | time | 両slot | 全frame | EOS | UTF8 | 種類 | parse | special rows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 7 | predicted | 0.164825604 | 57 | 43 | 0 | 0 | 1 | 1 | 26 | 120 | 0 |
| A | 7 | both_gold | 0.104085495 | 48 | 99 | 48 | 16 | 1 | 1 | 20 | 99 | 0 |
| A | 17 | predicted | 0.190100016 | 44 | 52 | 0 | 0 | 0.96 | 0.96 | 31 | 114 | 0 |
| A | 17 | both_gold | 0.151850218 | 46 | 114 | 46 | 12 | 0.96 | 0.96 | 22 | 114 | 0 |
| A | 29 | predicted | 0.181583041 | 27 | 43 | 0 | 0 | 1 | 1 | 31 | 102 | 0 |
| A | 29 | both_gold | 0.113778301 | 64 | 103 | 64 | 20 | 1 | 1 | 22 | 103 | 0 |
| B | 7 | predicted | 0.160202307 | 39 | 35 | 0 | 0 | 1 | 1 | 25 | 120 | 0 |
| B | 7 | participant_gold | 0.128506885 | 112 | 35 | 34 | 6 | 1 | 1 | 25 | 120 | 0 |
| B | 7 | time_gold | 0.134376196 | 10 | 114 | 10 | 2 | 1 | 1 | 24 | 120 | 0 |
| B | 7 | both_gold | 0.103684178 | 67 | 91 | 61 | 12 | 1 | 1 | 19 | 97 | 0 |
| B | 17 | predicted | 0.195328469 | 38 | 39 | 0 | 0 | 1 | 1 | 24 | 92 | 0 |
| B | 17 | participant_gold | 0.170182175 | 56 | 39 | 18 | 3 | 1 | 1 | 25 | 77 | 0 |
| B | 17 | time_gold | 0.175990987 | 38 | 72 | 18 | 3 | 1 | 1 | 30 | 92 | 0 |
| B | 17 | both_gold | 0.152637954 | 56 | 72 | 36 | 6 | 1 | 1 | 24 | 92 | 0 |
| B | 29 | predicted | 0.160970533 | 39 | 43 | 0 | 0 | 1 | 1 | 31 | 89 | 0 |
| B | 29 | participant_gold | 0.140670864 | 45 | 43 | 0 | 0 | 1 | 1 | 31 | 88 | 0 |
| B | 29 | time_gold | 0.140627042 | 19 | 89 | 19 | 6 | 1 | 1 | 35 | 89 | 0 |
| B | 29 | both_gold | 0.120607829 | 47 | 90 | 47 | 17 | 1 | 1 | 30 | 90 | 0 |

### 3seed算術平均

各countは150例あたりの平均件数。n=3、拡張seed20/pair seed28は共通。

| arm | condition | LM | 人物 | time | 両slot | 全frame | EOS | UTF8 | 種類 |
|---|---|---|---|---|---|---|---|---|---|
| A | predicted | 0.17883622 | 42.6666667 | 46 | 0 | 0 | 0.986666667 | 0.986666667 | 29.3333333 |
| A | both_gold | 0.123238005 | 52.6666667 | 105.333333 | 52.6666667 | 16 | 0.986666667 | 0.986666667 | 21.3333333 |
| B | predicted | 0.172167103 | 38.6666667 | 39 | 0 | 0 | 1 | 1 | 26.6666667 |
| B | participant_gold | 0.146453308 | 71 | 39 | 17.3333333 | 3 | 1 | 1 | 27 |
| B | time_gold | 0.150331408 | 22.3333333 | 91.6666667 | 15.6666667 | 3.66666667 | 1 | 1 | 29.6666667 |
| B | both_gold | 0.12564332 | 56.6666667 | 84.3333333 | 48 | 11.6666667 | 1 | 1 | 24.3333333 |

## Bの25-way head

trainの未見群0例とvalidationの既見群0例は率null。train診断とvalidationを直接比較して汎化差の因果効果とはしない。
gold rankは1始まり、同確率はCartesian index昇順。Brierは25class平方誤差和のrow平均。

| seed | split | group | n | accuracy | NLL | Brier | ECE10 | gold probability mean | gold rank mean |
|---|---|---|---|---|---|---|---|---|---|
| 7 | train | all | 450 | 0.991111111 | 0.530461938 | 0.233745267 | 0.384214657 | 0.607960852 | 1.00888889 |
| 7 | train | train_seen | 450 | 0.991111111 | 0.530461938 | 0.233745267 | 0.384214657 | 0.607960852 | 1.00888889 |
| 7 | train | train_unseen | 0 | null | null | null | null | null | null |
| 7 | validation | all | 150 | 0 | 7.67287601 | 1.20836507 | 0.339384718 | 0.000677773597 | 21.2733333 |
| 7 | validation | train_seen | 0 | null | null | null | null | null | null |
| 7 | validation | train_unseen | 150 | 0 | 7.67287601 | 1.20836507 | 0.339384718 | 0.000677773597 | 21.2733333 |
| 17 | train | all | 450 | 1 | 0.468195318 | 0.187869822 | 0.356193437 | 0.643806563 | 1 |
| 17 | train | train_seen | 450 | 1 | 0.468195318 | 0.187869822 | 0.356193437 | 0.643806563 | 1 |
| 17 | train | train_unseen | 0 | null | null | null | null | null | null |
| 17 | validation | all | 150 | 0 | 8.03111009 | 1.22623707 | 0.379119043 | 0.000445243593 | 20.7266667 |
| 17 | validation | train_seen | 0 | null | null | null | null | null | null |
| 17 | validation | train_unseen | 150 | 0 | 8.03111009 | 1.22623707 | 0.379119043 | 0.000445243593 | 20.7266667 |
| 29 | train | all | 450 | 0.973333333 | 0.440474583 | 0.176854494 | 0.30178947 | 0.670112565 | 1.03555556 |
| 29 | train | train_seen | 450 | 0.973333333 | 0.440474583 | 0.176854494 | 0.30178947 | 0.670112565 | 1.03555556 |
| 29 | train | train_unseen | 0 | null | null | null | null | null | null |
| 29 | validation | all | 150 | 0 | 8.74349551 | 1.31551065 | 0.469788048 | 0.000363137933 | 21.6066667 |
| 29 | validation | train_seen | 0 | null | null | null | null | null | null |
| 29 | validation | train_unseen | 150 | 0 | 8.74349551 | 1.31551065 | 0.469788048 | 0.000363137933 | 21.6066667 |

| 3seed mean split | accuracy | NLL | Brier | ECE10 | gold probability | gold rank |
|---|---|---|---|---|---|---|
| train | 0.988148148 | 0.479710613 | 0.199489861 | 0.347399188 | 0.64062666 | 1.01481481 |
| validation | 0 | 8.14916054 | 1.2500376 | 0.39609727 | 0.000495385041 | 21.2022222 |

## 独立headとjoint marginal

jointは同じrowで両argmaxが正しい件数。Bの独立headはdecoderに投入しない。

| arm | seed | split | head | field | accuracy | balanced | Brier | NLL | ECE10 | joint count |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 7 | validation | independent | participant | 0.38 | 0.38 | 0.766639184 | 1.5635621 | 0.213591435 | 0 |
| A | 7 | validation | independent | time | 0.406666667 | 0.406666667 | 0.862014994 | 1.5953941 | 0.300854457 | 0 |
| A | 17 | validation | independent | participant | 0.386666667 | 0.386666667 | 0.722243442 | 1.5687264 | 0.207475566 | 2 |
| A | 17 | validation | independent | time | 0.393333333 | 0.393333333 | 0.736530818 | 1.57080532 | 0.208370125 | 2 |
| A | 29 | validation | independent | participant | 0.153333333 | 0.153333333 | 0.945633479 | 1.82924671 | 0.362314173 | 0 |
| A | 29 | validation | independent | time | 0.533333333 | 0.533333333 | 0.79163819 | 1.54121449 | 0.324676981 | 0 |
| B | 7 | train | independent | participant | 1 | 1 | 0.0182183523 | 0.0850938874 | 0.0788203967 | 449 |
| B | 7 | train | independent | time | 0.997777778 | 0.997777778 | 0.057764684 | 0.166427001 | 0.14349353 | 449 |
| B | 7 | train | marginals | participant | 0.997777778 | 0.997777778 | 0.0751296933 | 0.215911362 | 0.183008593 | 447 |
| B | 7 | train | marginals | time | 0.993333333 | 0.993333333 | 0.143646169 | 0.32293112 | 0.258419081 | 447 |
| B | 7 | validation | independent | participant | 0.38 | 0.38 | 0.795550584 | 1.62551481 | 0.207335849 | 0 |
| B | 7 | validation | independent | time | 0.28 | 0.28 | 0.979242121 | 1.81808986 | 0.327879573 | 0 |
| B | 7 | validation | marginals | participant | 0.333333333 | 0.333333333 | 0.754093699 | 1.43197419 | 0.209252098 | 0 |
| B | 7 | validation | marginals | time | 0.32 | 0.32 | 0.869831617 | 1.5336263 | 0.233420783 | 0 |
| B | 17 | train | independent | participant | 1 | 1 | 0.0232165987 | 0.0955235248 | 0.0871818366 | 450 |
| B | 17 | train | independent | time | 1 | 1 | 0.0233067434 | 0.104586416 | 0.0966566894 | 450 |
| B | 17 | train | marginals | participant | 1 | 1 | 0.0817002673 | 0.236273797 | 0.203287503 | 450 |
| B | 17 | train | marginals | time | 1 | 1 | 0.0837430593 | 0.239322801 | 0.206814763 | 450 |
| B | 17 | validation | independent | participant | 0.413333333 | 0.413333333 | 0.727002379 | 1.57060585 | 0.293763773 | 29 |
| B | 17 | validation | independent | time | 0.766666667 | 0.766666667 | 0.423955451 | 0.790630236 | 0.280363298 | 29 |
| B | 17 | validation | marginals | participant | 0.413333333 | 0.413333333 | 0.728376096 | 1.49431858 | 0.15621868 | 3 |
| B | 17 | validation | marginals | time | 0.58 | 0.58 | 0.617970517 | 1.11989424 | 0.240361343 | 3 |
| B | 29 | train | independent | participant | 0.993333333 | 0.993333333 | 0.0350789328 | 0.116885196 | 0.0983195895 | 444 |
| B | 29 | train | independent | time | 0.993333333 | 0.993333333 | 0.038427154 | 0.125986479 | 0.105063205 | 444 |
| B | 29 | train | marginals | participant | 0.982222222 | 0.982222222 | 0.0958985097 | 0.244314309 | 0.179712915 | 438 |
| B | 29 | train | marginals | time | 0.988888889 | 0.988888889 | 0.0864503581 | 0.231767625 | 0.186212987 | 438 |
| B | 29 | validation | independent | participant | 0.306666667 | 0.306666667 | 0.890615222 | 1.91116998 | 0.264900661 | 24 |
| B | 29 | validation | independent | time | 0.74 | 0.74 | 0.533418567 | 1.0393588 | 0.309688344 | 24 |
| B | 29 | validation | marginals | participant | 0.36 | 0.36 | 0.798437783 | 1.74615429 | 0.269810916 | 0 |
| B | 29 | validation | marginals | time | 0.586666667 | 0.586666667 | 0.703155477 | 1.21344646 | 0.363076138 | 0 |

| 3seed mean split | head | field | accuracy | balanced | Brier | NLL | ECE10 | joint count |
|---|---|---|---|---|---|---|---|---|
| train | independent | participant | 0.997777778 | 0.997777778 | 0.025504628 | 0.0991675359 | 0.0881072743 | 447.666667 |
| train | independent | time | 0.997037037 | 0.997037037 | 0.0398328605 | 0.132333299 | 0.115071141 | 447.666667 |
| train | marginals | participant | 0.993333333 | 0.993333333 | 0.0842428234 | 0.23216649 | 0.18866967 | 445 |
| train | marginals | time | 0.994074074 | 0.994074074 | 0.104613195 | 0.264673849 | 0.217148944 | 445 |
| validation | independent | participant | 0.366666667 | 0.366666667 | 0.804389395 | 1.70243021 | 0.255333428 | 17.6666667 |
| validation | independent | time | 0.595555556 | 0.595555556 | 0.645538713 | 1.2160263 | 0.305977072 | 17.6666667 |
| validation | marginals | participant | 0.368888889 | 0.368888889 | 0.760302526 | 1.55748235 | 0.211760564 | 1 |
| validation | marginals | time | 0.495555556 | 0.495555556 | 0.730319204 | 1.288989 | 0.278952755 | 1 |

## Bの25factorial

各cellはvalidation source150例に同じ指定pairを与え、その指定への追随を採点。通常正解率ではない。

### Seed 7

| pair | 人物 | time | status | train | validation | 人物正解 | time正解 | 両slot | 全frame | parse |
|---|---|---|---|---|---|---|---|---|---|---|
| 0,0 | 先輩 | 今日 | train_seen | 30 | 0 | 120 | 120 | 120 | 31 | 120 |
| 0,1 | 先輩 | 明日 | neither_train_nor_validation | 0 | 0 | 120 | 91 | 91 | 16 | 120 |
| 0,2 | 先輩 | 来月 | train_seen | 30 | 0 | 120 | 120 | 120 | 31 | 120 |
| 0,3 | 先輩 | 来週 | train_seen | 30 | 0 | 120 | 120 | 120 | 31 | 120 |
| 0,4 | 先輩 | 週末 | validation_unseen | 0 | 30 | 0 | 120 | 0 | 0 | 120 |
| 1,0 | 友人 | 今日 | neither_train_nor_validation | 0 | 0 | 120 | 120 | 120 | 31 | 120 |
| 1,1 | 友人 | 明日 | validation_unseen | 0 | 30 | 120 | 91 | 91 | 16 | 120 |
| 1,2 | 友人 | 来月 | train_seen | 30 | 0 | 120 | 120 | 120 | 31 | 120 |
| 1,3 | 友人 | 来週 | train_seen | 30 | 0 | 120 | 120 | 120 | 31 | 120 |
| 1,4 | 友人 | 週末 | train_seen | 30 | 0 | 120 | 120 | 120 | 31 | 120 |
| 2,0 | 同僚 | 今日 | validation_unseen | 0 | 30 | 121 | 121 | 121 | 31 | 121 |
| 2,1 | 同僚 | 明日 | train_seen | 30 | 0 | 121 | 91 | 91 | 16 | 121 |
| 2,2 | 同僚 | 来月 | neither_train_nor_validation | 0 | 0 | 121 | 121 | 121 | 31 | 121 |
| 2,3 | 同僚 | 来週 | train_seen | 30 | 0 | 121 | 121 | 121 | 31 | 121 |
| 2,4 | 同僚 | 週末 | train_seen | 30 | 0 | 121 | 121 | 121 | 31 | 121 |
| 3,0 | 知人 | 今日 | train_seen | 30 | 0 | 120 | 120 | 120 | 31 | 120 |
| 3,1 | 知人 | 明日 | train_seen | 30 | 0 | 120 | 91 | 91 | 16 | 120 |
| 3,2 | 知人 | 来月 | train_seen | 30 | 0 | 120 | 120 | 120 | 31 | 120 |
| 3,3 | 知人 | 来週 | validation_unseen | 0 | 30 | 0 | 0 | 0 | 0 | 0 |
| 3,4 | 知人 | 週末 | neither_train_nor_validation | 0 | 0 | 0 | 120 | 0 | 0 | 120 |
| 4,0 | 隣人 | 今日 | train_seen | 30 | 0 | 120 | 120 | 120 | 31 | 120 |
| 4,1 | 隣人 | 明日 | train_seen | 30 | 0 | 120 | 91 | 91 | 16 | 120 |
| 4,2 | 隣人 | 来月 | validation_unseen | 0 | 30 | 87 | 120 | 87 | 15 | 120 |
| 4,3 | 隣人 | 来週 | neither_train_nor_validation | 0 | 0 | 0 | 121 | 0 | 0 | 121 |
| 4,4 | 隣人 | 週末 | train_seen | 30 | 0 | 120 | 120 | 120 | 31 | 120 |

### Seed 17

| pair | 人物 | time | status | train | validation | 人物正解 | time正解 | 両slot | 全frame | parse |
|---|---|---|---|---|---|---|---|---|---|---|
| 0,0 | 先輩 | 今日 | train_seen | 30 | 0 | 92 | 92 | 92 | 15 | 92 |
| 0,1 | 先輩 | 明日 | neither_train_nor_validation | 0 | 0 | 92 | 92 | 92 | 15 | 92 |
| 0,2 | 先輩 | 来月 | train_seen | 30 | 0 | 92 | 92 | 92 | 15 | 92 |
| 0,3 | 先輩 | 来週 | train_seen | 30 | 0 | 92 | 92 | 92 | 15 | 92 |
| 0,4 | 先輩 | 週末 | validation_unseen | 0 | 30 | 82 | 10 | 0 | 0 | 92 |
| 1,0 | 友人 | 今日 | neither_train_nor_validation | 0 | 0 | 92 | 92 | 92 | 15 | 92 |
| 1,1 | 友人 | 明日 | validation_unseen | 0 | 30 | 92 | 92 | 92 | 15 | 92 |
| 1,2 | 友人 | 来月 | train_seen | 30 | 0 | 92 | 92 | 92 | 15 | 92 |
| 1,3 | 友人 | 来週 | train_seen | 30 | 0 | 92 | 92 | 92 | 15 | 92 |
| 1,4 | 友人 | 週末 | train_seen | 30 | 0 | 92 | 10 | 10 | 1 | 92 |
| 2,0 | 同僚 | 今日 | validation_unseen | 0 | 30 | 91 | 91 | 91 | 15 | 91 |
| 2,1 | 同僚 | 明日 | train_seen | 30 | 0 | 91 | 91 | 91 | 15 | 91 |
| 2,2 | 同僚 | 来月 | neither_train_nor_validation | 0 | 0 | 0 | 92 | 0 | 0 | 92 |
| 2,3 | 同僚 | 来週 | train_seen | 30 | 0 | 91 | 91 | 91 | 15 | 91 |
| 2,4 | 同僚 | 週末 | train_seen | 30 | 0 | 91 | 10 | 10 | 1 | 91 |
| 3,0 | 知人 | 今日 | train_seen | 30 | 0 | 92 | 92 | 92 | 15 | 92 |
| 3,1 | 知人 | 明日 | train_seen | 30 | 0 | 92 | 92 | 92 | 15 | 92 |
| 3,2 | 知人 | 来月 | train_seen | 30 | 0 | 92 | 92 | 92 | 15 | 92 |
| 3,3 | 知人 | 来週 | validation_unseen | 0 | 30 | 0 | 90 | 0 | 0 | 90 |
| 3,4 | 知人 | 週末 | neither_train_nor_validation | 0 | 0 | 0 | 10 | 0 | 0 | 90 |
| 4,0 | 隣人 | 今日 | train_seen | 30 | 0 | 92 | 92 | 92 | 15 | 92 |
| 4,1 | 隣人 | 明日 | train_seen | 30 | 0 | 92 | 92 | 92 | 15 | 92 |
| 4,2 | 隣人 | 来月 | validation_unseen | 0 | 30 | 0 | 92 | 0 | 0 | 92 |
| 4,3 | 隣人 | 来週 | neither_train_nor_validation | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4,4 | 隣人 | 週末 | train_seen | 30 | 0 | 10 | 10 | 10 | 1 | 92 |

### Seed 29

| pair | 人物 | time | status | train | validation | 人物正解 | time正解 | 両slot | 全frame | parse |
|---|---|---|---|---|---|---|---|---|---|---|
| 0,0 | 先輩 | 今日 | train_seen | 30 | 0 | 88 | 88 | 88 | 40 | 88 |
| 0,1 | 先輩 | 明日 | neither_train_nor_validation | 0 | 0 | 95 | 95 | 95 | 43 | 95 |
| 0,2 | 先輩 | 来月 | train_seen | 30 | 0 | 99 | 99 | 99 | 46 | 99 |
| 0,3 | 先輩 | 来週 | train_seen | 30 | 0 | 104 | 104 | 104 | 47 | 104 |
| 0,4 | 先輩 | 週末 | validation_unseen | 0 | 30 | 93 | 93 | 93 | 42 | 93 |
| 1,0 | 友人 | 今日 | neither_train_nor_validation | 0 | 0 | 0 | 84 | 0 | 0 | 84 |
| 1,1 | 友人 | 明日 | validation_unseen | 0 | 30 | 46 | 84 | 46 | 13 | 84 |
| 1,2 | 友人 | 来月 | train_seen | 30 | 0 | 84 | 84 | 84 | 38 | 84 |
| 1,3 | 友人 | 来週 | train_seen | 30 | 0 | 84 | 84 | 84 | 38 | 84 |
| 1,4 | 友人 | 週末 | train_seen | 30 | 0 | 85 | 85 | 85 | 38 | 85 |
| 2,0 | 同僚 | 今日 | validation_unseen | 0 | 30 | 70 | 87 | 70 | 26 | 87 |
| 2,1 | 同僚 | 明日 | train_seen | 30 | 0 | 96 | 96 | 96 | 43 | 96 |
| 2,2 | 同僚 | 来月 | neither_train_nor_validation | 0 | 0 | 31 | 85 | 31 | 25 | 85 |
| 2,3 | 同僚 | 来週 | train_seen | 30 | 0 | 95 | 95 | 95 | 43 | 95 |
| 2,4 | 同僚 | 週末 | train_seen | 30 | 0 | 96 | 96 | 96 | 43 | 96 |
| 3,0 | 知人 | 今日 | train_seen | 30 | 0 | 86 | 86 | 86 | 37 | 86 |
| 3,1 | 知人 | 明日 | train_seen | 30 | 0 | 85 | 85 | 85 | 37 | 85 |
| 3,2 | 知人 | 来月 | train_seen | 30 | 0 | 85 | 85 | 85 | 37 | 85 |
| 3,3 | 知人 | 来週 | validation_unseen | 0 | 30 | 0 | 93 | 0 | 0 | 93 |
| 3,4 | 知人 | 週末 | neither_train_nor_validation | 0 | 0 | 0 | 84 | 0 | 0 | 84 |
| 4,0 | 隣人 | 今日 | train_seen | 30 | 0 | 84 | 84 | 84 | 37 | 84 |
| 4,1 | 隣人 | 明日 | train_seen | 30 | 0 | 84 | 84 | 84 | 37 | 84 |
| 4,2 | 隣人 | 来月 | validation_unseen | 0 | 30 | 0 | 84 | 0 | 0 | 84 |
| 4,3 | 隣人 | 来週 | neither_train_nor_validation | 0 | 0 | 0 | 86 | 0 | 0 | 86 |
| 4,4 | 隣人 | 週末 | train_seen | 30 | 0 | 84 | 84 | 84 | 37 | 84 |

## 再現性

Report SHA256: `902545c14f91c68845d45b0af880432aa8c004137e5ff9cbd458ae84506f69c1`。
Aは3seedの全保存評価とstateが厳密一致。Bは3checkpointのstate/旧concept/独立head/pair/marginal/logits/greedyが完全一致。

- seed7: initial `e704a4ff2e79c53ace85b69a3d2a813a13757b0e4442c5e83e01a8db972719e8`; schedule `6c3ae94191450c6c60d8ea3375711f56b241475b5da890940855f999d7e95e66`。
  checkpoint `566e5ce56552e616327d7330f6543f877b36ae71302db06f724d2d0d91aa52c5`; state `68bbbf3ba5677196b546e0fc1c994b3cbb3b28dafcb6e813d6bce655f7c6e5e8`。
  600更新の学習時間 44.714377秒。
- seed17: initial `60a550f0c923da4efbc1a99ba54e2939a99d995538b66688f47f9070ebf3f9d1`; schedule `aff617f4afe36027c5ab34c00f01f5bda171ce15fdb6b6f9ba099dd572240261`。
  checkpoint `d89b3d34481be53e2c04751573a9873a33f3c2176b5f386ed24c012d54199fc6`; state `becdfd7640088dc01ca9fbc79ea0cc9d1cf5f83b2e0c63db9b05cf2de355d1db`。
  600更新の学習時間 45.779140秒。
- seed29: initial `8e4676280822312767f4eeaca81a1d3e97436057c7458073c5423b7b62c291d7`; schedule `d606e8d0cbffeb457d55c1c53ddda54aacfdc26db103a111ce695f3af493b0b3`。
  checkpoint `96580387b458f1d581db7bd0bf6fe1f20af11ca7ff00fed91423f12e2e5f0223`; state `7991c21c4f8e3c215ca46ead892c136326c4493598207a22ded2a315085823bc`。
  600更新の学習時間 45.303141秒。

各rowの分布・gold rank・confusion・ECE bins・生成token・学習traceはignored reportに保存。
既存evaluate_conditionのteacher_forcedフィールドはnull（到達不能コード）。今回はbyte診断を測定済みとはしない。
将来修復時は新旧測定を明示して比較する。LMと自己履歴生成はこの欠落と独立に実測している。
