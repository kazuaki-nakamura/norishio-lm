# Head prediction and decoder following results (Issue #26)

事前計画 `c95375e`、実装 `0c0141abaee77dc594b64cc35809afaabfffd14c`。
条件・式・境界は[事前計画](head-following.md)。測定日2026-09-11。
Dの42689parameter・既消費prefixによるgateを固定。seed7は過去Dの8条件/25pairとstateを厳密再現。
追加17/29はcommonモデル初期化とsampling seedを変更。拡張head/projection seed20は全seed共通。
各600更新、batch16、Adam .003、clip1、CPU1thread、既存4損失各1 + head CE各1。
train450のみfit、validation150全未見pair、test未評価。追加探索・再校正なし。

## 主要観測

- seed7/17/29の通常両slot一致は全て0/150、両head goldは48/46/64（平均52.667/150）。
  部分的なoracle追随は3seedで残るが、通常経路での同時保持は未解決。
- source head joint exactは0/2/0（/150）。人物とtimeの誤り相関は約-0.648/-0.583/-0.455。
  同じrowでの成功が少ない。ただしdecoderにはsoft分布を渡しているので、argmax指標だけで情報の欠如や原因を断定しない。
- confident-correct train263例でも生成両slotは210/263、全frame56/263。confident-wrongは0例で比較不能。
- seed7のevent入力zero + 両head goldは両slot48→72だが全frame16→12、LMも悪化。
  slot保持だけを最適化して意味入力の有効性を判断しない。

## 各seedの通常生成と両head gold

以下は全validation150例。head jointは常にsource予測を採点し、goldを加点しない。

| seed | condition | LM | 人物 | time | 両方 | 全frame | EOS rate | UTF8 rate | 種類 | parse | head joint /150 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 | predicted | 0.164825604 | 57 | 43 | 0 | 0 | 1.000000000 | 1.000000000 | 26 | 120 | 0 |
| 7 | both_gold | 0.104085495 | 48 | 99 | 48 | 16 | 1.000000000 | 1.000000000 | 20 | 99 | 0 |
| 17 | predicted | 0.190100016 | 44 | 52 | 0 | 0 | 0.960000000 | 0.960000000 | 31 | 114 | 2 |
| 17 | both_gold | 0.151850218 | 46 | 114 | 46 | 12 | 0.960000000 | 0.960000000 | 22 | 114 | 2 |
| 29 | predicted | 0.181583041 | 27 | 43 | 0 | 0 | 1.000000000 | 1.000000000 | 31 | 102 | 0 |
| 29 | both_gold | 0.113778301 | 64 | 103 | 64 | 20 | 1.000000000 | 1.000000000 | 22 | 103 | 0 |

### 3 seedの算術平均

各countは150例あたりの平均正解数。n=3、独立大標本や一般性能の証明ではない。

| condition | lm | eos_rate | valid_utf8_rate | unique_sequences | participant_correct | time_correct | both_correct | fullframe_correct | head_joint_correct |
|---|---|---|---|---|---|---|---|---|---|
| predicted | 0.178836220 | 0.986666667 | 0.986666667 | 29.333333333 | 42.666666667 | 46.000000000 | 0.000000000 | 0.000000000 | 0.666666667 |
| both_gold | 0.123238005 | 0.986666667 | 0.986666667 | 21.333333333 | 52.666666667 | 105.333333333 | 52.666666667 | 16.000000000 | 0.666666667 |

## 未調整headのcalibrationと誤り相関

Brierは5class平方誤差の和をrow平均、NLLは自然対数。ECEはconfidence10等幅bin、温度調整なし。

| seed | head | accuracy | balanced | Brier | NLL | ECE |
|---|---|---|---|---|---|---|
| 7 | participant | 0.380000000 | 0.380000000 | 0.766639184 | 1.563562098 | 0.213591435 |
| 7 | time | 0.406666667 | 0.406666667 | 0.862014994 | 1.595394096 | 0.300854457 |
| 17 | participant | 0.386666667 | 0.386666667 | 0.722243442 | 1.568726396 | 0.207475566 |
| 17 | time | 0.393333333 | 0.393333333 | 0.736530818 | 1.570805321 | 0.208370125 |
| 29 | participant | 0.153333333 | 0.153333333 | 0.945633479 | 1.829246711 | 0.362314173 |
| 29 | time | 0.533333333 | 0.533333333 | 0.791638190 | 1.541214491 | 0.324676981 |

| mean head | accuracy | balanced | Brier | NLL | ECE |
|---|---|---|---|---|---|
| participant | 0.306666667 | 0.306666667 | 0.811505369 | 1.653845068 | 0.261127058 |
| time | 0.444444444 | 0.444444444 | 0.796728001 | 1.569137969 | 0.277967187 |

| seed | both correct | 人物only | time only | both wrong | error covariance | error correlation |
|---|---|---|---|---|---|---|
| 7 | 0 | 57 | 61 | 32 | -0.154533333 | -0.648135587 |
| 17 | 2 | 56 | 57 | 35 | -0.138755556 | -0.583281613 |
| 29 | 0 | 23 | 80 | 47 | -0.081777778 | -0.454944094 |

joint確率は2つのheadの外積による独立近似で、学習したjoint分布ではない。
gold joint logprobは各headのgold確率を個別に1e-12でfloorしてlogの和、entropyは各head entropyの和。
marginは外積25classのtop1-top2。各rowの値・確率・gold/predicted IDとECE全binはreport保存。

| seed | mean gold joint logprob | mean joint entropy | mean joint top margin |
|---|---|---|---|
| 7 | -3.158956195 | 2.050871580 | 0.165193049 |
| 17 | -3.139531717 | 1.960644608 | 0.116888161 |
| 29 | -3.370461201 | 1.966844286 | 0.125336750 |

### Gold pair別のhead失敗

表はvalidationにある5pair。reportには25cell全てを保存し、存在しないcellはcount0/ratesnull。

| seed | 人物 | time | rows | both correct | 人物only | time only | both wrong |
|---|---|---|---|---|---|---|---|
| 7 | 先輩 | 週末 | 30 | 0 | 24 | 0 | 6 |
| 7 | 友人 | 明日 | 30 | 0 | 0 | 23 | 7 |
| 7 | 同僚 | 今日 | 30 | 0 | 24 | 4 | 2 |
| 7 | 知人 | 来週 | 30 | 0 | 9 | 4 | 17 |
| 7 | 隣人 | 来月 | 30 | 0 | 0 | 30 | 0 |
| 17 | 先輩 | 週末 | 30 | 2 | 28 | 0 | 0 |
| 17 | 友人 | 明日 | 30 | 0 | 0 | 30 | 0 |
| 17 | 同僚 | 今日 | 30 | 0 | 23 | 0 | 7 |
| 17 | 知人 | 来週 | 30 | 0 | 5 | 0 | 25 |
| 17 | 隣人 | 来月 | 30 | 0 | 0 | 27 | 3 |
| 29 | 先輩 | 週末 | 30 | 0 | 17 | 0 | 13 |
| 29 | 友人 | 明日 | 30 | 0 | 0 | 29 | 1 |
| 29 | 同僚 | 今日 | 30 | 0 | 6 | 0 | 24 |
| 29 | 知人 | 来週 | 30 | 0 | 0 | 21 | 9 |
| 29 | 隣人 | 来月 | 30 | 0 | 0 | 30 | 0 |

## Seed7 decoder追随をgold pair別に分離

各pair30例、parse失敗も分母に残す。

| condition | 人物 | time | rows | 人物正解 | time正解 | 両方 | 全frame |
|---|---|---|---|---|---|---|---|
| predicted | 先輩 | 週末 | 30 | 23 | 0 | 0 | 0 |
| predicted | 友人 | 明日 | 30 | 4 | 15 | 0 | 0 |
| predicted | 同僚 | 今日 | 30 | 24 | 0 | 0 | 0 |
| predicted | 知人 | 来週 | 30 | 6 | 4 | 0 | 0 |
| predicted | 隣人 | 来月 | 30 | 0 | 24 | 0 | 0 |
| participant_gold | 先輩 | 週末 | 30 | 24 | 0 | 0 | 0 |
| participant_gold | 友人 | 明日 | 30 | 24 | 15 | 15 | 3 |
| participant_gold | 同僚 | 今日 | 30 | 24 | 0 | 0 | 0 |
| participant_gold | 知人 | 来週 | 30 | 20 | 0 | 0 | 0 |
| participant_gold | 隣人 | 来月 | 30 | 3 | 24 | 3 | 2 |
| time_gold | 先輩 | 週末 | 30 | 0 | 24 | 0 | 0 |
| time_gold | 友人 | 明日 | 30 | 0 | 24 | 0 | 0 |
| time_gold | 同僚 | 今日 | 30 | 20 | 24 | 20 | 6 |
| time_gold | 知人 | 来週 | 30 | 0 | 24 | 0 | 0 |
| time_gold | 隣人 | 来月 | 30 | 0 | 24 | 0 | 0 |
| both_gold | 先輩 | 週末 | 30 | 0 | 24 | 0 | 0 |
| both_gold | 友人 | 明日 | 30 | 24 | 24 | 24 | 8 |
| both_gold | 同僚 | 今日 | 30 | 24 | 24 | 24 | 8 |
| both_gold | 知人 | 来週 | 30 | 0 | 3 | 0 | 0 |
| both_gold | 隣人 | 来月 | 30 | 0 | 24 | 0 | 0 |

### Factorial 25pair追随 /150

各validation sourceへ指定one-hot pairを介入したcounterfactual。通常reference精度と混同しない。
T=train seen、V=validation unseen、N=neither train nor validation。Nのtest rowは参照しない。

| 人物 | 今日 | 明日 | 来月 | 来週 | 週末 |
|---|---|---|---|---|---|
| 先輩 | 120 T | 120 N | 120 T | 120 T | 0 V |
| 友人 | 120 N | 120 V | 120 T | 120 T | 120 T |
| 同僚 | 120 V | 120 T | 120 N | 120 T | 120 T |
| 知人 | 120 T | 120 T | 120 T | 0 V | 0 N |
| 隣人 | 120 T | 120 T | 0 V | 0 N | 120 T |

## Confident train subsets

両head max>=0.8を満たすtrain rowだけでjoint正誤を分け、自分自身のtrain source/baseで生成。
validationラベルでsubsetを作らず、閾値の引下げもない。選択済みtrain診断で一般化評価ではない。

| subset | train rows | LM | 人物 | time | 両方 | 全frame | EOS rate | UTF8 rate | 種類 | parse |
|---|---|---|---|---|---|---|---|---|---|---|
| confident_correct | 263 | 0.083577629 | 210 | 210 | 210 | 56 | 1.000000000 | 1.000000000 | 46 | 210 |
| confident_wrong | 0 | null | null | null | null | null | null | null | null | null |

両subsetに入らないtrain row: 187/450。空群の成績はnull。

## Seed7 semantic base入力ablation

retained21全zero、event群zero、operators群zeroを推論時だけ比較。元の人物/time旧群は常に切断。
確率を再正規化せず入力zeroとし、base projection biasは残す。head予測・重みは不変。
上流の意味層全体の有効性を示す実験ではなく、旧concept入力groupの除去診断。

| condition | LM | 人物 | time | 両方 | 全frame | EOS rate | UTF8 rate | 種類 | parse |
|---|---|---|---|---|---|---|---|---|---|
| full_predicted | 0.164825604 | 57 | 43 | 0 | 0 | 1.000000000 | 1.000000000 | 26 | 120 |
| full_both_gold | 0.104085495 | 48 | 99 | 48 | 16 | 1.000000000 | 1.000000000 | 20 | 99 |
| base_zero_predicted | 0.449051149 | 0 | 0 | 0 | 0 | 1.000000000 | 1.000000000 | 1 | 0 |
| base_zero_both_gold | 0.408637086 | 0 | 0 | 0 | 0 | 1.000000000 | 1.000000000 | 1 | 0 |
| event_zero_predicted | 0.181177319 | 62 | 36 | 0 | 0 | 1.000000000 | 1.000000000 | 13 | 111 |
| event_zero_both_gold | 0.118093764 | 72 | 96 | 72 | 12 | 1.000000000 | 1.000000000 | 14 | 96 |
| operators_zero_predicted | 0.169646553 | 59 | 43 | 0 | 0 | 1.000000000 | 1.000000000 | 20 | 120 |
| operators_zero_both_gold | 0.107049367 | 48 | 96 | 48 | 12 | 1.000000000 | 1.000000000 | 14 | 96 |

byte NLL/rank/argmaxは正解過去履歴でreport保存。self baselineのため感度項0は介入不変の証拠ではない。

全base入力zeroはpredicted/both-goldともparse0、生成1種類。保存tokenを既知文頭と照合する補足集計で、
認識可能prefixは各0/150（full both-goldは150/150）だった。gateは後方を再検索しないため、
この自由生成では文頭の崩壊に伴いslot注入自体が開始しない。正解headが供給されても使われない経路であり、
「slot情報だけでは意味を表せない」という一般的結論にはしない。正解過去履歴のLMではgate条件が異なる。

## 保存と検証

report SHA256: `964ab4d7dd7fc8b2fdddac816af51ac2ea8ab28a806a33dff32e40631250170f`

### seed17

- 初期state: `fc27e573f5145510038a967534eb293ea59b4e5f3ebd6c2d421b5dc33a794f1f`
- sampling: `aff617f4afe36027c5ab34c00f01f5bda171ce15fdb6b6f9ba099dd572240261`
- checkpoint SHA256: `bb3dc4c5d83d0e6622bdaf0f94bfbd8f925d2cb29460091d7a27c162a3be5614`
- 最終state: `e95fb37a3986ad5916115e32a2efb7e56e8946b29332dcc384af6ee269978978`
- CPU学習秒数: 44.881（速度比較の証明ではない）
- 再読込: `{'state': True, 'old_concepts': True, 'heads': True, 'logits': True, 'greedy': True}`

### seed29

- 初期state: `e586768c5ed9b3e726c2be0dfe55f87ff947fcbaaa85636dac6f8eea693e460b`
- sampling: `d606e8d0cbffeb457d55c1c53ddda54aacfdc26db103a111ce695f3af493b0b3`
- checkpoint SHA256: `fd6d79852acb39ea2f7c0c56571744449085449e0f19f05946866dabea56803e`
- 最終state: `3ab8909a77182f5b5b6754050b51db01b48bb557e4383a731e408d55150a6992`
- CPU学習秒数: 44.661（速度比較の証明ではない）
- 再読込: `{'state': True, 'old_concepts': True, 'heads': True, 'logits': True, 'greedy': True}`

seed7のstate不変、歴史評価完全一致、新seedのstate/head/logits/greedy再読込完全一致。
出力はignored `codex/work_output/issue26-fixed-v1/`。コマンド・テスト結果は[handoff](handoff.md)。
3seedでも既知文法・同じデータ・共通拡張seed20に依存する。誤り相関は因果関係を意味しない。
入力zeroは分布外介入で、再学習ablationの代替ではない。人の検証印や一般意味理解の主張を追加しない。
