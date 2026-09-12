---
type: OpenQuestion
title: 未解決課題と次の実験
status: draft
generated: { by: ai-assisted-source-review, at: 2026-09-12 }
stale_after: 2026-10-12
sources:
  - id: handoff
    resource: docs/handoff.md
    title: 未実装と検証上の限界 / 次の実装候補
  - id: architecture
    resource: docs/architecture.md
    title: Candidate losses / Key falsification experiments
---

# 未解決課題と次の実験

- 実装済み: 辞書入力と関係三つ組の検証、由来・出典・版の追跡。根拠のない信頼度は補わない。
- 未解決: 出典本文の妥当性確認、関係端点のオントロジー検証、評価可能な信頼度の定義。
- 未実装: 外部辞書・形態素解析アダプター、文脈による語義選択。
- 要確認: 曖昧語・誤誘導する字形・未知複合語・言い換えを含む評価セットと分割方法。
- 実装済み: レコードの補助層を除外する API。
- 実装済み: 出典と候補を保持する10チャネルのテンソル化、独立encoder、masked gated fusion、CPU forward。
- 実装済み: authored toy用CPU学習、A/B/C比較、損失再学習ablation、チャネル除去・概念介入測定。
- 未実装: 順序/グラフを反映するencoder、外部辞書による意味層の比較。
- 実装済み: 同初期値の定数concept再学習とvalidation自由生成診断。短期学習では正常生成に失敗。
- 観測: 60/600更新比較でbyte/EOS失敗は600で回復、初期学習量不足の寄与を支持。
- 観測: 凍結encoderの線形probeは多数派を超え、encoder/soft conceptは150通りだが生成は1通り。
- 観測: 初期状態だけの条件注入ではgold conceptでも同一文となる。participant/timeと否定作用域、複数seedは未解決。
- 観測: per-step加算の同容量対照では9種類へ分化し、後半conditioning感度が残る。
- 未解決: per-stepでも全slot一致0/150、人物・時点保持と文型外出力、oracle介入の崩壊。
- 観測: 凍結per-stepの人物/時点単独gold置換で自由生成の対象slotは回復しない。
- 観測: Issue #16の人物直前prefixでも人物0/150、時点直前は30/150。履歴修復だけでは回復しない。
- 未解決: decoderのslot表現・出力head・学習目的の原因分離。counterfactual prefix、長さ/内容効果、soft/oracle分布差。
- 観測: slot CE重み1の同容量対照は人物26/時点25へ微増、全frame0のまま、全体LM/EOSは悪化。
- 実装済み: 専用slot head対照、head介入とlogit感度、C保存物再現。
- 観測: C通常は人物36/time25、全frame0。単独oracleの対象slot回復と他slot崩壊が併存。
- 未解決: 人物/time同時保持、組合せ汎化、元conceptと専用headの重複、容量/目的の寄与分離。
- 観測: Issue #22はvalidation全150例がunseen pairで、seen群は0例。分割を変更せず率nullと記録。
- 観測: 数学的に同値な投影分割Bは人物40/time25、両slot0。人物head goldでtime1となり干渉が残る。
- 未解決: 直交注入、重複除去と局所注入の交絡分離。seen/unseen差は現分割で推定不可。
- 実装済み: Issue #24の旧人物/time除去C、既生成prefixに基づく局所注入D。未来やgold spanは参照しない。
- 観測: 通常両slotはC/Dとも0、両head goldはC6/D48（/150）。Dでも未見5組中3組の追随失敗は残る。
- 未解決: source headの同時予測（補足集計D0/150）とdecoderの未見pair追随、文法prior依存とh0/注入時刻の原因分離。
- 観測: Issue #26はseed7/17/29で通常両slot全て0、両head gold48/46/64。head jointは0/2/0。
- 観測: confident train correct263例でも両slot210、wrong群は0例で比較不能。全base zeroは文頭崩壊でgate未開始。
- 未解決: soft/hard headの情報差、headの組合せ一般化と文法prefix依存の分離。入力group推論ablationと意味層再学習は別。
- 未確認: 拡張headのseedも独立にした再現、別作者/未見テンプレート、soft conceptの余剰情報と因果的意味。
- 観測: Issue28の25-way headはtrain高精度だが未見pair全seed0、通常生成両slotも0。gold平均48/150。
- 未解決: 非等価なcompositional目的の設計、容量/損失/入力分布の効果分離。単純外積はIssue30で等価性を確認。
- 修復済み: Issue26の到達不能byte診断をIssue30で明示opt-in v2にしseed7を初測定。旧report/nullは保持。
- 観測: 外積CはAと数学的に同じ周辺を返し、全3seed生成一致。未見pair top10 C90%でも通常両slot0は不変。
- 未実施: 既存2CEの和と非等価な組合せ目的/表現設計。factorized NLLの単純追加を新しい学習制約と呼ばない。
- 未確認: 明示的な各意味層が単純なベースラインより役立つか。デモや単体テストの成功で代替しない。

実験の順序は [意味層の分離](semantic-separation.md) と既存 architecture に従う。
