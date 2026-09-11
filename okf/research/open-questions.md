---
type: OpenQuestion
title: 未解決課題と次の実験
status: draft
generated: { by: ai-assisted-source-review, at: 2026-09-11 }
stale_after: 2026-10-11
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
- 未解決: gold conceptでも同一文となるdecoderの条件利用、participant/timeと否定作用域、複数seedでの再現。
- 未確認: 複数seed・別作者/未見テンプレートでの再現、soft conceptの余剰情報と因果的意味。
- 未確認: 明示的な各意味層が単純なベースラインより役立つか。デモや単体テストの成功で代替しない。

実験の順序は [意味層の分離](semantic-separation.md) と既存 architecture に従う。
