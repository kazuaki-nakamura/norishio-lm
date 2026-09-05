---
type: OpenQuestion
title: 未解決課題と次の実験
status: draft
generated: { by: ai-assisted-source-review, at: 2026-09-05 }
stale_after: 2026-10-05
sources:
  - id: handoff
    resource: docs/handoff.md
    title: 未実装と検証上の限界 / 次の実装候補
  - id: architecture
    resource: docs/architecture.md
    title: Candidate losses / Key falsification experiments
---

# 未解決課題と次の実験

- 未解決: 辞書入力の厳密な検証、関係三つ組の検証、出典と信頼度の追跡。
- 未実装: 外部辞書・形態素解析アダプター、文脈による語義選択。
- 要確認: 曖昧語・誤誘導する字形・未知複合語・言い換えを含む評価セットと分割方法。
- 未実装: 層の有効・無効設定、テンソル化、CPU ベースライン、同条件の ablation。
- 未確認: 明示的な各意味層が単純なベースラインより役立つか。デモや単体テストの成功で代替しない。

実験の順序は [意味層の分離](semantic-separation.md) と既存 architecture に従う。
