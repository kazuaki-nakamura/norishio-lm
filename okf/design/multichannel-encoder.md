---
type: Implementation
title: Multi-channel encoder の実装範囲
status: draft
generated: { by: ai-assisted-source-review, at: 2026-09-10 }
stale_after: 2026-10-10
sources:
  - id: tensorizer
    resource: src/norishio_lm/tensorizer.py
    title: SemanticTensorizer / Feature / SemanticBatch
  - id: encoder
    resource: src/norishio_lm/encoder.py
    title: MultiChannelEncoder / EncoderState
  - id: tests
    resource: tests/test_encoder.py
    title: Masks, gradients and semantic independence
  - id: tensor-tests
    resource: tests/test_tensorizer.py
    title: Vocabulary, provenance and candidate boundaries
  - id: offline-tests
    resource: tests/test_encoder_offline.py
    title: Offline forward/backward and candidate ID separation
  - id: contract
    resource: docs/multichannel-encoder.md
    title: Channel contract and limitations
  - id: observations
    resource: docs/handoff.md
    title: Issue 2 verification observations
---

# Multi-channel encoder の実装範囲

表記から関係まで10チャネルを独立語彙でtensor化。出典と候補対応はmetadataに保持する。
PADとUNKは別ID。fit後のencodeで語彙は増やさず、training splitだけでfitする。
独立Embedding・masked mean・projectionから入力依存gateを計算して融合する。
字形/字源は語義ラベルを書き換えず、欠損・無効チャネルの寄与はゼロ。
全無効時の有限ゼロ、勾配分離、オフライン動作をテストで確認する。

ランダム初期値のforwardのみ。性能改善、WSD、学習済み意味理解の根拠ではない。
順序と候補対応を構造として学習するencoder、graph encoder、bottleneck、生成は未実装。
意味層の有効性は [比較実験](../research/semantic-separation.md) で検証が必要。
