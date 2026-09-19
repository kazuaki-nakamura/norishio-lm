# Post-v3 research status

Status at merge `9331eda9d918adbc1b415e01fddeae62249cb5ed` (PR #41).
This note separates implemented wiring, measured observations, unavailable
corrections, unmeasured questions, and research hypotheses. The v1 concept-toy,
benchmark v2, and benchmark v3 fixtures differ; their scores are not one
performance-improvement curve.

| Area | Status | Evidence boundary | Implementation / result provenance |
| --- | --- | --- | --- |
| Semantic schema and compiler | **Implemented** | Schema 1.0 validates records, ambiguity, provenance, JSON round trips, and layer removal. Authored dictionary output is not learned output. | [`docs/semantic-schema.md` § 入力境界と版](semantic-schema.md#入力境界と版); `f590228a1bc86bac1d123226060341e38b3ec443` |
| Ten-channel tensorizer and encoder | **Implemented wiring** | Candidate associations and provenance are retained; the CPU demo is randomly initialized. Contextual sense selection and layer usefulness are unmeasured. | [`docs/multichannel-encoder.md` § Limits and next measurements](multichannel-encoder.md#limits-and-next-measurements); `86b4c913e7cbd19e82bdcfb0f427eead0b4d1367`, provenance fix `3e38c7bf94b3b90f2233cef189500dc3da2b0c2c` |
| v1 concept and slot diagnostics | **Measured on authored toy data** | The frozen encoder's source/concept vectors were decodable in the recorded toy probe, while ordinary generation repeatedly fails joint participant/time retention. Issue #32 isolates the dedicated-head LM gradient but yields only 1/450 ordinary joint successes in G1. This is not evidence for semantic-layer or LLM quality. | [`docs/handoff.md` § SourceEncoder/Conceptの観測](handoff.md#sourceencoderconceptの観測); [`docs/gradient-routing-results.md` §§ Generation and slot metrics / Limits and next step](gradient-routing-results.md#generation-and-slot-metrics); probe implementation `bd859c406a55e4a027705598e73aa9cbddcbfbd0`, routing implementation `e89abc240198eb401b1fbe3e19bf3d327750b25c`, result-document commit `e54e62c80a39b4dbcd7616590fad9466b1e2ac17`, correction `ea79500f91e524be4b0d020b614c541f8cc6b1e1` |
| Benchmark v2 tournament | **Implemented and measured** | Six arms × three seeds completed and the final holdout was consumed once. Arm E's historical intermediate/head/2×2 values are **uncorrectable** because the saved aggregate lacks raw logits, row predictions, and a full confusion matrix. Generation and the frozen ranking remain recorded. | [`docs/results/benchmark-v2-final.md` §§ Three-seed arithmetic means / Erratum: E intermediate label space](results/benchmark-v2-final.md#three-seed-arithmetic-means); protocol `19ce7145495a47b40a178255272050caa615dc79`, tournament `38e363e1b4e836b890f1c852f6d0a45fe3119296`, execution `565c2bf1f8b22435f21a74d5833dfeb4821d8174`, result-document commit `a51d1b8c707f7afff71fcc2641411616913d4d20`, artifact SHA-256 `1d21b4cfa950c3c96c50ca0647e33308bb4d0653c18bcc146560ce1da4266554` |
| Benchmark v3 tournament | **Implemented and measured** | Six arms × three seeds completed. The corrected primary is `all.generation_frame_exact.accuracy`. H1L1 ranks first on the authored fixture, but every source-conditioned arm remains much weaker on unseen pairs than on pair-seen/unseen-triple rows. | [`docs/results/benchmark-v3-final.md` §§ Frozen selection result / Support-group split](results/benchmark-v3-final.md#frozen-selection-result); fixture `ae8ea2727558a57f6fa7c0e23ad28fc767dcc859`, architecture `bc69ac3c132c77d9613481ae163b1e8c4342a9eb`, execution `50577091f1aaa5b90045ab55bace25b5f69f5a52`, development-result commit `5be2c9dde1714b5d9826e7e3569493b2540913af`, final-result commit `119c6298a49213a156923c435edfab7233d5e9d0`, final artifact SHA-256 `eb94b2b35d9de0c575b90342d517582e2304c766b3d8bb385ea22f2ac8c56a59` |
| v3 metric and intervention erratum | **Corrected by read-only audit** | Correcting the primary path does not change the ranking. Historical main-arm intervention is 0/48 target changes under **fixed class 0**. Baseline probability vectors were not retained, so baseline-argmax grouping is **unavailable and uncorrectable**; 0/48 is not alternate-class evidence. | [`docs/results/benchmark-v3-final.md` § Protocol erratum](results/benchmark-v3-final.md#protocol-erratum-issue-39); audit `f68d418b7e3bc5cf28f82d73ad3bd8e04158749d`, merged result `9331eda9d918adbc1b415e01fddeae62249cb5ed` |
| Future v3 execution contract | **Implemented protocol; unmeasured experiment** | A separate descriptor binds the corrected selection paths and validates future execution callbacks. The alternate rule selects `(baseline_argmax + 1) % width` and binds the actual one-hot class. No run has used it. | [`docs/factor-path-tournament-v3.md` § Post-merge protocol erratum and future contract](factor-path-tournament-v3.md#post-merge-protocol-erratum-and-future-contract); binding `77d24f1c7e8a0939bec5c689c50141f1067d51c0`, actual-class validation `d6e7451bd384e7649c464b2d15c1116df9443a26` |

## Current unknowns

- Whether a correctly predicted factor is actually followed by the decoder under
  a true alternate-class intervention is **unmeasured**.
- The cause of the unseen-pair gap is unresolved: factor-head error, decoder
  follow-through, surface realization, or fixture-specific grammatical priors
  may each contribute.
- External dictionary and morphology adapters, contextual lexical-sense
  selection, ordered/graph encoders, and sourced semantic-layer ablations remain
  unimplemented or unmeasured.
- The usefulness of morphology, sub-character structure, lexical senses,
  sememes, and concepts remains a **hypothesis**. Glyph and etymology channels
  cannot establish modern lexical meaning.

The next factor-path proposal is [review-unapproved](factor-path-next-experiment.md).
It does not authorize data generation, training, inference, or access to any
consumed final split.
