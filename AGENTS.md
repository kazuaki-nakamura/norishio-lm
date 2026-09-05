# AGENTS.md

## Project intent

Norishio-LM is a research codebase for testing a vertically integrated semantic representation stack:

`surface -> morphology -> sub-character -> lexical sense -> sememe -> concept -> generation`

The repository must remain experimentally falsifiable. Do not add semantic layers merely because they are elegant.

## Non-negotiable design rules

1. **Do not equate glyph decomposition with modern semantics.**
   - Example: `性 = 忄 + 生` is orthographic/etymological structure.
   - It must not be treated as proof that modern `性` literally means "a heart being born".
2. Keep separate channels for:
   - surface form
   - morphology
   - sub-character/glyph structure
   - etymology
   - modern lexical senses
   - sememes
   - concepts/relations
3. Every added semantic channel must have an ablation path.
4. Prefer interpretable intermediate schemas over magic constants.
5. Unit tests should include ambiguous words and intentionally misleading character decompositions.
6. v0.x should be CPU-runnable. Do not introduce large-model training as a prerequisite for basic tests.

## Initial implementation order

1. Semantic schema and compiler
2. Lexicon adapters
3. Layer encoders
4. Fusion module
5. Explicit concept bottleneck
6. Tiny causal LM baseline
7. Multi-task losses
8. Ablation harness
9. Japanese compositional benchmark

## First benchmark examples

- `性格`, `性質`, `性別`, `性器`, `可能性`, `人間性`
- paraphrase family: `また会いたい`, `明日もいてほしい`, `離れたくない`
- novel/rare compounds where character cues may or may not help

## Coding style

- Python 3.11+
- typed public APIs
- dataclasses / pydantic-like explicit schemas before tensorization
- pytest
- small, inspectable modules
- no hidden network access in tests

## AI project foundation

- Use the repository's Python environment (`.venv/Scripts/python.exe` on Windows).
- Start with `python codex/tools/check_knowledge.py --mode quick`. If the index is absent, run `python codex/tools/build_context.py`. Invalid indexes are not evidence: inspect original sources and resolve differences before rebuilding.
- Enter via `okf/index.md` and read only relevant concepts. With MCP, use `okf_search` -> `okf_get_concept` -> `okf_trace` as needed. Verify important claims against the cited source and section.
- OKF records development knowledge. It is separate from the model's lexical senses, sememes, and concept bottleneck; do not silently feed it into training or benchmarks.
- Distinguish implemented behavior, research hypotheses, test observations, and unverified claims. A hand-authored dictionary is not learned model output.
- When implementation or evidence changes, review affected OKF concepts, update sources, generated.at, status, stale_after and okf/log.md. Never add human verification without explicit human review.
- After source changes rebuild the metadata index; after OKF changes run `validate_okf.py` and `check_knowledge.py --mode full`. Use scripts for hashes and counts; do not load the entire manifest into conversation.
- Generated outputs belong in ignored `codex/work_output/`; do not edit them manually. Keep secrets, personal contacts, conversation transcripts, and training artifacts out of OKF and indexes.
- Record recurring operational failures as improvement candidates, without treating a missing metric as zero. Do not automatically escalate publishing, authentication, paid compute, or destructive actions.
- Run `python -m pytest` for both compiler and foundation tests, and the project MCP live check documented in `docs/ai-foundation.md` when MCP behavior or knowledge changes.

## Delegation to Luna

- Proactively delegate concrete, independent subtasks to `gpt-5.6-luna`. This is an explicit project request for sub-agent delegation, including automated Issue development.
- The parent owns requirement interpretation, decomposition, integration, final verification, and GitHub state changes. One Issue worker may coordinate several sub-agents; sub-agents must not claim other Issues or create automations/tasks.
- Give each sub-agent the exact worktree, applicable rules, bounded scope, owned files, acceptance criteria, and expected deliverable. Use focused context (`fork_turns="none"` or a bounded history) when selecting Luna explicitly.
- Prefer independent implementation, regression-test design, or review work that can proceed alongside useful parent work. Avoid overlapping file ownership. The parent inspects all returned changes and verifies the integrated result rather than assuming a sub-agent's success report is sufficient.
- Account for handoff, verification, and rework cost. Keep trivial edits and inseparable work local; briefly record the reason when a substantial task cannot use delegation.
- If Luna is unavailable, report that fact and continue locally within the authorized scope; do not silently claim delegation or substitute another model. Record actual delegate model, assigned scope, and verification in the Issue/PR result when delegation was used.
