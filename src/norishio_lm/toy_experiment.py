"""Seeded CPU conditional-generation measurement on authored fixtures.

Requires a source checkout containing data/issue3. Reports are observations,
never a claim of general language understanding.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time
from typing import Any

import torch

from .tensorizer import CHANNELS, SemanticTensorizer
from .encoder import EncoderConfig, MultiChannelEncoder
from .concept_model import (CONCEPT_FIELDS, ConceptBottleneck, ConceptVocabulary,
                            TinyConceptDecoder, encode_targets)
from .toy_adapter import source_record, misleading_glyph_fixture_record


def fixture_module(name: str) -> Any:
    path = Path(__file__).resolve().parents[2] / "data" / "issue3" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"norishio_fixture_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source-checkout fixture: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pad(rows: list[list[int]], fill: int = 0) -> torch.Tensor:
    """Right padding leaves causal histories unchanged; labels use -100."""
    if not rows or any(not row for row in rows):
        raise ValueError("nonempty sequences required")
    result = torch.full((len(rows), max(map(len, rows))), fill, dtype=torch.long)
    for i, row in enumerate(rows):
        result[i, :len(row)] = torch.tensor(row, dtype=torch.long)
    return result


class ToyModel(torch.nn.Module):
    """Composition root: C has only named concept probabilities at the boundary."""

    def __init__(self, pathway: str, tensorizer: SemanticTensorizer,
                 vocabulary: ConceptVocabulary, hidden_dim: int = 32) -> None:
        super().__init__()
        self.pathway = pathway
        self.encoder = None if pathway == "A" else MultiChannelEncoder(
            EncoderConfig(tensorizer.vocab_sizes, hidden_dim=hidden_dim))
        bottleneck = None if pathway == "A" else ConceptBottleneck(hidden_dim, vocabulary)
        self.decoder = TinyConceptDecoder(hidden_dim=hidden_dim, bottleneck=bottleneck)

    def forward(self, ids: torch.Tensor, semantic: Any, *,
                labels: torch.Tensor | None = None,
                channels: dict[str, bool] | None = None,
                intervention: str | torch.Tensor | None = None) -> tuple[Any, Any]:
        latent = self.encoder(semantic, ablation=channels).fused if self.encoder else None
        auxiliary = self.decoder.bottleneck(latent) if self.decoder.bottleneck else None
        kwargs: dict[str, Any] = {}
        if self.pathway == "B":
            kwargs["encoder_latent"] = latent
        if self.pathway == "C":
            probabilities = auxiliary.probabilities()
            if isinstance(intervention, torch.Tensor):
                probabilities = intervention
            elif intervention == "zero":
                probabilities = torch.zeros_like(probabilities)
            elif intervention == "hard":
                probabilities = torch.cat([
                    torch.nn.functional.one_hot(auxiliary.concept_logits[f].argmax(-1),
                                                auxiliary.concept_logits[f].shape[-1]).float()
                    for f in CONCEPT_FIELDS], dim=-1)
            elif intervention is not None:
                raise ValueError("unknown concept intervention")
            kwargs["concept_probs"] = probabilities
        elif intervention is not None:
            raise ValueError("concept interventions require C")
        return self.decoder(ids, pathway=self.pathway, labels=labels, **kwargs), auxiliary


def batch(rows: list[dict[str, Any]], pathway: str, tensorizer: SemanticTensorizer,
          corpus: Any, strict: Any) -> tuple[torch.Tensor, torch.Tensor, Any]:
    histories = [(strict.target_history(r) if pathway == "C" else corpus.teacher_forcing(r))
                 for r in rows]
    semantic = tensorizer.encode([source_record(corpus.model_inputs(r)) for r in rows])
    return (pad([h["input_ids"] for h in histories]),
            pad([h["labels"] for h in histories], -100), semantic)


@torch.no_grad()
def evaluate(model: ToyModel, rows: list[dict[str, Any]], tensorizer: SemanticTensorizer,
             vocabulary: ConceptVocabulary, corpus: Any, strict: Any,
             batch_size: int, **overrides: Any) -> dict[str, Any]:
    model.eval()
    lm_sum = 0.0
    token_count = 0
    loss_sums = {k: 0.0 for k in (*CONCEPT_FIELDS, "sense", "sememes")}
    counts = dict.fromkeys(loss_sums, 0)
    correct = dict.fromkeys(CONCEPT_FIELDS, 0)
    for start in range(0, len(rows), batch_size):
        part = rows[start:start + batch_size]
        ids, labels, semantic = batch(part, model.pathway, tensorizer, corpus, strict)
        output, auxiliary = model(ids, semantic, labels=labels, **overrides)
        n = int((labels != -100).sum())
        lm_sum += float(output["lm_loss"]) * n
        token_count += n
        if auxiliary is not None:
            target = encode_targets([r["targets"] for r in part], vocabulary)
            losses = model.decoder.bottleneck.loss(auxiliary, target)
            masks = {**target.masks, "sense": target.sense_mask, "sememes": target.sememe_mask}
            for key, mask in masks.items():
                count = int(mask.sum())
                counts[key] += count
                loss_sums[key] += float(losses[key]) * count
            for field in CONCEPT_FIELDS:
                correct[field] += int(((auxiliary.concept_logits[field].argmax(-1)
                                        == target.fields[field]) & target.masks[field]).sum())
    field_losses = {k: loss_sums[k] / counts[k] if counts[k] else None for k in loss_sums}
    present = [field_losses[f] for f in CONCEPT_FIELDS if field_losses[f] is not None]
    return {"rows": len(rows), "lm_tokens": token_count, "lm": lm_sum / token_count,
            "sense": field_losses["sense"], "sememe": field_losses["sememes"],
            "concept": sum(present) / len(present) if present else None,
            "concept_field_losses": {f: field_losses[f] for f in CONCEPT_FIELDS},
            "label_counts": counts,
            "concept_accuracy": sum(correct.values()) / sum(counts[f] for f in CONCEPT_FIELDS)
            if sum(counts[f] for f in CONCEPT_FIELDS) else None,
            "concept_field_accuracy": {f: correct[f] / counts[f] if counts[f] else None
                                       for f in CONCEPT_FIELDS}}


def train(pathway: str, weights: dict[str, float], rows: list[dict[str, Any]],
          tensorizer: SemanticTensorizer, vocabulary: ConceptVocabulary,
          corpus: Any, strict: Any, *, seed: int, steps: int,
          batch_size: int) -> tuple[ToyModel, dict[str, Any]]:
    torch.manual_seed(seed)
    model = ToyModel(pathway, tensorizer, vocabulary)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    generator = torch.Generator().manual_seed(seed)
    trace = []
    started = time.perf_counter()
    model.train()
    for step in range(steps):
        indices = torch.randint(len(rows), (batch_size,), generator=generator).tolist()
        part = [rows[i] for i in indices]
        ids, labels, semantic = batch(part, pathway, tensorizer, corpus, strict)
        target = encode_targets([r["targets"] for r in part], vocabulary)
        optimizer.zero_grad(set_to_none=True)
        output, auxiliary = model(ids, semantic, labels=labels)
        total = model.decoder.losses(output, targets=target, bottleneck_output=auxiliary,
                                     weights=weights)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == steps:
            aux = model.decoder.bottleneck.loss(auxiliary, target) if auxiliary else {}
            trace.append({"step": step + 1, "total": float(total.detach()),
                          "lm": float(output["lm_loss"].detach()),
                          **{k: float(aux[v].detach()) if v in aux else None
                             for k, v in (("sense", "sense"), ("sememe", "sememes"),
                                          ("concept", "concept"))}})
    return model, {"pathway": pathway, "weights": weights, "steps": steps,
                   "training_seconds": time.perf_counter() - started,
                   "parameters": sum(p.numel() for p in model.parameters()), "trace": trace}


@torch.no_grad()
def diagnostics(model: ToyModel, rows: list[dict[str, Any]], tensorizer: SemanticTensorizer,
                vocabulary: ConceptVocabulary, corpus: Any, strict: Any) -> dict[str, Any]:
    # Fixed concepts deliberately isolate the decoder from a changed source state.
    ids, labels, original = batch(rows[:1], "C", tensorizer, corpus, strict)
    _, aux = model(ids, original)
    concepts = aux.probabilities()
    changed = tensorizer.encode([source_record({"context": "changed", "text": "別のソース"})])
    first, _ = model(ids, original, intervention=concepts)
    second, _ = model(ids, changed, intervention=concepts)
    zero, _ = model(ids, original, intervention="zero")
    hard, _ = model(ids, original, intervention="hard")
    core = {"fixed_concept_changed_source_max_delta": float((first["logits"] - second["logits"]).abs().max()),
            "zero_concept_max_delta": float((first["logits"] - zero["logits"]).abs().max()),
            "hard_concept_max_delta": float((first["logits"] - hard["logits"]).abs().max())}
    # The diagnostic source pair is equal; only the explicit false glyph is added.
    fixture_row = next(r for r in rows if r["metadata"].get("perturbation"))
    ids, _, normal = batch([fixture_row], "C", tensorizer, corpus, strict)
    altered = tensorizer.encode([misleading_glyph_fixture_record(corpus.model_inputs(fixture_row))])
    for enabled in (True, False):
        switches = {"subcharacters": enabled}
        plain, _ = model(ids, normal, channels=switches)
        fixture, _ = model(ids, altered, channels=switches)
        core[f"false_glyph_enabled_{enabled}_max_delta"] = float((plain["logits"] - fixture["logits"]).abs().max())
    core["false_glyph_note"] = "untrained unknown feature diagnostic; not real etymology"
    return core


def run(*, seed: int = 7, steps: int = 60, batch_size: int = 16) -> dict[str, Any]:
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be positive")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    corpus, strict = fixture_module("toy_corpus"), fixture_module("strict_decoder")
    bundle = corpus.build()
    corpus.validate(bundle)
    tensorizer = SemanticTensorizer.fit([source_record(corpus.model_inputs(r)) for r in bundle["train"]])
    vocabulary = ConceptVocabulary.fit([r["targets"] for r in bundle["train"]])
    defaults = {"lm": 1.0, "sense": 1.0, "sememe": 1.0, "concept": 1.0}
    variants = [("A", "A", defaults), ("B", "B", defaults), ("C", "C", defaults)]
    variants += [(f"C_no_{key}", "C", {**defaults, key: 0.0}) for key in defaults]
    variants += [("C_lm_only", "C", {"lm": 1.0, "sense": 0.0, "sememe": 0.0, "concept": 0.0})]
    report: dict[str, Any] = {"seed": seed, "steps": steps, "batch_size": batch_size,
        "torch": torch.__version__, "device": "cpu", "threads": 1,
        "dataset": corpus.validate(bundle), "feature_vocab_sizes": tensorizer.vocab_sizes,
        "concept_vocabulary": {f: [{"value": v, "id": i} for v, i in vocabulary.fields[f].items()]
                               for f in CONCEPT_FIELDS},
        "scope": "single-seed teacher-forced conditional toy experiment, no model selection",
        "variants": {}}
    trained = {}
    # All configurations are predetermined. Test is untouched until every model is trained.
    for name, pathway, weights in variants:
        print(f"training {name}: {steps} CPU steps", flush=True)
        model, record = train(pathway, weights, bundle["train"], tensorizer, vocabulary,
                              corpus, strict, seed=seed, steps=steps, batch_size=batch_size)
        record["train"] = evaluate(model, bundle["train"], tensorizer, vocabulary, corpus, strict, batch_size)
        record["validation"] = evaluate(model, bundle["validation"], tensorizer, vocabulary, corpus, strict, batch_size)
        report["variants"][name] = record
        trained[name] = model
    model = trained["C"]
    report["validation_channel_removal"] = {
        name: evaluate(model, bundle["validation"], tensorizer, vocabulary, corpus, strict,
                       batch_size, channels={name: False}) for name in CHANNELS}
    report["validation_concept_interventions"] = {
        name: evaluate(model, bundle["validation"], tensorizer, vocabulary, corpus, strict,
                       batch_size, intervention=name) for name in ("zero", "hard")}
    report["diagnostic"] = evaluate(model, bundle["diagnostic"], tensorizer, vocabulary,
                                     corpus, strict, batch_size)
    report["boundary_diagnostics"] = diagnostics(model, bundle["diagnostic"], tensorizer, vocabulary, corpus, strict)
    # Readable, per-row predictions; authored gold is labeled separately for inspection.
    sample = bundle["validation"][:3]
    ids, _, semantic = batch(sample, "C", tensorizer, corpus, strict)
    with torch.no_grad():
        _, aux = model(ids, semantic)
    report["concept_examples"] = [
        {"source": corpus.model_inputs(r), "authored_gold": r["targets"]["concept"],
         "predicted": vocabulary.inspect({f: logits[i:i+1] for f, logits in aux.concept_logits.items()})}
        for i, r in enumerate(sample)]
    for name, model in trained.items():
        report["variants"][name]["test"] = evaluate(model, bundle["test"], tensorizer,
                                                       vocabulary, corpus, strict, batch_size)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists; choose a new report path")
    report = run(seed=args.seed, steps=args.steps, batch_size=args.batch_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()
