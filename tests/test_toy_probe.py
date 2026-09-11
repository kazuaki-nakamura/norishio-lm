from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.concept_model import CONCEPT_FIELDS, ConceptVocabulary
from norishio_lm.toy_probe import train_probe


def _targets(values: list[int | None]) -> list[dict]:
    return [{"concept": {"event": value}} for value in values]


def test_probe_detaches_inputs_and_keeps_seven_ordered_heads() -> None:
    targets = _targets(["a", "b"])
    vocab = ConceptVocabulary.fit(targets)
    latents = torch.tensor([[-2.0, 0.0], [2.0, 0.0]], requires_grad=True)
    before = latents.detach().clone()
    probe, metadata = train_probe(latents, targets, vocab, steps=2)
    output = probe(latents)
    assert list(output) == list(CONCEPT_FIELDS)
    assert all(output[f].shape == (2, len(vocab.fields[f]) + 1) for f in CONCEPT_FIELDS)
    assert latents.grad is None
    assert torch.equal(latents, before)
    assert metadata["input"]["detached"] is True


def test_probe_is_deterministic_for_same_seed() -> None:
    targets = _targets(["a", "b", "a", "b"])
    vocab = ConceptVocabulary.fit(targets)
    latents = torch.tensor([[-2.0], [2.0], [-1.0], [1.0]])
    one, _ = train_probe(latents, targets, vocab, seed=19, steps=20)
    two, _ = train_probe(latents, targets, vocab, seed=19, steps=20)
    assert torch.equal(one(latents)["event"], two(latents)["event"])


def test_probe_learns_simple_separable_training_data() -> None:
    targets = _targets(["left"] * 8 + ["right"] * 8)
    vocab = ConceptVocabulary.fit(targets)
    latents = torch.tensor([[-3.0]] * 8 + [[3.0]] * 8)
    probe, metadata = train_probe(latents, targets, vocab, steps=100)
    predicted = probe(latents)["event"].argmax(-1)
    expected = torch.tensor([vocab.fields["event"]["left"]] * 8 + [vocab.fields["event"]["right"]] * 8)
    assert (predicted == expected).float().mean() > 0.9
    assert metadata["loss"]["final"] < metadata["loss"]["first"]


def test_missing_labels_are_masked_and_validation_uses_training_stats() -> None:
    targets = _targets(["a", None, "b"])
    vocab = ConceptVocabulary.fit(targets)
    train = torch.tensor([[0.0], [10.0], [20.0]])
    probe, metadata = train_probe(train, targets, vocab, steps=0)
    assert metadata["train_normalization"]["mean"] == [10.0]
    assert metadata["train_normalization"]["std"] == [8.164965629577637]
    assert torch.allclose(probe.normalize(torch.tensor([[10.0], [30.0]])), torch.tensor([[0.0], [2.4494898]]), atol=1e-5)


def test_all_labels_missing_has_differentiable_zero_loss() -> None:
    targets = [{"concept": {}} for _ in range(2)]
    vocab = ConceptVocabulary.fit(targets)
    probe, metadata = train_probe(torch.ones(2, 3), targets, vocab, steps=2)
    assert metadata["loss"]["first"] == 0.0
    assert metadata["loss"]["final"] == 0.0


@pytest.mark.parametrize("bad", [torch.ones(2, 1, dtype=torch.float64),
                                  torch.ones(2, 1, device="meta"),
                                  torch.tensor([[0.0], [float("nan")]], dtype=torch.float32)])
def test_input_must_be_finite_cpu_float32(bad) -> None:
    vocab = ConceptVocabulary.fit(_targets(["a", "b"]))
    with pytest.raises(ValueError):
        train_probe(bad, _targets(["a", "b"]), vocab, steps=0)


@pytest.mark.parametrize("bad_steps", [-1, True, 1.0])
def test_steps_must_be_nonnegative_integer(bad_steps) -> None:
    vocab = ConceptVocabulary.fit(_targets(["a", "b"]))
    with pytest.raises(ValueError):
        train_probe(torch.ones(2, 1), _targets(["a", "b"]), vocab, steps=bad_steps)
