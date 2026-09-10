import pytest
torch = pytest.importorskip("torch")

from norishio_lm.concept_model import (
    ConceptBottleneck, ConceptVocabulary, TinyConceptDecoder, encode_targets,
)


def targets():
    return [{"sense": "meet_person", "sememes": ["MEET", "HUMAN_INTERACTION"], "concept": {
        "event": "MEET", "operators": ["WANT", "NOT"], "agent": "SELF",
        "participant": "A", "time": "TODAY", "location": "UNSPECIFIED", "repeat_marked": False,
    }}, {"sense": None, "sememes": None, "concept": None}]


def test_vocab_preserves_operator_order_and_masks_nulls():
    vocab = ConceptVocabulary.fit(targets())
    assert ("WANT", "NOT") in vocab.fields["operators"]
    encoded = encode_targets(targets(), vocab)
    assert encoded.masks["event"].tolist() == [True, False]
    assert encoded.masks["location"].tolist() == [True, False]
    assert encoded.sense_mask.tolist() == [True, False]


def test_c_pathway_uses_only_history_and_intervenable_concepts():
    torch.manual_seed(7)
    vocab = ConceptVocabulary.fit(targets())
    bottleneck = ConceptBottleneck(vocabulary=vocab)
    decoder = TinyConceptDecoder(bottleneck=bottleneck).eval()
    concept = bottleneck(torch.zeros(1, 32)).probabilities()
    history = torch.tensor([[1, 9, 10]])
    changed_source_is_irrelevant = decoder.decode_with_concept_intervention(history, concept)
    assert torch.equal(changed_source_is_irrelevant["logits"],
                       decoder.decode_with_concept_intervention(history, concept)["logits"])
    altered = concept + torch.randn_like(concept) * .1
    assert not torch.equal(changed_source_is_irrelevant["logits"], decoder.decode_with_concept_intervention(history, altered)["logits"])


def test_c_rejects_latent_and_a_is_causal():
    decoder = TinyConceptDecoder()
    ids = torch.tensor([[1, 9, 10, 11]])
    early = decoder(ids, pathway="A")["logits"]
    changed_future = decoder(torch.tensor([[1, 9, 10, 12]]), pathway="A")["logits"]
    assert torch.equal(early[:, :3], changed_future[:, :3])
    with pytest.raises(ValueError):
        decoder(ids[:, :2], pathway="C", concept_probs=torch.zeros(1, 1), encoder_latent=torch.zeros(1, 32))


def test_auxiliary_outputs_do_not_enter_concept_probabilities():
    vocab = ConceptVocabulary.fit(targets())
    bottleneck = ConceptBottleneck(vocabulary=vocab)
    output = bottleneck(torch.zeros(1, 32))
    assert output.probabilities().shape[-1] == sum(len(x) + 1 for x in vocab.fields.values())


def test_a_baseline_and_all_masked_lm_loss_are_safe():
    decoder = TinyConceptDecoder()
    output = decoder(torch.tensor([[1, 4]]), labels=torch.full((1, 2), -100))
    output["lm_loss"].backward()
    assert output["lm_loss"].item() == 0


def test_loss_switches_match_named_components_and_validate_weights():
    vocab = ConceptVocabulary.fit(targets())
    bottleneck = ConceptBottleneck(vocabulary=vocab)
    decoder = TinyConceptDecoder(bottleneck=bottleneck)
    encoded = encode_targets(targets(), vocab)
    bo = bottleneck(torch.zeros(2, 32))
    lm = decoder(torch.tensor([[1, 4], [1, 5]]), labels=torch.tensor([[4, 2], [5, 2]]))
    output = {**lm, "lm_loss": lm["lm_loss"]}
    total = decoder.losses(output, targets=encoded, bottleneck_output=bo,
                           weights={"lm": 2, "sense": 3, "sememe": 5, "concept": 7})
    parts = bottleneck.loss(bo, encoded)
    expected = 2 * lm["lm_loss"] + 3 * parts["sense"] + 5 * parts["sememes"] + 7 * parts["concept"]
    assert torch.allclose(total, expected)
    with pytest.raises(ValueError):
        decoder.losses(output, weights={"unknown": 1})
    with pytest.raises(ValueError):
        decoder.losses(output, weights={"lm": -1})


def test_missing_auxiliary_labels_still_backpropagate_c_lm():
    vocab = ConceptVocabulary.fit(targets())
    bottleneck = ConceptBottleneck(vocabulary=vocab)
    decoder = TinyConceptDecoder(bottleneck=bottleneck)
    missing = encode_targets([{"sense": None, "sememes": None, "concept": None}], vocab)
    bo = bottleneck(torch.randn(1, 32))
    result = decoder(torch.tensor([[1, 4, 5]]), pathway="C", concept_probs=bo.probabilities(),
                     labels=torch.tensor([[4, 5, 2]]))
    total = decoder.losses(result, targets=missing, bottleneck_output=bo,
                           weights={"sense": 0, "sememe": 0, "concept": 0})
    total.backward()
    assert any(p.grad is not None and bool((p.grad.abs() > 0).any())
               for p in bottleneck.field_heads.parameters())


def test_fit_all_missing_labels_has_no_nan():
    missing = [{"sense": None, "sememes": None, "concept": None}]
    vocab = ConceptVocabulary.fit(missing)
    bottleneck = ConceptBottleneck(vocabulary=vocab)
    encoded = encode_targets(missing, vocab)
    losses = bottleneck.loss(bottleneck(torch.zeros(1, 32)), encoded)
    assert all(torch.isfinite(value) for value in losses.values())


def test_all_missing_auxiliary_labels_are_differentiable_zero():
    vocab = ConceptVocabulary.fit(targets())
    bottleneck = ConceptBottleneck(vocabulary=vocab)
    missing = encode_targets([{"sense": None, "sememes": None, "concept": None}], vocab)
    output = bottleneck(torch.zeros(1, 32))
    total = sum(bottleneck.loss(output, missing).values())
    total.backward()
    assert total.item() == 0
    assert all(p.grad is not None for p in bottleneck.parameters())
    assert all(torch.equal(p.grad, torch.zeros_like(p.grad)) for p in bottleneck.parameters())
