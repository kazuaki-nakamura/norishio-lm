import pytest
torch = pytest.importorskip("torch")

from norishio_lm.concept_model import TinyConceptDecoder
from norishio_lm.toy_generation import greedy_generate


def _decoder(mode="initial_only"):
    torch.manual_seed(12)
    decoder = TinyConceptDecoder(vocab_size=16, hidden_dim=6, conditioning_mode=mode)
    decoder.concept_projection = torch.nn.Linear(3, 6)
    return decoder.eval()


def test_modes_share_parameters_and_initial_only_is_legacy_path():
    initial = _decoder("initial_only")
    additive = _decoder("per_step_additive")
    additive.load_state_dict(initial.state_dict())
    assert sum(p.numel() for p in initial.parameters()) == sum(p.numel() for p in additive.parameters())
    assert initial.state_dict().keys() == additive.state_dict().keys()
    ids = torch.tensor([[1, 4, 5, 6]])
    concepts = torch.randn(1, 3)
    projected = torch.tanh(initial.concept_projection(concepts)).unsqueeze(0)
    hidden, _ = initial.gru(initial.embedding(ids), projected)
    expected = initial.lm_head(hidden)
    assert torch.equal(initial(ids, pathway="C", concept_probs=concepts)["logits"], expected)


def test_future_tokens_do_not_change_earlier_logits_and_additive_reaches_late_steps():
    concepts = torch.tensor([[1.0, -2.0, .5]])
    for mode in ("initial_only", "per_step_additive"):
        decoder = _decoder(mode)
        first = decoder(torch.tensor([[1, 4, 5, 6]]), pathway="C", concept_probs=concepts)["logits"]
        changed = decoder(torch.tensor([[1, 4, 5, 9]]), pathway="C", concept_probs=concepts)["logits"]
        assert torch.equal(first[:, :3], changed[:, :3])
    initial = _decoder("initial_only")
    additive = _decoder("per_step_additive")
    additive.load_state_dict(initial.state_dict())
    baseline = initial(torch.tensor([[1, 4, 5, 6]]), pathway="C", concept_probs=concepts)["logits"]
    stepped = additive(torch.tensor([[1, 4, 5, 6]]), pathway="C", concept_probs=concepts)["logits"]
    assert not torch.equal(baseline[:, 1:], stepped[:, 1:])


def test_additive_concepts_change_late_logits():
    decoder = _decoder("per_step_additive")
    prefix = torch.tensor([[1] + [4] * 32])
    first = decoder(prefix, pathway="C", concept_probs=torch.tensor([[1.0, 0.0, 0.0]]))["logits"]
    second = decoder(prefix, pathway="C", concept_probs=torch.tensor([[0.0, 1.0, 0.0]]))["logits"]
    assert bool((first[:, -1] - second[:, -1]).abs().max() > 1e-6)


def test_incremental_generation_matches_full_history_for_both_modes():
    concepts = torch.randn(2, 3)
    for mode in ("initial_only", "per_step_additive"):
        decoder = _decoder(mode)
        rows = greedy_generate(decoder, concepts, max_new_tokens=5)
        for index, row in enumerate(rows):
            history = torch.tensor([[1]])
            expected = []
            for _ in range(5):
                logits = decoder(history, pathway="C", concept_probs=concepts[index:index + 1])["logits"]
                token = int(logits[0, -1].argmax())
                expected.append(token)
                history = torch.cat((history, torch.tensor([[token]])), dim=1)
                if token == 2:
                    break
            assert row.token_ids == expected


def test_per_step_rejects_non_c_pathways_and_latent_inputs():
    decoder = _decoder("per_step_additive")
    with pytest.raises(ValueError):
        decoder(torch.tensor([[1, 2]]), pathway="A")
    with pytest.raises(ValueError):
        decoder(torch.tensor([[1, 2]]), pathway="B", encoder_latent=torch.zeros(1, 6))
    with pytest.raises(ValueError):
        decoder(torch.tensor([[1, 2]]), pathway="C", concept_probs=torch.zeros(1, 3),
                encoder_latent=torch.zeros(1, 6))


def _checkpoint_model():
    from norishio_lm.concept_model import ConceptVocabulary
    from norishio_lm.tensorizer import SemanticTensorizer
    from norishio_lm.toy_adapter import source_record
    from norishio_lm.toy_experiment import ToyModel
    vocab = ConceptVocabulary.fit([{"concept": {"event": "MEET"}, "sense": "meet", "sememes": ["MEET"]}])
    tensorizer = SemanticTensorizer.fit([source_record({"text": "x", "context": ""})])
    return ToyModel("C", tensorizer, vocab, conditioning_mode="per_step_additive").eval(), tensorizer, vocab


def test_additive_checkpoint_restores_mode_logits_and_greedy(tmp_path):
    from norishio_lm.toy_checkpoint import save_checkpoint, load_checkpoint
    model, tensorizer, vocab = _checkpoint_model()
    path = tmp_path / "step.pt"
    save_checkpoint(path, model, tensorizer, vocab, dataset_version="norishio-toy-1.0")
    restored = load_checkpoint(path)
    assert restored.model.decoder.conditioning_mode == "per_step_additive"
    concepts = torch.randn(2, model.decoder.concept_projection.in_features)
    ids = torch.tensor([[1, 4, 5], [1, 5, 4]])
    assert torch.equal(model.decoder.decode_with_concept_intervention(ids, concepts)["logits"],
                       restored.model.decoder.decode_with_concept_intervention(ids, concepts)["logits"])
    assert greedy_generate(model.decoder, concepts, 12) == greedy_generate(restored.model.decoder, concepts, 12)


@pytest.mark.parametrize("edit", ["legacy", "missing_required", "unknown_mode"])
def test_checkpoint_config_legacy_and_strict_validation(tmp_path, edit):
    from norishio_lm.toy_checkpoint import save_checkpoint, load_checkpoint, _digest
    model, tensorizer, vocab = _checkpoint_model()
    model.decoder.conditioning_mode = "initial_only"
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, tensorizer, vocab, dataset_version="norishio-toy-1.0")
    raw = torch.load(path, weights_only=True, map_location="cpu")
    if edit == "legacy":
        del raw["config"]["conditioning_mode"]
    elif edit == "missing_required":
        del raw["config"]["encoder"]
    else:
        raw["config"]["conditioning_mode"] = "unsupported"
    raw["manifest"]["sha256"] = _digest(raw["config"], raw["vocabulary"], raw["tensorizer"])
    torch.save(raw, path)
    if edit == "legacy":
        restored = load_checkpoint(path)
        assert restored.model.decoder.conditioning_mode == "initial_only"
        assert all(torch.equal(v, restored.model.state_dict()[k]) for k, v in model.state_dict().items())
    else:
        with pytest.raises(ValueError):
            load_checkpoint(path)
