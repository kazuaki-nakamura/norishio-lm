import pytest

torch = pytest.importorskip("torch")

from norishio_lm.encoder import EncoderConfig, MultiChannelEncoder
from norishio_lm.schema import LexicalSense, SemanticRecord
from norishio_lm.tensorizer import CHANNELS, ChannelBatch, SemanticBatch, SemanticTensorizer


def _sense(sense_id: str, gloss: str, sememe: str = "trait") -> LexicalSense:
    return LexicalSense(sense_id, gloss, (sememe,), ("person",))


def _records() -> list[SemanticRecord]:
    return [
        SemanticRecord(
            surface="性格",
            tokens=("性格",),
            morphemes=("性", "格"),
            characters=("性", "格"),
            subcharacters={"性": ("忄", "生"), "格": ("木", "各")},
            etymology_notes={"性": "shape and sound history"},
            senses=(_sense("personality", "personality"), _sense("nature", "nature", "quality")),
            relations=(("性格", "is-a", "attribute"),),
        ),
        SemanticRecord(
            surface="",
            tokens=("unseen-token",),
            characters=(),
            senses=(),
        ),
    ]


def _model_and_batch(records=None):
    records = _records() if records is None else records
    tx = SemanticTensorizer.fit(records)
    model = MultiChannelEncoder(
        EncoderConfig(vocab_sizes=tx.vocab_sizes, embedding_dim=8, hidden_dim=12)
    )
    return tx, model, tx.encode(records)


def test_forward_state_shapes_and_default_enabled_channels():
    _, model, batch = _model_and_batch()
    state = model(batch)
    assert state.fused.shape == (2, 12)
    assert state.channel_states.shape == (2, len(CHANNELS), 12)
    assert state.gates.shape == (2, len(CHANNELS))
    assert state.channel_mask.shape == (2, len(CHANNELS))
    assert state.channel_mask.dtype == torch.bool
    assert torch.isfinite(state.fused).all()
    assert torch.allclose(state.gates.sum(dim=1), torch.ones(2))
    assert state.channel_mask[:, CHANNELS.index("surface")].tolist() == [True, True]
    assert state.channel_mask[:, CHANNELS.index("senses")].tolist() == [True, False]


@pytest.mark.parametrize("channel", CHANNELS)
def test_each_channel_can_be_disabled_and_is_exactly_zero(channel):
    _, model, batch = _model_and_batch()
    state = model(batch, ablation={channel: False})
    index = CHANNELS.index(channel)
    assert torch.equal(state.channel_states[:, index], torch.zeros_like(state.channel_states[:, index]))
    assert torch.equal(state.gates[:, index], torch.zeros_like(state.gates[:, index]))
    assert not state.channel_mask[:, index].any()


def test_all_channels_disabled_is_finite_zero_state():
    _, model, batch = _model_and_batch()
    state = model(batch, ablation={channel: False for channel in CHANNELS})
    assert torch.equal(state.fused, torch.zeros_like(state.fused))
    assert torch.equal(state.channel_states, torch.zeros_like(state.channel_states))
    assert torch.equal(state.gates, torch.zeros_like(state.gates))
    assert torch.isfinite(state.fused).all()


def test_disabled_channel_input_ids_cannot_change_fused_output():
    tx, model, batch = _model_and_batch()
    channel = "characters"
    altered = dict(batch.channels)
    original = altered[channel]
    altered[channel] = ChannelBatch(
        ids=torch.full_like(original.ids, tx.vocab_sizes[channel] - 1),
        mask=original.mask, features=original.features
    )
    changed = SemanticBatch(altered)
    kwargs = {channel: False}
    left = model(batch, ablation=kwargs)
    right = model(changed, ablation=kwargs)
    assert torch.equal(left.fused, right.fused)
    assert torch.equal(left.channel_states, right.channel_states)


def test_masked_padding_does_not_change_output():
    _, model, batch = _model_and_batch()
    padded = {}
    for name, channel in batch.channels.items():
        extra_ids = torch.zeros((channel.ids.shape[0], 3), dtype=torch.long)
        extra_mask = torch.zeros((channel.mask.shape[0], 3), dtype=torch.bool)
        padded[name] = ChannelBatch(
            torch.cat((channel.ids, extra_ids), dim=1),
            torch.cat((channel.mask, extra_mask), dim=1),
            channel.features,
        )
    left = model(batch)
    right = model(SemanticBatch(padded))
    assert torch.allclose(left.fused, right.fused, atol=1e-6, rtol=1e-6)
    assert torch.allclose(left.channel_states, right.channel_states, atol=1e-6, rtol=1e-6)


def test_gradients_zero_for_disabled_channel_and_present_for_enabled_channel_and_gates():
    _, model, batch = _model_and_batch()
    disabled, enabled = "characters", "surface"
    model.zero_grad(set_to_none=True)
    state = model(batch, ablation={disabled: False})
    state.fused.square().sum().backward()
    named = dict(model.named_parameters())
    disabled_grads = [p.grad for n, p in named.items() if f".{disabled}." in f".{n}." and p.requires_grad]
    enabled_grads = [p.grad for n, p in named.items() if f".{enabled}." in f".{n}." and p.requires_grad]
    gate_grads = [p.grad for n, p in named.items() if "gate" in n.lower() and p.requires_grad]
    assert disabled_grads and all(g is None or torch.equal(g, torch.zeros_like(g)) for g in disabled_grads)
    assert enabled_grads and any(g is not None and torch.any(g != 0) for g in enabled_grads)
    assert gate_grads and any(g is not None and torch.any(g != 0) for g in gate_grads)


def test_state_dict_roundtrip_reproduces_outputs_after_vocab_roundtrip():
    tx, model, batch = _model_and_batch()
    restored_tx = SemanticTensorizer.from_dict(tx.to_dict())
    restored_model = MultiChannelEncoder(
        EncoderConfig(vocab_sizes=restored_tx.vocab_sizes, embedding_dim=8, hidden_dim=12)
    )
    restored_model.load_state_dict(model.state_dict())
    with torch.no_grad():
        left, right = model(batch), restored_model(restored_tx.encode(_records()))
    assert torch.equal(left.fused, right.fused)
    assert torch.equal(left.channel_states, right.channel_states)
    assert torch.equal(left.gates, right.gates)


def test_glyph_and_etymology_changes_do_not_change_sense_representation_or_labels():
    base = _records()[0]
    altered = SemanticRecord(
        surface=base.surface, tokens=base.tokens, morphemes=base.morphemes,
        characters=("別", "格"), subcharacters={"性": ("月", "心"), "格": ("金", "各")},
        etymology_notes={"性": "deliberately misleading modern gloss"},
        senses=base.senses, relations=base.relations,
    )
    tx = SemanticTensorizer.fit([base, altered])
    model = MultiChannelEncoder(EncoderConfig(vocab_sizes=tx.vocab_sizes, embedding_dim=8, hidden_dim=12))
    states = model(tx.encode([base, altered]))
    sense_index = CHANNELS.index("senses")
    assert [f.sense_id for f in tx.encode([base, altered]).channels["senses"].features[0]] == ["personality", "nature"]
    assert torch.equal(states.channel_states[0, sense_index], states.channel_states[1, sense_index])


def test_malformed_masks_and_unknown_ablation_are_rejected():
    _, model, batch = _model_and_batch()
    channels = dict(batch.channels)
    channel = channels["surface"]
    channels["surface"] = ChannelBatch(channel.ids, torch.ones((2, 2), dtype=torch.bool), channel.features)
    with pytest.raises(ValueError):
        model(SemanticBatch(channels))
    with pytest.raises(ValueError):
        model(batch, ablation={"not-a-channel": False})
    with pytest.raises(ValueError):
        model(batch, ablation={"surface": 0})


def test_oov_is_valid_unknown_and_empty_or_glyph_only_rows_have_no_fabricated_senses():
    fit_record = SemanticRecord(surface="既知", tokens=("known",))
    oov_record = SemanticRecord(surface="未知", tokens=("unseen-after-fit",))
    empty = SemanticRecord(surface="")
    glyph_only = SemanticRecord(surface="性", characters=("性",), subcharacters={"性": ("忄",)})
    tx = SemanticTensorizer.fit([fit_record])
    batch = tx.encode([oov_record, empty, glyph_only])
    assert batch.channels["tokens"].ids[0, 0].item() == 1
    assert batch.channels["tokens"].mask[0, 0].item()
    assert not batch.channels["senses"].mask.any()
    model = MultiChannelEncoder(EncoderConfig(vocab_sizes=tx.vocab_sizes, embedding_dim=8, hidden_dim=12))
    state = model(batch)
    assert not state.channel_mask[:, CHANNELS.index("senses")].any()
    assert torch.equal(state.channel_states[:, CHANNELS.index("senses")], torch.zeros(3, 12))


def test_invalid_ids_and_configuration_are_rejected():
    tx, model, batch = _model_and_batch()
    channels = dict(batch.channels)
    surface = channels["surface"]
    bad_ids = surface.ids.clone()
    bad_ids[0, 0] = tx.vocab_sizes["surface"]
    channels["surface"] = ChannelBatch(bad_ids, surface.mask, surface.features)
    with pytest.raises(ValueError):
        model(SemanticBatch(channels))
    with pytest.raises((ValueError, TypeError)):
        MultiChannelEncoder(EncoderConfig(vocab_sizes={"surface": 1}, embedding_dim=8, hidden_dim=12))
