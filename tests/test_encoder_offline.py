"""End-to-end wiring checks independent of the per-module regression tests."""
import socket

import pytest

torch = pytest.importorskip("torch")

from norishio_lm import SemanticCompiler
from norishio_lm.encoder import EncoderConfig, MultiChannelEncoder
from norishio_lm.schema import LexicalSense, SemanticRecord
from norishio_lm.tensorizer import CHANNELS, SemanticTensorizer


def test_cpu_forward_backward_and_vocab_restore_without_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("model wiring must not access the network")

    monkeypatch.setattr(socket, "socket", forbidden)
    compiler = SemanticCompiler({"x": {"tokens": ["x"], "senses": [
        {"sense_id": "x:a", "gloss": "candidate", "sememes": ["A"]}]}})
    records = [compiler.compile("x"), compiler.compile("unknown"), compiler.compile("")]
    tx = SemanticTensorizer.fit(records[:1])
    tx = SemanticTensorizer.from_dict(tx.to_dict())
    model = MultiChannelEncoder(EncoderConfig(tx.vocab_sizes))
    state = model(tx.encode(records).to("cpu"))
    state.fused.square().sum().backward()
    assert torch.isfinite(state.fused).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_sememe_and_concept_inputs_do_not_encode_sense_identity():
    # Sense IDs belong only to the sense channel and trace metadata. Removing
    # that channel cannot leak the ID through categorical sememe/concept values.
    records = [SemanticRecord(surface="x", senses=(
        LexicalSense(sense_id, "same", ("A",), ("B",)),)) for sense_id in ("one", "two")]
    tx = SemanticTensorizer.fit(records)
    model = MultiChannelEncoder(EncoderConfig(tx.vocab_sizes))
    batch = tx.encode(records)
    state = model(batch, ablation={"senses": False})
    torch.testing.assert_close(state.fused[0], state.fused[1])
    assert not state.channel_mask[:, CHANNELS.index("senses")].any()
