from dataclasses import asdict
import json

import pytest
torch = pytest.importorskip("torch")

from norishio_lm.encoder import EncoderConfig, MultiChannelEncoder
from norishio_lm.schema import LexicalSense, Provenance, SemanticRecord
from norishio_lm.semantic_compiler import SemanticCompiler
from norishio_lm.tensorizer import CHANNELS, SemanticTensorizer


def compiled_record(**entry) -> SemanticRecord:
    """Build through the real compiler and normalize through the schema boundary."""
    raw = {"語": {"tokens": ["語"], "characters": ["語"], **entry}}
    return SemanticRecord.from_dict(SemanticCompiler(raw).compile("語").to_dict())


def p(kind: str, source: str | None = None, revision: str | None = None) -> Provenance:
    return Provenance(kind, source, revision)


@pytest.mark.parametrize("channel", ["senses", "sememes", "concepts"])
@pytest.mark.parametrize("mode", ["layer_only", "candidate_only", "both_different", "both_unknown"])
def test_candidate_and_layer_provenance_are_kept_separate(channel: str, mode: str) -> None:
    layer_origin = (p("sourced", "layer-source", "layer-r2")
                    if mode in ("layer_only", "both_different") else p("unknown"))
    candidate_origin = (p("authored_demo", "candidate-source", "candidate-r1")
                        if mode == "candidate_only" else
                        p("inferred", "candidate-source", "candidate-r2")
                        if mode == "both_different" else p("unknown"))
    senses = (
        LexicalSense("a", "first", ("sememe-a",), ("concept-a",), provenance={
            **({channel if channel != "senses" else "sense": (candidate_origin,)}
               if mode in ("candidate_only", "both_different") else {}),
        }),
        LexicalSense("b", "second", ("sememe-b",), ("concept-b",), provenance={}),
    )
    sense_dicts = [json.loads(json.dumps(asdict(sense), ensure_ascii=False)) for sense in senses]
    record_provenance = ({channel: [vars(layer_origin)]}
                         if mode in ("layer_only", "both_different") else {})
    record = compiled_record(
        senses=sense_dicts,
        provenance=record_provenance,
    )
    tx = SemanticTensorizer.fit([record])
    batch = tx.encode([record])
    layer = batch.channels[channel]
    assert layer.layer_provenance == ((layer_origin,),)
    assert layer.features[0][0].provenance == (candidate_origin,)
    assert layer.features[0][1].provenance == (p("unknown"),)
    moved = batch.to("cpu")
    assert moved.channels[channel].layer_provenance == layer.layer_provenance
    assert moved.channels[channel].features == layer.features


def test_layer_provenance_is_row_aligned_for_multiple_and_empty_rows() -> None:
    first = compiled_record(tokens=["first"], provenance={
        "tokens": [{"kind": "sourced", "source": "tokens-a"}],
        "etymology_notes": [{"kind": "authored_demo"}],
    })
    second = compiled_record(tokens=[], etymology_notes={}, provenance={
        "tokens": [{"kind": "inferred"}],
        "etymology_notes": [{"kind": "sourced", "source": "etymology-b"}],
    })
    batch = SemanticTensorizer.fit([first, second]).encode([first, second])
    assert batch.channels["tokens"].layer_provenance == (
        (p("sourced", "tokens-a"),), (p("inferred"),)
    )
    assert batch.channels["etymology"].layer_provenance == (
        (p("authored_demo"),), (p("sourced", "etymology-b"),)
    )
    assert batch.channels["tokens"].features[1] == ()
    assert batch.channels["etymology"].features[1] == ()


def test_etymology_channel_uses_etymology_notes_provenance() -> None:
    origin = p("sourced", "etymology-source")
    record = compiled_record(
        etymology_notes={"語": "orthographic note"},
        provenance={"etymology_notes": [vars(origin)]},
    )
    batch = SemanticTensorizer.fit([record]).encode([record])
    assert batch.channels["etymology"].layer_provenance == ((origin,),)
    assert batch.channels["etymology"].features[0][0].provenance == (origin,)


@pytest.mark.parametrize("channel", ["senses", "sememes", "concepts"])
def test_empty_candidate_layers_retain_declared_row_origins(channel: str) -> None:
    origin = p("sourced", f"empty-{channel}", "r1")
    first = compiled_record(
        senses=[{"sense_id": "a", "gloss": "candidate", "sememes": ["s"], "concepts": ["c"]}],
        provenance={channel: [vars(origin)]},
    )
    second = compiled_record(senses=[], provenance={channel: [vars(origin)]})
    batch = SemanticTensorizer.fit([first, second]).encode([first, second])
    assert batch.channels[channel].features[1] == ()
    assert batch.channels[channel].layer_provenance == ((origin,), (origin,))


def test_ablation_clears_candidate_features_and_their_declarations() -> None:
    record = compiled_record(
        senses=[{"sense_id": "a", "gloss": "candidate", "sememes": ["s"], "concepts": ["c"]}],
        provenance={name: [{"kind": "sourced", "source": name, "revision": "r1"}]
                    for name in ("senses", "sememes", "concepts")},
    ).without_layers(("senses",))
    batch = SemanticTensorizer.fit([record]).encode([record])
    unknown = (p("unknown"),)
    for channel in ("senses", "sememes", "concepts"):
        assert batch.channels[channel].features == ((),)
        assert batch.channels[channel].layer_provenance == (unknown,)


def test_to_cpu_keeps_layer_origins_and_original_mutation_cannot_change_output() -> None:
    record = compiled_record(tokens=["token"], provenance={
        "tokens": [{"kind": "sourced", "source": "before"}],
    })
    tx = SemanticTensorizer.fit([record])
    batch = tx.encode([record])
    expected = batch.channels["tokens"].layer_provenance
    record.provenance["tokens"] = (p("inferred"),)
    moved = batch.to("cpu")
    assert moved.channels["tokens"].layer_provenance == expected
    assert moved.channels["tokens"].features == batch.channels["tokens"].features


def test_provenance_does_not_change_ids_masks_or_encoder_outputs() -> None:
    values = {"tokens": ["same"], "characters": ["語"]}
    left = compiled_record(**values, provenance={"tokens": [{"kind": "unknown"}]})
    right = compiled_record(**values, provenance={"tokens": [{"kind": "sourced", "source": "changed"}]})
    tx = SemanticTensorizer.fit([left, right])
    left_batch, right_batch = tx.encode([left]), tx.encode([right])
    for channel in CHANNELS:
        assert torch.equal(left_batch.channels[channel].ids, right_batch.channels[channel].ids)
        assert torch.equal(left_batch.channels[channel].mask, right_batch.channels[channel].mask)
    torch.manual_seed(7)
    model = MultiChannelEncoder(EncoderConfig(tx.vocab_sizes, embedding_dim=4, hidden_dim=6)).eval()
    with torch.no_grad():
        left_state, right_state = model(left_batch), model(right_batch)
    assert torch.equal(left_state.fused, right_state.fused)
    assert torch.equal(left_state.channel_states, right_state.channel_states)
    assert torch.equal(left_state.gates, right_state.gates)
