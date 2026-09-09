import json
import pytest

torch = pytest.importorskip("torch")

from norishio_lm.schema import LexicalSense, Provenance, SemanticRecord
from norishio_lm.tensorizer import CHANNELS, SemanticTensorizer


def record(surface="性格", **kwargs):
    return SemanticRecord(surface=surface, **kwargs)


def test_shapes_masks_and_padding():
    tx = SemanticTensorizer.fit([record(tokens=("性格",)), record(surface="")])
    batch = tx.encode([record(tokens=("性格",)), record(surface="")])
    assert tuple(batch.channels) == CHANNELS
    assert batch.channels["tokens"].ids.shape == (2, 1)
    assert batch.channels["tokens"].mask.tolist() == [[True], [False]]
    assert batch.channels["surface"].mask.tolist() == [[True], [True]]
    assert batch.channels["surface"].features[1][0].value == ""
    assert batch.channels["tokens"].ids[1, 0].item() == 0


def test_unknown_is_masked_but_missing_is_pad():
    tx = SemanticTensorizer.fit([record(tokens=("known",))])
    batch = tx.encode([record(tokens=("new",)), record(tokens=())])
    channel = batch.channels["tokens"]
    assert channel.ids[:, 0].tolist() == [1, 0]
    assert channel.mask[:, 0].tolist() == [True, False]


def test_ambiguity_and_provenance_are_preserved():
    p1 = Provenance("sourced", "dict", "r1")
    p2 = Provenance("authored_demo")
    senses = (LexicalSense("a", "character", ("personality",), ("trait",), provenance={
        "sense": (p1,), "sememes": (p1,), "concepts": (p2,)}),
              LexicalSense("b", "format", ("form",), ("type",), provenance={
                  "sense": (p2,), "sememes": (p2,), "concepts": (p1,)}))
    tx = SemanticTensorizer.fit([record(senses=senses)])
    features = tx.encode([record(senses=senses)]).channels
    assert [f.value for f in features["senses"].features[0]] == ["a", "b"]
    assert [f.sense_id for f in features["senses"].features[0]] == ["a", "b"]
    assert [f.sense_id for f in features["sememes"].features[0]] == ["a", "b"]
    assert features["sememes"].features[0][0].value == "personality"
    assert features["sememes"].features[0][0].provenance == (p1,)
    assert features["concepts"].features[0][1].provenance == (p1,)


def test_glyph_channel_is_independent_from_characters():
    r = record(characters=("性",), subcharacters={"性": ("忄", "生")})
    batch = SemanticTensorizer.fit([r]).encode([r])
    assert batch.channels["characters"].features[0][0].value == "性"
    assert batch.channels["subcharacters"].features[0][0].value == '["性","忄"]'


def test_all_non_sense_channels_trace_provenance_and_to_preserves_features():
    p = Provenance("sourced", "fixture", "1")
    r = record(tokens=("性格",), characters=("性",),
               subcharacters={"性": ("忄",)}, etymology_notes={"性": "note"},
               relations=(("性格", "is", "trait"),),
               provenance={"tokens": (p,), "characters": (p,), "subcharacters": (p,),
                           "etymology_notes": (p,), "relations": (p,)})
    tx = SemanticTensorizer.fit([r])
    batch = tx.encode([r])
    for channel in ("tokens", "characters", "subcharacters", "etymology", "relations"):
        assert batch.channels[channel].features[0][0].provenance == (p,)
    moved = batch.to("cpu")
    assert moved.channels["tokens"].features == batch.channels["tokens"].features


def test_deterministic_vocab_and_checkpoint_round_trip():
    records = [record(tokens=("b", "a")), record(tokens=("c",))]
    left = SemanticTensorizer.fit(records)
    right = SemanticTensorizer.fit(list(reversed(records)))
    assert left.vocabularies == right.vocabularies
    assert left.to_dict() == right.to_dict()
    restored = SemanticTensorizer.from_dict(left.to_dict())
    assert restored.vocabularies == left.vocabularies
    with pytest.raises(ValueError):
        SemanticTensorizer.from_dict({"version": "999", "channels": left.vocabularies})
    malformed = left.vocabularies
    malformed["tokens"] = {"<PAD>": 0, "<UNK>": 1, "x": 3}
    with pytest.raises(ValueError):
        SemanticTensorizer.from_dict({"version": "1", "channels": malformed})
    malformed["tokens"] = {"<PAD>": 0, "<UNK>": 1, "x": 2, "y": 2}
    with pytest.raises(ValueError):
        SemanticTensorizer.from_dict({"version": "1", "channels": malformed})


def test_reserved_and_json_like_literals_are_distinct():
    r = record(tokens=("<PAD>", "<UNK>", '"<PAD>"', "plain"))
    tx = SemanticTensorizer.fit([r])
    batch = tx.encode([r]).channels["tokens"]
    ids = batch.ids[0].tolist()
    assert all(value >= 2 for value in ids)
    assert len(set(ids)) == 4


def test_excluded_layers_do_not_fabricate_features():
    r = record(tokens=("token",), characters=("性",)).without_layers(("tokens",))
    batch = SemanticTensorizer.fit([r]).encode([r])
    assert batch.channels["tokens"].features == ((),)
    assert batch.channels["tokens"].mask.tolist() == [[False]]
    assert batch.channels["characters"].mask.tolist() == [[True]]


@pytest.mark.parametrize("value", [[], True, -1, 2.0])
def test_invalid_checkpoint_id_types_fail_explicitly(value):
    raw = SemanticTensorizer.fit([record()]).to_dict()
    raw["channels"]["tokens"][json.dumps("bad")] = value
    with pytest.raises(ValueError):
        SemanticTensorizer.from_dict(raw)


def test_equal_gloss_candidates_keep_distinct_ids_and_vocab_does_not_grow():
    r = record(senses=(LexicalSense("a", "same"), LexicalSense("b", "same")))
    tx = SemanticTensorizer.fit([r])
    before = tx.to_dict()
    batch = tx.encode([r, record(surface="held-out")])
    assert len(set(batch.channels["senses"].ids[0].tolist())) == 2
    assert batch.channels["surface"].ids[1, 0] == 1
    assert tx.to_dict() == before
    exported = tx.vocabularies
    exported["senses"].clear()
    assert tx.to_dict() == before
    with pytest.raises(ValueError):
        tx.encode([])
    with pytest.raises(ValueError):
        SemanticTensorizer.fit([])
