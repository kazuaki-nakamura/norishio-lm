from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.concept_model import ConceptVocabulary
from norishio_lm.pair_slot_checkpoint import (
    DATASET_VERSION, load_pair_checkpoint, save_pair_checkpoint,
)
from norishio_lm.pair_slot_model import PairSlotModel, source_pair_distributions
from norishio_lm.local_slot_generation import greedy_generate_local
from norishio_lm.tensorizer import CHANNELS, SemanticTensorizer
from norishio_lm.toy_experiment import ToyModel


def _setup():
    tensorizer = SemanticTensorizer({name: {"<PAD>": 0, "<UNK>": 1, '"x"': 2}
                                     for name in CHANNELS})
    counts = {"event": 4, "operators": 4, "agent": 4, "participant": 5,
              "time": 5, "location": 3, "repeat_marked": 1}
    vocabulary = ConceptVocabulary(
        {name: {f"{name}-{i}": i + 1 for i in range(count)}
         for name, count in counts.items()}, {"sense": 1}, {"sememe": 0})
    model = PairSlotModel(ToyModel("C", tensorizer, vocabulary, hidden_dim=32,
                                   conditioning_mode="per_step_additive"), vocabulary).eval()
    return model, tensorizer, vocabulary


def test_pair_checkpoint_roundtrip_and_exclusive_save(tmp_path: Path):
    model, tensorizer, vocabulary = _setup()
    path = tmp_path / "pair.pt"
    save_pair_checkpoint(path, model, tensorizer, vocabulary, dataset_version=DATASET_VERSION)
    rng = torch.get_rng_state().clone()
    loaded = load_pair_checkpoint(path, expected_dataset_version=DATASET_VERSION,
                                  expected_tensorizer=tensorizer,
                                  expected_vocabulary=vocabulary)
    assert torch.equal(rng, torch.get_rng_state())
    assert loaded.model.pair_seed == 28
    assert loaded.model.decoder.local is True
    assert all(torch.equal(value, loaded.model.state_dict()[key])
               for key, value in model.state_dict().items())
    sources = [{"context": "", "text": "x"}]
    a = source_pair_distributions(model, sources, tensorizer)
    b = source_pair_distributions(loaded.model, sources, tensorizer)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    x, y = torch.cat((a[0], a[3]), -1), torch.cat((b[0], b[3]), -1)
    ids = torch.tensor([[1, *[v + 4 for v in "私は、今日友人".encode()]]])
    with torch.no_grad():
        assert torch.equal(model.decoder.decode_with_concept_intervention(ids, x)["logits"],
                           loaded.model.decoder.decode_with_concept_intervention(ids, y)["logits"])
        assert greedy_generate_local(model.decoder, x) == greedy_generate_local(loaded.model.decoder, y)
    with pytest.raises(FileExistsError):
        save_pair_checkpoint(path, model, tensorizer, vocabulary,
                             dataset_version=DATASET_VERSION)
    model.decoder.local = False
    with pytest.raises(ValueError, match="local causal"):
        save_pair_checkpoint(tmp_path / "global.pt", model, tensorizer, vocabulary, dataset_version=DATASET_VERSION)


def test_pair_checkpoint_detects_state_and_metadata_tampering(tmp_path: Path):
    model, tensorizer, vocabulary = _setup()
    path = tmp_path / "pair.pt"
    save_pair_checkpoint(path, model, tensorizer, vocabulary, dataset_version=DATASET_VERSION)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    raw["metadata"]["pair_shape"][0] = 4
    metadata_path = tmp_path / "metadata.pt"
    torch.save(raw, metadata_path)
    with pytest.raises(ValueError, match="integrity"):
        load_pair_checkpoint(metadata_path)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    key = "pair_head.weight"
    raw["state_dict"][key] = raw["state_dict"][key].clone()
    raw["state_dict"][key].view(-1)[0] += 1
    state_path = tmp_path / "state.pt"
    torch.save(raw, state_path)
    with pytest.raises(ValueError, match="integrity"):
        load_pair_checkpoint(state_path)


def test_pair_checkpoint_replay_is_deterministic_and_rejects_wrong_vocab(tmp_path: Path):
    model, tensorizer, vocabulary = _setup()
    path = tmp_path / "pair.pt"
    save_pair_checkpoint(path, model, tensorizer, vocabulary, dataset_version=DATASET_VERSION)
    loaded = load_pair_checkpoint(path)
    wrong_fields = {name: dict(values) for name, values in vocabulary.fields.items()}
    wrong_fields["time"] = {f"other-{i}": i + 1 for i in range(5)}
    wrong = ConceptVocabulary(wrong_fields, vocabulary.senses, vocabulary.sememes)
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        load_pair_checkpoint(path, expected_vocabulary=wrong)
    assert loaded.model.pair_seed == model.pair_seed == 28
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.model.state_dict()[key])
