from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.concept_model import CONCEPT_FIELDS, ConceptVocabulary
from norishio_lm.tensorizer import CHANNELS, SemanticTensorizer
from norishio_lm.toy_experiment import ToyModel
from norishio_lm.explicit_slot_model import ExplicitSlotModel
from norishio_lm.explicit_slot_checkpoint import (
    DATASET_VERSION, load_explicit_checkpoint, save_explicit_checkpoint,
)


def _setup() -> tuple[ExplicitSlotModel, SemanticTensorizer, ConceptVocabulary]:
    tensorizer = SemanticTensorizer({name: {"<PAD>": 0, "<UNK>": 1, '"x"': 2}
                                     for name in CHANNELS})
    counts = {"event": 4, "operators": 4, "agent": 4, "participant": 5,
              "time": 5, "location": 3, "repeat_marked": 1}
    fields = {name: {f"{name}-{i}": i + 1 for i in range(count)}
              for name, count in counts.items()}
    vocab = ConceptVocabulary(fields, {"sense": 1}, {"sememe": 0})
    base = ToyModel("C", tensorizer, vocab, hidden_dim=32,
                    conditioning_mode="per_step_additive")
    return ExplicitSlotModel(base, vocab, new_seed=20).eval(), tensorizer, vocab


def test_roundtrip_preserves_state_and_metadata(tmp_path: Path):
    model, tensorizer, vocab = _setup()
    path = tmp_path / "explicit.pt"
    save_explicit_checkpoint(path, model, tensorizer, vocab, dataset_version=DATASET_VERSION)
    loaded = load_explicit_checkpoint(path, expected_dataset_version=DATASET_VERSION,
                                       expected_tensorizer=tensorizer, expected_vocabulary=vocab)
    assert loaded.dataset_version == DATASET_VERSION
    assert loaded.model.decoder.hidden_dim == model.decoder.hidden_dim
    assert loaded.model.slot_heads.keys() == model.slot_heads.keys()
    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded.model.state_dict()[key])


def test_metadata_and_dataset_are_checked(tmp_path: Path):
    model, tensorizer, vocab = _setup()
    path = tmp_path / "explicit.pt"
    save_explicit_checkpoint(path, model, tensorizer, vocab, dataset_version=DATASET_VERSION)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    raw["slots"]["order"] = ["time", "participant"]
    torch.save(raw, path)
    with pytest.raises(ValueError, match="integrity"):
        load_explicit_checkpoint(path)

    path2 = tmp_path / "wrong.pt"
    save_explicit_checkpoint(path2, model, tensorizer, vocab, dataset_version=DATASET_VERSION)
    with pytest.raises(ValueError, match="dataset"):
        load_explicit_checkpoint(path2, expected_dataset_version="other")

    wrong_fields = {name: dict(values) for name, values in vocab.fields.items()}
    wrong_fields["participant"] = {f"other-{i}": i + 1 for i in range(5)}
    wrong_vocab = ConceptVocabulary(wrong_fields, vocab.senses, vocab.sememes)
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        load_explicit_checkpoint(path2, expected_vocabulary=wrong_vocab)


def test_save_rejects_same_shape_vocabulary_not_used_by_model(tmp_path: Path):
    model, tensorizer, vocab = _setup()
    wrong_fields = {name: dict(values) for name, values in vocab.fields.items()}
    wrong_fields["time"] = {f"other-{i}": i + 1 for i in range(5)}
    wrong_vocab = ConceptVocabulary(wrong_fields, vocab.senses, vocab.sememes)
    with pytest.raises(ValueError, match="vocabulary"):
        save_explicit_checkpoint(tmp_path / "wrong.pt", model, tensorizer, wrong_vocab,
                                 dataset_version=DATASET_VERSION)


def test_save_is_exclusive_and_slot_dimensions_are_validated(tmp_path: Path):
    model, tensorizer, vocab = _setup()
    path = tmp_path / "explicit.pt"
    save_explicit_checkpoint(path, model, tensorizer, vocab, dataset_version=DATASET_VERSION)
    with pytest.raises(FileExistsError):
        save_explicit_checkpoint(path, model, tensorizer, vocab, dataset_version=DATASET_VERSION)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    key = next(key for key in raw["state_dict"] if key.endswith("slot_heads.participant.weight"))
    raw["state_dict"][key] = raw["state_dict"][key][:-1]
    torch.save(raw, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="integrity|state_dict"):
        load_explicit_checkpoint(tmp_path / "bad.pt")
