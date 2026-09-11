from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.concept_model import ConceptVocabulary
from norishio_lm.local_slot_checkpoint import (
    DATASET_VERSION, _metadata_digest, load_local_checkpoint, save_local_checkpoint,
)
from norishio_lm.local_slot_model import DuplicateRemovedModel
from norishio_lm.tensorizer import CHANNELS, SemanticTensorizer
from norishio_lm.toy_experiment import ToyModel


def _setup(pathway: str = "C", *, local: bool = True):
    tensorizer = SemanticTensorizer({name: {"<PAD>": 0, "<UNK>": 1, '"x"': 2}
                                     for name in CHANNELS})
    counts = {"event": 4, "operators": 4, "agent": 4, "participant": 5,
              "time": 5, "location": 3, "repeat_marked": 1}
    vocabulary = ConceptVocabulary(
        {name: {f"{name}-{i}": i + 1 for i in range(count)}
         for name, count in counts.items()}, {"sense": 1}, {"sememe": 0})
    model = DuplicateRemovedModel(
        ToyModel(pathway, tensorizer, vocabulary, hidden_dim=32,
                 conditioning_mode="per_step_additive"), vocabulary, local=local).eval()
    return model, tensorizer, vocabulary


def test_local_checkpoint_roundtrip_and_exclusive_save(tmp_path: Path):
    model, tensorizer, vocabulary = _setup(local=False)
    path = tmp_path / "local.pt"
    save_local_checkpoint(path, model, tensorizer, vocabulary,
                           dataset_version=DATASET_VERSION)
    loaded = load_local_checkpoint(path, expected_dataset_version=DATASET_VERSION,
                                   expected_tensorizer=tensorizer,
                                   expected_vocabulary=vocabulary)
    assert loaded.model.decoder.local is False
    assert loaded.model.decoder.concept_projection.base_proj.in_features == 21
    assert isinstance(loaded.model.kept_old_indices, tuple)
    assert all(torch.equal(value, loaded.model.state_dict()[key])
               for key, value in model.state_dict().items())
    with pytest.raises(FileExistsError):
        save_local_checkpoint(path, model, tensorizer, vocabulary,
                              dataset_version=DATASET_VERSION)


def test_local_checkpoint_detects_metadata_and_state_tampering(tmp_path: Path):
    model, tensorizer, vocabulary = _setup()
    path = tmp_path / "local.pt"
    save_local_checkpoint(path, model, tensorizer, vocabulary,
                           dataset_version=DATASET_VERSION)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    raw["metadata"]["kept_indices"][0] += 1
    torch.save(raw, tmp_path / "metadata.pt")
    with pytest.raises(ValueError, match="integrity"):
        load_local_checkpoint(tmp_path / "metadata.pt")

    raw = torch.load(path, map_location="cpu", weights_only=True)
    key = next(iter(raw["state_dict"]))
    raw["state_dict"][key] = raw["state_dict"][key].clone()
    raw["state_dict"][key].view(-1)[0] += 1
    torch.save(raw, tmp_path / "state.pt")
    with pytest.raises(ValueError, match="integrity"):
        load_local_checkpoint(tmp_path / "state.pt")


def test_local_checkpoint_rejects_rehashed_wrong_rule_and_indices(tmp_path: Path):
    model, tensorizer, vocabulary = _setup()
    path = tmp_path / "local.pt"
    save_local_checkpoint(path, model, tensorizer, vocabulary,
                           dataset_version=DATASET_VERSION)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    raw["config"]["rule_version"] = "wrong-rule"
    raw["manifest"]["sha256"] = _metadata_digest(
        raw["config"], raw["vocabulary"], raw["tensorizer"], raw["metadata"])
    torch.save(raw, tmp_path / "wrong-rule.pt")
    with pytest.raises(ValueError, match="architecture metadata"):
        load_local_checkpoint(tmp_path / "wrong-rule.pt")

    raw = torch.load(path, map_location="cpu", weights_only=True)
    raw["metadata"]["kept_indices"][0] += 1
    raw["manifest"]["sha256"] = _metadata_digest(
        raw["config"], raw["vocabulary"], raw["tensorizer"], raw["metadata"])
    torch.save(raw, tmp_path / "wrong-indices.pt")
    with pytest.raises(ValueError, match="architecture metadata"):
        load_local_checkpoint(tmp_path / "wrong-indices.pt")


def test_local_checkpoint_rejects_nonfinite_state_and_vocabulary_mismatch(tmp_path: Path):
    model, tensorizer, vocabulary = _setup()
    path = tmp_path / "local.pt"
    save_local_checkpoint(path, model, tensorizer, vocabulary,
                           dataset_version=DATASET_VERSION)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    key = next(key for key, value in raw["state_dict"].items() if value.is_floating_point())
    raw["state_dict"][key] = raw["state_dict"][key].clone()
    raw["state_dict"][key].view(-1)[0] = float("nan")
    raw["manifest"]["state_sha256"] = "tampered"
    torch.save(raw, tmp_path / "nan.pt")
    with pytest.raises(ValueError, match="finite|integrity"):
        load_local_checkpoint(tmp_path / "nan.pt")

    altered = {name: dict(values) for name, values in vocabulary.fields.items()}
    altered["time"] = {f"other-{i}": i + 1 for i in range(5)}
    wrong = ConceptVocabulary(altered, vocabulary.senses, vocabulary.sememes)
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        load_local_checkpoint(path, expected_vocabulary=wrong)


def test_local_checkpoint_roundtrip_pathway_d(tmp_path: Path):
    model, tensorizer, vocabulary = _setup("C", local=True)
    path = tmp_path / "local-d.pt"
    save_local_checkpoint(path, model, tensorizer, vocabulary,
                           dataset_version=DATASET_VERSION)
    loaded = load_local_checkpoint(path, expected_dataset_version=DATASET_VERSION,
                                   expected_tensorizer=tensorizer,
                                   expected_vocabulary=vocabulary)
    assert loaded.model.pathway == "C"
    assert loaded.model.decoder.local is True
    assert all(torch.equal(value, loaded.model.state_dict()[key])
               for key, value in model.state_dict().items())
