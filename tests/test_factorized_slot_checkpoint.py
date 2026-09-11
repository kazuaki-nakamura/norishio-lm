from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.factorized_slot_checkpoint import (
    DATASET_VERSION, load_factorized_checkpoint, save_factorized_checkpoint,
)
from norishio_lm.factorized_slot_model import FactorizedSlotModel
from norishio_lm.toy_experiment import ToyModel
from test_explicit_slot_checkpoint import _setup


def test_factorized_checkpoint_roundtrip_and_exclusive_save(tmp_path: Path):
    _, tensorizer, vocabulary = _setup()
    model = FactorizedSlotModel(
        ToyModel("C", tensorizer, vocabulary, hidden_dim=32,
                 conditioning_mode="per_step_additive"), vocabulary).eval()
    path = tmp_path / "factorized.pt"
    save_factorized_checkpoint(path, model, tensorizer, vocabulary,
                               dataset_version=DATASET_VERSION)
    loaded = load_factorized_checkpoint(path, expected_dataset_version=DATASET_VERSION,
                                        expected_tensorizer=tensorizer,
                                        expected_vocabulary=vocabulary)
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(),
                                                  loaded.model.state_dict().values()))
    with pytest.raises(FileExistsError):
        save_factorized_checkpoint(path, model, tensorizer, vocabulary,
                                   dataset_version=DATASET_VERSION)


def test_factorized_checkpoint_detects_state_tampering(tmp_path: Path):
    _, tensorizer, vocabulary = _setup()
    model = FactorizedSlotModel(
        ToyModel("C", tensorizer, vocabulary, hidden_dim=32,
                 conditioning_mode="per_step_additive"), vocabulary).eval()
    path = tmp_path / "factorized.pt"
    save_factorized_checkpoint(path, model, tensorizer, vocabulary,
                               dataset_version=DATASET_VERSION)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    key = next(iter(raw["factorized_state"]))
    raw["factorized_state"][key] = raw["factorized_state"][key].clone()
    raw["factorized_state"][key].view(-1)[0] += 1
    torch.save(raw, path)
    with pytest.raises(ValueError, match="integrity"):
        load_factorized_checkpoint(path)
