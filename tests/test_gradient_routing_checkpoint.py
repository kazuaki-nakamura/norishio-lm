from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.local_slot_checkpoint import DATASET_VERSION, load_local_checkpoint, save_local_checkpoint
from norishio_lm.local_slot_model import DuplicateRemovedModel
from norishio_lm.toy_adapter import source_record
from norishio_lm.toy_experiment import fixture_module
from test_explicit_slot_model import _setup


def test_stop_slot_lm_checkpoint_preserves_routing_and_forward(tmp_path: Path):
    initial, tensorizer, vocabulary = _setup()
    model = DuplicateRemovedModel(initial, vocabulary, new_seed=20, local=True,
                                  gradient_routing="stop_slot_lm").eval()
    path = tmp_path / "stop.pt"
    save_local_checkpoint(path, model, tensorizer, vocabulary, dataset_version=DATASET_VERSION)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert raw["config"]["gradient_routing"] == "stop_slot_lm"
    assert raw["metadata"]["gradient_routing"] == "stop_slot_lm"
    loaded = load_local_checkpoint(path, expected_dataset_version=DATASET_VERSION,
                                   expected_tensorizer=tensorizer, expected_vocabulary=vocabulary)
    assert loaded.model.gradient_routing == "stop_slot_lm"
    semantic = tensorizer.encode([source_record({"context": "", "text": "x"})])
    ids = torch.tensor([[1, 8, 9, 10]], dtype=torch.long)
    labels = torch.tensor([[8, 9, 10, 11]], dtype=torch.long)
    with torch.no_grad():
        original = model(ids, semantic, labels=labels)[0]["logits"]
        restored = loaded.model(ids, semantic, labels=labels)[0]["logits"]
    assert torch.equal(original, restored)


def test_legacy_checkpoint_defaults_to_end_to_end():
    path = Path("codex/work_output/issue26-fixed-v1/D-seed17.pt")
    if not path.is_file():
        pytest.skip("historical ignored checkpoint unavailable")
    corpus = fixture_module("toy_corpus")
    bundle = corpus.build()
    train = bundle["train"]
    tensorizer = __import__("norishio_lm.tensorizer", fromlist=["SemanticTensorizer"]).SemanticTensorizer.fit(
        [source_record(corpus.model_inputs(row)) for row in train])
    vocabulary = __import__("norishio_lm.concept_model", fromlist=["ConceptVocabulary"]).ConceptVocabulary.fit(
        [row["targets"] for row in train])
    loaded = load_local_checkpoint(path, expected_dataset_version=corpus.VERSION,
                                   expected_tensorizer=tensorizer, expected_vocabulary=vocabulary)
    assert loaded.model.gradient_routing == "end_to_end"
