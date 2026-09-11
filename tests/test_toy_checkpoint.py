from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from norishio_lm.concept_model import CONCEPT_FIELDS, ConceptVocabulary
from norishio_lm.tensorizer import CHANNELS, SemanticTensorizer
from norishio_lm.toy_checkpoint import DATASET_VERSION, load_checkpoint, save_checkpoint
from norishio_lm.toy_experiment import ToyModel
from norishio_lm.toy_generation import greedy_generate


def _setup() -> tuple[ToyModel, SemanticTensorizer, ConceptVocabulary]:
    tensorizer = SemanticTensorizer({name: {"<PAD>": 0, "<UNK>": 1, '"x"': 2, '"y"': 3}
                                     for name in CHANNELS})
    fields = {name: {("left", "right") if name == "operators" else (True if name == "repeat_marked" else "value"): 1}
              for name in CONCEPT_FIELDS}
    vocab = ConceptVocabulary(fields, {"sense": 1}, {"sememe": 0})
    return ToyModel("C", tensorizer, vocab, hidden_dim=4).eval(), tensorizer, vocab


@pytest.fixture
def artifact_dir(tmp_path):
    return tmp_path


def test_reload_reproduces_logits_and_greedy_tokens(artifact_dir):
    model, tensorizer, vocab = _setup()
    with torch.no_grad():
        model.decoder.lm_head.weight.zero_()
        model.decoder.lm_head.bias.fill_(-100)
        model.decoder.lm_head.bias[5] = 1
        model.decoder.lm_head.bias[2] = 2
    constant = torch.zeros(1, model.decoder.concept_projection.in_features)
    path = artifact_dir / "model.pt"
    save_checkpoint(path, model, tensorizer, vocab, dataset_version=DATASET_VERSION, constant=constant)
    loaded = load_checkpoint(path, expected_dataset_version=DATASET_VERSION,
                             expected_tensorizer=tensorizer, expected_vocabulary=vocab)
    ids = torch.tensor([[1, 4, 5]], dtype=torch.long)
    with torch.no_grad():
        before = model.decoder(ids, pathway="C", concept_probs=constant)["logits"]
        after = loaded.model.decoder(ids, pathway="C", concept_probs=loaded.constant)["logits"]
    assert torch.equal(before, after)
    assert [r.token_ids for r in greedy_generate(model.decoder, constant, 4)] == [r.token_ids for r in greedy_generate(loaded.model.decoder, loaded.constant, 4)]


def test_vocabulary_preserves_tuple_and_bool_keys(artifact_dir):
    model, tensorizer, vocab = _setup()
    path = artifact_dir / "model.pt"
    save_checkpoint(path, model, tensorizer, vocab, dataset_version=DATASET_VERSION)
    restored = load_checkpoint(path).vocabulary
    assert restored.fields["operators"] == vocab.fields["operators"]
    assert restored.fields["repeat_marked"] == vocab.fields["repeat_marked"]


def test_existing_path_is_refused(artifact_dir):
    model, tensorizer, vocab = _setup()
    path = artifact_dir / "model.pt"
    save_checkpoint(path, model, tensorizer, vocab, dataset_version=DATASET_VERSION)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, tensorizer, vocab, dataset_version=DATASET_VERSION)


def test_manifest_rejects_same_size_tensorizer_id_swap(artifact_dir):
    model, tensorizer, vocab = _setup()
    path = artifact_dir / "model.pt"
    save_checkpoint(path, model, tensorizer, vocab, dataset_version=DATASET_VERSION)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    raw["tensorizer"]["channels"][CHANNELS[0]]['"x"'] = 3
    raw["tensorizer"]["channels"][CHANNELS[0]]['"y"'] = 2
    torch.save(raw, path)
    with pytest.raises(ValueError, match="integrity"):
        load_checkpoint(path)


def test_byte_spec_and_dataset_validation(artifact_dir):
    model, tensorizer, vocab = _setup()
    path = artifact_dir / "model.pt"
    save_checkpoint(path, model, tensorizer, vocab, dataset_version=DATASET_VERSION)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    raw["byte_spec"]["BOS"] = 9
    torch.save(raw, path)
    with pytest.raises(ValueError, match="byte specification"):
        load_checkpoint(path)


def test_constant_is_finite_and_roundtrips_immutably(artifact_dir):
    model, tensorizer, vocab = _setup()
    width = model.decoder.concept_projection.in_features
    original = torch.arange(1.0, width + 1).view(1, -1)
    expected = original.clone()
    path = artifact_dir / "model.pt"
    save_checkpoint(path, model, tensorizer, vocab, dataset_version=DATASET_VERSION, constant=original)
    original[0, 0] = 99
    loaded = load_checkpoint(path)
    assert torch.equal(loaded.constant, expected)
    loaded.constant[0, 0] = -4
    assert torch.equal(load_checkpoint(path).constant, expected)
    with pytest.raises(ValueError, match="finite"):
        save_checkpoint(artifact_dir / "bad.pt", model, tensorizer, vocab,
                        dataset_version=DATASET_VERSION,
                        constant=torch.full((1, width), float("nan")))


def test_publication_race_does_not_replace_existing_file(artifact_dir, monkeypatch):
    import norishio_lm.toy_checkpoint as checkpoint
    model, tensorizer, vocab = _setup()
    path = artifact_dir / "race.pt"
    operation = "rename" if checkpoint.os.name == "nt" else "link"
    original = getattr(checkpoint.os, operation)
    def competing_write(source, target):
        Path(target).write_bytes(b"existing result")
        return original(source, target)
    monkeypatch.setattr(checkpoint.os, operation, competing_write)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, tensorizer, vocab, dataset_version=DATASET_VERSION)
    assert path.read_bytes() == b"existing result"


def test_wrong_constant_dtype_and_model_dtype_rejected(artifact_dir):
    model, tensorizer, vocab = _setup()
    width = model.decoder.concept_projection.in_features
    with pytest.raises(ValueError):
        save_checkpoint(artifact_dir / "double.pt", model, tensorizer, vocab,
                        dataset_version=DATASET_VERSION, constant=torch.zeros(1, width, dtype=torch.float64))
    with pytest.raises(ValueError, match="float32"):
        save_checkpoint(artifact_dir / "model.pt", model.double(), tensorizer, vocab, dataset_version=DATASET_VERSION)


def test_expected_concept_ids_and_dataset_version_are_checked(artifact_dir):
    model, tensorizer, vocab = _setup()
    path = artifact_dir / "model.pt"
    save_checkpoint(path, model, tensorizer, vocab, dataset_version=DATASET_VERSION)
    with pytest.raises(ValueError, match="dataset"):
        load_checkpoint(path, expected_dataset_version="another-dataset")
    altered = ConceptVocabulary({**vocab.fields, "operators": {("right", "left"): 1}}, vocab.senses, vocab.sememes)
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        load_checkpoint(path, expected_vocabulary=altered)
