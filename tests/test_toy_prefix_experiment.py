import hashlib
import json

import pytest

pytest.importorskip("torch")

from norishio_lm.toy_prefix_experiment import boundary_positions, checked_json, run


def test_six_boundaries_use_raw_byte_positions_even_mid_codepoint():
    spans = [{"participant": {"start": 6, "end": 12}, "time": {"start": 15, "end": 21}}]
    assert boundary_positions(spans) == {
        "participant_before": [5], "participant_start": [6], "participant_end": [12],
        "time_before": [14], "time_start": [15], "time_end": [21]}
    with pytest.raises(ValueError, match="preceding byte"):
        boundary_positions([{"participant": {"start": 0, "end": 6}, "time": {"start": 6, "end": 12}}])


def test_report_hash_rejects_tampering(tmp_path):
    path = tmp_path / "report.json"
    raw = json.dumps({"metric": 1}).encode()
    path.write_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()
    assert checked_json(path, expected) == {"metric": 1}
    path.write_text('{"metric": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        checked_json(path, expected)


def test_invalid_prior_report_stops_before_output_or_generation(tmp_path, monkeypatch):
    import norishio_lm.toy_prefix_experiment as experiment
    called = []
    monkeypatch.setattr(experiment, "prefix_generate", lambda *a, **k: called.append(True))
    prior = tmp_path / "wrong.json"
    prior.write_text("{}", encoding="utf-8")
    out = tmp_path / "output"
    with pytest.raises(ValueError, match="hash mismatch"):
        run(tmp_path / "absent.pt", tmp_path / "absent.json", prior, out)
    assert not out.exists()
    assert not called


def test_real_template_switch_reads_only_prefix_and_self_generated_bytes(monkeypatch):
    import torch
    from norishio_lm.concept_model import TinyConceptDecoder
    from norishio_lm.prefix_generation import prefix_generate
    from norishio_lm.toy_experiment import fixture_module
    from norishio_lm.toy_spans import authored_slot_spans
    corpus = fixture_module("toy_corpus")
    target = corpus.build()["validation"][0]["targets"]
    j = authored_slot_spans(target, corpus.seed_data())["participant"]["start"]
    reference = [b + 4 for b in target["text"].encode("utf-8")]
    decoder = TinyConceptDecoder(vocab_size=260, hidden_dim=5, conditioning_mode="per_step_additive")
    decoder.concept_projection = torch.nn.Linear(3, 5)
    decoder.eval()
    with torch.no_grad():
        decoder.lm_head.weight.zero_()
        decoder.lm_head.bias.fill_(-100)
        decoder.lm_head.bias[259] = 10  # deliberately unlike the gold suffix
    observed = []
    handle = decoder.embedding.register_forward_pre_hook(lambda module, args: observed.append(args[0].clone()))
    result = prefix_generate(decoder, torch.zeros(1, 3), [reference[:j]], max_total_tokens=j + 3)[0]
    handle.remove()
    # Warmup and state recovery see the same prefix; all subsequent calls see
    # emitted token259, never the authored continuation at positionj or later.
    assert observed[0].tolist() == [[1, *reference[:j]]]
    assert observed[1].tolist() == [[1, *reference[:j]]]
    assert all(x.tolist() == [[259]] for x in observed[2:])
    assert result["generated_token_ids"] == [259, 259, 259]
