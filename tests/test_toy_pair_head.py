import pytest

torch = pytest.importorskip("torch")

from norishio_lm.toy_pair_head import generation_cases, scalar_means


def test_oracles_replace_only_named_slots_without_mutating_source():
    old = torch.arange(66).reshape(2, 33).float()
    heads = torch.full((2, 10), .2)
    gold = torch.zeros_like(heads)
    gold[:, 1] = gold[:, 9] = 1
    cases = generation_cases(old, heads, gold)
    assert torch.equal(cases["predicted"], torch.cat((old, heads), -1))
    assert torch.equal(cases["participant_gold"][:, 33:38], gold[:, :5])
    assert torch.equal(cases["participant_gold"][:, 38:], heads[:, 5:])
    assert torch.equal(cases["time_gold"][:, 33:38], heads[:, :5])
    assert torch.equal(cases["time_gold"][:, 38:], gold[:, 5:])
    assert all(torch.equal(value[:, :33], old) for value in cases.values())
    assert torch.equal(cases["both_gold"][:, 33:], gold)
    assert torch.equal(heads, torch.full((2, 10), .2))


def test_means_do_not_drop_missing_seed_or_select_best():
    assert scalar_means([{"score": 0}, {"score": 2}, {"score": 4}]) == {"score": 2}
    assert scalar_means([{"score": 0}, {"score": None}, {"score": 4}]) == {"score": None}
