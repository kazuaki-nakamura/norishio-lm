import pytest

torch = pytest.importorskip("torch")

from norishio_lm.toy_joint_slots import head_cases


def test_single_head_interventions_hold_other_channels_fixed():
    old = torch.arange(66).reshape(2, 33).float()
    heads = torch.arange(20).reshape(2, 10).float()
    gold = heads + 100
    cases = head_cases(old, heads, gold, [1, 0])
    assert len(cases) == 8
    for name, value in cases.items():
        assert torch.equal(value[:, :33], old)
        if name.startswith("participant_"):
            assert torch.equal(value[:, 38:], heads[:, 5:])
        if name.startswith("time_"):
            assert torch.equal(value[:, 33:38], heads[:, :5])
    assert torch.equal(cases["participant_permuted"][:, 33:38], heads[[1, 0], :5])
    assert torch.equal(cases["time_gold"][:, 38:], gold[:, 5:])
    assert not cases["participant_zero"][:, 33:38].any()
    assert not cases["time_zero"][:, 38:].any()
    assert torch.equal(cases["both_gold"][:, 33:], gold)
    assert torch.equal(heads, torch.arange(20).reshape(2, 10).float())
