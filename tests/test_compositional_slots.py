import math
import pytest

torch = pytest.importorskip("torch")
from norishio_lm.compositional_slots import factorized_pairs, factorized_log_probs, pair_space_metrics
from norishio_lm.pair_slot_model import pair_marginals
from norishio_lm.concept_model import ConceptVocabulary


def test_outer_product_recovers_marginals_and_rejects_invalid_probabilities():
    logits = torch.randn(4, 2, 5, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    heads = logits.softmax(-1).reshape(4, 10)
    pairs = factorized_pairs(heads)
    assert torch.allclose(pair_marginals(pairs), heads, atol=1e-14, rtol=1e-14)
    assert torch.allclose(pairs, factorized_log_probs(logits[:, 0], logits[:, 1]).exp())
    assert factorized_pairs(torch.empty(0, 10)).shape == (0, 25)
    for bad in (torch.ones(1, 10), torch.full((1, 10), float('nan')), torch.ones(1, 25)):
        with pytest.raises(ValueError):
            factorized_pairs(bad)


def test_pair_nll_is_sum_of_independent_ce_in_value_and_gradients():
    p = torch.tensor([[1000., -1000, 1, 0, -2], [0, 2, 1, -1, 4]], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([[0., 1, -4, 3, 2], [1000, -1000, 0, 1, 2]], dtype=torch.float64, requires_grad=True)
    pi, ti = torch.tensor([1, 2]), torch.tensor([3, 1])
    independent = torch.nn.functional.cross_entropy(p, pi) + torch.nn.functional.cross_entropy(t, ti)
    pair_loss = torch.nn.functional.nll_loss(factorized_log_probs(p, t), pi * 5 + ti)
    assert torch.allclose(pair_loss, independent, atol=1e-12, rtol=1e-12)
    a = torch.autograd.grad(pair_loss, (p, t), retain_graph=True)
    b = torch.autograd.grad(independent, (p, t))
    assert all(torch.allclose(x, y, atol=1e-12, rtol=1e-12) for x, y in zip(a, b))


def test_pair_space_entropy_rank_coverage_and_absent_groups():
    targets = [{'concept': {'participant': f'p{i}', 'time': f't{i}'}} for i in range(5)]
    vocabulary = ConceptVocabulary.fit(targets)
    support = [[1] * 5 for _ in range(5)]
    support[1][1] = 0
    result = pair_space_metrics(torch.full((1, 25), 1 / 25, dtype=torch.float64),
                                [targets[1]], vocabulary, support)
    group = result['all']
    assert group['rows'][0]['gold_rank'] == 7
    assert group['top_k']['5']['coverage'] == 0
    assert group['top_k']['10']['coverage'] == 1
    assert group['entropy_mean'] == pytest.approx(math.log(25))
    assert group['train_unseen_mass_mean'] == pytest.approx(.04)
    assert result['train_seen']['entropy_mean'] is None
    assert result['train_seen']['top_k']['1']['coverage'] is None
