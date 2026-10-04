import pytest
import torch
from verl.trainer.spft import normalized_backward_loss, spft_token_weights, token_weights


def test_reference_equality_and_lambda_zero():
    logp = torch.tensor([0.01, 0.3, 0.8]).log().requires_grad_()
    assert torch.allclose(spft_token_weights(logp, logp, 0.1), logp.exp() / 2)
    assert torch.allclose(spft_token_weights(logp, logp - 2, 0), logp.exp() / 2)
    assert not spft_token_weights(logp, logp).requires_grad
    torch.testing.assert_close(token_weights(logp, "sft"), torch.ones_like(logp))


def test_preference_direction_and_stability():
    ref = torch.tensor([0.5]).log()
    below, above = torch.tensor([0.2]).log(), torch.tensor([0.8]).log()
    assert (spft_token_weights(below, ref, 0.3) / below.exp()).item() > 0.5
    assert (spft_token_weights(above, ref, 0.3) / above.exp()).item() < 0.5
    result = spft_token_weights(torch.tensor([-1000., 0.]), torch.tensor([0., -1000.]))
    assert torch.isfinite(result).all()
    assert (result >= 0).all()


@pytest.mark.parametrize("lambda_,eps", [(-1, 1e-6), (float("nan"), 1e-6), (0.1, 0), (0.1, 0.5)])
def test_invalid_parameters(lambda_, eps):
    with pytest.raises(ValueError):
        spft_token_weights(torch.zeros(2), torch.zeros(2), lambda_, eps)


@pytest.mark.parametrize("mode", ["sft", "dft", "spft"])
@pytest.mark.parametrize("micro", [1, 2, 4])
def test_accumulation_matches_global_token_gradient(mode, micro):
    torch.manual_seed(7)
    logits = torch.randn(4, 3, 5, requires_grad=True)
    labels = torch.randint(5, (4, 3))
    mask = torch.tensor([[1, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 0]], dtype=torch.bool)
    ref = torch.log_softmax(torch.randn(4, 3, 5), -1).gather(-1, labels[..., None]).squeeze(-1)

    def objective(value, y, m, r):
        ce = torch.nn.functional.cross_entropy(value.flatten(0, 1), y.flatten(), reduction="none").reshape_as(y)
        return (ce * token_weights(-ce, mode, r) * m).sum()

    baseline = objective(logits, labels, mask, ref) / mask.sum()
    expected = torch.autograd.grad(baseline, logits)[0]
    for start in range(0, 4, micro):
        sl = slice(start, start + micro)
        normalized_backward_loss(objective(logits[sl], labels[sl], mask[sl], ref[sl]), mask.sum(), 1).backward()
    torch.testing.assert_close(logits.grad, expected)
