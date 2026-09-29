"""Real two-process CPU collective test with unequal valid-token counts."""
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl.trainer.spft import normalized_backward_loss, token_weights


def worker(rank, rendezvous, destination):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    torch.manual_seed(3)
    parameter = torch.randn(3, 5, requires_grad=True)
    inputs = torch.randn(4, 2, 3)
    labels = torch.randint(5, (4, 2))
    mask = torch.tensor([[1, 0], [1, 1], [0, 0], [1, 1]], dtype=torch.bool)
    ref = torch.full((4, 2), -1.7)
    for mode in ("dft", "spft"):
        parameter.grad = None
        count = mask[rank::2].sum()
        dist.all_reduce(count)
        for i in range(rank, 4, 2):
            ce = torch.nn.functional.cross_entropy(inputs[i] @ parameter, labels[i], reduction="none")
            value = (ce * token_weights(-ce, mode, ref[i]) * mask[i]).sum()
            normalized_backward_loss(value, count, 2).backward()
        # Simulate FSDP's average after summing micro-batch gradients.
        dist.all_reduce(parameter.grad)
        parameter.grad /= 2
        ce = torch.nn.functional.cross_entropy((inputs @ parameter).reshape(-1, 5), labels.flatten(), reduction="none").reshape(4, 2)
        expected = torch.autograd.grad((ce * token_weights(-ce, mode, ref) * mask).sum() / mask.sum(), parameter)[0]
        torch.testing.assert_close(parameter.grad, expected)
    if rank == 0:
        Path(destination).write_text("ok")
    dist.destroy_process_group()


def test_two_rank_global_token_normalization(tmp_path):
    result = tmp_path / "result"
    mp.spawn(worker, args=(str(tmp_path / "rendezvous"), str(result)), nprocs=2, join=True)
    assert result.read_text() == "ok"
