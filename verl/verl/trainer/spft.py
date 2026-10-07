"""Detached token weights; no optimizer or distributed side effects."""

import math

import torch


def spft_token_weights(log_probs, reference_log_probs, lambda_=0.1, eps=1e-6):
    if log_probs.shape != reference_log_probs.shape:
        raise ValueError("Policy and reference log probabilities must have identical shapes")
    if not math.isfinite(lambda_) or lambda_ < 0 or not 0 < eps < 0.5:
        raise ValueError("Require finite lambda >= 0 and 0 < eps < 0.5")
    with torch.no_grad():
        p = log_probs.float().exp().clamp(eps, 1 - eps)
        p0 = reference_log_probs.float().exp().clamp(eps, 1 - eps)
        log_odds_delta = torch.logit(p) - torch.logit(p0)
        return (p * torch.sigmoid(-lambda_ * log_odds_delta)).detach()


def psft_token_losses(log_probs, reference_log_probs, clip_ratio_low=0.2, clip_ratio_high=0.28):
    """Proximal SFT token objective from PSFT's PPO-style clipped surrogate."""
    if log_probs.shape != reference_log_probs.shape:
        raise ValueError("Policy and reference log probabilities must have identical shapes")
    if not math.isfinite(clip_ratio_low) or not math.isfinite(clip_ratio_high):
        raise ValueError("PSFT clip ratios must be finite")
    if clip_ratio_low < 0 or clip_ratio_high < 0:
        raise ValueError("PSFT clip ratios must be non-negative")
    ratio = (log_probs - reference_log_probs).exp()
    clipped = ratio.clamp(1 - clip_ratio_low, 1 + clip_ratio_high)
    return -torch.minimum(ratio, clipped)


def token_weights(log_probs, mode, reference_log_probs=None, lambda_=0.1, eps=1e-6):
    if mode == "sft":
        return torch.ones_like(log_probs, dtype=torch.float32)
    if mode == "dft":
        return log_probs.detach().float().exp()
    if mode == "spft" and reference_log_probs is not None:
        return spft_token_weights(log_probs, reference_log_probs, lambda_, eps)
    raise ValueError("Use sft, dft, or spft with reference log probabilities")


def normalized_backward_loss(weighted_token_sum, global_token_count, world_size):
    """FSDP averages gradients: undo that average, then divide by global tokens.

    The denominator covers the ENTIRE optimizer batch, not one micro-batch.
    Summing these losses across micro-batches gives a global token mean.
    """
    return weighted_token_sum * world_size / global_token_count.clamp_min(1)
