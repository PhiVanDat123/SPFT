# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
# Adapted from yongliang-wu/DFT @ 11e395d49ed1b9da7dc5d957224d942d90af5bf6.
"""Single-node, multi-GPU DFT/SPFT. Full fine-tuning, FSDP and AdamW only."""

import functools
import hashlib
import json
import os
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullStateDictConfig, FullyShardedDataParallel as FSDP,
    MixedPrecision, ShardingStrategy, StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

from .data import ValidationSampler, build_dataset
from .spft import normalized_backward_loss, psft_token_losses, token_weights


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sufficient_statistics(ce, objective, mask, weights, threshold):
    """Token sums, later reduced over micro-batches AND all ranks."""
    valid = mask.bool()
    ce, objective, weights = ce[valid].double(), objective[valid].double(), weights[valid].double()
    return torch.stack([
        objective.sum(), ce.sum(), ce.new_tensor(ce.numel()),
        weights.sum(), (weights > threshold).double().sum(),
    ])


def reduce_metrics(stats, prefix):
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    denominator = stats[2].clamp_min(1)
    return {
        f"{prefix}/loss": (stats[0] / denominator).item(),
        f"{prefix}/original_loss": (stats[1] / denominator).item(),
        f"{prefix}/valid_tokens": stats[2].item(),
        f"{prefix}/weight_mean": (stats[3] / denominator).item(),
        f"{prefix}/w_ratio": (stats[4] / denominator).item(),
    }


def token_losses(model, reference, batch, config, device):
    inputs = {key: value.to(device) for key, value in batch.items() if key != "loss_mask"}
    labels = inputs["input_ids"][:, 1:]
    ref_log_probs = None
    if reference is not None:
        with torch.no_grad():
            if config.optim.spft.reference_cpu_offload:
                reference.to(device)
            logits = reference(**inputs, use_cache=False).logits[:, :-1]
            ref_log_probs = -F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none",
            ).reshape_as(labels)
            del logits
            if config.optim.spft.reference_cpu_offload:
                reference.to("cpu")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**inputs, use_cache=False).logits[:, :-1]
        ce = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="none",
        ).reshape_as(labels)
    log_probs = -ce
    if config.optim.loss_mode == "psft":
        objective = psft_token_losses(
            log_probs, ref_log_probs,
            config.optim.psft.clip_ratio_low, config.optim.psft.clip_ratio_high,
        )
        weights = (log_probs - ref_log_probs).detach().exp()
    else:
        weights = token_weights(
            log_probs, config.optim.loss_mode, ref_log_probs,
            config.optim.spft["lambda"], config.optim.spft.eps,
        )
        objective = ce * weights
    return ce, objective, weights, batch["loss_mask"].to(device)


def save_checkpoint(model, tokenizer, root, step, config):
    destination = root / f"global_step_{step}"
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        state = model.state_dict()
    if dist.get_rank() == 0:
        destination.mkdir(parents=True, exist_ok=True)
        model.module.save_pretrained(destination, state_dict=state, safe_serialization=True)
        tokenizer.save_pretrained(destination)
        OmegaConf.save(config, destination / "training_config.yaml")
    dist.barrier()


def validate(model, reference, loader, config, device):
    model.eval()
    stats = torch.zeros(5, dtype=torch.float64, device=device)
    with torch.no_grad():
        for batch in loader:
            ce, objective, weights, mask = token_losses(model, reference, batch, config, device)
            stats += sufficient_statistics(ce, objective, mask, weights, config.optim.spft.weight_threshold)
    result = reduce_metrics(stats, "val")
    model.train()
    return result


def run(config):
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA; use torchrun via the sweep scripts")
    if config.optim.loss_mode not in {"sft", "dft", "spft", "psft"}:
        raise ValueError("Only sft, dft, spft and psft are supported")
    if not 0 <= config.optim.warmup_steps_ratio <= 1 or config.optim.lr <= 0:
        raise ValueError("Invalid learning rate or warmup ratio")
    # Validate SPFT hyperparameters before allocating a model.
    token_weights(torch.zeros(1), "spft", torch.zeros(1),
                  config.optim.spft["lambda"], config.optim.spft.eps)
    psft_token_losses(torch.zeros(1), torch.zeros(1),
                      config.optim.psft.clip_ratio_low, config.optim.psft.clip_ratio_high)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", local_rank)
    set_seed(config.trainer.seed)
    global_batch = config.data.train_batch_size
    micro_batch = config.data.micro_batch_size_per_gpu
    if micro_batch < 1 or global_batch < 1 or global_batch % (world_size * micro_batch):
        raise ValueError("Global batch must be divisible by world_size * micro_batch_size_per_gpu")
    if config.trainer.total_epochs < 1 or (config.trainer.max_steps is not None and config.trainer.max_steps < 1):
        raise ValueError("Epochs and max_steps must be positive")
    local_batch = global_batch // world_size
    root = Path(config.trainer.default_local_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Prevent mixing checkpoint/log artifacts from independent experiments.
    occupied = torch.tensor(int(rank == 0 and (root / "run_manifest.json").exists()), device=device)
    dist.broadcast(occupied, src=0)
    if occupied.item():
        raise FileExistsError(f"Existing run: {root}. Choose a fresh SAVE_PATH.")

    model_kwargs = dict(
        revision=config.model.revision, torch_dtype=torch.bfloat16,
        attn_implementation=config.model.attention, trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model.partial_pretrain, revision=config.model.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset_type = getattr(config.data, "dataset_type", "numina")
    train_data = build_dataset(dataset_type, config.data.train_files, tokenizer, config.data.max_length)
    val_data = build_dataset(dataset_type, config.data.val_files, tokenizer, config.data.max_length)
    train_sampler = DistributedSampler(
        train_data, num_replicas=world_size, rank=rank, seed=config.trainer.seed, drop_last=True,
    )
    loader_kwargs = {
        "num_workers": config.data.num_workers, "pin_memory": True,
    }
    if config.data.num_workers > 0:
        loader_kwargs.update(multiprocessing_context="spawn", persistent_workers=True)
    train_loader = DataLoader(
        train_data, sampler=train_sampler, batch_size=local_batch, drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_data, sampler=ValidationSampler(len(val_data), rank, world_size),
        batch_size=micro_batch, **loader_kwargs,
    )
    if not len(train_loader):
        raise ValueError("Training dataset is smaller than the global batch")
    total_steps = len(train_loader) * config.trainer.total_epochs
    if config.trainer.max_steps is not None:
        total_steps = min(total_steps, config.trainer.max_steps)

    policy = AutoModelForCausalLM.from_pretrained(config.model.partial_pretrain, **model_kwargs)
    # Pin the reference to the exact resolved policy revision, if loaded from Hub.
    resolved_revision = getattr(policy.config, "_commit_hash", None)
    if resolved_revision:
        model_kwargs["revision"] = resolved_revision
    reference = None
    if config.optim.loss_mode in {"spft", "psft"}:
        reference = AutoModelForCausalLM.from_pretrained(config.model.partial_pretrain, **model_kwargs)
        reference.requires_grad_(False).eval()
        if not config.optim.spft.reference_cpu_offload:
            reference.to(device)
    if config.model.use_liger:
        from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

        _apply_liger_kernel_to_instance(model=policy)
    if config.model.gradient_checkpointing:
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    layer_names = set(policy._no_split_modules or [])
    layer_classes = {type(module) for module in policy.modules() if type(module).__name__ in layer_names}
    if not layer_classes:
        raise ValueError("Model has no recognized transformer layers for FSDP wrapping")
    model = FSDP(
        policy, device_id=device, sync_module_states=True, use_orig_params=True,
        auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_classes),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.optim.lr, betas=tuple(config.optim.betas),
        eps=config.optim.eps, weight_decay=config.optim.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * config.optim.warmup_steps_ratio), total_steps,
    )
    tracker = None
    if rank == 0:
        import importlib.metadata

        manifest = {
            "upstream": "yongliang-wu/DFT@11e395d49ed1b9da7dc5d957224d942d90af5bf6",
            "world_size": world_size, "global_batch_size": global_batch,
            "local_batch_size": local_batch, "accumulation_steps": local_batch // micro_batch,
            "train_rows": len(train_data), "val_rows": len(val_data), "total_steps": total_steps,
            "train_sha256": file_sha256(config.data.train_files),
            "val_sha256": file_sha256(config.data.val_files), "model_revision": resolved_revision,
            "normalization": "global_token_mean_over_entire_optimizer_batch",
            "config": OmegaConf.to_container(config, resolve=True),
            "versions": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "datasets"]},
        }
        (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        OmegaConf.save(config, root / "config.yaml")
        if config.trainer.wandb:
            import wandb

            tracker = wandb.init(project=config.trainer.project_name, name=config.trainer.experiment_name, config=manifest)

    def log(values, step):
        if rank == 0:
            values = {"step": step, **values}
            print(json.dumps(values), flush=True)
            with (root / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(values) + "\n")
            if tracker:
                tracker.log(values, step=step)

    # Step zero separates initial model/data differences from training behavior.
    log(validate(model, reference, val_loader, config, device), 0)
    step = 0
    for epoch in range(config.trainer.total_epochs):
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            count = batch["loss_mask"].sum().to(device=device, dtype=torch.float64)
            dist.all_reduce(count)
            if count.item() == 0:
                raise ValueError("Global batch contains no answer tokens after truncation")
            stats = torch.zeros(5, dtype=torch.float64, device=device)
            for start in range(0, local_batch, micro_batch):
                micro = {key: value[start:start + micro_batch] for key, value in batch.items()}
                ce, objective, weights, mask = token_losses(model, reference, micro, config, device)
                loss = normalized_backward_loss((objective * mask).sum(), count, world_size)
                loss.backward()
                stats += sufficient_statistics(ce.detach(), objective.detach(), mask, weights, config.optim.spft.weight_threshold)
            grad_norm = model.clip_grad_norm_(config.optim.clip_grad)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Nonfinite gradient norm")
            optimizer.step()
            scheduler.step()
            step += 1
            metrics = reduce_metrics(stats, "train")
            metrics.update({"train/lr": scheduler.get_last_lr()[0], "train/grad_norm": grad_norm.item()})
            if step == total_steps or (config.trainer.test_freq > 0 and step % config.trainer.test_freq == 0):
                metrics.update(validate(model, reference, val_loader, config, device))
            log(metrics, step)
            if step == total_steps or (config.trainer.save_freq > 0 and step % config.trainer.save_freq == 0):
                save_checkpoint(model, tokenizer, root, step, config)
            if step >= total_steps:
                if tracker:
                    tracker.finish()
                return


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    try:
        run(config)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
