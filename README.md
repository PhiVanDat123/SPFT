# Final SPFT

DFT/SPFT + AdamW, adapted from upstream DFT `11e395d4`. Source and license details: `UPSTREAM.md`.

## Math Reasoning

```bash
python -m pip install -r requirements.txt
python -m pip install flash-attn --no-build-isolation
bash verl/download_datasets.sh
N_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 bash verl/sweep_dft_1gpu.sh
N_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 bash verl/sweep_spft_1gpu.sh
python -m pytest -q tests
```

- Default training dataset: 100,000 Numina train examples. Numina test is used as validation. Set `VAL_FILE` to use another parquet validation file with `extra_info.question/answer`.
- Global batch is 256 and micro-batch/GPU is 8 by default. Override with `TRAIN_BATCH_SIZE` and `MICRO_BATCH_SIZE_PER_GPU`. Global batch must be divisible by `N_GPUS * MICRO_BATCH_SIZE_PER_GPU`.
- Sweeps: `OPTIM_LRS`, `OPTIM_WEIGHT_DECAYS`, `EPOCHS_LIST`, `SPFT_LAMBDAS`. Change model with `MODEL_NAME`; set `DRY_RUN=1` to print the launch command. Extra Hydra overrides can be appended to the command.
- Liger is off by default. `USE_LIGER=true` requires `liger-kernel`; `USE_WANDB=true` requires `wandb`. If FlashAttention is unavailable, append `model.attention=sdpa`.
- Full fine-tuning uses FSDP1. SPFT keeps a fixed reference model, one replica per GPU, with optional `SPFT_REFERENCE_CPU_OFFLOAD=true`. Saved checkpoints are Hugging Face checkpoints and do not include optimizer state/resume state.
- Gradient accumulation and token normalization are computed over the full global batch. Validation covers every example exactly once across ranks. `weight_threshold` is metric-only.

Math eval, preferably in a separate environment:

```bash
python -m pip install -r requirements-eval.txt
bash verl/download_datasets.sh --eval-only
MODEL_NAME_OR_PATH=/path/to/checkpoint EVAL_CUDA_VISIBLE_DEVICES=0,1 bash verl/eval_dft.sh
```

Supported math eval sets: `math`, `math_oai`, `minerva_math`, `olympiadbench`, `aime24`, `amc23`. Defaults: qwen-boxed prompt, `n=16`, temperature 1, top-p 1, max tokens 4096. Override with `EVAL_DATASETS`, `EVAL_N_SAMPLING`, `EVAL_TEMPERATURE`, `EVAL_TOP_P`, `EVAL_MAX_TOKENS`, `EVAL_SEED`, `OUTPUT_DIR`. ANTLR 4.11.1 is required. Keep GPU count fixed when comparing sampling runs; `mean_acc` is not pass@16.

## Code Generation

This adds the exploratory setup from paper section 4.3: UltraFeedback SFT data, one epoch, learning rate `5e-5`, warmup ratio `0.05`, global batch size `16`, and HumanEval/HumanEval+/MultiPL-E eval scripts.

```bash
python -m pip install -r requirements.txt
bash verl/download_codegen_dataset.sh
N_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 CODEGEN_MODELS=Qwen/Qwen2.5-Coder-3B bash verl/sweep_codegen_dft.sh
N_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 CODEGEN_MODELS=Qwen/Qwen2.5-Coder-3B bash verl/sweep_codegen_sft.sh
N_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 CODEGEN_MODELS=Qwen/Qwen2.5-Coder-3B bash verl/sweep_codegen_spft.sh
```

- Dataset output: `verl/data/ultrafeedback_codegen/{train,test}.parquet`.
- Downloader source defaults to `openbmb/UltraFeedback`. It samples usable rows, selects the response with the highest average numeric score, and writes 10,000 train rows plus 500 validation rows by default.
- Downloader knobs: `ULTRAFEEDBACK_SOURCE`, `ULTRAFEEDBACK_SPLIT`, `ULTRAFEEDBACK_REVISION`, `CODEGEN_TRAIN_SIZE`, `CODEGEN_VAL_SIZE`, `SEED`.
- Training wrappers set `DATASET_TYPE=ultrafeedback`, `TRAIN_BATCH_SIZE=16`, `MICRO_BATCH_SIZE_PER_GPU=1`, `WARMUP_STEPS_RATIO=0.05`, `TOTAL_EPOCHS=1`, and `OPTIM_LR=5e-5` by default. Use `sweep_codegen_sft.sh` for the SFT baseline, `sweep_codegen_dft.sh` for DFT, and `sweep_codegen_spft.sh` for SPFT.
- Model sweep knob: `CODEGEN_MODELS`, for example `Qwen/Qwen2.5-3B`, `Qwen/Qwen2.5-Coder-3B`, or `Qwen/Qwen2.5-Coder-7B`.

HumanEval/HumanEval+ eval uses EvalPlus with vLLM by default:

```bash
python -m pip install -r requirements-eval.txt
MODEL_NAME_OR_PATH=/path/to/checkpoint EVAL_CUDA_VISIBLE_DEVICES=0,1 bash verl/eval_codegen.sh
```

MultiPL-E eval requires the official checkout and Docker or Podman for executing generated code:

```bash
git clone https://github.com/nuprl/MultiPL-E /path/to/MultiPL-E
MODEL_NAME_OR_PATH=/path/to/checkpoint CODEGEN_EVALS=humaneval,multiple MULTIPLE_REPO=/path/to/MultiPL-E bash verl/eval_codegen.sh
```

`MULTIPLE_LANGS` defaults to `py,cpp,java,php,ts,cs,sh,js`, matching the paper columns Python/C++/Java/PHP/TS/C#/Bash/JS. EvalPlus and MultiPL-E download their benchmark data as needed. Outputs default to `verl/codegen_eval_outputs/...`.
MultiPL-E generation uses `automodel_vllm.py` by default (`MULTIPLE_BACKEND=vllm`) and passes `MULTIPLE_NUM_GPUS` to vLLM, defaulting to `EVAL_TP`/`N_GPUS`/`1`. Set `MULTIPLE_BACKEND=transformers` to use MultiPL-E's non-vLLM `automodel.py`.

If Docker/Podman is unavailable on the GPU server, generate completions there and run the execution step later on a local machine with Docker:

```bash
# On the GPU server: generate MultiPL-E completions only.
MODEL_NAME_OR_PATH=/path/to/checkpoint CODEGEN_EVALS=multiple MULTIPLE_REPO=/path/to/MultiPL-E \
  MULTIPLE_RUN_TESTS=false OUTPUT_DIR=/path/to/codegen_eval_outputs bash verl/eval_codegen.sh

# Copy /path/to/codegen_eval_outputs/multiple to a machine that can run Docker/Podman.
# On the local machine, from the MultiPL-E checkout:
docker run --rm --network none -v /path/to/codegen_eval_outputs/multiple:/out:rw \
  ghcr.io/nuprl/multipl-e-evaluation --dir /out --output-dir /out --recursive
python pass_k.py /path/to/codegen_eval_outputs/multiple/*
```
