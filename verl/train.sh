#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
export PYTHONPATH="${script_dir}${PYTHONPATH:+:${PYTHONPATH}}"
python_bin="${PYTHON_BIN:-python}"
gpus="${N_GPUS:-1}"
global_batch="${TRAIN_BATCH_SIZE:-256}"
micro_batch="${MICRO_BATCH_SIZE_PER_GPU:-8}"
for number in "$gpus" "$global_batch" "$micro_batch"; do
    [[ "$number" =~ ^[1-9][0-9]*$ ]] || { echo 'GPU and batch counts must be positive integers' >&2; exit 2; }
done
(( global_batch % (gpus * micro_batch) == 0 )) || {
    echo 'TRAIN_BATCH_SIZE must be divisible by N_GPUS * MICRO_BATCH_SIZE_PER_GPU' >&2; exit 2;
}
mode="${LOSS_MODE:-dft}"
[[ "$mode" == dft || "$mode" == spft ]] || { echo 'LOSS_MODE must be dft or spft' >&2; exit 2; }
run_name="${EXPERIMENT_NAME:-numina-${mode}-$(date +%Y%m%d-%H%M%S)-$$}"
train_file="${TRAIN_FILE:-${script_dir}/data/numina_cot/train.parquet}"
val_file="${VAL_FILE:-${script_dir}/data/numina_cot/test.parquet}"
save_path="${SAVE_PATH:-${script_dir}/checkpoints/${run_name}}"
cmd=("$python_bin" -m torch.distributed.run --standalone --nnodes=1 "--nproc_per_node=$gpus"
    -m verl.trainer.fsdp_dft_trainer
    "data.dataset_type='${DATASET_TYPE:-numina}'"
    "data.train_files='$train_file'" "data.val_files='$val_file'"
    "data.train_batch_size=$global_batch" "data.micro_batch_size_per_gpu=$micro_batch"
    "data.max_length=${MAX_LENGTH:-2048}"
    "model.partial_pretrain='${MODEL_NAME:-Qwen/Qwen2.5-Math-1.5B}'"
    "model.use_liger=${USE_LIGER:-false}"
    "optim.loss_mode=$mode" "optim.lr=${OPTIM_LR:-5e-5}"
    "optim.weight_decay=${OPTIM_WEIGHT_DECAY:-0.01}"
    "optim.spft.lambda=${SPFT_LAMBDA:-0.1}"
    "optim.spft.reference_cpu_offload=${SPFT_REFERENCE_CPU_OFFLOAD:-false}"
    "trainer.total_epochs=${TOTAL_EPOCHS:-1}" "trainer.seed=${SEED:-1}"
    "trainer.test_freq=${TEST_FREQ:-10}" "trainer.save_freq=${SAVE_FREQ:--1}"
    "trainer.default_local_dir='$save_path'" "trainer.experiment_name='$run_name'"
    "trainer.project_name='${WANDB_PROJECT:-final-spft}'" "trainer.wandb=${USE_WANDB:-false}"
    "$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    exit 0
fi
[[ -f "$train_file" && -f "$val_file" ]] || {
    echo 'Missing parquet files. Run bash download_datasets.sh first.' >&2; exit 2;
}
# torchrun owns all distributed rank/rendezvous variables.
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE DIST_INIT_METHOD
exec "${cmd[@]}"
