#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for model in ${CODEGEN_MODELS:-Qwen/Qwen2.5-Coder-3B}; do
    safe_model="${model//\//--}"
    for epochs in ${EPOCHS_LIST:-1}; do
        for lambda in ${SPFT_LAMBDAS:-0.1}; do
            for lr in ${OPTIM_LRS:-5e-5}; do
                for wd in ${OPTIM_WEIGHT_DECAYS:-0.01}; do
                    name="ultrafeedback-codegen-spft_${safe_model}_lr${lr}_lambda${lambda}_wd${wd}_${epochs}ep_seed${SEED:-1}_$(date +%Y%m%d-%H%M%S)-$$"
                    DATASET_TYPE=ultrafeedback MODEL_NAME="$model" LOSS_MODE=spft SPFT_LAMBDA="$lambda" \
                        TRAIN_FILE="${TRAIN_FILE:-${script_dir}/data/ultrafeedback_codegen/train.parquet}" \
                        VAL_FILE="${VAL_FILE:-${script_dir}/data/ultrafeedback_codegen/test.parquet}" \
                        TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}" MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}" \
                        OPTIM_LR="$lr" OPTIM_WEIGHT_DECAY="$wd" TOTAL_EPOCHS="$epochs" \
                        EXPERIMENT_NAME="$name" SAVE_PATH="${SAVE_PATH:-${script_dir}/checkpoints/${name}}" \
                        bash "$script_dir/train.sh" "optim.warmup_steps_ratio=${WARMUP_STEPS_RATIO:-0.05}" "$@"
                done
            done
        done
    done
done
