#!/usr/bin/env bash
# Historical filename; N_GPUS controls 1, 2, 4, ... GPUs on one node.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for epochs in ${EPOCHS_LIST:-1}; do
    for lr in ${OPTIM_LRS:-5e-5}; do
        for wd in ${OPTIM_WEIGHT_DECAYS:-0.01}; do
            name="numina-sft_adamw_lr${lr}_wd${wd}_${epochs}ep_seed${SEED:-1}_$(date +%Y%m%d-%H%M%S)-$$"
            LOSS_MODE=sft OPTIM_LR="$lr" OPTIM_WEIGHT_DECAY="$wd" TOTAL_EPOCHS="$epochs" \
                EXPERIMENT_NAME="$name" bash "$script_dir/train.sh" "$@"
        done
    done
done
