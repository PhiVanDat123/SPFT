#!/usr/bin/env bash
# Historical filename; N_GPUS controls 1, 2, 4, ... GPUs on one node.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for epochs in ${EPOCHS_LIST:-1}; do
    for lambda in ${SPFT_LAMBDAS:-0.1}; do
        for lr in ${OPTIM_LRS:-1e-4}; do
            for wd in ${OPTIM_WEIGHT_DECAYS:-0.01}; do
                name="numina-spft_adamw_lr${lr}_lambda${lambda}_wd${wd}_${epochs}ep_seed${SEED:-1}_$(date +%Y%m%d-%H%M%S)-$$"
                LOSS_MODE=spft SPFT_LAMBDA="$lambda" OPTIM_LR="$lr" OPTIM_WEIGHT_DECAY="$wd" \
                    TOTAL_EPOCHS="$epochs" EXPERIMENT_NAME="$name" bash "$script_dir/train.sh" "$@"
            done
        done
    done
done
