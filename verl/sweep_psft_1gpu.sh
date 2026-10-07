#!/usr/bin/env bash
# Historical filename; N_GPUS controls 1, 2, 4, ... GPUs on one node.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for epochs in ${EPOCHS_LIST:-1}; do
    for clip_low in ${PSFT_CLIP_RATIO_LOWS:-0.2}; do
        for clip_high in ${PSFT_CLIP_RATIO_HIGHS:-0.28}; do
            for lr in ${OPTIM_LRS:-5e-5}; do
                for wd in ${OPTIM_WEIGHT_DECAYS:-0.1}; do
                    name="numina-psft_adamw_lr${lr}_clip${clip_low}-${clip_high}_wd${wd}_${epochs}ep_seed${SEED:-1}_$(date +%Y%m%d-%H%M%S)-$$"
                    LOSS_MODE=psft PSFT_CLIP_RATIO_LOW="$clip_low" PSFT_CLIP_RATIO_HIGH="$clip_high" \
                        OPTIM_LR="$lr" OPTIM_WEIGHT_DECAY="$wd" TOTAL_EPOCHS="$epochs" \
                        EXPERIMENT_NAME="$name" bash "$script_dir/train.sh" "$@"
                done
            done
        done
    done
done
