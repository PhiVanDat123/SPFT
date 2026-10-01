#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$script_dir/.." && pwd)"
if [[ "${1:-}" == --help ]]; then
    echo 'MODEL_NAME_OR_PATH=/path/to/checkpoint EVAL_CUDA_VISIBLE_DEVICES=0,1 bash verl/eval_codegen.sh'
    echo 'Set CODEGEN_EVALS=humaneval,multiple to include MultiPL-E when MULTIPLE_REPO points to a checkout.'
    exit 0
fi
: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to a saved Hugging Face checkpoint or model ID}"
export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
export TOKENIZERS_PARALLELISM=false
if [[ -d "$MODEL_NAME_OR_PATH" ]]; then
    MODEL_NAME_OR_PATH="$(cd "$MODEL_NAME_OR_PATH" && pwd)"
fi
evals="${CODEGEN_EVALS:-humaneval}"
output="${OUTPUT_DIR:-$script_dir/codegen_eval_outputs/$(date +%Y%m%d-%H%M%S)-$$}"
mkdir -p "$output"
output="$(cd "$output" && pwd)"
IFS=',' read -ra names <<< "$evals"
for name in "${names[@]}"; do
    case "$name" in humaneval|multiple) ;;
        *) echo "Unsupported codegen eval: $name" >&2; exit 2 ;;
    esac
done

for name in "${names[@]}"; do
    case "$name" in
        humaneval)
            evalplus.evaluate \
                --model "$MODEL_NAME_OR_PATH" --dataset humaneval \
                --backend "${EVALPLUS_BACKEND:-vllm}" --greedy \
                --tp "${EVAL_TP:-${N_GPUS:-1}}" --root "$output/evalplus"
            ;;
        multiple)
            : "${MULTIPLE_REPO:?Set MULTIPLE_REPO to a local MultiPL-E checkout for CODEGEN_EVALS=multiple}"
            multiple_repo="$(cd "$MULTIPLE_REPO" && pwd)"
            languages="${MULTIPLE_LANGS:-py,cpp,java,php,ts,cs,sh,js}"
            mkdir -p "$output/multiple"
            (
                cd "$multiple_repo"
                IFS=',' read -ra langs <<< "$languages"
                result_dirs=()
                for lang in "${langs[@]}"; do
                    lang_out="$output/multiple/$lang"
                    mkdir -p "$lang_out"
                    "${PYTHON_BIN:-python}" automodel.py \
                        --name "$MODEL_NAME_OR_PATH" \
                        --root-dataset humaneval \
                        --lang "$lang" \
                        --temperature "${MULTIPLE_TEMPERATURE:-0.2}" \
                        --batch-size "${MULTIPLE_BATCH_SIZE:-20}" \
                        --completion-limit "${MULTIPLE_COMPLETION_LIMIT:-20}" \
                        --output-dir-prefix "$lang_out"
                    generated="$(find "$lang_out" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
                    [[ -n "$generated" ]] || { echo "No MultiPL-E output for $lang" >&2; exit 2; }
                    if [[ "${MULTIPLE_RUN_TESTS:-true}" == true ]]; then
                        if command -v docker >/dev/null 2>&1; then
                            docker run --rm --network none -v "$generated:/out:rw" ghcr.io/nuprl/multipl-e-evaluation \
                                --dir /out --output-dir /out
                        elif command -v podman >/dev/null 2>&1; then
                            podman run --rm --network none -v "$generated:/out:rw" ghcr.io/nuprl/multipl-e-evaluation \
                                --dir /out --output-dir /out
                        else
                            echo "Install Docker/Podman or set MULTIPLE_RUN_TESTS=false to skip execution" >&2
                            exit 2
                        fi
                    fi
                    result_dirs+=("$generated")
                done
                "${PYTHON_BIN:-python}" pass_k.py "${result_dirs[@]}" > "$output/multiple/pass_k.txt"
            )
            ;;
    esac
done
echo "Code generation eval outputs: $output"
