#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$script_dir/.." && pwd)"
if [[ "${1:-}" == --help ]]; then
    echo 'MODEL_NAME_OR_PATH=/path/to/checkpoint EVAL_CUDA_VISIBLE_DEVICES=0,1 bash eval_dft.sh'
    exit 0
fi
: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to a saved Hugging Face checkpoint or model ID}"
export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
export PYTHONPATH="$root/math_evaluation/latex2sympy${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
data_names="${EVAL_DATASETS:-math,math_oai,minerva_math,olympiadbench,aime24,amc23}"
output="${OUTPUT_DIR:-$script_dir/eval_outputs/$(date +%Y%m%d-%H%M%S)-$$}"
if [[ -d "$MODEL_NAME_OR_PATH" ]]; then
    MODEL_NAME_OR_PATH="$(cd "$MODEL_NAME_OR_PATH" && pwd)"
fi
mkdir -p "$output"
output="$(cd "$output" && pwd)"
IFS=',' read -ra names <<< "$data_names"
for name in "${names[@]}"; do
    case "$name" in math|math_oai|minerva_math|olympiadbench|aime24|amc23) ;;
        *) echo "Unsupported benchmark: $name" >&2; exit 2;;
    esac
    [[ -f "$root/math_evaluation/data/$name/test.jsonl" ]] || {
        echo 'Missing benchmarks. Run bash download_datasets.sh --eval-only' >&2; exit 2;
    }
    [[ ! -e "$output/${name}_metrics.json" && ! -e "$output/${name}_final_results.json" ]] || {
        echo "Existing results in $output; choose a fresh OUTPUT_DIR" >&2; exit 2;
    }
done
cd "$root/math_evaluation"
"${PYTHON_BIN:-python}" -u math_eval.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" --data_names "$data_names" \
    --output_dir "$output" --data_dir "$root/math_evaluation/data" \
    --prompt_type qwen-boxed --num_shots 0 --split test --seed "${EVAL_SEED:-0}" \
    --temperature "${EVAL_TEMPERATURE:-1}" --top_p "${EVAL_TOP_P:-1}" \
    --n_sampling "${EVAL_N_SAMPLING:-16}" --max_tokens_per_call "${EVAL_MAX_TOKENS:-4096}" \
    --num_test_sample "${EVAL_NUM_TEST_SAMPLE:--1}" --use_vllm
"${PYTHON_BIN:-python}" "$script_dir/summarize_eval.py" "$output" "$data_names"
