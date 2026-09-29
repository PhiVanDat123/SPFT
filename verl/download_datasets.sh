#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="${HF_HOME:-${script_dir}/.cache/huggingface}"
exec "${PYTHON_BIN:-python}" "$script_dir/download_datasets.py" "$@"
