# Provenance

Base repository: https://github.com/yongliang-wu/DFT

Pinned master commit: `11e395d49ed1b9da7dc5d957224d942d90af5bf6`.

- The lightweight trainer is a refactoring of the upstream DFT training path: HF causal LM, response-token masking, detached probability-weighted cross entropy, FSDP, AdamW, cosine schedule and HF checkpoint export. Multi-GPU normalization and validation handling intentionally differ; see README.
- `math_evaluation/{evaluate,parser,grader,utils,examples,trajectory,python_executor,model_utils,data_loader}.py` are copied unchanged from that commit.
- `math_evaluation/math_eval.py` is copied with explicit spawn, visible-device mapping, vLLM seed, empty-input and worker-failure handling fixes.
- `math_evaluation/latex2sympy` contains only the upstream runtime Python files and their license, not the Java compiler, tests or binaries.
- Training code license: root `LICENSE` (Apache-2.0). Evaluator license: `math_evaluation/LICENSE` (MIT). LaTeX parser license: `math_evaluation/latex2sympy/LICENSE.txt`.
- No historical checkpoints, credentials, run logs or datasets were imported.
