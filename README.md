# Final SPFT

DFT/SPFT + AdamW, từ DFT gốc `11e395d4`. Chi tiết nguồn và license: `UPSTREAM.md`.

```bash
python -m pip install -r requirements.txt
python -m pip install flash-attn --no-build-isolation
bash verl/download_datasets.sh
N_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 bash verl/sweep_dft_1gpu.sh
N_GPUS=2 CUDA_VISIBLE_DEVICES=0,1 bash verl/sweep_spft_1gpu.sh
python -m pytest -q tests
```

- Dataset mặc định: 100.000 mẫu Numina train, Numina test làm validation. Đặt `VAL_FILE` để dùng parquet validation khác (`extra_info.question/answer`). Manifest lưu revision và SHA256.
- Global batch 256, micro-batch/GPU 8; chỉnh bằng `TRAIN_BATCH_SIZE`, `MICRO_BATCH_SIZE_PER_GPU`. Global batch phải chia hết cho số GPU × micro-batch. Tên `_1gpu.sh` vẫn hỗ trợ nhiều GPU trên một node.
- Sweep: `OPTIM_LRS`, `OPTIM_WEIGHT_DECAYS`, `EPOCHS_LIST`, `SPFT_LAMBDAS`. Chỉnh model bằng `MODEL_NAME`; `DRY_RUN=1` chỉ in lệnh. Hydra overrides thêm cuối lệnh.
- Liger mặc định tắt. `USE_LIGER=true` cần cài liger-kernel; `USE_WANDB=true` cần wandb. Nếu không cài FlashAttention, thêm `model.attention=sdpa`.
- Full fine-tuning FSDP1; reference SPFT cố định, một replica/GPU. Có `SPFT_REFERENCE_CPU_OFFLOAD=true`. Checkpoint HF không chứa optimizer state/resume.
- Đã sửa gradient accumulation và chuẩn hóa theo tổng token của global batch; validation tổng hợp mọi rank, không bỏ/trùng mẫu. Vì vậy không tái lập bit-for-bit run cũ. `weight_threshold` chỉ dùng cho metric.

Eval (nên dùng môi trường riêng):

```bash
python -m pip install -r requirements-eval.txt
bash verl/download_datasets.sh --eval-only
MODEL_NAME_OR_PATH=/path/to/checkpoint EVAL_CUDA_VISIBLE_DEVICES=0,1 bash verl/eval_dft.sh
```

Sáu bộ: `math` (5.000 câu), `math_oai` (500 câu), Minerva, OlympiadBench, AIME24, AMC23. Mặc định qwen-boxed, n=16, temperature=1, top-p=1, max tokens=4096. Chỉnh bằng `EVAL_DATASETS` (dấu phẩy), `EVAL_N_SAMPLING`, `EVAL_TEMPERATURE`, `EVAL_TOP_P`, `EVAL_MAX_TOKENS`, `EVAL_SEED`, `OUTPUT_DIR`. ANTLR 4.11.1 bắt buộc. Giữ cùng số GPU khi so sampling; `mean_acc` không phải pass@16.
