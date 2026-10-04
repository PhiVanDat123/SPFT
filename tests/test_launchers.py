import json
import os
from pathlib import Path
import shlex
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode", ["dft", "spft"])
@pytest.mark.parametrize("gpus", [1, 2, 4])
def test_sweep_dry_run(mode, gpus):
    result = subprocess.run(
        ["bash", str(ROOT / "verl" / f"sweep_{mode}_1gpu.sh"), "trainer.max_steps=2"],
        env={**os.environ, "DRY_RUN": "1", "N_GPUS": str(gpus)},
        check=True, capture_output=True, text=True,
    )
    args = shlex.split(result.stdout)
    assert f"--nproc_per_node={gpus}" in args
    assert f"optim.loss_mode={mode}" in args
    assert "trainer.max_steps=2" in args
    assert any(str(ROOT) in arg for arg in args)


@pytest.mark.parametrize("mode", ["sft", "dft", "spft"])
def test_codegen_sweep_dry_run(mode):
    result = subprocess.run(
        ["bash", str(ROOT / "verl" / f"sweep_codegen_{mode}.sh"), "trainer.max_steps=2"],
        env={**os.environ, "DRY_RUN": "1", "N_GPUS": "1"},
        check=True, capture_output=True, text=True,
    )
    args = shlex.split(result.stdout)
    assert "data.dataset_type=ultrafeedback" in args
    assert f"optim.loss_mode={mode}" in args
    assert "data.train_batch_size=16" in args
    assert "optim.warmup_steps_ratio=0.05" in args
    assert "trainer.total_epochs=1" in args


def test_reject_invalid_batch():
    result = subprocess.run(
        ["bash", str(ROOT / "verl/train.sh")],
        env={**os.environ, "DRY_RUN": "1", "N_GPUS": "3"}, capture_output=True,
    )
    assert result.returncode == 2


def test_summary_ignores_unrequested_files(tmp_path):
    for name, value in (("a", 20), ("b", 40), ("stale", 100)):
        (tmp_path / f"{name}_metrics.json").write_text(json.dumps({"acc": value, "mean_acc": value}))
    subprocess.run([os.sys.executable, str(ROOT / "verl/summarize_eval.py"), str(tmp_path), "a,b"], check=True)
    summary = json.loads((tmp_path / "eval_metrics_summary.json").read_text())
    assert summary["avg"]["mean_acc"] == 30
    assert "stale" not in summary


def test_codegen_eval_multiple_defaults_to_vllm():
    script = (ROOT / "verl" / "eval_codegen.sh").read_text()
    assert 'multiple_backend="${MULTIPLE_BACKEND:-vllm}"' in script
    assert "automodel_vllm.py" in script
    assert "--num-gpus" in script
    assert "MULTIPLE_BACKEND=transformers" in script
