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
