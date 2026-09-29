import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_downloader(tmp_path):
    path = Path(__file__).resolve().parents[1] / "verl/download_datasets.py"
    spec = importlib.util.spec_from_file_location("download_datasets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path / "verl"
    return module


def test_numina_subset_provenance_and_reuse(tmp_path, monkeypatch):
    import datasets
    import huggingface_hub

    module = load_downloader(tmp_path)
    source = datasets.Dataset.from_list([{"problem": f"q{i}", "solution": f"a{i}"} for i in range(5)])
    calls = []

    def fake_load(name, revision):
        calls.append((name, revision))
        return {"train": source, "test": source.select([3, 4])}

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    monkeypatch.setattr(huggingface_hub.HfApi, "dataset_info", lambda *args, **kwargs: SimpleNamespace(sha="pinned"))
    args = SimpleNamespace(train_size=3, revision="main", force=False)
    module.download_numina(args)
    module.download_numina(args)
    assert calls == [("AI-MO/NuminaMath-CoT", "pinned")]
    manifest = json.loads((module.ROOT / "data/numina_cot/manifest.json").read_text())
    assert manifest["files"]["train"]["rows"] == 3
    assert manifest["files"]["test"]["rows"] == 2
    assert not (tmp_path / "math_evaluation").exists()
    args.train_size = 4
    with pytest.raises(ValueError, match="subset/revision"):
        module.download_numina(args)


def test_benchmark_is_explicit_pinned_and_verified(tmp_path, monkeypatch):
    module = load_downloader(tmp_path)
    urls = []

    def fake_urlopen(url, timeout):
        urls.append(url)
        return io.BytesIO(b'{"problem":"1+1", "solution":"2"}\n')

    monkeypatch.setattr(module, "urlopen", fake_urlopen)
    args = SimpleNamespace(eval_datasets="math_oai", force=False)
    module.download_benchmarks(args)
    module.download_benchmarks(args)
    assert len(urls) == 1 and module.UPSTREAM in urls[0]
    assert not (module.ROOT / "data/numina_cot").exists()
    path = tmp_path / "math_evaluation/data/math_oai/test.jsonl"
    path.write_text("modified")
    with pytest.raises(ValueError, match="checksum"):
        module.download_benchmarks(args)
