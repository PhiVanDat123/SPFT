"""Download Numina only by default; benchmark download is explicit."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
UPSTREAM = "11e395d49ed1b9da7dc5d957224d942d90af5bf6"
BENCHMARKS = ("math", "math_oai", "minerva_math", "olympiadbench", "aime24", "amc23")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_numina(args):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    target = ROOT / "data/numina_cot"
    manifest_path = target / "manifest.json"
    if target.exists() and any(target.iterdir()) and not args.force:
        if not manifest_path.exists():
            raise ValueError("Data exists without a manifest; choose --force to regenerate")
        previous = json.loads(manifest_path.read_text())
        if previous["requested_train_rows"] != args.train_size or previous["requested_revision"] != args.revision:
            raise ValueError("Existing subset/revision differs. Use --force explicitly.")
        for split in ("train", "test"):
            if digest(target / f"{split}.parquet") != previous["files"][split]["sha256"]:
                raise ValueError("Dataset checksum mismatch; use --force to regenerate")
        print(f"Verified existing Numina dataset: {manifest_path}")
        return
    source = "AI-MO/NuminaMath-CoT"
    revision = HfApi().dataset_info(source, revision=args.revision).sha
    dataset = load_dataset(source, revision=revision)
    train = dataset["train"].select(range(min(args.train_size, len(dataset["train"]))))
    target.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": source, "revision": revision, "requested_revision": args.revision,
        "requested_train_rows": args.train_size, "selection": "first N; no shuffle", "files": {},
        "validation": "official Numina test split; NOT MATH/Math500",
    }
    for split, data in (("train", train), ("test", dataset["test"])):
        data = data.map(
            lambda row, i: {"extra_info": {"question": row["problem"], "answer": row["solution"], "index": i}},
            with_indices=True, remove_columns=data.column_names,
        )
        path = target / f"{split}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        data.to_parquet(str(temporary))
        temporary.replace(path)
        manifest["files"][split] = {"rows": len(data), "sha256": digest(path)}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def download_benchmarks(args):
    for name in args.eval_datasets.split(","):
        if name not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark {name!r}; choose {BENCHMARKS}")
        path = ROOT.parent / "math_evaluation/data" / name / "test.jsonl"
        url = f"https://raw.githubusercontent.com/yongliang-wu/DFT/{UPSTREAM}/math_evaluation/data/{name}/test.jsonl"
        manifest_path = path.parent / "manifest.json"
        if path.exists() and not args.force:
            if not manifest_path.exists():
                raise ValueError(f"No provenance for {path}; use --force to download the pinned version")
            old = json.loads(manifest_path.read_text())
            if old["sha256"] != digest(path) or old["source"] != url:
                raise ValueError(f"Benchmark checksum/source mismatch: {path}")
            print(f"Verified {path}")
            continue
        with urlopen(url, timeout=120) as response:
            payload = response.read()
        rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
        manifest_path.write_text(json.dumps({"source": url, "rows": len(rows), "sha256": digest(path)}, indent=2))
        print(f"{name}: {len(rows)} questions -> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=int(os.getenv("NUMINA_TRAIN_END", "100000")))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help="Download benchmarks instead of Numina")
    parser.add_argument("--eval-datasets", default=",".join(BENCHMARKS))
    args = parser.parse_args()
    if args.train_size < 1:
        parser.error("--train-size must be positive")
    if args.eval_only:
        download_benchmarks(args)
    else:
        download_numina(args)


if __name__ == "__main__":
    main()
