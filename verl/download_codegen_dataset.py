"""Build the UltraFeedback SFT subset used by the code-generation experiment."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def numeric_values(value):
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        values = []
        for item in value.values():
            values.extend(numeric_values(item))
        return values
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(numeric_values(item))
        return values
    return []


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, dict):
        return None
    for key in ("response", "text", "content", "answer", "completion"):
        value = completion.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def completion_score(completion):
    if not isinstance(completion, dict):
        return 0.0
    for key in ("overall_score", "score", "rating", "helpfulness", "preference_score"):
        value = completion.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    for key in ("annotations", "critique", "scores"):
        values = numeric_values(completion.get(key))
        if values:
            return sum(values) / len(values)
    values = numeric_values(completion)
    return sum(values) / len(values) if values else 0.0


def best_answer(row):
    completions = row.get("completions") or row.get("responses") or row.get("answers")
    if isinstance(completions, list) and completions:
        ranked = []
        for completion in completions:
            text = completion_text(completion)
            if text:
                ranked.append((completion_score(completion), text))
        if ranked:
            return max(ranked, key=lambda item: item[0])[1]
    for key in ("chosen", "response", "answer", "completion"):
        value = row.get(key)
        text = completion_text(value)
        if text:
            return text
    return None


def prompt_text(row):
    for key in ("instruction", "prompt", "question"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
    return None


def split_rows(rows, val_size, seed):
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    val_count = min(val_size, max(1, len(rows) // 20))
    val_indices = set(indices[:val_count])
    train, val = [], []
    for index, row in enumerate(rows):
        (val if index in val_indices else train).append(row)
    return train, val


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=os.getenv("ULTRAFEEDBACK_SOURCE", "openbmb/UltraFeedback"))
    parser.add_argument("--split", default=os.getenv("ULTRAFEEDBACK_SPLIT", "train"))
    parser.add_argument("--revision", default=os.getenv("ULTRAFEEDBACK_REVISION", "main"))
    parser.add_argument("--train-size", type=int, default=int(os.getenv("CODEGEN_TRAIN_SIZE", "10000")))
    parser.add_argument("--val-size", type=int, default=int(os.getenv("CODEGEN_VAL_SIZE", "500")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("SEED", "1")))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "ultrafeedback_codegen"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.train_size < 1 or args.val_size < 1:
        parser.error("train and validation sizes must be positive")

    from datasets import Dataset, load_dataset
    from huggingface_hub import HfApi

    target = Path(args.output_dir)
    manifest_path = target / "manifest.json"
    if target.exists() and any(target.iterdir()) and not args.force:
        if not manifest_path.exists():
            raise ValueError("Data exists without a manifest; choose --force to regenerate")
        previous = json.loads(manifest_path.read_text())
        expected = {
            "source": args.source, "requested_revision": args.revision,
            "split": args.split, "train_rows": args.train_size,
            "val_rows": args.val_size, "seed": args.seed,
        }
        if any(previous.get(key) != value for key, value in expected.items()):
            raise ValueError("Existing UltraFeedback subset differs. Use --force explicitly.")
        for split in ("train", "test"):
            path = target / f"{split}.parquet"
            if digest(path) != previous["files"][split]["sha256"]:
                raise ValueError("Dataset checksum mismatch; use --force to regenerate")
        print(f"Verified existing UltraFeedback codegen dataset: {manifest_path}")
        return

    revision = HfApi().dataset_info(args.source, revision=args.revision).sha
    dataset = load_dataset(args.source, split=args.split, revision=revision).shuffle(seed=args.seed)
    rows = []
    for row in dataset:
        question = prompt_text(row)
        answer = best_answer(row)
        if question and answer:
            rows.append({"extra_info": {"question": question, "answer": answer, "source_index": len(rows)}})
        if len(rows) >= args.train_size + args.val_size:
            break
    if len(rows) < args.train_size + 1:
        raise ValueError(f"Only found {len(rows)} usable rows; cannot build a {args.train_size}-row train set")

    train, val = split_rows(rows[:args.train_size + args.val_size], args.val_size, args.seed)
    train = train[:args.train_size]
    target.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": args.source, "revision": revision, "requested_revision": args.revision,
        "split": args.split, "train_rows": len(train), "val_rows": len(val), "seed": args.seed,
        "selection": "seeded shuffled usable rows; best completion by average numeric score; seeded validation split",
        "files": {},
    }
    for split, data in (("train", train), ("test", val)):
        path = target / f"{split}.parquet"
        temporary = path.with_suffix(".parquet.tmp")
        Dataset.from_list(data).to_parquet(str(temporary))
        temporary.replace(path)
        manifest["files"][split] = {"rows": len(data), "sha256": digest(path)}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
