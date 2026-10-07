"""Summarize EvalPlus HumanEval/HumanEval+ outputs."""

import argparse
import json
import math
from pathlib import Path


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_rate(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def metric_keys(pass_k):
    return (f"pass@{pass_k}", f"pass_at_{pass_k}")


def find_precomputed_rates(obj, pass_k):
    if not isinstance(obj, dict):
        return None
    candidates = []
    metric = next((obj.get(key) for key in metric_keys(pass_k) if key in obj), None)
    if isinstance(metric, dict):
        he = as_rate(metric.get("base") or metric.get("HE") or metric.get("humaneval"))
        hep = as_rate(metric.get("plus") or metric.get("HE+") or metric.get("humaneval_plus"))
        if he is not None and hep is not None:
            candidates.append((he, hep))
    for key in ("base", "plus"):
        if key in obj and isinstance(obj[key], dict):
            he = next((as_rate(obj["base"].get(metric_key)) for metric_key in metric_keys(pass_k) if metric_key in obj["base"]), None)
            hep = next((as_rate(obj["plus"].get(metric_key)) for metric_key in metric_keys(pass_k) if metric_key in obj["plus"]), None)
            if he is not None and hep is not None:
                candidates.append((he, hep))
                break
    return candidates[0] if candidates else None


def iter_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from iter_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_dicts(value)


def is_pass(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"pass", "passed", "ok", "success", "true"}
    return False


def task_id_of(row):
    task_id = row.get("task_id") or row.get("task")
    if isinstance(task_id, str):
        return task_id
    return None


def pass_at_k(n, c, k):
    if n < k:
        return None
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / i for i in range(n - c + 1, n + 1))


def find_per_task_rates(obj, pass_k):
    by_task = {}
    for row in iter_dicts(obj):
        task_id = task_id_of(row)
        if task_id is None:
            continue
        base_keys = ("base_status", "base_result", "base_passed", "passed_base")
        plus_keys = ("plus_status", "plus_result", "plus_passed", "passed_plus")
        if not any(key in row for key in base_keys + plus_keys):
            continue
        base = any(is_pass(row.get(key)) for key in base_keys)
        plus = any(is_pass(row.get(key)) for key in plus_keys)
        previous = by_task.get(task_id, [0, 0, 0])
        previous[0] += 1
        previous[1] += int(base)
        previous[2] += int(plus)
        by_task[task_id] = previous
    if not by_task:
        return None
    base_rates = []
    plus_rates = []
    for n, base_correct, plus_correct in by_task.values():
        base_rate = pass_at_k(n, base_correct, pass_k)
        plus_rate = pass_at_k(n, plus_correct, pass_k)
        if base_rate is not None and plus_rate is not None:
            base_rates.append(base_rate)
            plus_rates.append(plus_rate)
    total = len(base_rates)
    if total == 0:
        return None
    he = sum(base_rates) / total
    hep = sum(plus_rates) / total
    return he, hep, total


def summarize(root, pass_k=1):
    json_paths = sorted(Path(root).rglob("*.json"))
    errors = []
    for path in json_paths:
        try:
            obj = load_json(path)
        except Exception as exc:  # pragma: no cover - best-effort diagnostics
            errors.append(f"{path}: {exc}")
            continue
        precomputed = find_precomputed_rates(obj, pass_k)
        if precomputed is not None:
            he, hep = precomputed
            return {"HE": he * 100, "HE+": hep * 100, "pass_k": pass_k, "source": str(path), "num_problems": None}
        per_task = find_per_task_rates(obj, pass_k)
        if per_task is not None:
            he, hep, total = per_task
            return {"HE": he * 100, "HE+": hep * 100, "pass_k": pass_k, "source": str(path), "num_problems": total}
    raise FileNotFoundError(
        f"No EvalPlus pass@{pass_k} summary/results found under {root}. "
        f"Scanned {len(json_paths)} JSON files. Errors: {errors[:3]}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="EvalPlus output directory")
    parser.add_argument("--pass-k", type=int, default=1, help="pass@k metric to summarize")
    args = parser.parse_args()
    result = summarize(args.root, pass_k=args.pass_k)
    root = Path(args.root)
    (root / "humaneval_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    csv = "Benchmark,Metric,Score\nHE,pass@{pass_k},{HE:.2f}\nHE+,pass@{pass_k},{HE+:.2f}\n".format(**result)
    (root / "humaneval_summary.csv").write_text(csv, encoding="utf-8")
    print(csv, end="")
    print(f"Source: {result['source']}")


if __name__ == "__main__":
    main()
