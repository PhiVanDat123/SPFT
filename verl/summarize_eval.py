"""Average exactly the requested benchmarks, never stale files from other runs."""

import json
from pathlib import Path
import sys


def summarize(output, names):
    if not names or len(set(names)) != len(names):
        raise ValueError("Provide a non-empty, unique benchmark list")
    result = {}
    for name in names:
        result[name] = json.loads((output / f"{name}_metrics.json").read_text())
    result["avg"] = {
        key: sum(result[name][key] for name in names) / len(names)
        for key in ("acc", "mean_acc")
    }
    (output / "eval_metrics_summary.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1]), sys.argv[2].split(",")), indent=2))
