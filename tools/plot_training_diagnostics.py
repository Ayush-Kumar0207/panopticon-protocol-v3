#!/usr/bin/env python3
"""Generate raw and five-point-smoothed canonical optimizer diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_repro import ReproducibilityError, atomic_write_json, load_spec


def rolling(values: list[float], window: int = 5) -> list[float]:
    return [sum(values[max(0, i - window + 1):i + 1]) / min(i + 1, window) for i in range(len(values))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--spec", default="training_specs/security_first_v5.json")
    args = parser.parse_args()
    try:
        spec, _ = load_spec(args.spec)
        root = Path(args.run_dir).resolve()
        lock = json.loads((root / spec["outputs"]["run_lock"]).read_text(encoding="utf-8"))
        rows = []
        for line_number, line in enumerate((root / spec["outputs"]["events"]).read_text(encoding="utf-8").splitlines(), start=1):
            row = json.loads(line)
            if row.get("run_fingerprint") != lock["run_fingerprint"]:
                raise ReproducibilityError(f"event line {line_number} belongs to another experiment")
            if row.get("event") == "train_progress" and row.get("loss") is not None:
                values = [row.get(key) for key in ("loss", "grad_norm", "learning_rate")]
                if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
                    raise ReproducibilityError(f"non-finite optimizer diagnostic at event line {line_number}")
                rows.append(row)
        levels = spec["trajectory"]["levels"]
        missing = [level for level in levels if not any(row.get("level") == level for row in rows)]
        if missing:
            raise ReproducibilityError(f"optimizer diagnostics omit levels: {missing}")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        for axis, metric in zip(axes, ("loss", "grad_norm", "learning_rate")):
            for level in levels:
                points = [row for row in rows if row.get("level") == level]
                x = [float(row.get("curriculum_pct", 0)) for row in points]
                y = [float(row[metric]) for row in points]
                axis.plot(x, y, alpha=0.22, linewidth=0.8)
                axis.plot(x, rolling(y), linewidth=1.4, label=f"{level} smooth(5)")
            axis.set_ylabel(metric)
            axis.grid(alpha=0.2)
        axes[-1].set_xlabel("curriculum completion (%)")
        axes[0].legend(ncol=3, fontsize=8)
        figure.suptitle("Panopticon canonical optimizer diagnostics (raw curves retained)")
        figure.tight_layout()
        output_dir = root / spec["outputs"]["training_plots_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_dir / "optimizer_diagnostics.png", dpi=170)
        plt.close(figure)
        atomic_write_json(output_dir / "optimizer_diagnostics.json", {
            "schema_version": 1,
            "run_fingerprint": lock["run_fingerprint"],
            "smoothing": {"method": "trailing_mean", "window": 5},
            "raw_points": len(rows),
            "levels": {level: sum(row.get("level") == level for row in rows) for level in levels},
            "interpretation": "Plots are diagnostics, not acceptance evidence. Raw event values remain authoritative.",
        })
        print(output_dir)
    except Exception as exc:
        print(f"STOP: cannot generate training diagnostics: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
