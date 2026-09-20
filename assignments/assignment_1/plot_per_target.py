from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = {
    "mutation": "Mutation only",
    "crossover": "Crossover + mutation",
}
COLORS = {
    "mutation": "#2563eb",
    "crossover": "#dc2626",
}


def load_target_runs(directory: Path, method: str) -> np.ndarray:
    """Load per-target distances for all seeds of one method."""
    paths = sorted(directory.glob(f"{method}_seed_*_targets.csv"))
    runs = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        runs.append(np.asarray([float(row["distance"]) for row in rows]))

    return np.asarray(runs)


def plot_per_target(directory: Path, output: Path) -> None:
    """Create the final per-target comparison plot."""
    mutation = load_target_runs(directory, "mutation")
    crossover = load_target_runs(directory, "crossover")

    if mutation.shape[1] != crossover.shape[1]:
        raise ValueError("mutation and crossover runs contain different numbers of targets")

    mutation_mean = mutation.mean(axis=0)
    mutation_std = mutation.std(axis=0)
    crossover_mean = crossover.mean(axis=0)
    crossover_std = crossover.std(axis=0)

    num_targets = mutation.shape[1]
    x = np.arange(num_targets)
    width = 0.36

    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)

    axis.bar(
        x - width / 2,
        mutation_mean,
        width,
        yerr=mutation_std,
        capsize=4,
        color=COLORS["mutation"],
        alpha=0.8,
        label=METHODS["mutation"],
    )
    axis.bar(
        x + width / 2,
        crossover_mean,
        width,
        yerr=crossover_std,
        capsize=4,
        color=COLORS["crossover"],
        alpha=0.8,
        label=METHODS["crossover"],
    )

    axis.set_title("Assignment 1: final performance on each target")
    axis.set_xlabel("Target body")
    axis.set_ylabel("Tree edit distance, lower is better")
    axis.set_xticks(x)
    axis.set_xticklabels([f"Target {index + 1}" for index in range(num_targets)])
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)

    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    """Generate the per-target result figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path.cwd() / "__data__" / "A1_template_2026" / "experiments",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or args.results / "per_target.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    plot_per_target(args.results, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()