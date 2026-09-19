"""Plot Assignment 1 convergence and final-score results.

Run this after A1_template_2026.py has generated CSV files in
``__data__/A1_template_2026/experiments``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


METHODS = {
    "mutation": "Mutation only",
    "crossover": "Crossover + mutation",
    "random": "Random search",
}
COLORS = {
    "mutation": "#2563eb",
    "crossover": "#dc2626",
    "random": "#6b7280",
}


def read_run(path: Path) -> dict[str, np.ndarray]:
    """Read one result CSV into numeric arrays."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        key: np.asarray([float(row[key]) for row in rows])
        for key in ("generation", "best", "mean", "std")
    }


def load_runs(directory: Path, method: str) -> list[dict[str, np.ndarray]]:
    """Load all seed files for one method."""
    paths = sorted(directory.glob(f"{method}_seed_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no {method} CSV files found in {directory}")
    return [read_run(path) for path in paths]


def summarize_ea(
    runs: list[dict[str, np.ndarray]],
    population_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Put EA best-so-far values on the evaluation axis."""
    generations = runs[0]["generation"]
    x = generations * population_size
    best = np.asarray([run["best"] for run in runs])
    return x, best.mean(axis=0), best.std(axis=0)


def summarize_random(
    runs: list[dict[str, np.ndarray]],
    population_size: int,
    generations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample random-search best-so-far at EA generation boundaries."""
    x = np.arange(generations + 1) * population_size
    best = np.asarray([
        run["best"][x.astype(int)]
        for run in runs
    ])
    return x, best.mean(axis=0), best.std(axis=0)


def plot_convergence(
    directory: Path,
    output: Path,
    population_size: int,
    generations: int,
) -> None:
    """Create the convergence plot with mean and across-seed spread."""
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for method in METHODS:
        runs = load_runs(directory, method)
        if method == "random":
            x, mean, spread = summarize_random(
                runs, population_size, generations,
            )
        else:
            x, mean, spread = summarize_ea(runs, population_size)
        color = COLORS[method]
        axis.plot(
            x,
            mean,
            label=f"{METHODS[method]}: mean best fitness",
            color=color,
            linewidth=2,
        )
        axis.fill_between(
            x,
            mean - spread,
            mean + spread,
            color=color,
            alpha=0.15,
        )

    axis.set_title("Assignment 1: morphology search convergence")
    axis.set_xlabel("Body evaluations")
    axis.set_ylabel("Best fitness so far, lower is better")
    axis.grid(True, alpha=0.25)
    axis.legend(
        handles=[
            Line2D([0], [0], color=COLORS["mutation"], linewidth=2,
                   label="Mutation only: mean best fitness"),
            Line2D([0], [0], color=COLORS["crossover"], linewidth=2,
                   label="Crossover + mutation: mean best fitness"),
            Line2D([0], [0], color=COLORS["random"], linewidth=2,
                   label="Random search: mean best fitness"),
            Patch(facecolor="black", alpha=0.15,
                  label="Shading: +/- 1 SD across seeds"),
        ],
        frameon=False,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_final_scores(directory: Path, output: Path) -> None:
    """Create a final-score distribution plot across independent seeds."""
    values: list[np.ndarray] = []
    labels: list[str] = []
    colors: list[str] = []
    for method in METHODS:
        runs = load_runs(directory, method)
        values.append(np.asarray([run["best"][-1] for run in runs]))
        labels.append(METHODS[method])
        colors.append(COLORS[method])

    figure, axis = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    boxplot = axis.boxplot(
        values,
        patch_artist=True,
        tick_labels=labels,
        flierprops={
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": "black",
            "markersize": 5,
        },
    )
    for box, color in zip(boxplot["boxes"], colors, strict=True):
        box.set_facecolor(color)
        box.set_alpha(0.7)
    for index, scores in enumerate(values, start=1):
        jitter = np.linspace(-0.08, 0.08, len(scores))
        axis.scatter(
            index + jitter,
            scores,
            color="black",
            s=28,
            zorder=3,
        )

    axis.set_title("Assignment 1: final best fitness across seeds")
    axis.set_ylabel("Final best fitness, lower is better")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(
        handles=[
            Patch(facecolor="lightgray", edgecolor="black",
                  label="Box: middle 50% of seed results"),
            Line2D([0], [0], color="black", linewidth=1.5,
                   label="Line: median seed result"),
            Line2D([0], [0], marker="o", color="black", linestyle="None",
                   markersize=5, label="Dots: individual seeds"),
                 Line2D([0], [0], marker="o", color="black", linestyle="None",
                     markerfacecolor="none", markersize=5,
                     label="Open circles: boxplot outliers"),
        ],
        frameon=False,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    """Generate both result figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path.cwd() / "__data__" / "A1_template_2026" / "experiments",
    )
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output or args.results
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_convergence(
        args.results,
        output_dir / "convergence.png",
        args.population,
        args.generations,
    )
    plot_final_scores(args.results, output_dir / "final_scores.png")
    print(f"Wrote {output_dir / 'convergence.png'}")
    print(f"Wrote {output_dir / 'final_scores.png'}")


if __name__ == "__main__":
    main()
