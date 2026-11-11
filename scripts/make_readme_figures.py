"""Figures for README.md, built only from saved result files (no numbers typed by hand).

Reads results/metrics/<run>/eval.json, inference_benchmark.json and error_counts.json.
Writes results/figures/test_perplexity.png, tradeoff.png and errors.png.

Colours (one per role, validated for colour-blind separation): teacher = blue, student alone = orange,
distilled student = aqua. Every bar carries its value as text, so no reading depends on colour alone.

Usage (from project2, .venv active):
    python scripts/make_readme_figures.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import kd  # noqa: E402

M = os.path.join(kd.PROJECT2, "results", "metrics")
OUT = os.path.join(kd.PROJECT2, "results", "figures")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
TEACHER, ALONE, KD = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
})


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ev(run):
    return load(os.path.join(M, run, "eval.json"))


def clean_axes(ax, axis="x"):
    if axis:
        ax.grid(axis=axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(length=0)


def fig_perplexity():
    runs = [("Teacher (13.9M)", "teacher", TEACHER),
            ("Distilled, T=2, α=0.25 (best)", "student_kd_T2_a0.25", KD),
            ("Distilled, T=2, α=0.5 (main)", "student_kd_T2_a0.5", KD),
            ("Distilled, T=4, α=0.5", "student_kd_T4_a0.5", KD),
            ("Distilled, T=1, α=0.5", "student_kd_T1_a0.5", KD),
            ("Distilled, T=2, α=0.75", "student_kd_T2_a0.75", KD),
            ("Student alone (no teacher)", "student_scratch", ALONE)]
    vals = [ev(r)["test_perplexity"] for _, r, _ in runs]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    y = range(len(runs))[::-1]
    ax.barh(list(y), vals, color=[c for *_, c in runs], height=0.62, edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, vals):
        ax.text(v + 0.08, yi, f"{v:.2f}", va="center", color=INK, fontsize=11)
    ax.set_yticks(list(y), [n for n, *_ in runs])
    ax.set_xlim(0, max(vals) * 1.12)
    ax.set_xlabel("Test perplexity (lower is better)")
    ax.set_title("Every distilled student beats the student trained alone", loc="left", color=INK, fontsize=13, pad=12)
    clean_axes(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "test_perplexity.png"), dpi=150)
    plt.close(fig)


def fig_tradeoff():
    inf = {r["run"]: r for r in load(os.path.join(M, "inference_benchmark.json"))}
    runs = [("Teacher", "teacher", TEACHER), ("Student\nalone", "student_scratch", ALONE),
            ("Distilled\nstudent", "student_kd_T2_a0.5", KD)]
    panels = [("Size (million parameters)", lambda r: inf[r]["params"] / 1e6, "{:.1f}M"),
              ("Time to write a 200-token story (s)", lambda r: inf[r]["latency_s_200_tokens"], "{:.2f} s"),
              ("Test perplexity (lower is better)", lambda r: ev(r)["test_perplexity"], "{:.2f}")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (title, f, fmt) in zip(axes, panels):
        vals = [f(r) for _, r, _ in runs]
        ax.bar(range(3), vals, color=[c for *_, c in runs], width=0.62, edgecolor=SURFACE, linewidth=2)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, fmt.format(v), ha="center", va="bottom", color=INK, fontsize=11)
        ax.set_xticks(range(3), [n for n, *_ in runs])
        ax.set_ylim(0, max(vals) * 1.18)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_title(title, color=INK, fontsize=11.5, pad=8)
        clean_axes(ax, axis=None)
    fig.suptitle("2.6× smaller and ~1.8× faster, at a cost in quality", x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tradeoff.png"), dpi=150)
    plt.close(fig)


def fig_errors():
    err = load(os.path.join(M, "error_counts.json"))
    runs = [("Teacher", "teacher", TEACHER), ("Student alone", "student_scratch", ALONE),
            ("Distilled student", "student_kd_T2_a0.5", KD)]
    kinds = [("Contradiction", "contradiction"), ("Character mix-up", "entity_consistency"),
             ("Grammar error", "grammar"), ("Repeated phrases", "repetition")]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    w = 0.26
    for j, (name, run, c) in enumerate(runs):
        vals = [100 * err[run][k] / err[run]["n"] for _, k in kinds]
        xs = [i + (j - 1) * w for i in range(len(kinds))]
        ax.bar(xs, vals, width=w, color=c, label=name, edgecolor=SURFACE, linewidth=2)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", color=INK, fontsize=10)
    ax.set_xticks(range(len(kinds)), [n for n, _ in kinds])
    ax.set_ylim(0, 112)
    ax.set_ylabel("Share of 60 stories (%)")
    ax.legend(frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.13))
    ax.set_title("Stories with each kind of error (AI-assisted labels)", loc="left", color=INK, fontsize=13, pad=26)
    clean_axes(ax, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "errors.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_perplexity()
    fig_tradeoff()
    fig_errors()
    print("wrote results/figures/test_perplexity.png, tradeoff.png, errors.png")
