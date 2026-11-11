"""Phases 5, 8, 9: training curves and a summary table for several runs.

Left: validation perplexity. Right: KL(teacher || student) on the validation windows (how far the student's
guesses are from the teacher's). Dashed line on the left: the teacher's validation perplexity (5.74).

Usage (from project2, .venv active):
    python scripts/plot_curves.py --run student_scratch --run student_kd_T2_a0.5 [--out results/figures/curves.png]
Prints a table (best validation perplexity, final KL, time) and writes it to results/metrics/ablation_table.md.
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import kd  # noqa: E402

TEACHER_VAL_PPL = 5.74


def read_run(run: str):
    d = os.path.join(kd.PROJECT2, "results", "metrics", run)
    with open(os.path.join(d, "log.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    summary = None
    if os.path.exists(os.path.join(d, "train_summary.json")):
        with open(os.path.join(d, "train_summary.json"), encoding="utf-8") as f:
            summary = json.load(f)
    return rows, summary


def label(run: str) -> str:
    if run == "student_scratch":
        return "scratch (no teacher)"
    return run.replace("student_kd_", "KD ").replace("_a", ", α=").replace("T", "T=")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--out", default="results/figures/curves.png")
    args = ap.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    table = ["| Run | Best val perplexity | Final val KL to teacher | Final train CE | Time | Avg tokens/s |",
             "|---|---|---|---|---|---|"]
    for run in args.run:
        rows, summary = read_run(run)
        ev = [r for r in rows if r["val_loss"]]
        steps = [int(r["step"]) for r in ev]
        ax1.plot(steps, [float(r["val_ppl"]) for r in ev], marker="o", ms=3, label=label(run))
        ax2.plot(steps, [float(r["val_kl_teacher"]) for r in ev], marker="o", ms=3, label=label(run))
        best = min(float(r["val_ppl"]) for r in ev)
        last_ce = sum(float(r["train_ce"]) for r in rows[-4:]) / len(rows[-4:])
        tps = sum(float(r["tokens_per_sec"]) for r in rows) / len(rows)
        hours = f"{summary['wall_clock_s'] / 3600:.2f} h" if summary else "running"
        table.append(f"| {label(run)} | {best:.2f} | {float(ev[-1]['val_kl_teacher']):.4f} | {last_ce:.3f} | "
                     f"{hours} | {tps:,.0f} |")
    ax1.axhline(TEACHER_VAL_PPL, color="gray", ls="--", lw=1, label=f"teacher ({TEACHER_VAL_PPL})")
    ax1.set(xlabel="step", ylabel="validation perplexity", title="Validation perplexity (lower is better)",
            yscale="log")
    ax2.set(xlabel="step", ylabel="KL(teacher ‖ student), T = 1", title="Distance to the teacher (lower = closer)",
            yscale="log")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(kd.PROJECT2, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130)
    print("\n".join(table))
    with open(os.path.join(kd.PROJECT2, "results", "metrics", "ablation_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(table) + "\n")
    print(f"\nwrote {args.out} and results/metrics/ablation_table.md")


if __name__ == "__main__":
    main()
