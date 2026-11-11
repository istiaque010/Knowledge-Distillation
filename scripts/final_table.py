"""Phase 14: the final comparison table, built only from saved results (no numbers copied by hand).

Reads results/metrics/<run>/eval.json, train_summary.json, results/metrics/inference_benchmark.json and
error_counts.json. Writes results/final_comparison.md and final_comparison.json.

Usage (from project2, .venv active):
    python scripts/final_table.py --distilled student_kd_T2_a0.5 [--ablation student_kd_T1_a0.5 ...]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kd  # noqa: E402

M = os.path.join(kd.PROJECT2, "results", "metrics")


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distilled", required=True, help="the main distilled run")
    ap.add_argument("--ablation", action="append", default=[], help="other distilled runs, for the ablation table")
    args = ap.parse_args()
    runs = ["teacher", "student_scratch", args.distilled]
    ev = {r: load(os.path.join(M, r, "eval.json")) for r in runs + args.ablation}
    tr = {r: load(os.path.join(M, r, "train_summary.json")) for r in runs + args.ablation}
    tr["teacher"] = load(os.path.join(kd.TEACHER_DIR, "results", "train_summary.json"))  # Project 1 (read only)
    if tr["teacher"]:
        tr["teacher"]["tokens"] = tr["teacher"]["steps"] * 64 * 256  # teacher: 64 x 256 tokens per step too
    inf = {r["run"]: r for r in (load(os.path.join(M, "inference_benchmark.json")) or [])}
    err = load(os.path.join(M, "error_counts.json")) or {}

    def cell(run, f, default="—"):
        try:
            return f(run)
        except (KeyError, TypeError):
            return default

    rows = [
        ("Parameters", lambda r: f"{ev[r]['params']['total'] / 1e6:.2f}M"),
        ("Compression vs teacher", lambda r: f"{inf[r]['compression_ratio']:.2f}×"),
        ("Weights on disk (fp32)", lambda r: f"{inf[r]['weights_mb']:.1f} MB"),
        ("**Test perplexity**", lambda r: f"**{ev[r]['test_perplexity']:.2f}**"),
        ("Test loss", lambda r: f"{ev[r]['test_loss']:.4f}"),
        ("KL(teacher ‖ model), test", lambda r: "0" if r == "teacher" else f"{ev[r]['closeness']['kl_teacher']:.4f}"),
        ("Same top-1 token as teacher", lambda r: "100%" if r == "teacher" else f"{ev[r]['closeness']['top1_agreement']:.1%}"),
        ("Distinct-2, top-p 0.95 (real 0.764)", lambda r: f"{ev[r]['generation']['distinct_2']:.3f}"),
        ("Flagged repetitive (real 5%)", lambda r: f"{ev[r]['generation']['repetition_flag_rate']:.1%}"),
        ("Stories that end by themselves", lambda r: f"{ev[r]['generation']['eos_rate']:.1%}"),
        ("Made-up words (real 0.02%)", lambda r: f"{ev[r]['generation']['madeup_rate']:.2%}"),
        ("Contradiction (hand-labelled)", lambda r: f"{err[r]['contradiction'] / err[r]['n']:.0%}"),
        ("Entity consistency errors", lambda r: f"{err[r]['entity_consistency'] / err[r]['n']:.0%}"),
        ("Grammar errors", lambda r: f"{err[r]['grammar'] / err[r]['n']:.0%}"),
        ("Stories with no error", lambda r: f"{err[r]['no_error']}/{err[r]['n']}"),
        ("Peak RAM, generating", lambda r: f"{inf[r]['peak_ram_mb']:.0f} MB"),
        ("Latency, 200-token story", lambda r: f"{inf[r]['latency_s_200_tokens']:.2f} s"),
        ("Generation tokens/s", lambda r: f"{inf[r]['gen_tokens_per_s']:.0f}"),
        ("Speedup (generation)", lambda r: f"{inf[r]['speedup_generation']:.2f}×"),
        ("Scoring tokens/s (batch 16)", lambda r: f"{inf[r]['score_tokens_per_s']:,.0f}"),
        ("Training (CPU)", lambda r: f"{tr[r]['steps']:,} steps, {tr[r]['tokens'] / 1e6:.0f}M tokens, "
                                     f"{tr[r]['wall_clock_s'] / 3600:.1f} h"),
    ]
    head = ["Teacher", "Student from scratch", f"Distilled student ({args.distilled.replace('student_kd_', '')})"]
    L = ["# Final comparison", "", "| Metric | " + " | ".join(head) + " |", "|---|---|---|---|"]
    L += [f"| {name} | " + " | ".join(cell(r, f) for r in runs) + " |" for name, f in rows]

    p_t, p_s, p_d = (ev[r]["test_perplexity"] for r in runs)
    gap = (p_s - p_d) / (p_s - p_t)
    summary = {"teacher_ppl": p_t, "scratch_ppl": p_s, "distilled_ppl": p_d, "gap_closed": gap,
               "ppl_increase_scratch_pct": 100 * (p_s / p_t - 1), "ppl_increase_distilled_pct": 100 * (p_d / p_t - 1)}
    L += ["", f"- Perplexity vs teacher: scratch **+{summary['ppl_increase_scratch_pct']:.1f}%**, distilled "
              f"**+{summary['ppl_increase_distilled_pct']:.1f}%**.",
          f"- Gap closed by distillation = (PPL_scratch − PPL_distilled) / (PPL_scratch − PPL_teacher) = "
          f"({p_s:.2f} − {p_d:.2f}) / ({p_s:.2f} − {p_t:.2f}) = **{gap:.1%}**."]
    if args.ablation:
        L += ["", "## All students (same 3,000-step budget)", "",
              "| Run | Test perplexity | KL(teacher ‖ model) | Same top-1 as teacher | Distinct-2 | Training time |",
              "|---|---|---|---|---|---|"]
        for r in ["student_scratch", args.distilled] + args.ablation:
            if ev.get(r):
                L.append(f"| {r} | {ev[r]['test_perplexity']:.2f} | {ev[r]['closeness']['kl_teacher']:.4f} | "
                         f"{ev[r]['closeness']['top1_agreement']:.1%} | {ev[r]['generation']['distinct_2']:.3f} | "
                         f"{tr[r]['wall_clock_s'] / 3600:.1f} h |")
    with open(os.path.join(kd.PROJECT2, "results", "final_comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(kd.PROJECT2, "results", "final_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n".join(L))


if __name__ == "__main__":
    main()
