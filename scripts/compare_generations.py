"""Phase 12: the three models' stories side by side, for the same prompts.

All stories come from kd/evaluate.py with the same decoding (top-p 0.95), the same length limit (512 new
tokens) and the same seed (1343), so only the model differs. The first --num_samples stories of each prompt
are shown for every model.

Usage (from project2, .venv active), after kd/evaluate.py has run for each model:
    python scripts/compare_generations.py --run teacher --run student_scratch --run student_kd_T2_a0.5 --num_samples 2
Writes results/generations/comparison.md
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kd  # noqa: E402

NAMES = {"teacher": "Teacher (13.9M)", "student_scratch": "Student from scratch (5.3M)"}


def name(run: str) -> str:
    return NAMES.get(run, f"Distilled student (5.3M, {run.replace('student_kd_', '')})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--num_samples", type=int, default=2)
    args = ap.parse_args()

    by_run = {}
    for run in args.run:
        with open(os.path.join(kd.PROJECT2, "results", "generations", run, "samples.jsonl"), encoding="utf-8") as f:
            stories = collections.defaultdict(list)
            for line in f:
                r = json.loads(line)
                stories[r["prompt"]].append(r)
        by_run[run] = stories

    prompts = list(by_run[args.run[0]])
    L = ["# Stories side by side", "",
         "Same prompts, same decoding (top-p 0.95), same seed (1343), up to 512 new tokens. Only the model differs.",
         "Prompt text is in **bold**. ⏹ = the model ended the story itself; ✂ = stopped at the 512-token limit.", ""]
    for p in prompts:
        L += [f"## {p}", ""]
        for k in range(args.num_samples):
            if args.num_samples > 1:
                L += [f"### Sample {k + 1}", ""]
            for run in args.run:
                r = by_run[run][p][k]
                end = "⏹" if r["ended_with_eos"] else "✂"
                L += [f"**{name(run)}** ({r['new_tokens']} tokens {end})", "",
                      f"> **{r['prompt']}**{r['completion']}", ""]
    out = os.path.join(kd.PROJECT2, "results", "generations", "comparison.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"wrote results/generations/comparison.md ({len(prompts)} prompts x {args.num_samples} x {len(args.run)} models)")


if __name__ == "__main__":
    main()
