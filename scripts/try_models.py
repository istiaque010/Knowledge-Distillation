"""Try the models yourself: give a story beginning, and the teacher and the students each finish it.

All models get the same prompt, the same seed and the same decoding (top-p 0.95), so only the model differs.

Usage (from project2, .venv active):
    python scripts/try_models.py                                   # default prompt
    python scripts/try_models.py --prompt "Tom found a red box"    # your own beginning
    python scripts/try_models.py --interactive                     # type prompts until you enter nothing
Options: --seed 1   --max_new_tokens 200   --models teacher scratch kd best
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import kd  # noqa: E402
from kd.model import count_parameters, load_checkpoint  # noqa: E402
from kd.teacher import load_frozen_teacher  # noqa: E402
from src.generate import generate_text  # noqa: E402

CKPT = {
    "scratch": ("Student, no teacher (5.3M)", "student_scratch"),
    "kd": ("Distilled student, T=2 a=0.5 (5.3M)", "student_kd_T2_a0.5"),
    "best": ("Distilled student, T=2 a=0.25 (5.3M, best)", "student_kd_T2_a0.25"),
}


def load_models(names):
    teacher, tok = load_frozen_teacher()
    models = []
    for n in names:
        if n == "teacher":
            models.append((f"Teacher ({count_parameters(teacher)["total"] / 1e6:.1f}M)", teacher))
            continue
        label, run = CKPT[n]
        path = os.path.join(kd.PROJECT2, "checkpoints", run, "model_best.pt")
        if not os.path.exists(path):
            print(f"[skip] {label}: {path} not found (train it first)")
            continue
        models.append((label, load_checkpoint(path)[0]))
    return models, tok


def write(models, tok, prompt, seed, max_new_tokens):
    print("\n" + "=" * 78 + f"\nPROMPT: {prompt}\n" + "=" * 78)
    for label, model in models:
        torch.manual_seed(seed)  # same seed for every model, so only the model differs
        r = generate_text(model, tok, prompt, max_new_tokens, "cpu", method="top_p", top_p=0.95)
        end = "ended by itself" if r["ended_with_eos"] else "stopped at the length limit"
        print(f"\n--- {label}  [{r['new_tokens']} tokens, {r['latency_s']:.2f} s, {end}]\n{r['text']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Once upon a time, there was a little")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--models", nargs="+", default=["teacher", "scratch", "kd"],
                    choices=["teacher", "scratch", "kd", "best"])
    args = ap.parse_args()
    models, tok = load_models(args.models)
    if not args.interactive:
        write(models, tok, args.prompt, args.seed, args.max_new_tokens)
        return
    while True:
        try:
            p = input("\nStory beginning (Enter to quit): ").strip()
        except EOFError:
            break
        if not p:
            break
        write(models, tok, p, args.seed, args.max_new_tokens)


if __name__ == "__main__":
    main()
