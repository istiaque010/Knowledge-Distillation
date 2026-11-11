"""Phases 10 and 13: evaluate the teacher and the students in exactly the same way.

Quality (Phase 10)
  - test and validation loss / perplexity: full sets, 256-token windows, batch 16, fp32 (Project 1's method)
  - closeness to the teacher on the full test set: KL(teacher || model) at T = 1, and top-1 agreement
    (how often the model's most likely next token is the teacher's most likely next token)
  - 40 stories with top-p 0.95 (8 prompts x 5): Distinct-1/2, Rep-4 repetition flag, EOS rate, made-up words
Error-analysis sheet (Phase 13)
  - 60 stories with top-p 0.90, written to error_analysis.csv for hand labelling

Seeds copy Project 1's src/evaluate.py exactly (top-p 0.95 stories: seed 1343; error stories: seed 1344), so the
teacher reproduces its Project 1 stories and its hand labels from Project 1 can be reused.
The real-story reference values (Distinct, Rep-4 threshold) are Project 1's, read from the teacher package.

Usage (from project2, .venv active):
    python -m kd.evaluate --model teacher
    python -m kd.evaluate --ckpt checkpoints/student_scratch/model_best.pt
Outputs: results/metrics/<run>/eval.json, eval_report.md; results/generations/<run>/samples.jsonl,
         error_analysis.csv
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import kd
from kd.model import count_parameters, load_checkpoint
from kd.teacher import load_frozen_teacher, teacher_logits
from src.dataset import NextTokenDataset, load_tokens
from src.evaluate import auto_flags, grouped_distinct, repeated_ngram_rate, write_error_csv
from src.generate import generate_text
from src.tokenizer import StoryTokenizer
from src.train import evaluate_loss
from src.utils import env_info, peak_memory_mb, set_seed

SEED = 1337
STORY_SETTING = {"name": "top_p_0.95", "method": "top_p", "top_p": 0.95, "seed_offset": 6}  # Project 1: setting 6
ERROR_SETTING = {"name": "top_p_0.90", "method": "top_p", "top_p": 0.90, "seed_offset": 7}  # Project 1: after 7
WORD = re.compile(r"[a-z]+")


def known_words(tok: StoryTokenizer, n_stories: int = 300_000) -> set[str]:
    """Words in the first N training stories (as Project 1's madeup_words.py), decoded from train.bin."""
    cache = os.path.join(kd.PROJECT2, "results", "metrics", f"known_words_{n_stories}.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return set(json.load(f))
    tokens = load_tokens(kd.DATA_DIR, "train")
    eos = np.flatnonzero(np.asarray(tokens[: 90_000_000]) == tok.eos_id)  # 300K stories fit in ~62M tokens
    end = int(eos[n_stories - 1])
    ids = np.asarray(tokens[: end + 1]).astype(np.int64)
    known: set[str] = set()
    for story in np.split(ids, eos[:n_stories] + 1)[:n_stories]:
        known.update(WORD.findall(tok.decode(story.tolist()).lower()))
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(sorted(known), f)
    return known


def madeup_rate(texts: list[str], known: set[str]) -> tuple[float, list[str]]:
    words = [w for t in texts for w in WORD.findall(t.lower())]
    bad = [w for w in words if w not in known]
    return len(bad) / max(1, len(words)), sorted(set(bad))[:8]


@torch.no_grad()
def closeness_to_teacher(model, teacher, loader) -> dict:
    """KL(teacher || model) at T = 1 per token, and how often both pick the same most likely token."""
    model.eval()
    kl_sum, agree, n_tok = 0.0, 0, 0
    for x, _ in loader:
        z_m, _ = model(x)
        z_t = teacher_logits(teacher, x)
        log_pm, log_pt = torch.log_softmax(z_m, -1), torch.log_softmax(z_t, -1)
        kl_sum += (log_pt.exp() * (log_pt - log_pm)).sum().item()
        agree += (z_m.argmax(-1) == z_t.argmax(-1)).sum().item()
        n_tok += x.numel()
    return {"kl_teacher": kl_sum / n_tok, "top1_agreement": agree / n_tok, "tokens": n_tok}


def stories(model, tok, prompts, setting, n_groups, max_new_tokens) -> list[list[dict]]:
    kw = {k: v for k, v in setting.items() if k in ("method", "top_p")}
    torch.manual_seed(SEED + setting["seed_offset"])
    return [[{**generate_text(model, tok, p, max_new_tokens, "cpu", **kw), "setting": setting["name"]}
             for p in prompts] for _ in range(n_groups)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["teacher"], default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--samples_per_prompt", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--n_error", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()
    if (args.model is None) == (args.ckpt is None):
        ap.error("give exactly one of --model teacher or --ckpt")

    set_seed(SEED)
    torch.set_num_threads(os.cpu_count() or 1)
    teacher, tok = load_frozen_teacher()
    if args.model == "teacher":
        model, run, step = teacher, "teacher", 10_000
    else:
        model, ckpt = load_checkpoint(args.ckpt)
        run, step = ckpt["run"]["run_name"], ckpt["step"]
    metrics_dir = os.path.join(kd.PROJECT2, "results", "metrics", run)
    gen_dir = os.path.join(kd.PROJECT2, "results", "generations", run)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    with open(os.path.join(kd.TEACHER_DIR, "results", "eval.json"), encoding="utf-8") as f:
        reference = json.load(f)["reference_real_stories"]

    # --- perplexity: full validation and test sets, Project 1's method ---
    result = {"run": run, "checkpoint": args.ckpt or "teacher.pt", "step": step}
    for split in ("val", "test"):
        loader = DataLoader(NextTokenDataset(load_tokens(kd.DATA_DIR, split), 256), batch_size=args.batch_size)
        t0 = time.time()
        loss = evaluate_loss(model, loader, "cpu", "fp32")
        result[f"{split}_loss"], result[f"{split}_perplexity"] = loss, math.exp(loss)
        print(f"{split:>4} loss {loss:.4f} | perplexity {math.exp(loss):.2f} ({time.time() - t0:.0f}s)")
        if split == "test" and model is not teacher:
            result["closeness"] = closeness_to_teacher(model, teacher, loader)
            print(f"     KL(teacher||model) {result['closeness']['kl_teacher']:.4f} | "
                  f"top-1 agreement {result['closeness']['top1_agreement']:.1%}")

    # --- stories, top-p 0.95 ---
    prompts = [p.strip() for p in open(os.path.join(kd.TEACHER_DIR, "configs", "prompts.txt"), encoding="utf-8")
               if p.strip()]
    groups = stories(model, tok, prompts, STORY_SETTING, args.samples_per_prompt, args.max_new_tokens)
    rows = [r for g in groups for r in g]
    flags = [auto_flags(r, reference["rep4_flag_threshold"]) for r in rows]
    known = known_words(tok)
    rate, examples = madeup_rate([r["completion"] for r in rows], known)
    d1, d1_std = grouped_distinct([[r["completion"] for r in g] for g in groups], 1)
    d2, d2_std = grouped_distinct([[r["completion"] for r in g] for g in groups], 2)
    result["generation"] = {
        "setting": STORY_SETTING["name"], "n": len(rows), "distinct_1": d1, "distinct_1_std": d1_std,
        "distinct_2": d2, "distinct_2_std": d2_std,
        "rep4_rate": statistics.mean(repeated_ngram_rate(r["completion"]) for r in rows),
        "repetition_flag_rate": sum(f["repetition"] for f in flags) / len(rows),
        "eos_rate": sum(r["ended_with_eos"] for r in rows) / len(rows),
        "hit_token_limit_rate": sum(f["hit_token_limit"] for f in flags) / len(rows),
        "avg_new_tokens": statistics.mean(r["new_tokens"] for r in rows),
        "madeup_rate": rate, "madeup_examples": examples, "known_words": len(known),
    }
    g = result["generation"]
    print(f"stories ({g['setting']}, {g['n']}) | D1 {d1:.3f} D2 {d2:.3f} | flagged {g['repetition_flag_rate']:.0%} "
          f"| eos {g['eos_rate']:.0%} | made-up words {rate:.2%} {examples}")
    with open(os.path.join(gen_dir, "samples.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- error-analysis sheet, top-p 0.90 ---
    err = stories(model, tok, [prompts[i % len(prompts)] for i in range(args.n_error)], ERROR_SETTING, 1,
                  args.max_new_tokens)[0]
    err_rows = [{"id": i, "prompt": r["prompt"], "text": r["text"], **auto_flags(r, reference["rep4_flag_threshold"])}
                for i, r in enumerate(err)]
    err_path = write_error_csv(os.path.join(gen_dir, "error_analysis.csv"), err_rows)
    result["error_analysis"] = {"setting": ERROR_SETTING["name"], "n": len(err_rows), "csv": os.path.relpath(err_path, kd.PROJECT2),
                                "auto_flag_counts": {k: sum(r[k] for r in err_rows)
                                                     for k in ("repetition", "incomplete", "hit_token_limit")}}

    params = count_parameters(model)
    result.update({"params": params, "model_size_mb_fp32": params["total"] * 4 / 2**20,
                   "reference_real_stories": reference, "peak_mem_mb": peak_memory_mb("cpu"), "env": env_info("cpu")})
    with open(os.path.join(metrics_dir, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    write_report(os.path.join(metrics_dir, "eval_report.md"), result, rows)
    print(f"wrote results/metrics/{run}/eval.json, eval_report.md and results/generations/{run}/")


def write_report(path: str, r: dict, rows: list[dict]) -> None:
    g, ref = r["generation"], r["reference_real_stories"]
    L = [f"# Evaluation — {r['run']}", "",
         f"- Checkpoint: `{r['checkpoint']}` (step {r['step']:,})",
         f"- Parameters: {r['params']['total']:,} ({r['params']['non_embedding']:,} non-embedding), "
         f"{r['model_size_mb_fp32']:.1f} MB fp32",
         f"- **Test** loss **{r['test_loss']:.4f}**, perplexity **{r['test_perplexity']:.2f}** "
         f"(validation {r['val_loss']:.4f} / {r['val_perplexity']:.2f}); full sets, 256-token windows, batch 16, fp32"]
    if "closeness" in r:
        c = r["closeness"]
        L.append(f"- Closeness to the teacher (test set, {c['tokens']:,} tokens): KL(teacher ‖ model) "
                 f"**{c['kl_teacher']:.4f}**, same top-1 token as the teacher **{c['top1_agreement']:.1%}**")
    L += ["", f"## Stories ({g['setting']}, {g['n']}, up to 512 new tokens)", "",
          "| | Distinct-1 | Distinct-2 | Rep-4 rate | Flagged repetitive | EOS rate | Hit token limit | Avg tokens | Made-up words |",
          "|---|---|---|---|---|---|---|---|---|",
          f"| real stories | {ref['distinct_1']:.3f} | {ref['distinct_2']:.3f} | {ref['rep4_rate']:.3f} | 5% by definition | — | — | — | 0.02% |",
          f"| {r['run']} | {g['distinct_1']:.3f} ± {g['distinct_1_std']:.3f} | {g['distinct_2']:.3f} ± {g['distinct_2_std']:.3f} | "
          f"{g['rep4_rate']:.3f} | {g['repetition_flag_rate']:.0%} | {g['eos_rate']:.0%} | {g['hit_token_limit_rate']:.0%} | "
          f"{g['avg_new_tokens']:.0f} | {g['madeup_rate']:.2%} |", "",
          f"Made-up word examples: {', '.join(g['madeup_examples']) or 'none'}", "",
          f"## Error-analysis sheet", "",
          f"{r['error_analysis']['n']} stories ({r['error_analysis']['setting']}) in `{r['error_analysis']['csv']}`. "
          f"Auto flags: " + ", ".join(f"{k} {v}" for k, v in r["error_analysis"]["auto_flag_counts"].items()), "",
          "## First story for each prompt", ""]
    for row in rows[:8]:
        L += [f"> {row['text']}", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
