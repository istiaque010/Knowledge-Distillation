"""Phase 4: measure training speed and memory for the student, alone and with the teacher, and estimate run time.

A distillation step = teacher forward pass (no gradient) + student forward + backward + AdamW. This script times
real micro-batches (batch 8 x 256 tokens from train.bin) after a few warm-up ones, then estimates the time for
the config's max_steps.

Usage (from project2, .venv active):
    python scripts/benchmark.py --config configs/student.yaml                   # student alone (scratch)
    python scripts/benchmark.py --config configs/student.yaml --with_teacher    # distillation step
    python scripts/benchmark.py --config configs/student.yaml --n_layer 3       # try another depth
    python scripts/benchmark.py --config configs/student.yaml --teacher_only    # teacher forward only
Appends one row per run to results/metrics/benchmark.csv. Run each one in its own process (peak memory is per process).
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import kd  # noqa: E402
from kd.model import build_student, count_parameters  # noqa: E402
from kd.teacher import load_frozen_teacher, teacher_logits  # noqa: E402
from src.dataset import NextTokenDataset, load_tokens  # noqa: E402
from src.train import make_optimizer  # noqa: E402
from src.utils import load_config, peak_memory_mb, set_seed  # noqa: E402


def hms(seconds: float) -> str:
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h}h {m:02d}m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--with_teacher", action="store_true", help="add the teacher forward pass and a KD loss")
    ap.add_argument("--teacher_only", action="store_true", help="time the teacher forward pass alone")
    ap.add_argument("--n_layer", type=int, default=None, help="override model.n_layer")
    ap.add_argument("--steps", type=int, default=30, help="measured micro-batches")
    ap.add_argument("--warmup", type=int, default=5, help="micro-batches ignored at the start")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tc, mc = cfg["train"], dict(cfg["model"])
    if args.n_layer is not None:
        mc["n_layer"] = args.n_layer
    set_seed(cfg["seed"])
    torch.set_num_threads(os.cpu_count() or 1)

    student = build_student(mc).train()
    optimizer = make_optimizer(student, tc)
    teacher = load_frozen_teacher()[0] if (args.with_teacher or args.teacher_only) else None

    ds = NextTokenDataset(load_tokens(os.path.join(kd.PROJECT2, cfg["data"]["data_dir"]), "train"), 256)
    batches = iter(DataLoader(ds, batch_size=tc["batch_size"], shuffle=True, drop_last=True))
    tokens_per_micro = tc["batch_size"] * 256

    for i in range(args.warmup + args.steps):
        if i == args.warmup:
            t0 = time.perf_counter()
        x, y = next(batches)
        if args.teacher_only:
            teacher_logits(teacher, x)
            continue
        z_s, loss = student(x, y)
        if teacher is not None:  # T = 2, alpha = 0.5: same amount of work as the real distillation loss
            z_t = teacher_logits(teacher, x)
            kl = F.kl_div(F.log_softmax(z_s / 2, -1), F.log_softmax(z_t / 2, -1), log_target=True,
                          reduction="batchmean")
            loss = 0.5 * loss + 0.5 * 4 * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), tc["grad_clip"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    elapsed = time.perf_counter() - t0

    tps = args.steps * tokens_per_micro / elapsed
    tokens_per_step = tokens_per_micro * tc["grad_accum_steps"]
    mode = "teacher forward" if args.teacher_only else ("student + teacher (KD)" if teacher else "student alone")
    row = {
        "mode": mode,
        "shape": f"{mc['n_layer']}L/{mc['n_head']}H/{mc['n_embd']}d",
        "student_params": count_parameters(student)["total"],
        "threads": torch.get_num_threads(),
        "batch": tc["batch_size"],
        "tokens_per_s": round(tps),
        "ms_per_micro_batch": round(1000 * elapsed / args.steps, 1),
        "peak_mem_mb": round(peak_memory_mb("cpu")),
        "max_steps": tc["max_steps"],
        "tokens_per_run": tc["max_steps"] * tokens_per_step,
        "est_run_time": hms(tc["max_steps"] * tokens_per_step / tps),
    }
    for k, v in row.items():
        print(f"{k:>20}: {v:,}" if isinstance(v, int) else f"{k:>20}: {v}")

    path = os.path.join(kd.PROJECT2, "results", "metrics", "benchmark.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    main()
