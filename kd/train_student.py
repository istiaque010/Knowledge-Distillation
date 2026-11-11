"""Phases 5, 8 and 9: train the student, from scratch or by distillation.

Both modes use the SAME config (configs/student.yaml): same data, data order, steps, batch, learning-rate schedule
and seed. The only difference is the loss:
    --mode scratch   L = cross-entropy                                       (Phase 5 baseline)
    --mode kd        L = alpha * CE + (1 - alpha) * T^2 * KL(teacher || student)   (Phases 8 and 9)

One distillation step:
    batch -> teacher forward (frozen, no gradient) -> teacher logits --+
          -> student forward                       -> student logits --+-> CE + KD loss
          -> backpropagate the STUDENT only -> AdamW update

Usage (from project2, .venv active):
    python -m kd.train_student --config configs/student.yaml --mode scratch
    python -m kd.train_student --config configs/student.yaml --mode kd --temperature 2 --alpha 0.5
    python -m kd.train_student --config configs/smoke.yaml --mode kd          # a few steps, to test
The run name is made from the settings (student_scratch, student_kd_T2_a0.5, ...) unless --run_name is given.

Every validation check also measures how close the student is to the teacher: KL(teacher || student) at T = 1 on
the same validation windows, for both modes, so the baseline and the distilled student can be compared on it.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader

import kd
from kd.distillation import distillation_loss, kd_kl
from kd.model import build_student, count_parameters
from kd.teacher import load_frozen_teacher, teacher_logits
from src.dataset import NextTokenDataset, load_tokens
from src.train import infinite, lr_at, make_optimizer
from src.utils import env_info, load_config, memory_mb, peak_memory_mb, resolve_device, set_seed


def fmt(v: float) -> str:
    return f"{v:g}"


@torch.no_grad()
def evaluate(student, teacher, loader, device: str, max_batches: int | None) -> dict:
    """Validation CE (-> perplexity) and KL(teacher || student) at T = 1, averaged per window."""
    student.eval()
    ce_sum = kl_sum = 0.0
    n = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        z_s, ce = student(x, y)
        ce_sum += ce.item() * x.size(0)
        kl_sum += kd_kl(z_s, teacher_logits(teacher, x), 1.0).item() * x.size(0)
        n += x.size(0)
    student.train()
    return {"ce": ce_sum / n, "kl_teacher": kl_sum / n}


def save_checkpoint(path, model, optimizer, step, best_val, cfg, run_cfg) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "best_val_loss": best_val, "model_config": model.cfg.to_dict(), "config": cfg, "run": run_cfg,
                "tokenizer": cfg["data"]["tokenizer"]}, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["scratch", "kd"], required=True)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--max_steps", type=int, default=None, help="override train.max_steps")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tc, dc = cfg["train"], cfg["data"]
    if args.max_steps is not None:
        tc["max_steps"] = args.max_steps
    T, alpha = (args.temperature, args.alpha) if args.mode == "kd" else (None, 1.0)
    default_name = "student_scratch" if args.mode == "scratch" else f"student_kd_T{fmt(T)}_a{fmt(alpha)}"
    run = args.run_name or (default_name if cfg["run_name"] == "student" else f"{cfg['run_name']}_{args.mode}")
    run_cfg = {"run_name": run, "mode": args.mode, "temperature": T, "alpha": alpha}

    set_seed(cfg["seed"])
    device = resolve_device(tc["device"])
    torch.set_num_threads(os.cpu_count() or 1)
    ckpt_dir = os.path.join(kd.PROJECT2, "checkpoints", run)
    metrics_dir = os.path.join(kd.PROJECT2, "results", "metrics", run)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # The teacher is loaded in both modes: in kd mode it gives the soft targets; in both modes it is used to
    # measure how close the student gets to it on the validation set.
    teacher, _ = load_frozen_teacher(device)
    student = build_student(cfg["model"]).to(device).train()
    params = count_parameters(student)
    print(f"run={run} mode={args.mode} T={T} alpha={alpha} device={device} "
          f"student params={params['total']:,} teacher params={count_parameters(teacher)['total']:,}")

    data_dir = os.path.join(kd.PROJECT2, dc["data_dir"])
    train_ds = NextTokenDataset(load_tokens(data_dir, "train"), 256)
    val_ds = NextTokenDataset(load_tokens(data_dir, "val"), 256)
    g = torch.Generator()
    train_loader = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True, drop_last=True,
                              num_workers=tc["num_workers"], generator=g)
    val_loader = DataLoader(val_ds, batch_size=tc["batch_size"], shuffle=False, num_workers=tc["num_workers"])
    print(f"train windows={len(train_ds):,} val windows={len(val_ds):,}")

    optimizer = make_optimizer(student, tc)
    start_step, best_val = 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step, best_val = ckpt["step"], ckpt["best_val_loss"]
        print(f"resumed from {args.resume} at step {start_step}")

    tokens_per_step = tc["batch_size"] * tc["grad_accum_steps"] * 256
    with open(os.path.join(metrics_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump({"run": run_cfg, "config": cfg, "model_config": student.cfg.to_dict(), "params": params,
                   "env": env_info(device), "effective_batch_seqs": tc["batch_size"] * tc["grad_accum_steps"],
                   "effective_batch_tokens": tokens_per_step}, f, indent=2)

    log_path = os.path.join(metrics_dir, "log.csv")
    new_log = not (args.resume and os.path.exists(log_path))
    log_f = open(log_path, "w" if new_log else "a", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["step", "train_loss", "train_ce", "train_kd", "val_loss", "val_ppl", "val_kl_teacher", "lr",
                      "tokens_per_sec", "mem_mb", "elapsed_s"])

    batches = infinite(train_loader, cfg["seed"] + start_step)
    t0 = t_last = time.time()
    run_total = run_ce = run_kd = 0.0
    since_log = 0
    for step in range(start_step, tc["max_steps"]):
        lr = lr_at(step, tc)
        for group in optimizer.param_groups:
            group["lr"] = lr

        for _ in range(tc["grad_accum_steps"]):
            x, y = next(batches)
            x, y = x.to(device), y.to(device)
            z_t = teacher_logits(teacher, x) if args.mode == "kd" else None
            z_s, _ = student(x)
            total, ce, kd_loss = distillation_loss(z_s, z_t, y, T or 1.0, alpha)
            (total / tc["grad_accum_steps"]).backward()
            run_total += total.item() / tc["grad_accum_steps"]
            run_ce += ce.item() / tc["grad_accum_steps"]
            run_kd += kd_loss.item() / tc["grad_accum_steps"]

        torch.nn.utils.clip_grad_norm_(student.parameters(), tc["grad_clip"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        since_log += 1

        done = step + 1
        is_eval = done % tc["eval_interval"] == 0 or done == tc["max_steps"]
        if done % tc["log_interval"] == 0 or is_eval:
            now = time.time()
            tps = tokens_per_step * since_log / max(1e-9, now - t_last)
            row = {"total": run_total / since_log, "ce": run_ce / since_log, "kd": run_kd / since_log}
            run_total = run_ce = run_kd = 0.0
            since_log = 0
            val = None
            if is_eval:
                val = evaluate(student, teacher, val_loader, device, tc["eval_batches"])
                if val["ce"] < best_val:
                    best_val = val["ce"]
                    save_checkpoint(os.path.join(ckpt_dir, "model_best.pt"), student, optimizer, done, best_val,
                                    cfg, run_cfg)
            mem = memory_mb(device)
            log.writerow([done, f"{row['total']:.4f}", f"{row['ce']:.4f}",
                          f"{row['kd']:.4f}" if args.mode == "kd" else "",
                          f"{val['ce']:.4f}" if val else "", f"{math.exp(val['ce']):.2f}" if val else "",
                          f"{val['kl_teacher']:.4f}" if val else "", f"{lr:.2e}", f"{tps:.0f}", f"{mem:.0f}",
                          f"{now - t0:.1f}"])
            log_f.flush()
            msg = (f"step {done:5d} | loss {row['total']:.4f} | ce {row['ce']:.4f}"
                   + (f" | kd {row['kd']:.4f}" if args.mode == "kd" else "")
                   + f" | lr {lr:.2e} | {tps:,.0f} tok/s | mem {mem:,.0f} MB")
            if val:
                msg += f" | val {val['ce']:.4f} ppl {math.exp(val['ce']):.2f} kl(teacher) {val['kl_teacher']:.4f}"
            print(msg, flush=True)
            t_last = time.time()  # don't count evaluation time in the throughput

        if done % tc["ckpt_interval"] == 0:
            save_checkpoint(os.path.join(ckpt_dir, f"model_step_{done}.pt"), student, optimizer, done, best_val,
                            cfg, run_cfg)
        if is_eval:
            save_checkpoint(os.path.join(ckpt_dir, "model_last.pt"), student, optimizer, done, best_val, cfg, run_cfg)

    log_f.close()
    elapsed = time.time() - t0
    summary = {"run": run_cfg, "best_val_loss": best_val, "best_val_ppl": math.exp(best_val),
               "steps": tc["max_steps"], "tokens": tc["max_steps"] * tokens_per_step,
               "wall_clock_s": round(elapsed, 1), "peak_mem_mb": round(peak_memory_mb(device), 1)}
    with open(os.path.join(metrics_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"done in {elapsed / 60:.1f} min | best val loss {best_val:.4f} (ppl {math.exp(best_val):.2f})")


if __name__ == "__main__":
    main()
