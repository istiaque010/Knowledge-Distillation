"""Phase 11: compression and inference cost of the teacher and the students.

For each model, in its own process (so peak memory belongs to that model alone):
  - parameters, and the size of its weights on disk (fp32 state_dict, as torch.save writes it)
  - generation: a fixed 200-token story with top-p 0.95 (no early stop), 5 runs -> latency and tokens/s
  - scoring throughput: forward passes over 16 x 256-token test windows (how fast it can score text, as in
    perplexity evaluation or as a teacher in distillation)
  - peak RAM of the process, and the RAM added by loading the model
  - other programs' CPU load before and during the timing (Project 1's check): above 15% = not reliable
Then compression ratio = teacher params / student params and speedup = teacher latency / student latency.

Usage (from project2, .venv active, on an otherwise idle PC):
    python scripts/benchmark_inference.py --model teacher --ckpt checkpoints/student_scratch/model_best.pt \
        --ckpt checkpoints/student_kd_T2_a0.5/model_best.pt
Writes results/metrics/inference_benchmark.json and inference_benchmark.md
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil  # noqa: E402
import torch  # noqa: E402

import kd  # noqa: E402

GEN_TOKENS, GEN_RUNS, SCORE_BATCHES, CPU_BUSY_LIMIT = 200, 5, 20, 15.0


def measure_one(which: str) -> dict:
    from torch.utils.data import DataLoader

    from kd.model import count_parameters, load_checkpoint
    from kd.teacher import load_frozen_teacher
    from src.dataset import NextTokenDataset, load_tokens
    from src.evaluate import OtherCpuLoad
    from src.generate import generate
    from src.tokenizer import StoryTokenizer
    from src.utils import peak_memory_mb

    torch.set_num_threads(os.cpu_count() or 1)
    proc = psutil.Process()
    rss_before = proc.memory_info().rss / 2**20
    if which == "teacher":
        model, _ = load_frozen_teacher()
        run = "teacher"
    else:
        model, ckpt = load_checkpoint(which)
        run = ckpt["run"]["run_name"]
    model.eval()
    rss_loaded = proc.memory_info().rss / 2**20
    tok = StoryTokenizer(kd.TOKENIZER)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)

    ids = torch.tensor([tok.encode("Once upon a time", add_bos=True)])
    windows = DataLoader(NextTokenDataset(load_tokens(kd.DATA_DIR, "test"), 256), batch_size=16)
    batches = [x for _, (x, _) in zip(range(SCORE_BATCHES), windows)]
    with torch.no_grad():
        generate(model, ids, GEN_TOKENS, method="top_p", top_p=0.95)  # warm-up, not timed
        model(batches[0])
        load = OtherCpuLoad()
        load.start()
        time.sleep(1.0)
        other_before = load.stop()
        load.start()
        torch.manual_seed(1337)
        t0 = time.perf_counter()
        for _ in range(GEN_RUNS):
            generate(model, ids, GEN_TOKENS, method="top_p", top_p=0.95)
        gen_s = (time.perf_counter() - t0) / GEN_RUNS
        t0 = time.perf_counter()
        for x in batches:
            model(x)
        score_s = time.perf_counter() - t0
        other_during = load.stop()
    other = max(other_before, other_during)
    p = count_parameters(model)
    return {"run": run, "params": p["total"], "non_embedding": p["non_embedding"],
            "weights_mb": len(buf.getvalue()) / 2**20,
            "latency_s_200_tokens": gen_s, "gen_tokens_per_s": GEN_TOKENS / gen_s,
            "score_tokens_per_s": SCORE_BATCHES * 16 * 256 / score_s,
            "peak_ram_mb": peak_memory_mb("cpu"), "ram_added_by_model_mb": rss_loaded - rss_before,
            "threads": torch.get_num_threads(), "other_cpu_before_pct": other_before,
            "other_cpu_during_pct": other_during, "reliable": other <= CPU_BUSY_LIMIT}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["teacher"], default=None)
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--single", default=None, help=argparse.SUPPRESS)  # internal: measure one model, print JSON
    args = ap.parse_args()
    if args.single:
        print("RESULT " + json.dumps(measure_one(args.single)))
        return

    targets = (["teacher"] if args.model else []) + args.ckpt
    results = []
    for t in targets:
        out = subprocess.run([sys.executable, os.path.abspath(__file__), "--single", t], capture_output=True,
                             text=True, cwd=kd.PROJECT2, check=True).stdout
        r = json.loads(next(line for line in out.splitlines() if line.startswith("RESULT "))[7:])
        results.append(r)
        print(f"{r['run']:>22} | {r['params'] / 1e6:5.2f}M params | {r['weights_mb']:5.1f} MB | "
              f"{r['latency_s_200_tokens']:.2f} s / 200 tokens ({r['gen_tokens_per_s']:.0f} tok/s) | "
              f"scoring {r['score_tokens_per_s']:,.0f} tok/s | peak RAM {r['peak_ram_mb']:.0f} MB | "
              f"other CPU {max(r['other_cpu_before_pct'], r['other_cpu_during_pct']):.0f}% "
              f"{'reliable' if r['reliable'] else 'NOT reliable'}")

    teacher = next((r for r in results if r["run"] == "teacher"), None)
    L = ["| Model | Parameters | Weights (MB) | Compression | Latency, 200 tokens | Generation tokens/s | Speedup | "
         "Scoring tokens/s | Peak RAM (MB) | RAM added by model (MB) | Reliable |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if teacher:
            r["compression_ratio"] = teacher["params"] / r["params"]
            r["speedup_generation"] = teacher["latency_s_200_tokens"] / r["latency_s_200_tokens"]
            r["speedup_scoring"] = r["score_tokens_per_s"] / teacher["score_tokens_per_s"]
        L.append(f"| {r['run']} | {r['params']:,} | {r['weights_mb']:.1f} | {r.get('compression_ratio', 1):.2f}× | "
                 f"{r['latency_s_200_tokens']:.2f} s | {r['gen_tokens_per_s']:.0f} | {r.get('speedup_generation', 1):.2f}× | "
                 f"{r['score_tokens_per_s']:,.0f} | {r['peak_ram_mb']:.0f} | {r['ram_added_by_model_mb']:.0f} | "
                 f"{'yes' if r['reliable'] else 'no'} |")
    path = os.path.join(kd.PROJECT2, "results", "metrics", "inference_benchmark")
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(path + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
