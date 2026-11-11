"""Phase 2: environment report. Checks that Project 2 runs the same versions as Project 1 and that training works.

1. versions: Python, PyTorch, tokenizers, numpy must equal the teacher's environment
2. smoke test: a tiny model learns one batch (loss must go down)
3. the teacher's code and tokenizer load from this environment

Usage (from the project2 folder, .venv active): python scripts/env_report.py
Writes results/metrics/env_report.json. Exit code 0 = everything matches.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy  # noqa: E402
import tokenizers  # noqa: E402
import torch  # noqa: E402

import kd  # noqa: E402  (makes the teacher package importable)
from src.model import GPT, GPTConfig, count_parameters  # noqa: E402
from src.tokenizer import StoryTokenizer  # noqa: E402
from src.utils import env_info, resolve_device  # noqa: E402

# Versions of the teacher's environment (teacher/requirements.lock.txt)
TEACHER_VERSIONS = {"python": "3.12.10", "torch": "2.14.1+cpu", "tokenizers": "0.23.3", "numpy": "2.5.4"}


def main() -> None:
    device = resolve_device("auto")
    info = env_info(device)
    info["tokenizers"] = tokenizers.__version__
    info["numpy"] = numpy.__version__

    # 1. same versions as the teacher
    info["version_check"] = {k: {"project2": info[k], "teacher": v, "same": info[k] == v}
                             for k, v in TEACHER_VERSIONS.items()}
    versions_ok = all(c["same"] for c in info["version_check"].values())

    # 2. smoke test: overfit one random batch
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=512, block_size=32, n_layer=2, n_head=2, n_embd=64, dropout=0.0)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randint(0, 512, (4, 33), device=device)
    losses = []
    for _ in range(20):
        _, loss = model(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    info["smoke_params"] = count_parameters(model)["total"]
    info["smoke_loss_first_last"] = [round(losses[0], 3), round(losses[-1], 3)]
    info["smoke_ok"] = losses[-1] < losses[0]

    # 3. teacher tokenizer loads and round-trips
    tok = StoryTokenizer(kd.TOKENIZER)
    text = "The little boy played with his dog."
    ids = tok.encode(text)
    info["tokenizer_ok"] = tok.vocab_size == 8192 and tok.decode(ids) == text
    info["tokenizer_sample_ids"] = ids

    info["all_ok"] = bool(versions_ok and info["smoke_ok"] and info["tokenizer_ok"])
    out = os.path.join(kd.PROJECT2, "results", "metrics", "env_report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    for k, c in info["version_check"].items():
        print(f"{k:11s} project2 {c['project2']:12s} teacher {c['teacher']:12s} {'same' if c['same'] else 'DIFFERENT'}")
    print(f"CPU: {info['processor']}, {info['cpu_count']} logical CPUs, {info['ram_gb']} GB RAM, "
          f"{info['torch_threads']} torch threads, device {device}")
    print(f"smoke test loss {losses[0]:.3f} -> {losses[-1]:.3f}  {'OK' if info['smoke_ok'] else 'FAILED'}")
    print(f"teacher tokenizer: '{text}' -> {ids} -> round trip {'OK' if info['tokenizer_ok'] else 'FAILED'}")
    print(f"\n{'ALL OK' if info['all_ok'] else 'SOMETHING DIFFERS'}  -> results/metrics/env_report.json")
    sys.exit(0 if info["all_ok"] else 1)


if __name__ == "__main__":
    main()
