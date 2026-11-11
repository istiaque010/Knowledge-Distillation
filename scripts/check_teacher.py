"""Phase 1: check that the Project 1 teacher still loads and gives exactly the same scores.

Everything Project 2 compares against depends on these numbers, so we check them before building anything.

1. every teacher file matches MANIFEST.sha256 (nothing missing or changed)
2. the data Project 2 trains on (project1/data/full) uses the same tokenizer and the same val/test files
3. the model loads with the expected number of parameters
4. test and validation loss/perplexity match expected.json (same method as Project 1)
5. it writes a story with the recommended decoding (top-p 0.95)

Project 1 is only READ, never changed. Results: results/metrics/teacher/check.json

Usage (from the project2 folder):
    python scripts/check_teacher.py            # full check, about 3 minutes
    python scripts/check_teacher.py --quick    # first 100 test batches only, about 15 s
Exit code 0 = all checks passed.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True  # don't write __pycache__ files into the project1 folder

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

PROJECT2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT1 = os.path.join(os.path.dirname(PROJECT2), "project1")
TEACHER_DIR = os.path.join(PROJECT1, "teacher")
DATA_DIR = os.path.join(PROJECT1, "data", "full")  # what Project 2 will train and evaluate on
OUT = os.path.join(PROJECT2, "results", "metrics", "teacher", "check.json")

sys.path.insert(0, TEACHER_DIR)
from load_teacher import load_teacher  # noqa: E402
from src.dataset import NextTokenDataset, load_tokens  # noqa: E402
from src.generate import generate_text  # noqa: E402
from src.model import count_parameters  # noqa: E402
from src.train import evaluate_loss  # noqa: E402
from src.utils import set_seed  # noqa: E402


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_manifest() -> list[str]:
    """Compare every file in the teacher package with the hash recorded when it was packaged."""
    problems = []
    with open(os.path.join(TEACHER_DIR, "MANIFEST.sha256"), encoding="utf-8") as f:
        for line in f:
            digest, rel = line.rstrip("\n").split("  ", 1)
            path = os.path.join(TEACHER_DIR, rel)
            if not os.path.exists(path):
                problems.append(f"missing {rel}")
            elif sha256(path) != digest:
                problems.append(f"changed {rel}")
    return problems


def check_same_data() -> dict:
    """The student trains on project1/data/full. Its tokenizer and held-out files must be the teacher's."""
    pairs = {
        "tokenizer": (os.path.join(TEACHER_DIR, "tokenizer.json"), os.path.join(PROJECT1, "data", "tokenizer.json")),
        "val.bin": (os.path.join(TEACHER_DIR, "data", "val.bin"), os.path.join(DATA_DIR, "val.bin")),
        "test.bin": (os.path.join(TEACHER_DIR, "data", "test.bin"), os.path.join(DATA_DIR, "test.bin")),
    }
    result = {name: sha256(a) == sha256(b) for name, (a, b) in pairs.items()}
    result["train.bin exists"] = os.path.exists(os.path.join(DATA_DIR, "train.bin"))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="score only the first 100 test batches")
    args = ap.parse_args()
    with open(os.path.join(TEACHER_DIR, "expected.json"), encoding="utf-8") as f:
        exp = json.load(f)
    report = {"teacher_dir": TEACHER_DIR, "data_dir": DATA_DIR, "quick": args.quick, "torch": torch.__version__}
    ok = True

    # 1. teacher files unchanged
    problems = check_manifest()
    report["manifest_problems"] = problems
    print(f"[1] teacher files vs MANIFEST.sha256: {'OK' if not problems else 'FAILED ' + ', '.join(problems)}")
    ok &= not problems

    # 2. Project 2's data = the teacher's data
    same = check_same_data()
    report["same_data"] = same
    good = all(same.values())
    print(f"[2] project1/data/full matches the teacher: "
          + ", ".join(f"{k} {'yes' if v else 'NO'}" for k, v in same.items()) + f"  {'OK' if good else 'FAILED'}")
    ok &= good

    # 3. model loads
    torch.set_num_threads(os.cpu_count() or 1)
    model, tok, info = load_teacher()
    counts = count_parameters(model)
    report["params"] = counts
    report["model_config"] = info["model_config"]
    good = counts["total"] == exp["params"]
    print(f"[3] model: {counts['total']:,} parameters ({counts['non_embedding']:,} non-embedding), "
          f"trained {info['step']:,} steps, vocab {tok.vocab_size}  {'OK' if good else 'FAILED'}")
    ok &= good

    # 4. perplexity, computed exactly as in Project 1 (256-token windows, batch 16, fp32)
    max_batches = 100 if args.quick else None
    tol = 0.05 if args.quick else 1e-3
    report["scores"] = {}
    for split in ("test",) if args.quick else ("test", "val"):
        ds = NextTokenDataset(load_tokens(DATA_DIR, split), model.cfg.block_size)
        t0 = time.time()
        loss = evaluate_loss(model, DataLoader(ds, batch_size=exp["eval_batch_size"], shuffle=False),
                             "cpu", exp["precision"], max_batches)
        want = exp[f"{split}_loss"]
        good = abs(loss - want) <= tol
        report["scores"][split] = {"loss": loss, "perplexity": math.exp(loss), "expected_loss": want,
                                   "windows": len(ds), "seconds": round(time.time() - t0, 1)}
        print(f"[4] {split}: loss {loss:.4f}, perplexity {math.exp(loss):.2f} "
              f"(expected {want:.4f} / {math.exp(want):.2f}{'; quick = first 100 batches' if args.quick else ''})"
              f"  {'OK' if good else 'FAILED'}")
        ok &= good

    # 5. one story with the recommended decoding
    set_seed(1337)
    d = exp["recommended_decoding"]
    out = generate_text(model, tok, "Once upon a time", max_new_tokens=200, method=d["method"],
                        top_p=d["top_p"], temperature=d["temperature"])
    report["story"] = out
    print(f"[5] story, top-p {d['top_p']} ({out['new_tokens']} new tokens, ended by itself: {out['ended_with_eos']}):")
    print(f"    {out['text']}")
    ok &= out["new_tokens"] > 0

    report["all_passed"] = bool(ok)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}  -> {os.path.relpath(OUT, PROJECT2)}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
