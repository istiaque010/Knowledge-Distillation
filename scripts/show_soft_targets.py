"""Phase 6 (learning aid): what temperature does to the teacher's next-token probabilities.

For a prompt, prints the teacher's top tokens at T = 1, 2, 4, and for each temperature:
  - entropy (bits): how spread out the distribution is (0 = certain; log2(8192) = 13 = completely random)
  - "effective choices" = 2^entropy: roughly how many tokens the teacher is choosing between
  - mass outside the top 10: how much of the signal sits in the long tail of the vocabulary
The hard label is just one token at 100%: everything else on these lists is what the student gains from a teacher.

Usage (from project2, .venv active):
    python scripts/show_soft_targets.py --prompt "One day, a little"
    python scripts/show_soft_targets.py --prompt "Tom was sad because" --top 8 --temps 1 2 4
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import kd  # noqa: E402
from kd.teacher import load_frozen_teacher, teacher_logits  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", action="append", help="can be given several times")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--temps", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    args = ap.parse_args()
    prompts = args.prompt or ["One day, a little", "Tom was sad because"]

    teacher, tok = load_frozen_teacher()
    out = []
    for prompt in prompts:
        ids = torch.tensor([tok.encode(prompt, add_bos=True)])
        z = teacher_logits(teacher, ids)[0, -1]  # scores for the next token
        order = torch.argsort(z, descending=True)[: args.top].tolist()
        names = [repr(tok.decode([i], skip_special=False)) for i in order]
        print(f'\nAfter "{prompt}" (teacher, top {args.top} tokens):')
        print(f"{'token':>12} " + " ".join(f"{'T=' + format(t, 'g'):>8}" for t in args.temps))
        probs = {t: torch.softmax(z / t, dim=-1) for t in args.temps}
        for i, name in zip(order, names):
            print(f"{name:>12} " + " ".join(f"{100 * probs[t][i].item():7.1f}%" for t in args.temps))
        rec = {"prompt": prompt, "top_tokens": names, "temps": {}}
        for t in args.temps:
            p = probs[t]
            ent = -(p * torch.log2(p.clamp_min(1e-30))).sum().item()
            tail = 1 - torch.topk(p, 10).values.sum().item()
            rec["temps"][format(t, "g")] = {"top_probs": [round(p[i].item(), 4) for i in order],
                                            "entropy_bits": round(ent, 2), "effective_choices": round(2**ent, 1),
                                            "mass_outside_top10": round(tail, 4)}
        print(f"{'entropy':>12} " + " ".join(f"{rec['temps'][format(t, 'g')]['entropy_bits']:6.2f} b" for t in args.temps))
        print(f"{'choices':>12} " + " ".join(f"{rec['temps'][format(t, 'g')]['effective_choices']:8.1f}" for t in args.temps))
        print(f"{'outside 10':>12} " + " ".join(f"{100 * rec['temps'][format(t, 'g')]['mass_outside_top10']:7.1f}%"
                                                for t in args.temps))
        out.append(rec)

    print(f"\n(completely random over {tok.vocab_size} tokens = {math.log2(tok.vocab_size):.0f} bits)")
    path = os.path.join(kd.PROJECT2, "results", "metrics", "soft_targets.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
