"""Phase 13: error rates for each model, from labelled error-analysis sheets (labels produced
with an AI assistant, blinded between the students; not yet human-verified).

Labels live in scripts/error_labels/<run>.py (a LABELS dict: id -> (contradiction, grammar,
entity_consistency, notes)), so they survive if a CSV is regenerated. This script writes them into
results/generations/<run>/error_analysis.csv and counts. The teacher's sheet is the one labelled in Project 1
(../project1/teacher/results/error_analysis.csv): same 60 prompts and seed, same rules.

Rules (the same as Project 1 Phase 13; one slip anywhere in a story counts):
- contradiction: something clashes with an earlier fact in the story or doesn't follow from it
- grammar: at least one broken sentence or invented word
- entity consistency: a character changes name, gender or species, or the wrong character speaks or acts
- repetition and incomplete are flagged automatically

Usage (from project2, .venv active):
    python scripts/error_counts.py --run teacher --run student_scratch --run student_kd_T2_a0.5
Writes results/metrics/error_counts.json and error_counts.md
"""
import argparse
import csv
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kd  # noqa: E402

HAND = ["contradiction", "grammar", "entity_consistency"]
AUTO = ["repetition", "incomplete", "hit_token_limit"]


def sheet_path(run: str) -> str:
    if run == "teacher":
        return os.path.join(kd.TEACHER_DIR, "results", "error_analysis.csv")
    return os.path.join(kd.PROJECT2, "results", "generations", run, "error_analysis.csv")


def apply_labels(run: str, path: str) -> None:
    """Write scripts/error_labels/<run>.py into the CSV (never for the teacher: Project 1 is read only)."""
    lab = os.path.join(kd.PROJECT2, "scripts", "error_labels", f"{run}.py")
    if run == "teacher" or not os.path.exists(lab):
        return
    spec = importlib.util.spec_from_file_location(f"labels_{run}", lab)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())
    for r in rows:
        c, g, e, note = mod.LABELS[int(r["id"])]
        r.update({"contradiction": "yes" if c else "no", "grammar": "yes" if g else "no",
                  "entity_consistency": "yes" if e else "no", "notes": note})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    args = ap.parse_args()

    out = {}
    for run in args.run:
        path = sheet_path(run)
        apply_labels(run, path)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not all(r["contradiction"] in ("yes", "no") for r in rows):
            print(f"{run}: not labelled yet ({path})")
            continue
        n = len(rows)
        yes = lambda r, k: r[k] in ("yes", "True")  # noqa: E731
        counts = {k: sum(yes(r, k) for r in rows) for k in HAND + AUTO}
        per_story = [sum(yes(r, k) for k in HAND + ["repetition", "incomplete"]) for r in rows]
        counts["no_error"] = sum(c == 0 for c in per_story)
        counts["n"] = n
        out[run] = counts
    if not out:
        return

    L = ["| Error (60 stories each, top-p 0.90) | " + " | ".join(out) + " |", "|---|" + "---|" * len(out)]
    for k, label in (("contradiction", "Contradiction"), ("entity_consistency", "Entity consistency"),
                     ("grammar", "Grammar"), ("repetition", "Repetition (auto)"),
                     ("incomplete", "Incomplete (auto)"), ("no_error", "**No error of any kind**")):
        L.append(f"| {label} | " + " | ".join(f"{c[k]}/{c['n']} ({c[k] / c['n']:.0%})" for c in out.values()) + " |")
    print("\n".join(L))
    with open(os.path.join(kd.PROJECT2, "results", "metrics", "error_counts.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(kd.PROJECT2, "results", "metrics", "error_counts.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
