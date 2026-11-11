# Knowledge Distillation of a Small Autoregressive Language Model

Project 2 of the TinyStories series. The 13.9M-parameter Transformer from Project 1
([Autoregressive-Transformer-Language-Model](https://github.com/istiaque010/Autoregressive-Transformer-Language-Model))
is the **teacher**. A smaller **student** (5.3M parameters, same architecture with 4 layers / 4 heads / width
256) learns to copy its next-token probabilities.

**Question:** how much can the model be compressed through knowledge distillation while keeping its
text-generation quality?

## Answer

**The teacher can be shrunk 2.6× (13.9M → 5.3M parameters, 53 MB → 20 MB) and made ~1.8× faster at writing
stories. Distillation makes that smaller model clearly better than training it alone, but it does not make it
as good as the teacher.**

- With the same data, steps and settings, the distilled student reaches test perplexity **8.59**. The same
  student trained without the teacher reaches **9.63**, and the teacher **5.74**. Distillation closes **26.7%**
  of the gap (30.7% with the best setting found, α = 0.25), for free at inference time: the cost is ~1.55× slower training on CPU.
- The distilled student also reaches the baseline's final quality in about **37% fewer steps**, and it is still
  slightly ahead when both are given the same wall-clock time.
- In blind hand-labelled stories it makes fewer grammar errors (67% vs 80% of stories) and fewer character
  mix-ups (73% vs 85%). The direction is consistent, but 60 stories per model are too few to call these two
  differences significant.
- **What compression loses most is story-level consistency.** All 120 student stories contain a contradiction
  (teacher: 88%), and distillation does not fix this. The skills that need the most capacity, keeping track of
  who is who and what is where over a whole story, are the ones a 2.6× smaller model cannot hold.
- The distilled student writes **in the teacher's style**, including the teacher's habit of repeating phrases
  (repetition 20% vs teacher 22%, scratch 7%).
- Temperature: **T = 2 and T = 4 work equally well** (8.59 / 8.60); T = 1 is weaker (8.75). Every temperature is
  far better than no teacher (9.63).
- Teacher weight: **the more weight on the teacher, the better.** With α (the real-data weight) at 0.75 / 0.5 /
  0.25, test perplexity is 9.08 / 8.59 / 8.43. The best setting found, **T = 2, α = 0.25, closes 30.7% of the
  gap.** The main comparison uses α = 0.5 because it was fixed before the runs.

## Results

| | Teacher | Student, no teacher | **Distilled student (T = 2, α = 0.5)** |
|---|---|---|---|
| Parameters | 13.89M | 5.32M | 5.32M |
| Weights (fp32) | 53.0 MB | 20.3 MB | 20.3 MB |
| **Test perplexity** | **5.74** | **9.63** | **8.59** |
| KL(teacher ‖ model) | 0 | 0.481 | 0.349 |
| Latency, 200-token story (CPU) | 1.04 s | 0.57 s | 0.59 s |
| Stories with a grammar error | 37% | 80% | 67% |
| Stories with a character error | 55% | 85% | 73% |
| Stories with a contradiction | 88% | 100% | 100% |
| Training | 10,000 steps, 9.9 h | 3,000 steps, 1.3 h | 3,000 steps, 2.0 h |

Full table (generated from saved results, no numbers typed by hand): [results/final_comparison.md](results/final_comparison.md).
Training curves: [results/figures/curves.png](results/figures/curves.png). Stories side by side:
[results/generations/comparison.md](results/generations/comparison.md).

**Limits:** one seed per setting; every student got the same 3,000-step budget (30% of the teacher's training), so
longer training would raise all students; error labels are from one labeller (blind between the two students, and
checked for consistency against Project 1's teacher labels); CPU only, fp32.

## Try it yourself

After setup (below) and training, the teacher and the students each finish the same story beginning:
```
python scripts\try_models.py --prompt "Tom found a red box"
python scripts\try_models.py --interactive
```
A beginner-friendly tutorial with interactive figures is in [project2.html](project2.html) (download it and open it in a browser).

## How it was done

Every phase (what to learn, files, commands, checks and the actual result) is in
[INSTRUCTIONS.md](INSTRUCTIONS.md).

- Loss: `α · CE(student, data) + (1 − α) · T² · KL(teacher_T ‖ student_T)`, in [kd/distillation.py](kd/distillation.py).
- The teacher is frozen (eval mode, no gradients, hash-checked against Project 1's manifest); tests in
  [tests/](tests/).
- The baseline and every distilled student use one config ([configs/student.yaml](configs/student.yaml)), the
  same seed and the same data order. The only difference is the loss.
- Evaluation reuses Project 1's code: full-test-set perplexity, the same story prompts and seeds, and the same
  error-labelling rules. The teacher's re-evaluation reproduces Project 1 exactly (5.7392).

## Setup

This folder sits next to `project1/` and reads the teacher and data from it (read only):
```
Projrcts List/
├── project1/    teacher/ (teacher.pt, tokenizer.json, model code), data/full/ (TinyStories token IDs)
└── project2/    this project
```

```
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest                                   # 12 tests
python scripts\check_teacher.py                    # teacher loads, test perplexity 5.74
python -m kd.train_student --config configs\student.yaml --mode scratch
python -m kd.train_student --config configs\student.yaml --mode kd --temperature 2 --alpha 0.5
powershell -ExecutionPolicy Bypass -File scripts\run_remaining.ps1   # other runs, all evaluations, speed
```
