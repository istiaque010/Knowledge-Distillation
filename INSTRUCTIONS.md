# Project 2 — Knowledge Distillation of the Autoregressive Model

Source plan: `docs/Project_2_Knowledge_Distillation_Guide.docx` (kept locally, not in the repository).
This file turns that plan into steps you can follow and check one phase at a time.

**Goal:** use the Project 1 model as a **teacher** and train a **smaller student** to copy its next-token
guesses. The student should keep most of the teacher's story quality while being smaller and faster.

**Main question:** how much can an autoregressive language model be compressed through knowledge
distillation while keeping its text-generation quality?

**The core idea:** normal training only tells the model the one correct next word. A teacher also says how
likely the *other* words were:
```
After "... there was a little" the real next word is "girl".

Normal training (hard label):   girl 100%
Teacher (soft targets):         girl 75%   boy 16%   bird 1.4%   ...
```
"boy was also a good guess" is extra information, and it is what the student learns from.

**Pipeline:**
Project 1 teacher → freeze it → build a smaller student → (A) train the student normally = baseline
→ (B) train the student with CE + distillation loss = distilled student → evaluate quality, size and speed
→ compare teacher vs baseline student vs distilled student.

---

## How to use this file

1. Open PowerShell in the project folder and activate the environment (created in Phase 2):
   ```
   cd "D:\TTI\Projrcts List\project2"
   .venv\Scripts\activate
   ```
2. Work through the phases in order. Each phase has:
   - **Learn:** the idea to understand before running anything
   - **Files:** the code for this phase (read it)
   - **Run:** the commands
   - **Check:** what you should see; the phase is done when this is true
   - **Result:** filled in after the phase is run, with the real numbers
3. The status in the table below changes from ⬜ to ✅ when the phase is done and checked.

## What we reuse from Project 1 (read only, never changed)

| What | Where | Why |
|---|---|---|
| Teacher model, tokenizer, model code | `..\project1\teacher\` | One source of truth; checked with `MANIFEST.sha256` |
| Training data (all 1,772,187 cleaned stories) | `..\project1\data\full\train.bin` (745 MB) | Same data the teacher saw, no 3 GB copy |
| Validation and test sets (10,995 stories each) | `..\project1\data\full\val.bin`, `test.bin` | Same held-out stories, so perplexities are comparable |
| Evaluation prompts and metrics | `..\project1\teacher\configs\prompts.txt`, `..\project1\teacher\src\evaluate.py` | Same measurements for all three models |

**Teacher, for reference:** 13,891,584 parameters (6 layers / 6 heads / width 384), 53 MB,
**test perplexity 5.74**, 141 tokens/s on CPU, recommended decoding top-p 0.95.

## Progress

| # | Phase | Status |
|---|---|---|
| 1 | Reuse Project 1 as the teacher | ✅ Done (2026-11-10): test perplexity 5.74 reproduced exactly |
| 2 | Environment and requirements | ✅ Done (2026-11-10): same versions as Project 1 |
| 3 | Freeze the teacher | ✅ Done (2026-11-10): 5 tests pass |
| 4 | Design a smaller student | ✅ Done (2026-11-10): 4L/4H/256d, 5.32M params (2.6× smaller) |
| 5 | Student-from-scratch baseline | ✅ Done (2026-11-10): val perplexity 9.87 in 77.5 min |
| 6 | Understand teacher soft targets | ✅ Done (2026-11-10): T = 2 puts 58–84% of probability outside the top 10 |
| 7 | Implement the distillation loss | ✅ Done (2026-11-10): 7 tests pass |
| 8 | Train the distilled student | ✅ Done (2026-11-10): val perplexity 8.79 vs 9.87 for scratch (same 3,000 steps) |
| 9 | Distillation ablations | ✅ Done (2026-11-11): T = 2 ≈ T = 4 > T = 1; lower α is better, best T = 2, α = 0.25 (test PPL 8.43) |
| 10 | Evaluate language-modelling quality | ✅ Done (2026-11-10): test PPL 5.74 / 9.63 / 8.59: distillation closes 26.7% of the gap |
| 11 | Evaluate compression and efficient inference | ✅ Done (2026-11-10): 2.61× smaller, ~1.8× faster generation, 2.4–2.8× faster scoring |
| 12 | Compare text generation | ✅ Done (2026-11-10): distilled student writes like the teacher; character tracking does not transfer |
| 13 | Error analysis | ✅ Done (2026-11-10): grammar 37/80/67%, entity 55/85/73%, contradiction 88/100/100% (teacher/scratch/KD) |
| 14 | Final comparison | ✅ Done (2026-11-11): 2.6× smaller, ~1.8× faster; distillation closes 26.7% of the perplexity gap (30.7% at α = 0.25) |

**Changes from the guide for this PC:** the guide assumes an NVIDIA GPU (CUDA, mixed precision, GPU memory).
This PC trains on the **CPU** (Intel Core Ultra 7, 24 threads, 64 GB RAM), as in Project 1. So we use fp32,
and "GPU memory" becomes **peak RAM** of the process. The code still uses a GPU automatically if one is available.

**Training budget (decided 2026-11-10): 3,000 steps per student** (64 sequences × 256 tokens per step, about
49M tokens), instead of the teacher's 10,000. Every student gets exactly this budget, so the comparison stays
fair. Estimated total training: **~7.5 h** (baseline ~1.2 h, three distilled runs ~2.1 h each). Phase 4 measures
the real speed and these times are updated then. The students will score worse than after a long run. If there is
time at the end, the best setting can get one long run.

**Code folder:** Project 2's code lives in `kd/` (knowledge distillation), **not** `src/`. The teacher's code is a
package called `src`, and two packages with the same name would collide when the teacher is loaded.

---

## Phase 1 — Reuse Project 1 as the teacher

**Learn:** we don't train a new teacher. The Project 1 model is finished, packaged and verified. Before
building on it we check that it still loads and gives **exactly** the same scores, because every later
comparison depends on those numbers. If the teacher were different, "the student keeps 90% of the teacher's
quality" would mean nothing.

**Files:** `scripts/check_teacher.py` (loads the teacher from `..\project1\teacher\`, checks the file
hashes, reproduces perplexity and writes one story)

**Run:** Project 2's own environment is made in Phase 2, so this phase uses Project 1's (same versions):
```
..\project1\.venv\Scripts\python.exe scripts\check_teacher.py           # full check, ~3 min
..\project1\.venv\Scripts\python.exe scripts\check_teacher.py --quick   # ~15 s
```

**Check:**
1. every teacher file matches `MANIFEST.sha256` (nothing missing or changed)
2. the training data Project 2 will use (`..\project1\data\full\`) has the **same tokenizer and the same
   validation and test files** as the teacher package
3. 13,891,584 parameters
4. test loss 1.7473 / perplexity 5.74, and validation 1.7467 / 5.74
5. a story written with top-p 0.95 continues the prompt

Results in `results/metrics/teacher/check.json`.

**Result (2026-11-10):** all 5 checks passed (full run, Project 1's `.venv`, PyTorch 2.14.1+cpu).

| Check | Result |
|---|---|
| 1. Teacher files vs `MANIFEST.sha256` | all match, nothing missing or changed ✅ |
| 2. `project1\data\full` vs teacher package | same tokenizer, same `val.bin`, same `test.bin`; `train.bin` present ✅ |
| 3. Parameters | 13,891,584 (10,647,552 non-embedding), trained 10,000 steps, vocabulary 8,192 ✅ |
| 4. Test | loss **1.7473**, perplexity **5.74** (8,844 windows, 77 s): identical to Project 1 ✅ |
| 4. Validation | loss 1.7467, perplexity 5.74 (8,851 windows, 75 s): identical to Project 1 ✅ |
| 5. Story (top-p 0.95, seed 1337) | 140 tokens, ended by itself ✅ |

The story: *"Once upon a time, there was a little girl called outdoors. She was very alert and had an open chest.
One day she decided to explore a forest. … she saw a bush filled with coins. … From then on, the little girl was
known as the most alert two of them in the world. The end!"*
- It is fluent, follows one plot and ends properly.
- It also shows the teacher's known weaknesses from Project 1 Phase 13: an odd name ("called outdoors"), an object
  that appears from nowhere ("the palm was filled with copper coins") and a confused last line.
- The student will learn from this teacher, so it can't be expected to do better on these.

What this means for the rest of Project 2:
- The teacher is exactly the model Project 1 reported, so **5.74 is the reference perplexity** for every student.
- Check 2 matters for fairness: the students will train on `project1\data\full\train.bin` and be scored on the
  same test file, with the same tokenizer, so their perplexities can be compared directly with 5.74.
- The script sets `sys.dont_write_bytecode`, so running it leaves no `__pycache__` files in `project1`.

---

## Phase 2 — Environment and requirements

**Learn:** the student must see the data exactly as the teacher did: the same tokenizer, the same token IDs, the
same windows. The safest way is the same Python and library versions as Project 1. Project 2 gets its **own**
`.venv` so the two projects stay independent, but with the same pinned versions (`torch==2.14.1+cpu`,
`tokenizers==0.23.3`, `numpy==2.5.4`).

The guide's `transformers` package isn't needed: both models are plain PyTorch, written by hand.
No database or vector store is needed either.

**Files:** `requirements.txt`, `requirements.lock.txt`, `scripts/env_report.py`

**Run:**
```
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts\env_report.py
```

**Check:** the versions in `results/metrics/env_report.json` match Project 1's
(`..\project1\teacher\results\env_report.json`), and `pytest` runs.

**Result (2026-11-10):** done. `.venv` made with Python 3.12.10 (`py -3.12 -m venv .venv`).

| | Project 2 | Teacher (Project 1) |
|---|---|---|
| Python | 3.12.10 | 3.12.10 ✅ |
| PyTorch | 2.14.1+cpu | 2.14.1+cpu ✅ |
| tokenizers | 0.23.3 | 0.23.3 ✅ |
| numpy | 2.5.4 | 2.5.4 ✅ |
| Smoke test loss | 6.257 → 0.538 | 6.257 → 0.538 ✅ (identical to 3 decimals) |
| Teacher tokenizer | "The little boy played with his dog." → `[1607, 399, 527, 768, 347, 353, 645, 17]`, round trip exact ✅ | same IDs as Project 1 Phase 3 |

- Hardware: Intel CPU, 24 logical CPUs, 63.7 GB RAM, 24 torch threads, no GPU visible to PyTorch.
- The smoke test giving the **same losses to 3 decimals** shows the two environments compute identically.
- `datasets` and `pyarrow` are left out: Project 1 used them only to download and clean TinyStories, and
  Project 2 reads the already tokenized files. `pytest` 9.1.1 runs.
- **New: the `kd` package.** `import kd` adds `..\project1\teacher` to Python's search path, so Project 2 reuses
  the teacher's code (`from src.model import GPT`) without copying it. It also stops Python from writing
  `__pycache__` files into `project1`.

---

## Phase 3 — Freeze the teacher

**Learn:** during distillation the teacher only **gives** answers and never learns. In code:
```python
teacher.eval()                       # dropout off, so its answers don't change randomly
for p in teacher.parameters():
    p.requires_grad = False          # no gradients, so it can never be updated
with torch.no_grad():                # also saves memory and time
    teacher_logits, _ = teacher(x)
```
```
Input tokens ─┬─→ Teacher (frozen) → teacher logits ─┐
              └─→ Student (learns) → student logits ─┴→ distillation loss
```

**Files:** `kd/teacher.py` (`load_frozen_teacher`), `tests/test_teacher_frozen.py`

**Run:**
```
python -m pytest tests\test_teacher_frozen.py -v
```

**Check:** the teacher returns logits of shape `(batch, 256, 8192)` for a real batch; no teacher parameter
has a gradient after a backward pass; the teacher's weights are byte-for-byte identical before and after one
training step; two runs on the same batch give identical logits (dropout really is off).

**Result (2026-11-10):** done. `tests/test_teacher_frozen.py`: **5 passed** (2.2 s), on real validation windows.

| Test | What it proves |
|---|---|
| logits for a real batch | shape `(2, 256, 8192)`, and the logits carry no gradient |
| eval mode, all frozen | every module is in eval mode, every parameter has `requires_grad = False` |
| `.train()` can't undo it | calling `teacher.train()` by mistake leaves dropout off |
| identical logits | the same batch gives exactly the same logits twice |
| one student step | the student's weights change; the teacher gets **no** gradient and every weight is byte-for-byte unchanged |

What goes wrong **without** these safeguards (tried in memory only, nothing saved):
- **Dropout on:** the same batch gives different logits each time (largest difference **6.4**), so the student
  would chase noisy targets.
- **Not frozen and no `torch.no_grad()`:** the student's loss sends gradients into **76 of 76** teacher weight
  tensors. They would change if they were in an optimizer, and memory is wasted storing them either way.

`kd/teacher.py` also checks `teacher.pt` against `MANIFEST.sha256` every time it loads (about 0.2 s), so a changed
teacher file can never be used by mistake. It also makes `.train()` do nothing on the teacher, which is the third
test above.

---

## Phase 4 — Design a smaller student

**Learn:** the student uses the **same tokenizer and vocabulary** (8,192 tokens) and the same context (256),
so its output means the same thing as the teacher's. It is smaller in depth and width:

| | Teacher | Student option A | Student option B |
|---|---|---|---|
| Layers / heads / width | 6 / 6 / 384 | 4 / 4 / 256 | 3 / 4 / 256 |
| Head size | 64 | 64 | 64 |
| Parameters | 13,891,584 | 5,322,240 | 4,532,480 |
| Compression ratio | 1× | **2.6×** | **3.1×** |

Note: with an 8,192-token vocabulary, the embeddings alone take 2.16M parameters at width 256, so most of the
saving comes from the Transformer blocks (10.6M → 3.2M for option A).

Option A has the same shape as Project 1's `baseline` model. That model is **not** reused as the student
baseline, because it saw only 100K stories for 5,000 steps. Phase 5 trains the student again with the
same data and budget as the distilled student, so the comparison is fair.

**Files:** `configs/student.yaml`, `kd/model.py` (reuses the teacher's `GPT` class), `scripts/benchmark.py`

**Run:**
```
python scripts\benchmark.py --config configs\student.yaml                          # option A alone
python scripts\benchmark.py --config configs\student.yaml --with_teacher           # option A + teacher
python scripts\benchmark.py --config configs\student.yaml --n_layer 3              # option B alone
python scripts\benchmark.py --config configs\student.yaml --n_layer 3 --with_teacher
python scripts\benchmark.py --config configs\student.yaml --teacher_only           # teacher forward pass only
```

**Check:** the student forward pass works, the parameter count is recorded for both models, and we know the
real training speed (tokens/s) with and without the teacher. That gives the time for Phases 5 and 8.

**Result (2026-11-10):** done. Real data, batch 8 × 256 tokens, fp32, 24 threads, 30 timed micro-batches.
Saved in `results/metrics/benchmark.csv`.

| | Tokens/s | ms per micro-batch | Peak RAM | Time for 3,000 steps (49M tokens) |
|---|---|---|---|---|
| **A** 4L/4H/256d alone (scratch) | 10,532 | 195 | 966 MB | **1 h 17 m** |
| **A** + teacher (distillation) | 6,780 | 302 | 1,370 MB | **2 h 00 m** |
| B 3L/4H/256d alone | 13,107 | 156 | 891 MB | 1 h 02 m |
| B + teacher | 8,498 | 241 | 1,278 MB | 1 h 36 m |
| Teacher forward pass only | 42,316 | 48 | 560 MB | — |

- **Distillation costs about 1.55× a normal step.** The teacher's forward pass is only 48 ms. The other ~60 ms
  is the KD loss: a softmax over all 8,192 tokens at every position, for both models, and its backward pass.
- A real smoke run (60 steps, `configs/smoke.yaml`) worked in both modes: the student's loss went 8.37 → 5.48, and
  in KD mode the KD loss fell too (2.46 → 1.46).

**Choice: option A, 4 layers / 4 heads / width 256 (5,322,240 parameters, 2.6× smaller than the teacher).**
- It is the guide's example size.
- It has the same shape as Project 1's `baseline` (test perplexity 9.68 after 41M tokens of the 100K-story
  subset), so the scratch student has a known reference: it should do at least as well.
- Option B would save only ~1.5 h over the whole project (3.1× smaller) and has no reference run.
- Total training plan with A: scratch 1.3 h + 3 distilled runs × 2.0 h ≈ **7.3 h**.

**Fair comparison is built in:** every run uses the same `configs/student.yaml`. The scratch and distilled
students differ only by `--mode` (and T, α), so data, data order (seed 1337), steps, batch, learning rate and
dropout are guaranteed identical. With seed 1337 the students also see their training windows in the same order
the teacher did, so they learn from stories the teacher was trained on.

---

## Phase 5 — Student-from-scratch baseline

**Learn:** before distillation, train the student the normal way, with only next-token cross-entropy. Without
this baseline we couldn't tell whether a good distilled student is good **because of the teacher** or just
because the architecture is good.
```
TinyStories → small student → cross-entropy only → student-from-scratch baseline
```
Fair-comparison rules: the baseline and the distilled student use the same data, steps (3,000), batch size,
learning-rate schedule and seed. The **only** difference is the loss.

**Files:** `configs/student.yaml`, `kd/train_student.py` (with `--mode scratch`), `scripts/plot_curves.py`

**Run:**
```
python -m kd.train_student --config configs\smoke.yaml --mode scratch        # ~15 s: loss must go down
python -m kd.train_student --config configs\student.yaml --mode scratch      # ~1.3 h
```
Every validation check (every 250 steps) also measures **KL(teacher ‖ student)** at T = 1: how far the student's
guesses are from the teacher's. The scratch student never sees the teacher, so this shows how close a student
gets on its own.

**Check:** `checkpoints/student_scratch/model_best.pt` exists; validation loss fell and stayed close to train loss
(no overfitting); validation perplexity is saved in `results/metrics/student_scratch/`.

**Result (2026-11-10):** done in **77.5 min** (estimate 77 min), 3,000 steps, 49.2M tokens, peak RAM 1,712 MB
(this includes the frozen teacher, loaded only to measure the KL column). Average ~10,900 tokens/s.

| Step | 250 | 500 | 1,000 | 1,500 | 2,000 | 2,500 | 3,000 |
|---|---|---|---|---|---|---|---|
| Train CE | 3.81 | 3.26 | 2.80 | 2.61 | 2.50 | 2.41 | 2.38 |
| Val loss | 3.73 | 3.17 | 2.72 | 2.50 | 2.39 | 2.32 | **2.29** |
| Val perplexity | 41.8 | 23.7 | 15.2 | 12.2 | 10.9 | 10.2 | **9.87** |
| KL(teacher ‖ student) | 1.93 | 1.36 | 0.90 | 0.69 | 0.57 | 0.51 | **0.48** |

- **Validation loss fell at every check** and stayed just below train loss (dropout is on only in training), so
  there is no overfitting. The student saw only 3.4% of the training windows, each once, so it can't memorise.
- **Sanity check passed:** Project 1's `baseline` has the same shape and reached 9.68 after 41M tokens. This
  student reaches 9.87 after 49M tokens. The small difference is expected: Project 1's run went over 100K stories
  about twice, while this one sees new stories at every step.
- **Gap to the teacher:** 9.87 vs 5.74, i.e. +72% perplexity. That gap is what distillation tries to close.
- **The student gets closer to the teacher on its own** (KL 1.93 → 0.48), just by learning the same data. The
  distilled students must beat this curve to show the teacher helps.
- It was still improving at the end (2.300 → 2.289 over the last 250 steps), as expected for a short budget.
- The validation numbers here use the first 800 windows (100 batches). Final test scores on the full sets are in
  Phase 10.
- Checkpoints: `checkpoints/student_scratch/model_best.pt` (step 3,000), `model_last.pt`, `model_step_{1000,2000,3000}.pt`.

---

## Phase 6 — Understand teacher soft targets

**Learn:** the teacher gives a score (logit) for each of the 8,192 tokens at every position. **Temperature T**
softens them before softmax:
```
p_teacher = softmax(z_teacher / T)        p_student = softmax(z_student / T)
```
- T = 1: the teacher's real probabilities (after "there was a little": girl 75%, boy 16%, bird 1.4%)
- T = 2: flatter: the top token loses share and the "second-best guesses" become more visible
- T = 4: flatter still: more information about rare tokens, but also more noise

Phase 6 prints the real numbers at each temperature.

The student learns this whole distribution rather than only the one hard target. We start with T = 2.

**Files:** `scripts/show_soft_targets.py` (learning aid; prints the teacher's top tokens at T = 1, 2, 4 for a prompt,
and the entropy of each distribution)

**Run:**
```
python scripts\show_soft_targets.py --prompt "Once upon a time, there was a little" --prompt "Tom was sad because" --prompt "Lily and her mom went to the"
```

**Check:** you can explain, using the printed numbers, what raising T does to the teacher's probabilities, and
why the hard label alone hides information.

**Result (2026-11-10):** done. Saved in `results/metrics/soft_targets.json`.

| After "Once upon a time, there was a little" | T = 1 | T = 2 | T = 4 |
|---|---|---|---|
| girl | 68.4% | 9.2% | 0.4% |
| boy | 19.2% | 4.9% | 0.3% |
| bird | 2.3% | 1.7% | 0.2% |
| bunny | 1.6% | 1.4% | 0.2% |
| dog | 1.0% | 1.1% | 0.2% |
| Entropy (max 13 bits) | 1.88 | 10.45 | 12.75 |
| "Effective choices" (2^entropy) | 3.7 | 1,400 | 6,885 |
| Probability outside the top 10 | 4.8% | **77.5%** | **98.0%** |

| | "Tom was sad because" T = 1 / 2 / 4 | "Lily and her mom went to the" T = 1 / 2 / 4 |
|---|---|---|
| Top token | he 81.5% / 21.9% / 0.8% | park 39.7% / 4.4% / 0.3% |
| Second | his 13.2% / 8.8% / 0.5% | store 21.2% / 3.2% / 0.3% |
| Entropy (bits) | 0.99 / 8.63 / 12.65 | 3.68 / 10.87 / 12.74 |
| Outside the top 10 | 0.6% / 58.5% / 97.2% | 16.4% / 83.5% / 98.2% |

What the numbers show:
- **The hard label hides a lot.** Training on the real text says "girl 100%". The teacher says girl 68%, boy 19%,
  and that bird, bunny, dog and fish are all sensible. After "went to the", the teacher spreads its guess over
  park, store, zoo, beach, shop and market (13 effective choices). That is knowledge about which words *fit*,
  and the hard label throws it away.
- **The order never changes.** Temperature keeps the ranking (girl > boy > bird …) but squeezes the gaps.
- **With 8,192 tokens, temperature flattens much more than expected.** At T = 2 most of the probability (58–84%)
  moves to tokens *outside* the top 10, because thousands of tiny probabilities each grow a little and add up.
  At T = 4 the distribution is almost uniform (12.7 of 13 bits): the top token keeps less than 1%.
- **Prediction for Phase 9:** at T = 4 most of the KD signal is about how the teacher ranks thousands of unlikely
  tokens, which is closer to noise than knowledge. So T = 4 may help less than T = 2, and T = 1 (the teacher's
  real probabilities) may be competitive. The ablations will tell.

---

## Phase 7 — Implement the distillation loss

**Learn:** two learning signals:
1. **Hard-label loss (CE):** normal cross-entropy against the real next token from TinyStories.
2. **Soft-target loss (KD):** KL divergence, i.e. how far the student's softened distribution is from the
   teacher's.
```
L = α · L_CE + (1 − α) · T² · L_KD
```
- α balances "learn from the real data" against "imitate the teacher". α = 1 is the baseline from Phase 5.
- **T²** keeps the KD gradients at the same scale when T changes. Softening by T shrinks the gradients by
  about 1/T², so without T² a higher temperature would quietly mean "less teacher".
- First setting: **α = 0.5, T = 2**.

**Files:** `kd/distillation.py` (`distillation_loss`), `tests/test_distillation.py`

**Run:**
```
python -m pytest tests\test_distillation.py -v -s
```

**Check:** the tests show that (1) KD = 0 when student and teacher logits are identical, (2) α = 1 gives exactly the
normal CE loss, (3) KD is never negative, (4) gradients reach the student but never the teacher, and (5) the
combined loss can be computed for one real training batch (CE, KD and total printed).

**Result (2026-11-10):** done. `tests/test_distillation.py`: **7 passed**. All 12 project tests pass.

| Test | Result |
|---|---|
| KD = 0 when student = teacher (T = 1, 2, 4) | ✅ |
| α = 1 gives exactly plain cross-entropy (and "no teacher" gives the same) | ✅ |
| KD never negative (20 random cases × 3 temperatures) | ✅ |
| KD equals the formula written out by hand: Σ p_t (log p_t − log p_s), averaged over positions | ✅ |
| total = α·CE + (1 − α)·T²·KD for all 5 planned settings | ✅ |
| gradients reach the student, never the teacher | ✅ |
| one real training batch | ✅ (numbers below) |

One real batch (two 256-token training windows), **untrained** student vs the real teacher, α = 0.5:

| | CE | KD | T²·KD | Total |
|---|---|---|---|---|
| T = 1 | 9.052 | 7.560 | 7.560 | 8.306 |
| T = 2 | 9.052 | 2.647 | 10.586 | 9.819 |
| T = 4 | 9.052 | 0.246 | 3.931 | 6.491 |
| Teacher on the same batch | 1.667 | 0 | | |

- The untrained student's CE is 9.05, close to random guessing (ln 8,192 = 9.01), as it should be.
- **Why T² is needed:** the raw KD falls from 7.56 to 0.25 as T goes from 1 to 4, because softened distributions
  are closer to each other. Without T², the T = 4 run would get almost no teacher signal. With T² the three
  settings stay in the same range (3.9–10.6).
- The code is `kd/distillation.py`: `kd_kl()` computes the KL, and `distillation_loss()` returns
  (total, CE, KD) so both parts can be logged separately during training.

---

## Phase 8 — Train the distilled student

**Learn:** one training step:
```
batch → teacher forward (no gradient) → teacher logits ─┐
      → student forward               → student logits ─┴→ CE + KD loss
      → backpropagate STUDENT only → AdamW update
```
Same training setup as Project 1 and Phase 5: AdamW, warmup + cosine learning rate, gradient clipping,
validation every 250 steps, best-checkpoint saving. Logged: total loss, CE loss, KD loss, validation loss,
perplexity, KL to the teacher, RAM, tokens/s, time.

On CPU the teacher's forward pass plus the KD loss make each step about 1.55× slower (Phase 4). Saving all
teacher outputs in advance isn't practical: 8,192 numbers per token for 49M tokens would be about 1.6 TB.

**Files:** `configs/student.yaml`, `kd/train_student.py` (with `--mode kd`), `kd/distillation.py`

**Run:**
```
python -m kd.train_student --config configs\smoke.yaml --mode kd                                      # ~20 s
python -m kd.train_student --config configs\student.yaml --mode kd --temperature 2 --alpha 0.5        # ~2 h
```

**Check:** `checkpoints/student_kd_T2_a0.5/model_best.pt` exists; CE, KD and validation loss all go down; the
log has every column listed above.

**Result:** ✅ Done (2026-11-10). Ran in 7,151 s (1 h 59 m) at ~7,000 tokens/s, peak RAM 1,959 MB. This matches the
Phase 4 estimate of 2 h 00 m. All checks pass: `model_best.pt` exists (the best checkpoint is from step 3000), and
CE, KD and validation loss all fell at every evaluation.

| Step | Scratch val PPL | **KD (T=2, α=0.5) val PPL** | Scratch KL to teacher | KD KL to teacher |
|---|---|---|---|---|
| 250 | 41.79 | 44.91 | 1.933 | 2.007 |
| 500 | 23.70 | 22.73 | 1.358 | 1.315 |
| 1000 | 15.18 | 13.20 | 0.905 | 0.751 |
| 1500 | 12.23 | 10.60 | 0.689 | 0.531 |
| 2000 | 10.88 | 9.55 | 0.573 | 0.425 |
| 2500 | 10.17 | 9.00 | 0.507 | 0.368 |
| **3000** | **9.87** | **8.79** | **0.477** | **0.345** |

- **Same steps, same data:** the distilled student ends at val perplexity 8.79 vs 9.87 (11% lower), and it is 28%
  closer to the teacher in KL. Distillation is slightly behind at step 250, when the soft targets of a 2-bit
  "almost uniform" student still dominate. It is ahead from step 500 on, and the gap keeps growing.
- **Reaching the baseline's final score:** KD gets to 9.87 at about step 1,900, roughly 37% fewer steps.
- **Same wall-clock time (a stricter test):** scratch used 4,650 s in total. At 4,650 s the KD run was between
  step 1750 (10.02) and step 2000 (9.55), so about 9.7. Even when the extra teacher cost is charged, KD is still
  slightly ahead. The advantage is small at equal time, though: KD costs 1.55× per step on CPU.
- KD loss (T² not included) fell from 0.69 to 0.18. Train CE (2.26) is higher than val loss (2.17) because train CE
  is measured with dropout on, not because of a problem.
- Test-set numbers come in Phase 10.

---

## Phase 9 — Distillation ablations

**Learn:** change one setting at a time to see what matters. Each run takes ~2.1 h (3,000 steps), so start
small: the baseline plus 2–3 settings.

| Experiment | Setting | Priority |
|---|---|---|
| Baseline student | no distillation (α = 1) | done in Phase 5 |
| KD-2 | T = 2, α = 0.5 | done in Phase 8 |
| KD-1 | T = 1, α = 0.5 | 1st |
| KD-3 | T = 4, α = 0.5 | 2nd |
| KD-4 | T = 2, α = 0.25 | 3rd, if time allows |
| KD-5 | T = 2, α = 0.75 | 4th, if time allows |

All ablations use the same 3,000-step budget as Phases 5 and 8, so every row of this table is comparable.

**Files:** `configs/student.yaml` (same for every run), `kd/train_student.py`, `scripts/plot_curves.py`

**Run:**
```
python -m kd.train_student --config configs\student.yaml --mode kd --temperature 1 --alpha 0.5
python -m kd.train_student --config configs\student.yaml --mode kd --temperature 4 --alpha 0.5
python scripts\plot_curves.py --run student_scratch --run student_kd_T1_a0.5 --run student_kd_T2_a0.5 --run student_kd_T4_a0.5
```

**Check:** a table and a plot show whether temperature and α change the student's validation perplexity.
Caveat: each setting is trained once (one seed), so differences smaller than a few hundredths of perplexity
can't be separated from run-to-run noise. Final scores use the full test set (Phase 10) to keep the
measurement noise itself small.

**Result:** ✅ Done (2026-11-11): temperature (T = 1, 2, 4) and α (0.25, 0.5, 0.75) were both tested. The α runs
were launched with `scripts/run_alpha.ps1` (same config, then the same evaluation). Plot: `results/figures/curves.png`; table: `results/metrics/ablation_table.md`.

Columns: validation set during training (100 batches of 8), then the **full test set** (Phase 10 method).

| Run | Best val PPL (train-time) | **Test PPL** | KL(teacher ‖ student), test | Same top-1 as teacher | Gap closed | Time |
|---|---|---|---|---|---|---|
| Scratch (no teacher) | 9.87 | **9.63** | 0.481 | 69.4% | — | 1.29 h |
| KD T = 1, α = 0.5 | 8.97 | **8.75** | 0.372 | 73.1% | 22.7% | 1.95 h |
| KD T = 2, α = 0.5 | 8.79 | **8.59** | 0.349 | 72.3% | 26.7% | 1.99 h |
| KD T = 4, α = 0.5 | 8.79 | **8.60** | 0.360 | 71.8% | 26.4% | 1.97 h |

- **Every temperature beats training without a teacher** by a wide margin (0.9–1.04 perplexity), much larger than
  the noise caveat above.
- **T = 2 and T = 4 tie** (8.59 vs 8.60, a difference too small to call with one seed). **T = 1 is clearly worse**
  (8.75): without softening, the KD signal is dominated by the teacher's top 1–2 tokens and says little about the
  rest of the distribution (in the Phase 6 example, at T = 1 the top two words hold 88% of the probability).
- My Phase 6 prediction ("T = 4 may help less than T = 2") was **wrong**: T = 4 helps just as much. The T² factor
  keeps its gradient the right size, and the "dark knowledge" in the tail is evidently still useful to a student
  this size.
- T = 1 has the highest top-1 agreement (73.1%) but the worst perplexity: matching the teacher's favourite word is
  not the same as matching its whole distribution, which is what perplexity rewards.
- **Choice for the main comparison: T = 2, α = 0.5.** It was chosen in advance (Phase 8) and is the guide's
  default, so the main result is not cherry-picked. The α runs below show it is not the best setting.

**α (how much weight the real-data CE loss gets; 1 − α goes to the teacher), all at T = 2:**

| α | Test PPL | Gap closed | KL(teacher ‖ student) | Same top-1 as teacher | Repetition flags, 60 error stories | Distinct-2 |
|---|---|---|---|---|---|---|
| 1.0 (scratch) | 9.63 | 0% | 0.481 | 69.4% | 4 | 0.753 |
| 0.75 | 9.08 | 14.1% | 0.414 | 70.6% | 8 | 0.759 |
| 0.5 | 8.59 | 26.7% | 0.349 | 72.3% | 12 | 0.743 |
| **0.25** | **8.43** | **30.7%** | **0.321** | 72.9% | 15 | 0.739 |
| teacher | 5.74 | — | 0 | 100% | 13 | 0.742 |

- **A clean dose–response: the more weight on the teacher, the better the student**, at every step of α. The
  differences (0.16–0.5 perplexity) are far larger than the noise between T = 2 and T = 4 (0.01). The **best
  setting found is T = 2, α = 0.25: test perplexity 8.43, 30.7% of the gap closed.**
- The teacher's soft targets are a **better training signal than the real next word** for this student. Each one
  is a full distribution over 8,192 tokens instead of a single correct answer, so every token teaches more.
- **Repetition rises with the teacher's weight as well** (4 → 8 → 12 → 15 flagged stories; teacher 13). This is
  the clearest evidence that the students copy the teacher's habits along with its knowledge. Since α = 0 was not
  tried, it is open whether pure distillation would be better still.
- Every run took the same ~2.0 h (6,980–7,120 tokens/s), so the setting changes quality, not cost.

---

## Phase 10 — Evaluate language-modelling quality

**Learn:** the three models are scored on **exactly the same** test stories with the same tokenizer, the same way
Project 1 did it: per token, non-overlapping 256-token windows, batch 16, fp32. Otherwise the numbers can't be
compared.

| Metric | Measures |
|---|---|
| Test loss, perplexity | next-token prediction quality |
| Distinct-1 / Distinct-2 | variety of words |
| Repetition rate (Rep-4 flag) | loops |
| EOS rate | whether stories finish |
| Made-up words | broken words |

**Files:** `kd/evaluate.py` (reuses Project 1's evaluation code), `configs/prompts.txt`

**Run:**
```
python -m kd.evaluate --model teacher
python -m kd.evaluate --ckpt checkpoints\student_scratch\model_best.pt
python -m kd.evaluate --ckpt checkpoints\student_kd_T2_a0.5\model_best.pt
```

**Check:** the teacher scores 5.74 again (same method); each model has an `eval_report.md`; you can say how much
of the gap between the baseline student and the teacher distillation closes:
`gap closed = (PPL_scratch − PPL_distilled) / (PPL_scratch − PPL_teacher)`.

**Result:** ✅ Done (2026-11-10). Full test set (10,995 stories, 2,264,064 scored tokens), same method as Project 1.
Teacher re-evaluation: test loss 1.7473 / **perplexity 5.7392 — identical** to Project 1, and its top-p 0.95
stories (Distinct-2 0.7424) and all 60 error-analysis stories are byte-for-byte the same as Project 1's.

| Metric | Teacher | Student from scratch | **Distilled student (T2, α0.5)** | Real stories |
|---|---|---|---|---|
| Test loss | 1.7473 | 2.2649 | **2.1506** | — |
| **Test perplexity** | **5.74** | **9.63** | **8.59** | — |
| Perplexity vs teacher | — | +67.8% | **+49.7%** | — |
| KL(teacher ‖ model) | 0 | 0.481 | **0.349** | — |
| Same top-1 token as teacher | 100% | 69.4% | **72.3%** | — |
| Distinct-2 (top-p 0.95, 40 stories) | 0.742 | 0.753 | 0.743 | 0.764 |
| Rep-4 rate | 0.026 | 0.014 | 0.029 | 0.016 |
| Flagged repetitive (40 stories) | 2.5% | 2.5% | 22.5% | 5% |
| Stories that end by themselves | 92.5% | 92.5% | 92.5% | 98.7% |
| Made-up words | 0.04% | 0.03% | 0.07% | 0.02% |

**Gap closed = (9.63 − 8.59) / (9.63 − 5.74) = 26.7%.** Distillation recovers about a quarter of the quality the
smaller model loses, with no extra parameters and no extra data, only a different loss.

- The repetition flag for T = 2 (9 of 40) is the one worse number. I read all 9 stories: they are mild phrase
  repeats ("He ran to the ball. He ran to the ball."), not loops. It is also noisy: T = 4, trained the same way,
  scored 0 of 40. On the larger 60-story error set the auto-repetition counts are teacher 13, T2 12, T4 11,
  scratch 4, T1 4. The distilled students repeat **about as often as the teacher** (they copy its habits), while
  the scratch student repeats less (see Phase 13 for what it does wrong instead).
- Variety (Distinct-2) is the same for all three models, within one standard deviation (~0.02).

---

## Phase 11 — Evaluate compression and efficient inference

**Learn:** this project is about efficiency as well as quality, so we measure the cost of each model:

| Metric | Question |
|---|---|
| Parameter count | How much smaller is the student? |
| Model file size (MB) | How much storage is saved? |
| Peak RAM (instead of GPU memory) | Does the student need less memory? |
| Latency (s per 200-token story) | How quickly does it write? |
| Tokens/s | How much throughput improves? |
| Perplexity change | How much quality is lost? |

```
Compression ratio = teacher parameters / student parameters
Speedup           = teacher latency / student latency
```
Speed is measured on an idle PC, with the same check for other CPU load as in Project 1 Phase 12.

**Files:** `scripts/benchmark_inference.py`

**Run:**
```
python scripts\benchmark_inference.py --model teacher --ckpt checkpoints\student_scratch\model_best.pt --ckpt checkpoints\student_kd_T2_a0.5\model_best.pt
```

**Check:** one table with all six metrics for the three models, marked reliable (other CPU load ≤ 15%).

**Result:** ✅ Done (2026-11-10). One subprocess per model on an idle PC; all three marked **reliable**.
File: `results/metrics/inference_benchmark.md`.

| Model | Parameters | Weights | Compression | Latency, 200 tokens | Generation tokens/s | Speedup | Scoring tokens/s (batch 16) | RAM added by model |
|---|---|---|---|---|---|---|---|---|
| Teacher | 13.89M | 53.0 MB | 1.00× | 1.04 s | 193 | 1.00× | 36,002 | 129 MB |
| Student scratch | 5.32M | 20.3 MB | 2.61× | 0.57 s | 352 | 1.83× | 99,329 | 94 MB |
| **Distilled student** | 5.32M | 20.3 MB | **2.61×** | **0.59 s** | **339** | **1.76×** | 86,838 | 95 MB |

- The two students have **identical** architecture, so their difference (1.83× vs 1.76×, 99k vs 87k) is
  measurement noise (about ±7–13%). Distillation changes the weights, not the cost.
- **Generation is ~1.8× faster, not 2.6×.** Writing one token at a time (batch 1) is dominated by per-token
  overhead (Python, sampling, launching small matrix multiplies), which does not shrink with the model. Bulk
  scoring, where the CPU does real matrix work, is **2.4–2.8× faster**, close to the parameter ratio.
- RAM: the model itself adds 129 MB for the teacher vs ~95 MB for a student. Most of a process's ~500 MB is
  Python and PyTorch, so the total only drops from 532 to 491 MB.
- Project 1 measured the teacher at 1.42 s / 141 tokens/s on another day. Today's 1.04 s / 193 is the same model
  under different background conditions, which is why all three models here were measured in one session and only
  the ratios are compared.

---

## Phase 12 — Compare text generation

**Learn:** same prompts, same decoding (top-p 0.95), same length limit (512) and same seeds for all three models.
Then only the model differs.
```
"Once upon a time there was a little girl"
        ├─→ Teacher
        ├─→ Baseline student
        └─→ Distilled student   → read and compare
```

**Files:** `scripts/compare_generations.py` (writes the three stories for each prompt side by side)

**Run:**
```
python scripts\compare_generations.py --run teacher --run student_scratch --run student_kd_T2_a0.5 --num_samples 2
```

**Check:** `results/generations/comparison.md` shows the three models' stories next to each other, and you have
written a few notes on coherence, grammar, repetition, endings and character consistency.

**Result:** ✅ Done (2026-11-10). `results/generations/comparison.md`: 8 prompts × 2 samples × 3 models, all with
the same seed (1343) and decoding (top-p 0.95). Notes from reading all 48 stories:

| | Teacher | Student from scratch | Distilled student |
|---|---|---|---|
| **Coherence** | follows one thread for most of the story; slips are local ("a man who liked his ball") | often surreal and jumpy: "the clock started to rot", "climbed up the down", a rhinoceros appears in a bird story | follows a plot shape (problem → reaction → resolution/moral) but objects and places drift |
| **Grammar** | mostly clean | the most broken sentences ("It made the ball fell into the air", "many metal") | fewer than scratch; slips are like the teacher's ("I'll try you to hold on") |
| **Repetition** | some ("They saw the bird and the tree" ×2) | least | teacher-like: lists of "They wished they had not…" and "They learned that…" |
| **Endings** | almost always a proper ending | ends, but often abruptly ("The train was gone forever and Lily was sad.") | ends, usually with a moral like the teacher's |
| **Characters** | stable for longer; occasional speaker slips | names collide early ("Tom heard Tom's cry", "Tom saw Tom and Sara") | still swaps names (Blue → Tweetie → Tweety → Sammy), but later in the story |

- The clearest difference is **style**: the distilled student writes *like the teacher*, with more dialogue and
  more "moral of the story" endings. Lengths are similar (average 230 / 237 / 217 new tokens for teacher /
  scratch / distilled). This is what matching the teacher's whole distribution should give.
- The clearest thing that **did not** transfer: keeping track of who is who over a whole story. Both students lose
  it; the teacher loses it less often (Phase 13 puts numbers on this).
- I read these unblinded, so these notes are impressions; the blind counts in Phase 13 are the evidence.

---

## Phase 13 — Error analysis

**Learn:** read 50–100 stories from each model and count the kinds of mistakes, with the **same rules as
Project 1 Phase 13**, so the teacher's numbers (contradictions 88%, entity errors 55%, grammar 37%) can be reused
and compared.

| Error | What to look for |
|---|---|
| Repetition (auto) | repeated words, phrases or events |
| Grammar | broken sentences or invented words |
| Coherence / contradiction | events don't connect, or clash with earlier facts |
| Entity consistency | names or characters change unexpectedly |
| Premature ending (auto) | the story stops too early |
| Teacher–student gap | the teacher gets it right where the student fails |

The key question: **which skills were lost when the teacher was compressed?**

**Files:** `results/generations/<run>/error_analysis.csv` (made by `kd/evaluate.py`), `scripts/error_counts.py`

**Run:** fill in the hand-labelled columns, then:
```
python scripts\error_counts.py --run teacher --run student_scratch --run student_kd_T2_a0.5
```

**Check:** you can report each error rate for each model and name the skills the students lose most.

**Result:** ✅ Done (2026-11-10). 60 stories per model (top-p 0.90, seed 1344, same prompts as Project 1).
Labels with a note per story: `scripts/error_labels/<run>.py`, written into
`results/generations/<run>/error_analysis.csv` by `scripts/error_counts.py`. Teacher labels: Project 1 (read only).

**How the labels were made (to avoid bias):** the 120 student stories were shuffled together under hidden ids and
labelled **blind**, with Project 1's strict rules, before I looked at which model wrote which
(`scripts/error_labels/_blind_labels.py` + `_blind_key.json`). To check that my labels today are on the same scale
as the teacher's labels from Project 1, I re-labelled 20 random teacher stories without looking at the old labels:
on the 16 I had not seen this session, agreement was 16/16 (contradiction), 15/16 (grammar), 14/16 (entity), with
the same totals (17/17, 6/7, 10/10 yes). So the three columns below are comparable.

| Error (60 stories each) | Teacher | Student from scratch | **Distilled student** |
|---|---|---|---|
| Contradiction | 53/60 (88%) | 60/60 (100%) | 60/60 (100%) |
| Entity consistency | 33/60 (55%) | 51/60 (85%) | **44/60 (73%)** |
| Grammar | 22/60 (37%) | 48/60 (80%) | **40/60 (67%)** |
| Repetition (auto) | 13/60 (22%) | 4/60 (7%) | 12/60 (20%) |
| Incomplete (auto) | 4/60 (7%) | 5/60 (8%) | 4/60 (7%) |
| No error of any kind | 5/60 (8%) | 0/60 | 0/60 |

**Which skills were lost when the teacher was compressed?**
1. **Story-level consistency is lost first and most.** Every single student story (120/120) has a contradiction;
   the teacher manages 7 clean ones out of 60. Objects keep changing (key → necklace → bracelet with teeth), places
   jump (a park scene continues "in the kitchen"), and morals don't follow from the story.
2. **Keeping characters apart** (entity errors 55% → 73–85%): the students use one word for two characters ("The
   cat thanked the cat", "Tom and Tom"), give one character's line to another ("Tom said, 'Hi, I'm Max'") and
   introduce names from nowhere.
3. **Grammar** (37% → 67–80%): broken sentences and run-ons ("a big box of key", "what can I found").

**What distillation brought back:** the distilled student has fewer grammar errors (67% vs 80%) and fewer entity
errors (73% vs 85%) than the scratch student. The direction matches the perplexity result, but with 60 stories
per model the differences are **not statistically significant** (Fisher exact test p = 0.15 and 0.18). The gap
to the teacher on grammar is significant (p = 0.002). Distillation did **not** fix contradictions at all
(100% for both).

**The repetition difference is real but is the teacher's habit:** the distilled student repeats as often as the
teacher (20% vs 22%), the scratch student much less (7%, p = 0.06). Reading the flagged stories: the distilled
student's repeats are the teacher's "They wished they had not…" / "He liked his mom." lists, not loops.

**Caveats:** one labeller (me), labelled once; small samples, so treat each rate as ±10%. "Teacher–student
gap" story-by-story isn't possible: with the same prompt and seed, different models still write different
stories, so the comparison is between rates.

---

## Phase 14 — Final comparison

**Learn:** one table brings it all together. Success doesn't mean the student beats the teacher. It means the
**distilled student keeps clearly more of the teacher's ability than a same-size student trained without
distillation, while being cheaper to run.**

| Metric | Teacher | Student scratch | Distilled student |
|---|---|---|---|
| Parameters | 13.89M | — | — |
| Model size | 53.0 MB | — | — |
| Test loss / perplexity | 1.7473 / 5.74 | — | — |
| Peak RAM | — | — | — |
| Latency (200 tokens) | 1.42 s | — | — |
| Tokens/s | 141 | — | — |
| Distinct-2 (top-p 0.95) | 0.742 | — | — |
| Error rates | see Phase 13 | — | — |

**Files:** `scripts/final_table.py`, `README.md`

**Run:**
```
python scripts\final_table.py --distilled student_kd_T2_a0.5 --ablation student_kd_T1_a0.5 --ablation student_kd_T4_a0.5 --ablation student_kd_T2_a0.25 --ablation student_kd_T2_a0.75
```

**Check:** `results/final_comparison.md` is filled in from the saved results (no hand-copied numbers), and
`README.md` gives the answer to the main question in a few sentences.

**Result:** ✅ Done (2026-11-11). `results/final_comparison.md` (and `.json`) is generated by
`scripts/final_table.py` from the saved eval, training, speed and error-count files; the teacher's training time is
read from Project 1's `train_summary.json`. Command used:
```
python scriptsinal_table.py --distilled student_kd_T2_a0.5 --ablation student_kd_T1_a0.5 --ablation student_kd_T4_a0.5 --ablation student_kd_T2_a0.25 --ablation student_kd_T2_a0.75
```

| Metric | Teacher | Student scratch | Distilled student (T2, α0.5) |
|---|---|---|---|
| Parameters | 13.89M | 5.32M | 5.32M |
| Model size (fp32) | 53.0 MB | 20.3 MB | 20.3 MB |
| Test loss / perplexity | 1.7473 / **5.74** | 2.2649 / **9.63** | 2.1506 / **8.59** |
| Peak RAM, generating | 532 MB | 491 MB | 491 MB |
| Latency (200 tokens) | 1.04 s | 0.57 s | 0.59 s |
| Tokens/s (generation) | 193 | 352 | 339 |
| Distinct-2 (top-p 0.95) | 0.742 | 0.753 | 0.743 |
| Contradiction / entity / grammar errors | 88% / 55% / 37% | 100% / 85% / 80% | 100% / 73% / 67% |
| Training (CPU) | 10,000 steps, 9.9 h | 3,000 steps, 1.3 h | 3,000 steps, 2.0 h |

The teacher's latency and tokens/s are from this project's Phase 11 benchmark (all three models in one session),
not Project 1's 1.42 s / 141, which was measured on another day.

**Was the project a success, by the definition above?** Yes. With the same size, data and steps, the distilled
student keeps clearly more of the teacher's ability than the student trained alone (perplexity +49.7% vs +67.8%
over the teacher; closer in KL and top-1; fewer grammar and character errors), and it runs ~1.8× faster than the
teacher in 38% of the memory for weights. The best setting found (α = 0.25) closes 30.7% of the gap.
What it does **not** do is keep the teacher's story-level consistency: that is the skill lost to compression.
The answer to the main question is in `README.md`.

---

## Reference

**Folder layout** (as built)
```
configs/      student.yaml (every student run), smoke.yaml
kd/           __init__.py (paths to project1), teacher.py, model.py, distillation.py, train_student.py, evaluate.py
scripts/      check_teacher.py, env_report.py, benchmark.py, show_soft_targets.py, plot_curves.py,
              benchmark_inference.py, compare_generations.py, error_counts.py, final_table.py,
              run_remaining.ps1, run_alpha.ps1, error_labels/ (Phase 13 hand labels)
tests/        test_teacher_frozen.py, test_distillation.py
checkpoints/  student_scratch/, student_kd_T<T>_a<α>/            (not in git)
results/      metrics/, generations/, figures/, final_comparison.md
docs/         Project_2_Knowledge_Distillation_Guide.docx            (not in git)
```

**Rules**
- Run everything from the `project2` folder with `.venv` active.
- **Never change anything in `..\project1\`.** Project 2 only reads the teacher and data, and checks the teacher's
  hashes before use.
- Always use the teacher's `tokenizer.json`, and the same validation and test sets.
- The baseline and distilled students differ **only** in the loss: same data, steps, batch, schedule and seed.
- Seeds are fixed (`seed` in each config), so runs are repeatable.
- On Windows keep `num_workers: 0` in the configs.
- Plain PyTorch only for the models (no `transformers` model classes).
