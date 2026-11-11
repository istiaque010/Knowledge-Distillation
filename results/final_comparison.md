# Final comparison

| Metric | Teacher | Student from scratch | Distilled student (T2_a0.5) |
|---|---|---|---|
| Parameters | 13.89M | 5.32M | 5.32M |
| Compression vs teacher | 1.00× | 2.61× | 2.61× |
| Weights on disk (fp32) | 53.0 MB | 20.3 MB | 20.3 MB |
| **Test perplexity** | **5.74** | **9.63** | **8.59** |
| Test loss | 1.7473 | 2.2649 | 2.1506 |
| KL(teacher ‖ model), test | 0 | 0.4809 | 0.3494 |
| Same top-1 token as teacher | 100% | 69.4% | 72.3% |
| Distinct-2, top-p 0.95 (real 0.764) | 0.742 | 0.753 | 0.743 |
| Flagged repetitive (real 5%) | 2.5% | 2.5% | 22.5% |
| Stories that end by themselves | 92.5% | 92.5% | 92.5% |
| Made-up words (real 0.02%) | 0.04% | 0.03% | 0.07% |
| Contradiction (hand-labelled) | 88% | 100% | 100% |
| Entity consistency errors | 55% | 85% | 73% |
| Grammar errors | 37% | 80% | 67% |
| Stories with no error | 5/60 | 0/60 | 0/60 |
| Peak RAM, generating | 532 MB | 491 MB | 491 MB |
| Latency, 200-token story | 1.04 s | 0.57 s | 0.59 s |
| Generation tokens/s | 193 | 352 | 339 |
| Speedup (generation) | 1.00× | 1.83× | 1.76× |
| Scoring tokens/s (batch 16) | 36,002 | 99,329 | 86,838 |
| Training (CPU) | 10,000 steps, 164M tokens, 9.9 h | 3,000 steps, 49M tokens, 1.3 h | 3,000 steps, 49M tokens, 2.0 h |

- Perplexity vs teacher: scratch **+67.8%**, distilled **+49.7%**.
- Gap closed by distillation = (PPL_scratch − PPL_distilled) / (PPL_scratch − PPL_teacher) = (9.63 − 8.59) / (9.63 − 5.74) = **26.7%**.

## All students (same 3,000-step budget)

| Run | Test perplexity | KL(teacher ‖ model) | Same top-1 as teacher | Distinct-2 | Training time |
|---|---|---|---|---|---|
| student_scratch | 9.63 | 0.4809 | 69.4% | 0.753 | 1.3 h |
| student_kd_T2_a0.5 | 8.59 | 0.3494 | 72.3% | 0.743 | 2.0 h |
| student_kd_T1_a0.5 | 8.75 | 0.3717 | 73.1% | 0.734 | 2.0 h |
| student_kd_T4_a0.5 | 8.60 | 0.3604 | 71.8% | 0.745 | 2.0 h |
| student_kd_T2_a0.25 | 8.43 | 0.3212 | 72.9% | 0.739 | 2.0 h |
| student_kd_T2_a0.75 | 9.08 | 0.4144 | 70.6% | 0.759 | 2.0 h |
