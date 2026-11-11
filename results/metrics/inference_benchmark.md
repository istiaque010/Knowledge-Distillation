| Model | Parameters | Weights (MB) | Compression | Latency, 200 tokens | Generation tokens/s | Speedup | Scoring tokens/s | Peak RAM (MB) | RAM added by model (MB) | Reliable |
|---|---|---|---|---|---|---|---|---|---|---|
| teacher | 13,891,584 | 53.0 | 1.00× | 1.04 s | 193 | 1.00× | 36,002 | 532 | 129 | yes |
| student_scratch | 5,322,240 | 20.3 | 2.61× | 0.57 s | 352 | 1.83× | 99,329 | 491 | 94 | yes |
| student_kd_T2_a0.5 | 5,322,240 | 20.3 | 2.61× | 0.59 s | 339 | 1.76× | 86,838 | 491 | 95 | yes |
