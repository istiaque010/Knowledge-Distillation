"""Phase 4: the student model.

The student is the teacher's own GPT class with a smaller shape, so the only difference between the two models is
size: same code, same tokenizer, same vocabulary, same context length.
"""
from __future__ import annotations

from . import TEACHER_DIR  # noqa: F401  (puts the teacher package on sys.path)

from src.model import GPT, GPTConfig, count_parameters, load_checkpoint  # noqa: E402,F401

TEACHER_SHAPE = {"vocab_size": 8192, "block_size": 256}


def build_student(model_cfg: dict) -> GPT:
    """A GPT with the config's shape. Vocabulary and context must equal the teacher's."""
    cfg = GPTConfig(vocab_size=TEACHER_SHAPE["vocab_size"], **model_cfg)
    if cfg.block_size != TEACHER_SHAPE["block_size"]:
        raise ValueError("the student must use the teacher's context length (256)")
    if cfg.n_embd % cfg.n_head:
        raise ValueError(f"n_embd {cfg.n_embd} is not divisible by n_head {cfg.n_head}")
    return GPT(cfg)
