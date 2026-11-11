"""Phase 3: load the Project 1 teacher and freeze it.

During distillation the teacher only GIVES answers (logits); it never learns:
    teacher.eval()                 -> dropout off, so the same input always gives the same logits
    requires_grad = False          -> no gradients are stored for its weights, so they can never change
    torch.no_grad() at call time   -> no computation graph: saves memory and time

Usage:
    from kd.teacher import load_frozen_teacher, teacher_logits
    teacher, tok = load_frozen_teacher()
    z_t = teacher_logits(teacher, x)          # (batch, length, 8192), no gradient
"""
from __future__ import annotations

import hashlib
import os
import types

import torch

from . import TEACHER_DIR  # importing kd puts the teacher package on sys.path

from load_teacher import load_teacher  # noqa: E402  (teacher package)


def _expected_hash(rel: str) -> str:
    with open(os.path.join(TEACHER_DIR, "MANIFEST.sha256"), encoding="utf-8") as f:
        for line in f:
            digest, name = line.rstrip("\n").split("  ", 1)
            if name == rel:
                return digest
    raise KeyError(rel)


def check_teacher_file() -> None:
    """Refuse to use a teacher.pt that differs from the one Project 1 packaged and verified (Phase 1)."""
    h = hashlib.sha256()
    with open(os.path.join(TEACHER_DIR, "teacher.pt"), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != _expected_hash("teacher.pt"):
        raise RuntimeError("teacher.pt does not match MANIFEST.sha256; run scripts/check_teacher.py")


def freeze(model: torch.nn.Module) -> torch.nn.Module:
    """Put a model in eval mode, turn off its gradients, and stop .train() from switching dropout back on."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # A frozen teacher must never go back to training mode (dropout on = noisy targets), even if some code
    # calls .train() on it by mistake. eval() itself calls train(False), which is still allowed.
    model.train = types.MethodType(lambda self, mode=True: self, model)
    return model


def load_frozen_teacher(device: str = "cpu", check_hash: bool = True):
    """Returns (teacher, tokenizer). The teacher is in eval mode with every parameter frozen."""
    if check_hash:
        check_teacher_file()
    model, tok, _ = load_teacher(device)
    return freeze(model), tok


@torch.no_grad()
def teacher_logits(teacher: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Teacher scores for every vocabulary token at every position: (batch, length, vocab). No gradient."""
    logits, _ = teacher(x)
    return logits
