"""Phase 3: the teacher gives logits but can never learn.

Run: python -m pytest tests/test_teacher_frozen.py -v
"""
import pytest
import torch
import torch.nn.functional as F

import kd
from kd.teacher import load_frozen_teacher, teacher_logits
from src.dataset import NextTokenDataset, load_tokens
from src.model import GPT, GPTConfig


@pytest.fixture(scope="module")
def teacher():
    torch.manual_seed(0)
    t, _ = load_frozen_teacher()
    return t


@pytest.fixture(scope="module")
def batch():
    """Two real 256-token windows from the validation set."""
    ds = NextTokenDataset(load_tokens(kd.DATA_DIR, "val"), 256)
    x, y = zip(ds[0], ds[1])
    return torch.stack(x), torch.stack(y)


def test_teacher_gives_logits_for_a_real_batch(teacher, batch):
    x, _ = batch
    z = teacher_logits(teacher, x)
    assert z.shape == (2, 256, 8192)
    assert not z.requires_grad, "teacher logits must not carry a gradient"


def test_teacher_is_in_eval_mode_and_every_parameter_is_frozen(teacher):
    assert not teacher.training
    assert all(not m.training for m in teacher.modules())
    assert all(not p.requires_grad for p in teacher.parameters())


def test_calling_train_cannot_switch_dropout_back_on(teacher, batch):
    x, _ = batch
    teacher.train()  # a mistake a training loop could make
    assert not teacher.training
    assert torch.equal(teacher_logits(teacher, x), teacher_logits(teacher, x))


def test_same_batch_gives_identical_logits(teacher, batch):
    """Dropout really is off: the teacher's targets don't change from one call to the next."""
    x, _ = batch
    assert torch.equal(teacher_logits(teacher, x), teacher_logits(teacher, x))


def test_one_student_training_step_never_changes_the_teacher(teacher, batch):
    """Train a tiny student towards the teacher for one step: student weights move, teacher weights don't."""
    x, y = batch
    before = {k: v.clone() for k, v in teacher.state_dict().items()}
    torch.manual_seed(0)
    student = GPT(GPTConfig(n_layer=1, n_head=2, n_embd=64))
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3)
    s_before = student.tok_emb.weight.clone()

    z_t = teacher_logits(teacher, x)
    z_s, ce = student(x, y)
    kl = F.kl_div(F.log_softmax(z_s, -1), F.log_softmax(z_t, -1), log_target=True, reduction="batchmean")
    (ce + kl).backward()
    opt.step()

    assert all(p.grad is None for p in teacher.parameters()), "a teacher parameter received a gradient"
    for k, v in teacher.state_dict().items():
        assert torch.equal(v, before[k]), f"teacher weight {k} changed"
    assert not torch.equal(student.tok_emb.weight, s_before), "the student should have learned something"
