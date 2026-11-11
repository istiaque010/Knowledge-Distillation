"""Phase 7: the distillation loss L = alpha * CE + (1 - alpha) * T^2 * KL(teacher || student).

Run: python -m pytest tests/test_distillation.py -v -s      (-s shows the real-batch numbers)
"""
import pytest
import torch
import torch.nn.functional as F

import kd
from kd.distillation import distillation_loss, kd_kl
from kd.model import build_student
from kd.teacher import load_frozen_teacher, teacher_logits
from src.dataset import NextTokenDataset, load_tokens

V = 8192


def rand_logits(seed, shape=(2, 16, V), scale=3.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g) * scale


def test_kd_is_zero_when_student_equals_teacher():
    z = rand_logits(0)
    for T in (1.0, 2.0, 4.0):
        assert kd_kl(z, z.clone(), T).item() == pytest.approx(0.0, abs=1e-6)


def test_alpha_one_is_exactly_normal_cross_entropy():
    z_s, z_t = rand_logits(1), rand_logits(2)
    y = torch.randint(0, V, (2, 16), generator=torch.Generator().manual_seed(3))
    total, ce, kd_loss = distillation_loss(z_s, z_t, y, temperature=2.0, alpha=1.0)
    assert torch.equal(total, F.cross_entropy(z_s.reshape(-1, V), y.reshape(-1)))
    assert kd_loss.item() == 0.0
    total_none, _, _ = distillation_loss(z_s, None, y)  # no teacher at all = the scratch baseline
    assert torch.equal(total_none, total)


def test_kd_is_never_negative():
    for seed in range(20):
        for T in (1.0, 2.0, 4.0):
            assert kd_kl(rand_logits(seed), rand_logits(seed + 100), T).item() >= 0.0


def test_kd_matches_the_formula_written_out_by_hand():
    """KL(p_t || p_s) = sum_v p_t(v) * (log p_t(v) - log p_s(v)), then the mean over token positions."""
    z_s, z_t, T = rand_logits(4), rand_logits(5), 2.0
    log_ps, log_pt = F.log_softmax(z_s / T, -1), F.log_softmax(z_t / T, -1)
    by_hand = (log_pt.exp() * (log_pt - log_ps)).sum(-1).mean()
    assert kd_kl(z_s, z_t, T).item() == pytest.approx(by_hand.item(), rel=1e-5)


def test_total_combines_ce_and_kd_with_t_squared():
    z_s, z_t = rand_logits(6), rand_logits(7)
    y = torch.randint(0, V, (2, 16), generator=torch.Generator().manual_seed(8))
    for T, a in ((1.0, 0.5), (2.0, 0.5), (4.0, 0.5), (2.0, 0.25), (2.0, 0.75)):
        total, ce, kd_loss = distillation_loss(z_s, z_t, y, T, a)
        assert total.item() == pytest.approx(a * ce.item() + (1 - a) * T * T * kd_loss.item(), rel=1e-5)


def test_gradients_reach_the_student_but_never_the_teacher():
    z_s = rand_logits(9).requires_grad_(True)
    z_t = rand_logits(10).requires_grad_(True)  # even if the teacher's logits had a graph...
    y = torch.randint(0, V, (2, 16), generator=torch.Generator().manual_seed(11))
    total, _, _ = distillation_loss(z_s, z_t.detach(), y)  # ...training always passes them detached
    total.backward()
    assert z_s.grad is not None and z_s.grad.abs().sum() > 0
    assert z_t.grad is None


def test_combined_loss_on_one_real_training_batch():
    """Real teacher, untrained student, two real 256-token training windows."""
    ds = NextTokenDataset(load_tokens(kd.DATA_DIR, "train"), 256)
    x, y = (torch.stack(t) for t in zip(ds[0], ds[1]))
    teacher, _ = load_frozen_teacher()
    torch.manual_seed(0)
    student = build_student({"n_layer": 4, "n_head": 4, "n_embd": 256, "block_size": 256, "dropout": 0.1})
    z_s, _ = student(x)
    z_t = teacher_logits(teacher, x)
    print()
    for T, a in ((2.0, 0.5), (1.0, 0.5), (4.0, 0.5)):
        total, ce, kd_loss = distillation_loss(z_s, z_t, y, T, a)
        print(f"  untrained student, T={T:g} alpha={a:g}: CE {ce.item():.3f}  KD {kd_loss.item():.4f}  "
              f"T^2*KD {T * T * kd_loss.item():.3f}  total {total.item():.3f}")
        assert torch.isfinite(total)
    # an untrained student guesses ~uniformly: CE close to ln(8192) = 9.01
    assert 8.5 < ce.item() < 9.6
    # the teacher itself on the same batch: much lower CE, and KD(teacher, teacher) = 0
    _, ce_t, kd_t = distillation_loss(z_t, z_t, y, 2.0, 0.5)
    print(f"  teacher on the same batch: CE {ce_t.item():.3f}  KD {kd_t.item():.4f}")
    assert ce_t.item() < 2.5 and kd_t.item() == pytest.approx(0.0, abs=1e-6)
