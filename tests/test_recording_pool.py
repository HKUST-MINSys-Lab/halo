"""Contracts for the optional learned recording-level pool."""

from __future__ import annotations

import torch

from model.tokenizer.encoder import RecordingAttentionPool


def test_recording_pool_ignores_padded_tokens_and_backpropagates():
    torch.manual_seed(3)
    pool = RecordingAttentionPool(8, 2, 0.0).eval()
    tokens = torch.randn(2, 3, 8, requires_grad=True)
    valid = torch.tensor([[True, True, False], [True, False, False]])
    first = pool(tokens, valid)
    changed = tokens.detach().clone()
    changed[~valid] = torch.randn_like(changed[~valid]) * 1_000
    second = pool(changed, valid)
    torch.testing.assert_close(first.detach(), second, atol=1e-6, rtol=1e-6)
    first.square().mean().backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    assert pool.query.grad is not None
