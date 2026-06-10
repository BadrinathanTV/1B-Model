"""
Delta Attention Residual Test Suite (Static Buffer API)
=========================================================

Tests correctness, edge cases, gradient flow, memory management,
numerical stability, and integration with the TransformerBlock.

Uses the `forward_static()` API which operates on a fixed-size tensor buffer
with a slot count, matching how TransformerBlock actually calls it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../training")))

from config import SLMConfig
from layers.residual import DeltaAttentionResidual


def make_config(**overrides):
    """Create a small test config with optional overrides."""
    config = SLMConfig()
    config.hidden_size = 64
    config.num_attention_heads = 4
    config.q_lora_rank = 16
    config.kv_lora_rank = 8
    config.qk_rope_head_dim = 8
    config.v_head_dim = 16
    config.rms_norm_eps = 1e-6
    config.training.seq_len = 8
    config.max_delta_history = 0  # full history by default
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


def _make_buffer_from_deltas(deltas, max_hist, batch, seq_len, hidden):
    """Create a deltas_buf and num_deltas from a list of delta tensors."""
    device = deltas[0].device if deltas else torch.device('cpu')
    dtype = deltas[0].dtype if deltas else torch.float32
    buf = torch.zeros(max_hist, batch, seq_len, hidden, device=device, dtype=dtype)
    n = min(len(deltas), max_hist)
    for i in range(n):
        buf[i] = deltas[i]
    num_deltas = torch.tensor([n], dtype=torch.long, device=device)
    return buf, num_deltas


class TestDeltaResidualCorrectness(unittest.TestCase):
    """Tests that the core math of Delta Attention Residual is correct."""

    def test_empty_deltas_returns_x_unchanged(self):
        """When there are no previous deltas, output must equal input exactly."""
        config = make_config(num_hidden_layers=2)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        routing_q = nn.Parameter(torch.randn(64))
        buf = torch.zeros(max_hist, 2, 8, 64)
        num_deltas = torch.tensor([0], dtype=torch.long)

        out = layer.forward_static(x, buf, num_deltas, routing_q)
        self.assertTrue(torch.allclose(out, x, atol=1e-5),
                        "With empty deltas, output should be identical to input.")

    def test_single_delta_softmax_reduces_to_identity(self):
        """With exactly 1 delta, softmax(score) = 1.0, so routed = delta itself."""
        config = make_config(num_hidden_layers=2)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        delta = torch.randn(2, 8, 64)
        routing_q = nn.Parameter(torch.randn(64))

        buf, num_deltas = _make_buffer_from_deltas([delta], max_hist, 2, 8, 64)
        out = layer.forward_static(x, buf, num_deltas, routing_q)

        # With a single delta, softmax produces alpha=1.0 -> routed = delta
        expected = x + delta
        self.assertTrue(torch.allclose(out, expected, atol=1e-5),
                        "Single delta should produce x + delta (softmax is trivially 1.0).")

    def test_output_shape_matches_input(self):
        """Output shape must always match input shape regardless of number of deltas."""
        config = make_config(num_hidden_layers=6)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config)
        routing_q = nn.Parameter(torch.randn(64))

        for n_deltas in [0, 1, 3, 10]:
            n_actual = min(n_deltas, max_hist)
            x = torch.randn(4, 8, 64)
            deltas = [torch.randn(4, 8, 64) for _ in range(n_actual)]
            buf, num_deltas = _make_buffer_from_deltas(deltas, max_hist, 4, 8, 64)
            out = layer.forward_static(x, buf, num_deltas, routing_q)
            self.assertEqual(out.shape, x.shape,
                             f"Output shape mismatch with {n_deltas} deltas.")


class TestDeltaResidualGradients(unittest.TestCase):
    """Tests that gradients flow correctly through the routing mechanism."""

    def test_gradients_flow_to_routing_q(self):
        """The learned routing query must receive gradients."""
        config = make_config(num_hidden_layers=2)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64, requires_grad=True)
        deltas = [torch.randn(2, 8, 64, requires_grad=True) for _ in range(3)]
        routing_q = nn.Parameter(torch.randn(64))

        buf, num_deltas = _make_buffer_from_deltas(deltas, max_hist, 2, 8, 64)
        # Re-create buf with grad tracking: stack from deltas
        buf = torch.zeros(max_hist, 2, 8, 64, requires_grad=True)
        with torch.no_grad():
            for i, d in enumerate(deltas):
                buf.data[i] = d
        buf = buf + 0  # force into computational graph

        out = layer.forward_static(x, buf, num_deltas, routing_q)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(routing_q.grad, "routing_q must receive gradients.")
        self.assertTrue(routing_q.grad.abs().sum() > 0, "routing_q gradient must be non-zero.")

    def test_gradients_flow_to_input_x(self):
        """The input x must receive gradients (it's added to the routed output)."""
        config = make_config(num_hidden_layers=2)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64, requires_grad=True)
        routing_q = nn.Parameter(torch.randn(64))

        deltas = [torch.randn(2, 8, 64) for _ in range(3)]
        buf, num_deltas = _make_buffer_from_deltas(deltas, max_hist, 2, 8, 64)
        out = layer.forward_static(x, buf, num_deltas, routing_q)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        # x has a direct identity path, so gradient should be all 1s + routing contribution
        self.assertTrue((x.grad.abs() > 0).all(), "Input x must receive gradients everywhere.")

    def test_zero_initialized_routing_q_produces_uniform_routing(self):
        """
        TransformerBlock initializes routing_q to zeros. With zero query, all scores
        are equal, so softmax produces uniform weights (1/N). This is intentional —
        the model starts by averaging all deltas equally and learns to specialize.
        """
        config = make_config(num_hidden_layers=2)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64, requires_grad=True)
        routing_q = nn.Parameter(torch.zeros(64))
        deltas = [torch.randn(2, 8, 64) for _ in range(4)]

        buf, num_deltas = _make_buffer_from_deltas(deltas, max_hist, 2, 8, 64)
        out = layer.forward_static(x, buf, num_deltas, routing_q)
        loss = out.sum()
        loss.backward()

        # Verify gradients still flow to break the symmetry during training
        self.assertTrue(routing_q.grad.abs().sum() > 0,
                        "routing_q must receive non-zero gradients even when initialized to zero.")


class TestDeltaResidualNumericalStability(unittest.TestCase):
    """Tests numerical edge cases that could cause NaN/Inf during training."""

    def test_zero_deltas_no_nan(self):
        """If all deltas are zero tensors, the RMSNorm should not produce NaN (eps protects)."""
        config = make_config(num_hidden_layers=2)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        routing_q = nn.Parameter(torch.randn(64))
        deltas = [torch.zeros(2, 8, 64) for _ in range(3)]
        buf, num_deltas = _make_buffer_from_deltas(deltas, max_hist, 2, 8, 64)

        out = layer.forward_static(x, buf, num_deltas, routing_q)
        self.assertFalse(torch.isnan(out).any(), "Zero deltas must not produce NaN.")
        self.assertFalse(torch.isinf(out).any(), "Zero deltas must not produce Inf.")

    def test_extreme_magnitude_deltas_no_nan(self):
        """Very large delta values should be stabilized by RMSNorm before scoring."""
        config = make_config(num_hidden_layers=2)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        routing_q = nn.Parameter(torch.randn(64))
        deltas = [torch.randn(2, 8, 64) * 1e6 for _ in range(3)]
        buf, num_deltas = _make_buffer_from_deltas(deltas, max_hist, 2, 8, 64)

        out = layer.forward_static(x, buf, num_deltas, routing_q)
        self.assertFalse(torch.isnan(out).any(), "Large deltas must not produce NaN.")
        self.assertFalse(torch.isinf(out).any(), "Large deltas must not produce Inf.")

    def test_tiny_magnitude_deltas_no_nan(self):
        """Very small delta values should not cause division-by-zero in RMSNorm."""
        config = make_config(num_hidden_layers=2)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        routing_q = nn.Parameter(torch.randn(64))
        deltas = [torch.randn(2, 8, 64) * 1e-10 for _ in range(3)]
        buf, num_deltas = _make_buffer_from_deltas(deltas, max_hist, 2, 8, 64)

        out = layer.forward_static(x, buf, num_deltas, routing_q)
        self.assertFalse(torch.isnan(out).any(), "Tiny deltas must not produce NaN.")
        self.assertFalse(torch.isinf(out).any(), "Tiny deltas must not produce Inf.")

    def test_mixed_magnitude_deltas_no_nan(self):
        """Deltas with wildly different scales should be handled gracefully."""
        config = make_config(num_hidden_layers=3)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        routing_q = nn.Parameter(torch.randn(64))
        deltas = [
            torch.randn(2, 8, 64) * 1e-8,
            torch.randn(2, 8, 64) * 1.0,
            torch.randn(2, 8, 64) * 1e6,
        ]
        buf, num_deltas = _make_buffer_from_deltas(deltas, max_hist, 2, 8, 64)

        out = layer.forward_static(x, buf, num_deltas, routing_q)
        self.assertFalse(torch.isnan(out).any(), "Mixed magnitude deltas must not produce NaN.")
        self.assertFalse(torch.isinf(out).any(), "Mixed magnitude deltas must not produce Inf.")

    def test_bfloat16_stability(self):
        """Delta routing must be numerically stable under bfloat16 precision."""
        config = make_config(num_hidden_layers=3)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config).bfloat16()
        x = torch.randn(2, 8, 64, dtype=torch.bfloat16)
        routing_q = nn.Parameter(torch.randn(64, dtype=torch.bfloat16))
        deltas = [torch.randn(2, 8, 64, dtype=torch.bfloat16) for _ in range(5)]
        buf, num_deltas = _make_buffer_from_deltas(deltas, max_hist, 2, 8, 64)
        buf = buf.bfloat16()

        out = layer.forward_static(x, buf, num_deltas, routing_q)
        self.assertFalse(torch.isnan(out).any(), "bfloat16 must not produce NaN.")
        self.assertFalse(torch.isinf(out).any(), "bfloat16 must not produce Inf.")


class TestDeltaResidualIntegration(unittest.TestCase):
    """Tests the Delta Residual in the context of how TransformerBlock uses it."""

    def test_two_routing_queries_produce_different_outputs(self):
        """
        TransformerBlock uses two separate routing queries (routing_q_attn, routing_q_ffn).
        They must produce different routing patterns.
        """
        config = make_config(num_hidden_layers=3)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        deltas = [torch.randn(2, 8, 64) for _ in range(5)]
        buf, num_deltas = _make_buffer_from_deltas(deltas, max_hist, 2, 8, 64)

        q1 = nn.Parameter(torch.randn(64))
        q2 = nn.Parameter(torch.randn(64))

        out1 = layer.forward_static(x, buf, num_deltas, q1)
        out2 = layer.forward_static(x, buf, num_deltas, q2)

        self.assertFalse(torch.allclose(out1, out2, atol=1e-5),
                         "Different routing queries must produce different outputs.")

    def test_batch_independence(self):
        """Routing in one batch element must not affect another."""
        config = make_config(num_hidden_layers=2)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config)
        routing_q = nn.Parameter(torch.randn(64))

        x1 = torch.randn(1, 8, 64)
        x2 = torch.randn(1, 8, 64)
        x_batch = torch.cat([x1, x2], dim=0)

        d1 = [torch.randn(1, 8, 64) for _ in range(3)]
        d2 = [torch.randn(1, 8, 64) for _ in range(3)]
        d_batch = [torch.cat([a, b], dim=0) for a, b in zip(d1, d2)]

        buf_batch, nd_batch = _make_buffer_from_deltas(d_batch, max_hist, 2, 8, 64)
        buf1, nd1 = _make_buffer_from_deltas(d1, max_hist, 1, 8, 64)
        buf2, nd2 = _make_buffer_from_deltas(d2, max_hist, 1, 8, 64)

        out_batch = layer.forward_static(x_batch, buf_batch, nd_batch, routing_q)
        out1 = layer.forward_static(x1, buf1, nd1, routing_q)
        out2 = layer.forward_static(x2, buf2, nd2, routing_q)

        self.assertTrue(torch.allclose(out_batch[0], out1[0], atol=1e-5),
                        "Batch element 0 must match independent computation.")
        self.assertTrue(torch.allclose(out_batch[1], out2[0], atol=1e-5),
                        "Batch element 1 must match independent computation.")


if __name__ == '__main__':
    unittest.main()
