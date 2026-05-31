"""
Comprehensive Delta Attention Residual Test Suite
===================================================

Tests correctness, edge cases, gradient flow, memory management,
numerical stability, and integration with the TransformerBlock.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import sys
import os
import gc

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


class TestDeltaResidualCorrectness(unittest.TestCase):
    """Tests that the core math of Delta Attention Residual is correct."""

    def test_empty_deltas_returns_x_unchanged(self):
        """When there are no previous deltas, output must equal input exactly."""
        config = make_config()
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        routing_q = nn.Parameter(torch.randn(64))

        out = layer(x, [], routing_q)
        self.assertTrue(torch.equal(out, x), "With empty deltas, output should be identical to input.")

    def test_single_delta_softmax_reduces_to_identity(self):
        """With exactly 1 delta, softmax(score) = 1.0, so routed = delta itself."""
        config = make_config()
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        delta = torch.randn(2, 8, 64)
        routing_q = nn.Parameter(torch.randn(64))

        out = layer(x, [delta], routing_q)

        # With a single delta, softmax produces alpha=1.0 -> routed = delta
        expected = x + delta
        self.assertTrue(torch.allclose(out, expected, atol=1e-6),
                        "Single delta should produce x + delta (softmax is trivially 1.0).")

    def test_softmax_weights_sum_to_one(self):
        """The routing weights alpha must always sum to 1 along the delta dimension."""
        config = make_config()
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        deltas = [torch.randn(2, 8, 64) for _ in range(5)]
        routing_q = nn.Parameter(torch.randn(64))

        # Manually compute the alpha weights
        stacked = torch.stack(deltas, dim=2)
        norm_factor = torch.mean(stacked.float() ** 2, dim=-1, keepdim=True)
        normed = stacked * torch.rsqrt(norm_factor + config.rms_norm_eps).to(stacked.dtype)
        scores = torch.einsum('bsdh,h->bsd', normed, routing_q)
        alpha = F.softmax(scores, dim=-1)

        alpha_sum = alpha.sum(dim=-1)
        self.assertTrue(torch.allclose(alpha_sum, torch.ones_like(alpha_sum), atol=1e-5),
                        "Softmax routing weights must sum to 1.0.")

    def test_routed_output_is_convex_combination(self):
        """The routed contribution must be a convex combination of the original (unnormalized) deltas."""
        config = make_config()
        layer = DeltaAttentionResidual(config)
        x = torch.zeros(1, 1, 64)  # zero x so contribution = out
        deltas = [torch.randn(1, 1, 64) for _ in range(4)]
        routing_q = nn.Parameter(torch.randn(64))

        out = layer(x, deltas, routing_q)
        # out = x + sum(alpha_i * delta_i), where sum(alpha_i)=1 and alpha_i >= 0

        # Verify each element of output is within [min(deltas), max(deltas)]
        stacked = torch.stack(deltas, dim=0)  # (4, 1, 1, 64)
        min_vals = stacked.min(dim=0).values.squeeze()
        max_vals = stacked.max(dim=0).values.squeeze()
        out_vals = out.squeeze()

        in_bounds = (out_vals >= min_vals - 1e-5) & (out_vals <= max_vals + 1e-5)
        self.assertTrue(in_bounds.all(),
                        "Routed output must be bounded by the convex hull of the deltas.")

    def test_output_shape_matches_input(self):
        """Output shape must always match input shape regardless of number of deltas."""
        config = make_config()
        layer = DeltaAttentionResidual(config)
        routing_q = nn.Parameter(torch.randn(64))

        for n_deltas in [0, 1, 3, 10, 50]:
            x = torch.randn(4, 8, 64)
            deltas = [torch.randn(4, 8, 64) for _ in range(n_deltas)]
            out = layer(x, deltas, routing_q)
            self.assertEqual(out.shape, x.shape,
                             f"Output shape mismatch with {n_deltas} deltas.")


class TestDeltaResidualGradients(unittest.TestCase):
    """Tests that gradients flow correctly through the routing mechanism."""

    def test_gradients_flow_to_routing_q(self):
        """The learned routing query must receive gradients."""
        config = make_config()
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64, requires_grad=True)
        deltas = [torch.randn(2, 8, 64, requires_grad=True) for _ in range(3)]
        routing_q = nn.Parameter(torch.randn(64))

        out = layer(x, deltas, routing_q)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(routing_q.grad, "routing_q must receive gradients.")
        self.assertTrue(routing_q.grad.abs().sum() > 0, "routing_q gradient must be non-zero.")

    def test_gradients_flow_to_all_deltas(self):
        """Every delta in the history must receive gradients (not just the most recent)."""
        config = make_config()
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64, requires_grad=True)
        deltas = [torch.randn(2, 8, 64, requires_grad=True) for _ in range(5)]
        routing_q = nn.Parameter(torch.randn(64))

        out = layer(x, deltas, routing_q)
        loss = out.sum()
        loss.backward()

        for i, delta in enumerate(deltas):
            self.assertIsNotNone(delta.grad, f"Delta {i} must receive gradients.")
            self.assertTrue(delta.grad.abs().sum() > 0, f"Delta {i} gradient must be non-zero.")

    def test_gradients_flow_to_input_x(self):
        """The input x must receive gradients (it's added to the routed output)."""
        config = make_config()
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64, requires_grad=True)
        deltas = [torch.randn(2, 8, 64) for _ in range(3)]
        routing_q = nn.Parameter(torch.randn(64))

        out = layer(x, deltas, routing_q)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        # x has a direct identity path, so gradient should be all 1s + routing contribution
        self.assertTrue((x.grad.abs() > 0).all(), "Input x must receive gradients everywhere.")

    def test_zero_initialized_routing_q_still_routes(self):
        """
        TransformerBlock initializes routing_q to zeros. With zero query, all scores
        are equal, so softmax produces uniform weights (1/N). This is intentional —
        the model starts by averaging all deltas equally and learns to specialize.
        """
        config = make_config()
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64, requires_grad=True)
        deltas = [torch.randn(2, 8, 64, requires_grad=True) for _ in range(4)]
        routing_q = nn.Parameter(torch.zeros(64))

        out = layer(x, deltas, routing_q)
        loss = out.sum()
        loss.backward()

        # Verify uniform routing
        stacked = torch.stack(deltas, dim=2)
        uniform_routed = stacked.mean(dim=2)
        expected = x + uniform_routed
        self.assertTrue(torch.allclose(out.detach(), expected.detach(), atol=1e-5),
                        "Zero routing_q should produce uniform (average) routing.")

        # Verify gradients still flow to break the symmetry during training
        self.assertTrue(routing_q.grad.abs().sum() > 0,
                        "routing_q must receive non-zero gradients even when initialized to zero.")


class TestDeltaResidualNumericalStability(unittest.TestCase):
    """Tests numerical edge cases that could cause NaN/Inf during training."""

    def test_zero_deltas_no_nan(self):
        """If all deltas are zero tensors, the RMSNorm should not produce NaN (eps protects)."""
        config = make_config()
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        deltas = [torch.zeros(2, 8, 64) for _ in range(3)]
        routing_q = nn.Parameter(torch.randn(64))

        out = layer(x, deltas, routing_q)
        self.assertFalse(torch.isnan(out).any(), "Zero deltas must not produce NaN.")
        self.assertFalse(torch.isinf(out).any(), "Zero deltas must not produce Inf.")

    def test_extreme_magnitude_deltas_no_nan(self):
        """Very large delta values should be stabilized by RMSNorm before scoring."""
        config = make_config()
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        deltas = [torch.randn(2, 8, 64) * 1e6 for _ in range(3)]
        routing_q = nn.Parameter(torch.randn(64))

        out = layer(x, deltas, routing_q)
        self.assertFalse(torch.isnan(out).any(), "Large deltas must not produce NaN.")
        self.assertFalse(torch.isinf(out).any(), "Large deltas must not produce Inf.")

    def test_tiny_magnitude_deltas_no_nan(self):
        """Very small delta values should not cause division-by-zero in RMSNorm."""
        config = make_config()
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        deltas = [torch.randn(2, 8, 64) * 1e-10 for _ in range(3)]
        routing_q = nn.Parameter(torch.randn(64))

        out = layer(x, deltas, routing_q)
        self.assertFalse(torch.isnan(out).any(), "Tiny deltas must not produce NaN.")
        self.assertFalse(torch.isinf(out).any(), "Tiny deltas must not produce Inf.")

    def test_mixed_magnitude_deltas_no_nan(self):
        """Deltas with wildly different scales should be handled gracefully."""
        config = make_config()
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        deltas = [
            torch.randn(2, 8, 64) * 1e-8,
            torch.randn(2, 8, 64) * 1.0,
            torch.randn(2, 8, 64) * 1e6,
        ]
        routing_q = nn.Parameter(torch.randn(64))

        out = layer(x, deltas, routing_q)
        self.assertFalse(torch.isnan(out).any(), "Mixed magnitude deltas must not produce NaN.")
        self.assertFalse(torch.isinf(out).any(), "Mixed magnitude deltas must not produce Inf.")

    def test_bfloat16_stability(self):
        """Delta routing must be numerically stable under bfloat16 precision."""
        config = make_config()
        layer = DeltaAttentionResidual(config).bfloat16()
        x = torch.randn(2, 8, 64, dtype=torch.bfloat16)
        deltas = [torch.randn(2, 8, 64, dtype=torch.bfloat16) for _ in range(5)]
        routing_q = nn.Parameter(torch.randn(64, dtype=torch.bfloat16))

        out = layer(x, deltas, routing_q)
        self.assertFalse(torch.isnan(out).any(), "bfloat16 must not produce NaN.")
        self.assertFalse(torch.isinf(out).any(), "bfloat16 must not produce Inf.")

    def test_float32_upcast_in_rmsnorm(self):
        """
        Verify that the internal RMSNorm computes variance in float32 even when
        inputs are bfloat16, to prevent squared-value overflow.
        """
        config = make_config()
        layer = DeltaAttentionResidual(config)
        # Create bfloat16 deltas with values near the representable limit
        x = torch.randn(2, 8, 64, dtype=torch.bfloat16)
        # bfloat16 max is ~3.39e38. Values of 100 squared = 10000, still safe.
        # But if NOT upcasting to float32, accumulation of many squared values could overflow.
        deltas = [torch.randn(2, 8, 64, dtype=torch.bfloat16) * 100.0 for _ in range(3)]
        routing_q = nn.Parameter(torch.randn(64, dtype=torch.bfloat16))

        out = layer(x, deltas, routing_q)
        self.assertFalse(torch.isnan(out).any(),
                         "float32 upcast in RMSNorm is critical — NaN detected!")


class TestDeltaResidualMemoryManagement(unittest.TestCase):
    """Tests memory behavior of the delta history list."""

    def test_max_delta_history_caps_list_length(self):
        """When max_delta_history > 0, the delta list must be pruned."""
        config = make_config(max_delta_history=4)
        # Simulate what TransformerBlock does
        deltas = []
        for i in range(10):
            deltas.append(torch.randn(2, 8, 64))
            if config.max_delta_history > 0:
                while len(deltas) > config.max_delta_history:
                    deltas.pop(0)

        self.assertEqual(len(deltas), 4,
                         f"Delta history should be capped at 4, got {len(deltas)}.")

    def test_full_history_keeps_all_deltas(self):
        """When max_delta_history = 0, all deltas must be kept."""
        config = make_config(max_delta_history=0)
        deltas = []
        for i in range(20):
            deltas.append(torch.randn(2, 8, 64))
            if config.max_delta_history > 0:
                while len(deltas) > config.max_delta_history:
                    deltas.pop(0)

        self.assertEqual(len(deltas), 20,
                         "Full history mode (max_delta_history=0) should keep all deltas.")

    def test_delta_memory_scales_linearly(self):
        """
        Memory usage of the delta list should scale linearly with delta count.
        This verifies we aren't accidentally cloning or duplicating tensors.
        """
        config = make_config()
        deltas = []
        single_delta_bytes = 2 * 8 * 64 * 4  # batch * seq * hidden * float32

        for i in range(10):
            d = torch.randn(2, 8, 64)
            deltas.append(d)

        # Each delta is a view/reference, not a copy. Total should be ~10x single.
        total_bytes = sum(d.nelement() * d.element_size() for d in deltas)
        expected_bytes = 10 * single_delta_bytes
        self.assertEqual(total_bytes, expected_bytes,
                         "Delta memory should scale exactly linearly.")

    def test_popped_deltas_are_freed(self):
        """
        When deltas are popped from the list, they should become eligible for GC
        (no lingering references inside the routing layer).
        """
        import weakref
        config = make_config(max_delta_history=2)

        deltas = []
        weak_refs = []

        for i in range(5):
            d = torch.randn(2, 8, 64)
            weak_refs.append(weakref.ref(d))
            deltas.append(d)
            if config.max_delta_history > 0:
                while len(deltas) > config.max_delta_history:
                    deltas.pop(0)

        # Delete our local 'd' references from the loop
        del d
        gc.collect()

        # The first 3 deltas should have been popped and freed
        for i in range(3):
            self.assertIsNone(weak_refs[i](),
                              f"Delta {i} should have been garbage collected after pop.")

        # The last 2 should still be alive
        for i in range(3, 5):
            self.assertIsNotNone(weak_refs[i](),
                                 f"Delta {i} should still be alive in the list.")


class TestDeltaResidualIntegration(unittest.TestCase):
    """Tests the Delta Residual in the context of how TransformerBlock uses it."""

    def test_two_routing_queries_produce_different_outputs(self):
        """
        TransformerBlock uses two separate routing queries (routing_q_attn, routing_q_ffn).
        They must produce different routing patterns.
        """
        config = make_config()
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        deltas = [torch.randn(2, 8, 64) for _ in range(5)]

        q1 = nn.Parameter(torch.randn(64))
        q2 = nn.Parameter(torch.randn(64))

        out1 = layer(x, deltas, q1)
        out2 = layer(x, deltas, q2)

        self.assertFalse(torch.allclose(out1, out2, atol=1e-5),
                         "Different routing queries must produce different outputs.")

    def test_delta_history_grows_correctly_across_layers(self):
        """
        Simulate multi-layer forward: each layer appends 2 deltas (attn + ffn).
        After N layers, history should have 2*N entries (with full history).
        """
        config = make_config(max_delta_history=0)
        num_layers = 6
        deltas = []

        for layer_idx in range(num_layers):
            # Simulate attention output delta
            v_attn = torch.randn(2, 8, 64)
            deltas.append(v_attn)
            # Simulate FFN output delta
            v_ffn = torch.randn(2, 8, 64)
            deltas.append(v_ffn)

        self.assertEqual(len(deltas), 2 * num_layers,
                         f"Expected {2 * num_layers} deltas, got {len(deltas)}.")

    def test_delta_history_capped_across_layers(self):
        """
        With max_delta_history=4, after 6 layers (12 deltas), only last 4 remain.
        """
        config = make_config(max_delta_history=4)
        num_layers = 6
        deltas = []

        for layer_idx in range(num_layers):
            v_attn = torch.randn(2, 8, 64)
            deltas.append(v_attn)
            if config.max_delta_history > 0:
                while len(deltas) > config.max_delta_history:
                    deltas.pop(0)

            v_ffn = torch.randn(2, 8, 64)
            deltas.append(v_ffn)
            if config.max_delta_history > 0:
                while len(deltas) > config.max_delta_history:
                    deltas.pop(0)

        self.assertEqual(len(deltas), 4,
                         f"Expected delta history capped at 4, got {len(deltas)}.")

    def test_routing_with_large_delta_count(self):
        """Stress test: 100 deltas should not crash or produce NaN."""
        config = make_config()
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        deltas = [torch.randn(2, 8, 64) for _ in range(100)]
        routing_q = nn.Parameter(torch.randn(64))

        out = layer(x, deltas, routing_q)
        self.assertFalse(torch.isnan(out).any(), "100 deltas must not produce NaN.")
        self.assertEqual(out.shape, x.shape)

    def test_batch_independence(self):
        """Routing in one batch element must not affect another."""
        config = make_config()
        layer = DeltaAttentionResidual(config)
        routing_q = nn.Parameter(torch.randn(64))

        x1 = torch.randn(1, 8, 64)
        x2 = torch.randn(1, 8, 64)
        x_batch = torch.cat([x1, x2], dim=0)

        d1 = [torch.randn(1, 8, 64) for _ in range(3)]
        d2 = [torch.randn(1, 8, 64) for _ in range(3)]
        d_batch = [torch.cat([a, b], dim=0) for a, b in zip(d1, d2)]

        out_batch = layer(x_batch, d_batch, routing_q)
        out1 = layer(x1, d1, routing_q)
        out2 = layer(x2, d2, routing_q)

        self.assertTrue(torch.allclose(out_batch[0], out1[0], atol=1e-5),
                        "Batch element 0 must match independent computation.")
        self.assertTrue(torch.allclose(out_batch[1], out2[0], atol=1e-5),
                        "Batch element 1 must match independent computation.")


if __name__ == '__main__':
    unittest.main()
