"""
Delta Attention Residual Test Suite (Dynamic Bottleneck Routing)
================================================================

Tests correctness, edge cases, gradient flow, memory management,
numerical stability, and integration with the TransformerBlock.

Covers both the list-based training API and the static buffer generation API.
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
    config.delta_routing_rank = 32  # small rank for tests
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


def _make_buffer_from_deltas(deltas, max_hist, batch, seq_len, hidden):
    """Create a deltas_buf and active_mask from a list of delta tensors."""
    device = deltas[0].device if deltas else torch.device('cpu')
    dtype = deltas[0].dtype if deltas else torch.float32
    buf = torch.zeros(max_hist, batch, seq_len, hidden, device=device, dtype=dtype)
    active_mask = torch.zeros(max_hist, batch, seq_len, dtype=torch.bool, device=device)
    n = min(len(deltas), max_hist)
    for i in range(n):
        buf[i] = deltas[i]
        active_mask[i] = True
    return buf, active_mask


class TestDeltaResidualCorrectness(unittest.TestCase):
    """Tests that the core math of Delta Attention Residual is correct."""

    def test_empty_deltas_returns_x_unchanged(self):
        """When there are no previous deltas, output must equal input exactly."""
        config = make_config(num_hidden_layers=2)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        routing_q = layer.compute_routing_q(x)
        buf = torch.zeros(max_hist, 2, 8, 64)
        active_mask = torch.zeros(max_hist, 2, 8, dtype=torch.bool)

        out = layer.forward(x, routing_q, delta_buffer=buf, active_mask=active_mask)
        self.assertTrue(torch.allclose(out, x, atol=1e-5),
                        "With empty deltas, output should be identical to input.")

    def test_empty_list_returns_x_unchanged_training(self):
        """Training phase: empty past_deltas list must return x unchanged."""
        config = make_config(num_hidden_layers=2)
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        routing_q = layer.compute_routing_q(x)

        out = layer.forward(x, routing_q, past_deltas=[])
        self.assertTrue(torch.allclose(out, x, atol=1e-5),
                        "Empty past_deltas list should return x unchanged.")

    def test_single_delta_softmax_reduces_to_identity(self):
        """With exactly 1 delta, softmax(score) = 1.0, so routed = delta itself."""
        config = make_config(num_hidden_layers=2)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        delta = torch.randn(2, 8, 64)
        routing_q = layer.compute_routing_q(x)

        buf, active_mask = _make_buffer_from_deltas([delta], max_hist, 2, 8, 64)
        out = layer.forward(x, routing_q, delta_buffer=buf, active_mask=active_mask)

        # With a single delta, softmax produces alpha=1.0 -> routed = delta
        expected = x + delta
        self.assertTrue(torch.allclose(out, expected, atol=1e-5),
                        "Single delta should produce x + delta (softmax is trivially 1.0).")

    def test_single_delta_training_path(self):
        """Training phase: single delta must also produce x + delta."""
        config = make_config(num_hidden_layers=2)
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        delta = torch.randn(2, 8, 64)
        routing_q = layer.compute_routing_q(x)

        out = layer.forward(x, routing_q, past_deltas=[delta])
        expected = x + delta
        self.assertTrue(torch.allclose(out, expected, atol=1e-4),
                        "Training path single delta should produce x + delta.")

    def test_output_shape_matches_input(self):
        """Output shape must always match input shape regardless of number of deltas."""
        config = make_config(num_hidden_layers=6)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config)

        for n_deltas in [0, 1, 3, 10]:
            n_actual = min(n_deltas, max_hist)
            x = torch.randn(4, 8, 64)
            routing_q = layer.compute_routing_q(x)
            deltas = [torch.randn(4, 8, 64) for _ in range(n_actual)]
            buf, active_mask = _make_buffer_from_deltas(deltas, max_hist, 4, 8, 64)
            out = layer.forward(x, routing_q, delta_buffer=buf, active_mask=active_mask)
            self.assertEqual(out.shape, x.shape,
                             f"Output shape mismatch with {n_deltas} deltas.")

    def test_training_vs_generation_consistency(self):
        """Training (list) and generation (buffer) paths must produce same results."""
        config = make_config(num_hidden_layers=3)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config)
        layer.eval()

        x = torch.randn(2, 8, 64)
        deltas = [torch.randn(2, 8, 64) for _ in range(4)]
        routing_q = layer.compute_routing_q(x)

        # Training path
        out_train = layer.forward(x, routing_q, past_deltas=deltas)

        # Generation path
        buf, active_mask = _make_buffer_from_deltas(deltas, max_hist, 2, 8, 64)
        out_gen = layer.forward(x, routing_q, delta_buffer=buf, active_mask=active_mask)

        self.assertTrue(torch.allclose(out_train, out_gen, atol=1e-4),
                        "Training and generation paths must produce consistent results.")


class TestDeltaResidualGradients(unittest.TestCase):
    """Tests that gradients flow correctly through the routing mechanism."""

    def test_gradients_flow_to_bottleneck_weights(self):
        """The bottleneck projection weights must receive gradients."""
        config = make_config(num_hidden_layers=2)
        layer = DeltaAttentionResidual(config)
        # Need non-zero routing_up for gradients to flow through to routing_down
        nn.init.normal_(layer.routing_up.weight, std=0.01)
        x = torch.randn(2, 8, 64, requires_grad=True)
        deltas = [torch.randn(2, 8, 64) for _ in range(3)]

        routing_q = layer.compute_routing_q(x)
        out = layer.forward(x, routing_q, past_deltas=deltas)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(layer.routing_down.weight.grad, 
                             "routing_down must receive gradients.")
        self.assertIsNotNone(layer.routing_up.weight.grad, 
                             "routing_up must receive gradients.")
        self.assertTrue(layer.routing_down.weight.grad.abs().sum() > 0, 
                        "routing_down gradient must be non-zero.")

    def test_gradients_flow_to_input_x(self):
        """The input x must receive gradients (it feeds both residual + routing query)."""
        config = make_config(num_hidden_layers=2)
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64, requires_grad=True)
        deltas = [torch.randn(2, 8, 64) for _ in range(3)]

        routing_q = layer.compute_routing_q(x)
        out = layer.forward(x, routing_q, past_deltas=deltas)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue((x.grad.abs() > 0).all(), "Input x must receive gradients everywhere.")

    def test_zero_initialized_routing_produces_uniform(self):
        """
        At init, routing_up is zero-initialized, so all scores are 0,
        softmax gives uniform weights (1/N). routing_up must receive
        gradients to break symmetry during training.
        """
        config = make_config(num_hidden_layers=2)
        layer = DeltaAttentionResidual(config)
        # Verify zero init
        self.assertTrue(torch.all(layer.routing_up.weight == 0),
                        "routing_up must be zero-initialized.")

        x = torch.randn(2, 8, 64, requires_grad=True)
        deltas = [torch.randn(2, 8, 64) for _ in range(4)]

        routing_q = layer.compute_routing_q(x)
        out = layer.forward(x, routing_q, past_deltas=deltas)
        loss = out.sum()
        loss.backward()

        # routing_up receives gradients from the scoring path even when zero-init
        self.assertIsNotNone(layer.routing_up.weight.grad,
                             "routing_up must receive gradients.")
        self.assertTrue(layer.routing_up.weight.grad.abs().sum() > 0,
                        "routing_up must receive non-zero gradients to break symmetry.")

    def test_gradients_flow_through_generation_path(self):
        """Generation path (buffer-based) must also support gradient flow."""
        config = make_config(num_hidden_layers=2)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64, requires_grad=True)
        deltas = [torch.randn(2, 8, 64, requires_grad=True) for _ in range(3)]

        routing_q = layer.compute_routing_q(x)
        buf, active_mask = _make_buffer_from_deltas(deltas, max_hist, 2, 8, 64)
        buf = buf + 0  # force into computational graph
        out = layer.forward(x, routing_q, delta_buffer=buf, active_mask=active_mask)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad, "x must receive gradients in generation path.")


class TestDeltaResidualNumericalStability(unittest.TestCase):
    """Tests numerical edge cases that could cause NaN/Inf during training."""

    def test_zero_deltas_no_nan(self):
        """If all deltas are zero tensors, the RMSNorm should not produce NaN (eps protects)."""
        config = make_config(num_hidden_layers=2)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        deltas = [torch.zeros(2, 8, 64) for _ in range(3)]
        routing_q = layer.compute_routing_q(x)
        buf, active_mask = _make_buffer_from_deltas(deltas, max_hist, 2, 8, 64)

        out = layer.forward(x, routing_q, delta_buffer=buf, active_mask=active_mask)
        self.assertFalse(torch.isnan(out).any(), "Zero deltas must not produce NaN.")
        self.assertFalse(torch.isinf(out).any(), "Zero deltas must not produce Inf.")

    def test_extreme_magnitude_deltas_no_nan(self):
        """Very large delta values should be stabilized by RMSNorm before scoring."""
        config = make_config(num_hidden_layers=2)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        deltas = [torch.randn(2, 8, 64) * 1e6 for _ in range(3)]
        routing_q = layer.compute_routing_q(x)
        buf, active_mask = _make_buffer_from_deltas(deltas, max_hist, 2, 8, 64)

        out = layer.forward(x, routing_q, delta_buffer=buf, active_mask=active_mask)
        self.assertFalse(torch.isnan(out).any(), "Large deltas must not produce NaN.")
        self.assertFalse(torch.isinf(out).any(), "Large deltas must not produce Inf.")

    def test_tiny_magnitude_deltas_no_nan(self):
        """Very small delta values should not cause division-by-zero in RMSNorm."""
        config = make_config(num_hidden_layers=2)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        deltas = [torch.randn(2, 8, 64) * 1e-10 for _ in range(3)]
        routing_q = layer.compute_routing_q(x)
        buf, active_mask = _make_buffer_from_deltas(deltas, max_hist, 2, 8, 64)

        out = layer.forward(x, routing_q, delta_buffer=buf, active_mask=active_mask)
        self.assertFalse(torch.isnan(out).any(), "Tiny deltas must not produce NaN.")
        self.assertFalse(torch.isinf(out).any(), "Tiny deltas must not produce Inf.")

    def test_mixed_magnitude_deltas_no_nan(self):
        """Deltas with wildly different scales should be handled gracefully."""
        config = make_config(num_hidden_layers=3)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        deltas = [
            torch.randn(2, 8, 64) * 1e-8,
            torch.randn(2, 8, 64) * 1.0,
            torch.randn(2, 8, 64) * 1e6,
        ]
        routing_q = layer.compute_routing_q(x)
        buf, active_mask = _make_buffer_from_deltas(deltas, max_hist, 2, 8, 64)

        out = layer.forward(x, routing_q, delta_buffer=buf, active_mask=active_mask)
        self.assertFalse(torch.isnan(out).any(), "Mixed magnitude deltas must not produce NaN.")
        self.assertFalse(torch.isinf(out).any(), "Mixed magnitude deltas must not produce Inf.")

    def test_bfloat16_stability(self):
        """Delta routing must be numerically stable under bfloat16 precision."""
        config = make_config(num_hidden_layers=3)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config).bfloat16()
        x = torch.randn(2, 8, 64, dtype=torch.bfloat16)
        deltas = [torch.randn(2, 8, 64, dtype=torch.bfloat16) for _ in range(5)]
        routing_q = layer.compute_routing_q(x)
        buf, active_mask = _make_buffer_from_deltas(deltas, max_hist, 2, 8, 64)
        buf = buf.bfloat16()

        out = layer.forward(x, routing_q, delta_buffer=buf, active_mask=active_mask)
        self.assertFalse(torch.isnan(out).any(), "bfloat16 must not produce NaN.")
        self.assertFalse(torch.isinf(out).any(), "bfloat16 must not produce Inf.")

    def test_training_path_zero_deltas_no_nan(self):
        """Training path: zero deltas must not produce NaN."""
        config = make_config(num_hidden_layers=2)
        layer = DeltaAttentionResidual(config)
        x = torch.randn(2, 8, 64)
        deltas = [torch.zeros(2, 8, 64) for _ in range(3)]
        routing_q = layer.compute_routing_q(x)

        out = layer.forward(x, routing_q, past_deltas=deltas)
        self.assertFalse(torch.isnan(out).any(), "Training: zero deltas must not produce NaN.")


class TestDeltaResidualIntegration(unittest.TestCase):
    """Tests the Delta Residual in the context of how TransformerBlock uses it."""

    def test_two_modules_produce_different_outputs(self):
        """
        TransformerBlock uses two separate DeltaAttentionResidual modules
        (delta_residual_attn, delta_residual_ffn). With different bottleneck
        weights, they must produce different routing patterns.
        """
        config = make_config(num_hidden_layers=3)
        max_hist = 2 * config.num_hidden_layers
        layer1 = DeltaAttentionResidual(config)
        layer2 = DeltaAttentionResidual(config)

        # Ensure different weights (both down and up)
        nn.init.normal_(layer1.routing_down.weight, std=0.1)
        nn.init.normal_(layer1.routing_up.weight, std=0.01)
        nn.init.normal_(layer2.routing_down.weight, std=0.1)
        nn.init.normal_(layer2.routing_up.weight, std=0.01)

        x = torch.randn(2, 8, 64)
        deltas = [torch.randn(2, 8, 64) for _ in range(5)]
        buf, active_mask = _make_buffer_from_deltas(deltas, max_hist, 2, 8, 64)

        q1 = layer1.compute_routing_q(x)
        q2 = layer2.compute_routing_q(x)
        out1 = layer1.forward(x, q1, delta_buffer=buf, active_mask=active_mask)
        out2 = layer2.forward(x, q2, delta_buffer=buf, active_mask=active_mask)

        self.assertFalse(torch.allclose(out1, out2, atol=1e-5),
                         "Different modules must produce different outputs.")

    def test_batch_independence(self):
        """Routing in one batch element must not affect another."""
        config = make_config(num_hidden_layers=2)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config)
        # Give non-trivial weights for dynamic routing
        nn.init.normal_(layer.routing_up.weight, std=0.01)

        x1 = torch.randn(1, 8, 64)
        x2 = torch.randn(1, 8, 64)
        x_batch = torch.cat([x1, x2], dim=0)

        d1 = [torch.randn(1, 8, 64) for _ in range(3)]
        d2 = [torch.randn(1, 8, 64) for _ in range(3)]
        d_batch = [torch.cat([a, b], dim=0) for a, b in zip(d1, d2)]

        # Training path (list-based)
        rq_batch = layer.compute_routing_q(x_batch)
        rq1 = layer.compute_routing_q(x1)
        rq2 = layer.compute_routing_q(x2)

        out_batch = layer.forward(x_batch, rq_batch, past_deltas=d_batch)
        out1 = layer.forward(x1, rq1, past_deltas=d1)
        out2 = layer.forward(x2, rq2, past_deltas=d2)

        self.assertTrue(torch.allclose(out_batch[0], out1[0], atol=1e-4),
                        "Batch element 0 must match independent computation.")
        self.assertTrue(torch.allclose(out_batch[1], out2[0], atol=1e-4),
                        "Batch element 1 must match independent computation.")

    def test_dynamic_routing_is_token_dependent(self):
        """Different tokens in the same sequence should produce different routing scores."""
        config = make_config(num_hidden_layers=2)
        layer = DeltaAttentionResidual(config)
        # Give non-trivial weights
        nn.init.normal_(layer.routing_down.weight, std=0.1)
        nn.init.normal_(layer.routing_up.weight, std=0.01)

        x = torch.randn(1, 8, 64)  # 8 different tokens
        routing_q = layer.compute_routing_q(x)  # [1, 8, 64]

        # Each token position should have a different routing query
        # (unless the input tokens happen to be identical, which is vanishingly unlikely)
        q_pos0 = routing_q[0, 0]
        q_pos4 = routing_q[0, 4]
        self.assertFalse(torch.allclose(q_pos0, q_pos4, atol=1e-6),
                         "Different tokens must produce different routing queries.")

    def test_bottleneck_param_count(self):
        """Verify the bottleneck adds the expected number of parameters."""
        config = make_config(delta_routing_rank=32)
        layer = DeltaAttentionResidual(config)
        
        # down: H * rank = 64 * 32 = 2048, up: rank * H = 32 * 64 = 2048
        expected = 64 * 32 + 32 * 64  # 4096
        actual = sum(p.numel() for p in layer.parameters())
        self.assertEqual(actual, expected,
                         f"Expected {expected} params, got {actual}.")

    def test_seq_len_1_generation(self):
        """Generation decode step: seq_len=1 must work correctly."""
        config = make_config(num_hidden_layers=2)
        max_hist = 2 * config.num_hidden_layers
        layer = DeltaAttentionResidual(config)

        x = torch.randn(1, 1, 64)  # single token decode
        routing_q = layer.compute_routing_q(x)
        deltas = [torch.randn(1, 1, 64) for _ in range(3)]
        buf, active_mask = _make_buffer_from_deltas(deltas, max_hist, 1, 1, 64)

        out = layer.forward(x, routing_q, delta_buffer=buf, active_mask=active_mask)
        self.assertEqual(out.shape, (1, 1, 64), "seq_len=1 decode must produce correct shape.")
        self.assertFalse(torch.isnan(out).any(), "seq_len=1 must not produce NaN.")


class TestDeltaResidualRingBuffer(unittest.TestCase):
    """Tests the ring buffer mechanism for generation."""

    def test_ring_buffer_overwrites_oldest(self):
        """When buffer is full, new deltas must overwrite the oldest entries."""
        config = make_config(num_hidden_layers=2)
        layer = DeltaAttentionResidual(config)
        max_deltas = 3
        buf, mask, idx = layer.init_state(1, 4, max_deltas, torch.float32, torch.device('cpu'))

        # Fill buffer
        for i in range(3):
            d = torch.ones(1, 4, 64) * (i + 1)
            buf, mask, idx = layer.update_state(d, buf, mask, idx)

        self.assertTrue(mask.all(), "All slots should be active after filling.")

        # 4th insert should overwrite slot 0
        d_new = torch.ones(1, 4, 64) * 99
        buf, mask, idx = layer.update_state(d_new, buf, mask, idx)
        self.assertTrue(torch.allclose(buf[0], d_new), "Slot 0 must be overwritten.")
        self.assertTrue(torch.allclose(buf[1], torch.ones(1, 4, 64) * 2), "Slot 1 must be unchanged.")


class TestDeltaResidualAdditionalEdgeCases(unittest.TestCase):
    """Verifies advanced edge cases including checkpointing, compilation, history=1, and extreme inputs."""

    def test_gradient_checkpointing_compatibility(self):
        """Verify layer works correctly within PyTorch's gradient checkpointing wrapper."""
        from torch.utils.checkpoint import checkpoint
        config = make_config(num_hidden_layers=2)
        layer = DeltaAttentionResidual(config)
        nn.init.normal_(layer.routing_down.weight, std=0.1)
        nn.init.normal_(layer.routing_up.weight, std=0.1)

        x = torch.randn(2, 8, 64, requires_grad=True)
        deltas = [torch.randn(2, 8, 64) for _ in range(3)]
        
        # Define a function to checkpoint that performs both query computation and forward pass
        def custom_checkpointed_forward(x_in, *deltas_in):
            rq = layer.compute_routing_q(x_in)
            return layer.forward(x_in, rq, past_deltas=list(deltas_in))

        # Run with checkpointing (use_reentrant=False)
        out = checkpoint(custom_checkpointed_forward, x, *deltas, use_reentrant=False)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(x.grad, "Input gradients must propagate back through checkpoint.")
        self.assertIsNotNone(layer.routing_down.weight.grad, "Checkpoint must compute gradients for layer parameters.")
        self.assertTrue(layer.routing_down.weight.grad.abs().sum() > 0)

    def test_max_history_of_one(self):
        """Verify correct behavior when the delta history is limited to exactly 1 entry."""
        config = make_config(max_delta_history=1)
        layer = DeltaAttentionResidual(config)
        nn.init.normal_(layer.routing_up.weight, std=0.01)

        x = torch.randn(2, 8, 64)
        rq = layer.compute_routing_q(x)
        
        # In training phase (Phase 1)
        deltas = [torch.randn(2, 8, 64)]
        out_train = layer.forward(x, rq, past_deltas=deltas)
        self.assertTrue(torch.allclose(out_train, x + deltas[0], atol=1e-4))

        # In generation phase (Phase 2)
        buf, active_mask = _make_buffer_from_deltas(deltas, 1, 2, 8, 64)
        out_gen = layer.forward(x, rq, delta_buffer=buf, active_mask=active_mask)
        self.assertTrue(torch.allclose(out_gen, x + deltas[0], atol=1e-4))

    def test_softmax_stability_extreme_values(self):
        """Verify softmax scoring handles extreme magnitude input weights without NaN/overflow."""
        config = make_config(num_hidden_layers=2)
        layer = DeltaAttentionResidual(config)
        # Create extreme weights to force extreme score logits
        nn.init.constant_(layer.routing_down.weight, 1e5)
        nn.init.constant_(layer.routing_up.weight, 1e5)

        x = torch.randn(2, 8, 64)
        rq = layer.compute_routing_q(x)
        deltas = [torch.randn(2, 8, 64) for _ in range(3)]
        
        out = layer.forward(x, rq, past_deltas=deltas)
        self.assertFalse(torch.isnan(out).any(), "Extreme weights must not produce NaNs.")
        self.assertFalse(torch.isinf(out).any(), "Extreme weights must not produce Infs.")

    def test_torch_compile_compatibility(self):
        """Verify DeltaAttentionResidual can be successfully compiled via torch.compile."""
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile is not available in this PyTorch version.")
            
        config = make_config(num_hidden_layers=2)
        layer = DeltaAttentionResidual(config)
        compiled_layer = torch.compile(layer, fullgraph=True)

        x = torch.randn(2, 8, 64)
        deltas = [torch.randn(2, 8, 64) for _ in range(2)]
        
        # Warmup and execute on compiled module
        rq = layer.compute_routing_q(x)
        
        # Phase 1: Training list path
        out_train = compiled_layer(x, rq, past_deltas=deltas)
        self.assertEqual(out_train.shape, x.shape)
        
        # Phase 2: Generation buffer path
        buf, active_mask = _make_buffer_from_deltas(deltas, 4, 2, 8, 64)
        out_gen = compiled_layer(x, rq, delta_buffer=buf, active_mask=active_mask)
        self.assertEqual(out_gen.shape, x.shape)


if __name__ == '__main__':
    unittest.main()
