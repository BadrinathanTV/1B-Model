import torch
import unittest
import sys
import os

# Add the directory to sys.path so we can import from the 1B-Model/training folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../training")))

from config import SLMConfig
from models.mtp import MTPProjection, MTPModule
from model import SLMModel
from train import compute_loss

class TestMTPProjection(unittest.TestCase):
    def setUp(self):
        self.config = SLMConfig()
        self.config.hidden_size = 64
        self.config.rms_norm_eps = 1e-6

    def test_mtp_projection_shape_and_residual(self):
        proj = MTPProjection(self.config)
        batch_size = 2
        seq_len = 8
        
        # Test input shape matching output shape
        x = torch.randn(batch_size, seq_len, self.config.hidden_size)
        y = proj(x)
        self.assertEqual(x.shape, y.shape)
        
        # Verify gradients propagate
        y.sum().backward()
        self.assertIsNotNone(proj.gate_proj.weight.grad)
        self.assertIsNotNone(proj.up_proj.weight.grad)
        self.assertIsNotNone(proj.down_proj.weight.grad)

class TestMTPModule(unittest.TestCase):
    def setUp(self):
        self.config = SLMConfig()
        self.config.vocab_size = 100
        self.config.hidden_size = 64
        self.config.mtp_depth = 3 # 1 main head + 2 MTP projections
        
        # Mock lm_head
        import torch.nn as nn
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def test_mtp_enabled(self):
        mtp = MTPModule(self.config, self.lm_head)
        self.assertEqual(len(mtp.projs), 2)
        
        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        # When use_mtp is True
        logits_list = mtp(x, use_mtp=True)
        
        # Should return mtp_depth (3) logit tensors
        self.assertEqual(len(logits_list), 3)
        for logits in logits_list:
            self.assertEqual(logits.shape, (batch_size, seq_len, self.config.vocab_size))

    def test_mtp_disabled(self):
        mtp = MTPModule(self.config, self.lm_head)
        
        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        # When use_mtp is False
        logits_list = mtp(x, use_mtp=False)
        
        # Should return exactly 1 logit tensor (main head)
        self.assertEqual(len(logits_list), 1)
        self.assertEqual(logits_list[0].shape, (batch_size, seq_len, self.config.vocab_size))

    def test_loss_computation_mtp_enabled(self):
        mtp = MTPModule(self.config, self.lm_head)
        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        logits_list = mtp(x, use_mtp=True)
        targets = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        
        # Should compute loss successfully without shape crashes
        loss = compute_loss(logits_list, targets, self.config)
        self.assertTrue(loss.item() > 0)

    def test_loss_computation_mtp_disabled(self):
        mtp = MTPModule(self.config, self.lm_head)
        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        logits_list = mtp(x, use_mtp=False)
        targets = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        
        # Should compute loss successfully even if MTP is disabled (only main logits used)
        loss = compute_loss(logits_list, targets, self.config)
        self.assertTrue(loss.item() > 0)

class TestMTPIntegration(unittest.TestCase):
    def test_full_model_mtp_toggle(self):
        from layers.rope import precompute_freqs_cis, precompute_cos_sin
        
        config = SLMConfig()
        config.vocab_size = 100
        config.hidden_size = 64
        config.num_hidden_layers = 2
        config.num_attention_heads = 2
        config.tst_group_size = 1 # Disable TST to isolate MTP
        config.mtp_depth = 3
        config.training.seq_len = 8
        config.max_delta_history = 0
        
        model = SLMModel(config)
        model.train()
        
        batch_size = 2
        inputs = torch.randint(0, config.vocab_size, (batch_size, config.training.seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, config.training.seq_len))
        
        freqs_cis = precompute_freqs_cis(config.qk_rope_head_dim, config.training.seq_len, config.rope_theta, inputs.device)
        cos_cache, sin_cache = precompute_cos_sin(config.qk_rope_head_dim, config.training.seq_len, config.rope_theta, inputs.device)
        
        # Case 1: MTP enabled end-to-end
        logits_mtp = model(inputs, freqs_cis=freqs_cis, cos_cache=cos_cache, sin_cache=sin_cache, use_mtp=True)
        self.assertEqual(len(logits_mtp), 3)
        loss_mtp = compute_loss(logits_mtp, targets, config)
        loss_mtp.backward()
        self.assertIsNotNone(model.embed.word_embeddings.weight.grad)
        
        # Zero out grads
        model.zero_grad()
        
        # Case 2: MTP disabled end-to-end
        logits_nomtp = model(inputs, freqs_cis=freqs_cis, cos_cache=cos_cache, sin_cache=sin_cache, use_mtp=False)
        self.assertEqual(len(logits_nomtp), 1)
        loss_nomtp = compute_loss(logits_nomtp, targets, config)
        loss_nomtp.backward()
        self.assertIsNotNone(model.embed.word_embeddings.weight.grad)

if __name__ == '__main__':
    unittest.main()
