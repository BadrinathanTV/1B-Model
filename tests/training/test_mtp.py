import torch
import unittest
import sys
import os

# Add the directory to sys.path so we can import from the 1B-Model/training folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../training")))

from config import SLMConfig
from models.mtp import DeepSeekMTPBlock, MTPModule
from model import SLMModel
from train import compute_loss

class TestDeepSeekMTPBlock(unittest.TestCase):
    def setUp(self):
        self.config = SLMConfig()
        self.config.hidden_size = 64
        self.config.rms_norm_eps = 1e-6

    def test_mtp_block_shape_and_residual(self):
        block = DeepSeekMTPBlock(self.config)
        batch_size = 2
        seq_len = 8
        
        # Test input shape matching output shape
        h = torch.randn(batch_size, seq_len, self.config.hidden_size)
        token_emb = torch.randn(batch_size, seq_len, self.config.hidden_size)
        y = block(h, token_emb)
        self.assertEqual(y.shape, h.shape)
        
        # Verify gradients propagate
        y.sum().backward()
        self.assertIsNotNone(block.concat_proj.weight.grad)
        self.assertIsNotNone(block.ffn.gate_up_proj.weight.grad)
        self.assertIsNotNone(block.ffn.down_proj.weight.grad)

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
        self.assertEqual(len(mtp.blocks), 2)
        
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
        
        hidden_states_list = mtp.forward_hidden(x, use_mtp=True)
        targets = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        
        # Should compute loss successfully without shape crashes
        loss, _ = compute_loss(hidden_states_list, targets, self.lm_head.weight, self.config)
        self.assertTrue(loss.item() > 0)

    def test_loss_computation_mtp_disabled(self):
        mtp = MTPModule(self.config, self.lm_head)
        batch_size = 2
        seq_len = 8
        x = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        hidden_states_list = mtp.forward_hidden(x, use_mtp=False)
        targets = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        
        # Should compute loss successfully even if MTP is disabled (only main logits used)
        loss, _ = compute_loss(hidden_states_list, targets, self.lm_head.weight, self.config)
        self.assertTrue(loss.item() > 0)

class TestMTPIntegration(unittest.TestCase):
    def test_full_model_mtp_toggle(self):
        from layers.rope import precompute_freqs_cis, precompute_cos_sin
        
        config = SLMConfig()
        config.vocab_size = 100
        config.hidden_size = 64
        config.num_hidden_layers = 2
        config.num_attention_heads = 2
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
        hidden_states_mtp = model(inputs, freqs_cis=freqs_cis, cos_cache=cos_cache, sin_cache=sin_cache, use_mtp=True, return_hidden_states=True)
        self.assertEqual(len(hidden_states_mtp), 3)
        loss_mtp, _ = compute_loss(hidden_states_mtp, targets, model.lm_head.weight, config)
        loss_mtp.backward()
        self.assertIsNotNone(model.embed.word_embeddings.weight.grad)
        
        # Zero out grads
        model.zero_grad()
        
        # Case 2: MTP disabled end-to-end
        hidden_states_nomtp = model(inputs, freqs_cis=freqs_cis, cos_cache=cos_cache, sin_cache=sin_cache, use_mtp=False, return_hidden_states=True)
        self.assertEqual(len(hidden_states_nomtp), 1)
        loss_nomtp, _ = compute_loss(hidden_states_nomtp, targets, model.lm_head.weight, config)
        loss_nomtp.backward()
        self.assertIsNotNone(model.embed.word_embeddings.weight.grad)

if __name__ == '__main__':
    unittest.main()
