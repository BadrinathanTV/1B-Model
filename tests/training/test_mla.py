import torch
import unittest
import sys
import os

# Add the directory to sys.path so we can import from the 1B-Model/training folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../training")))

from config import SLMConfig
from layers.attention import MultiHeadLatentAttention
from layers.rope import precompute_freqs_cis, precompute_cos_sin

class TestMLA(unittest.TestCase):
    def setUp(self):
        self.config = SLMConfig()
        self.config.hidden_size = 128
        self.config.num_attention_heads = 4
        self.config.q_lora_rank = 32
        self.config.kv_lora_rank = 16
        self.config.qk_rope_head_dim = 16
        self.config.v_head_dim = 32
        self.config.rms_norm_eps = 1e-6
        self.config.training.seq_len = 16
        
        self.batch_size = 2
        self.seq_len = self.config.training.seq_len

    def test_mla_forward_shape(self):
        mla = MultiHeadLatentAttention(self.config)
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        
        freqs_cis = precompute_freqs_cis(self.config.qk_rope_head_dim, self.seq_len, self.config.rope_theta)
        cos_cache, sin_cache = precompute_cos_sin(self.config.qk_rope_head_dim, self.seq_len, self.config.rope_theta)
        
        out = mla(x, freqs_cis=freqs_cis, cos_cache=cos_cache, sin_cache=sin_cache)
        self.assertEqual(out.shape, x.shape)
        
    def test_mla_backward(self):
        mla = MultiHeadLatentAttention(self.config)
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size, requires_grad=True)
        
        freqs_cis = precompute_freqs_cis(self.config.qk_rope_head_dim, self.seq_len, self.config.rope_theta)
        cos_cache, sin_cache = precompute_cos_sin(self.config.qk_rope_head_dim, self.seq_len, self.config.rope_theta)
        
        out = mla(x, freqs_cis=freqs_cis, cos_cache=cos_cache, sin_cache=sin_cache)
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        for name, param in mla.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")
            
    def test_mla_missing_rope(self):
        mla = MultiHeadLatentAttention(self.config)
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        with self.assertRaises(ValueError):
            mla(x)

    def test_mla_rope_sequence_exceeds_cache(self):
        mla = MultiHeadLatentAttention(self.config)
        long_seq_len = 32
        x = torch.randn(self.batch_size, long_seq_len, self.config.hidden_size)
        
        freqs_cis = precompute_freqs_cis(self.config.qk_rope_head_dim, self.seq_len, self.config.rope_theta)
        cos_cache, sin_cache = precompute_cos_sin(self.config.qk_rope_head_dim, self.seq_len, self.config.rope_theta)
        
        with self.assertRaises(ValueError):
            mla(x, freqs_cis=freqs_cis, cos_cache=cos_cache, sin_cache=sin_cache)
            
    def test_mla_causality(self):
        mla = MultiHeadLatentAttention(self.config)
        mla.eval()
        
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        freqs_cis = precompute_freqs_cis(self.config.qk_rope_head_dim, self.seq_len, self.config.rope_theta)
        cos_cache, sin_cache = precompute_cos_sin(self.config.qk_rope_head_dim, self.seq_len, self.config.rope_theta)
        
        out1 = mla(x, freqs_cis=freqs_cis, cos_cache=cos_cache, sin_cache=sin_cache)
        
        x2 = x.clone()
        x2[:, -1, :] = torch.randn(self.batch_size, self.config.hidden_size)
        out2 = mla(x2, freqs_cis=freqs_cis, cos_cache=cos_cache, sin_cache=sin_cache)
        
        torch.testing.assert_close(out1[:, :-1, :], out2[:, :-1, :])
        diff = (out1[:, -1, :] - out2[:, -1, :]).abs().mean()
        self.assertTrue(diff.item() > 1e-5)

    def test_mla_dtypes(self):
        # MLA should support float32, bfloat16, float16 (if hardware supports it)
        dtypes = [torch.float32, torch.bfloat16]
        if torch.cuda.is_available():
            dtypes.append(torch.float16)

        for dtype in dtypes:
            mla = MultiHeadLatentAttention(self.config).to(dtype)
            x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size, dtype=dtype)
            
            freqs_cis = precompute_freqs_cis(self.config.qk_rope_head_dim, self.seq_len, self.config.rope_theta)
            cos_cache, sin_cache = precompute_cos_sin(self.config.qk_rope_head_dim, self.seq_len, self.config.rope_theta, dtype=dtype)
            
            out = mla(x, freqs_cis=freqs_cis, cos_cache=cos_cache, sin_cache=sin_cache)
            self.assertEqual(out.dtype, dtype)
            self.assertEqual(out.shape, x.shape)

    def test_mla_batch_independence(self):
        # Changing elements in batch index 0 should NOT affect batch index 1
        mla = MultiHeadLatentAttention(self.config)
        mla.eval()
        
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        freqs_cis = precompute_freqs_cis(self.config.qk_rope_head_dim, self.seq_len, self.config.rope_theta)
        cos_cache, sin_cache = precompute_cos_sin(self.config.qk_rope_head_dim, self.seq_len, self.config.rope_theta)
        
        out1 = mla(x, freqs_cis=freqs_cis, cos_cache=cos_cache, sin_cache=sin_cache)
        
        x2 = x.clone()
        x2[0] = torch.randn(self.seq_len, self.config.hidden_size) # Mutate only batch index 0
        out2 = mla(x2, freqs_cis=freqs_cis, cos_cache=cos_cache, sin_cache=sin_cache)
        
        # Batch index 1 outputs must be identical
        torch.testing.assert_close(out1[1:], out2[1:])
        # Batch index 0 outputs must differ
        diff = (out1[0] - out2[0]).abs().mean()
        self.assertTrue(diff.item() > 1e-5)

    def test_mla_head_configurations(self):
        # Test extreme configuration: single head attention
        config = SLMConfig()
        config.hidden_size = 64
        config.num_attention_heads = 1
        config.q_lora_rank = 16
        config.kv_lora_rank = 8
        config.qk_rope_head_dim = 8
        config.v_head_dim = 16
        config.rms_norm_eps = 1e-6
        config.training.seq_len = 8
        
        mla = MultiHeadLatentAttention(config)
        x = torch.randn(self.batch_size, 8, config.hidden_size)
        
        freqs_cis = precompute_freqs_cis(config.qk_rope_head_dim, 8, config.rope_theta)
        cos_cache, sin_cache = precompute_cos_sin(config.qk_rope_head_dim, 8, config.rope_theta)
        
        out = mla(x, freqs_cis=freqs_cis, cos_cache=cos_cache, sin_cache=sin_cache)
        self.assertEqual(out.shape, x.shape)

    def test_mla_zero_inputs(self):
        # MLA should handle zero inputs without yielding NaNs
        mla = MultiHeadLatentAttention(self.config)
        x = torch.zeros(self.batch_size, self.seq_len, self.config.hidden_size)
        
        freqs_cis = precompute_freqs_cis(self.config.qk_rope_head_dim, self.seq_len, self.config.rope_theta)
        cos_cache, sin_cache = precompute_cos_sin(self.config.qk_rope_head_dim, self.seq_len, self.config.rope_theta)
        
        out = mla(x, freqs_cis=freqs_cis, cos_cache=cos_cache, sin_cache=sin_cache)
        self.assertFalse(torch.isnan(out).any())
        self.assertEqual(out.shape, x.shape)

    def test_liger_rope_mocking(self):
        # We can test both the PyTorch fallback path and Liger RoPE path by mocking LIGER_ROPE
        import layers.rope as rope
        
        # Keep original LIGER_ROPE setting
        orig_liger = rope.LIGER_ROPE
        
        try:
            # Force fallback mode by setting LIGER_ROPE to False
            rope.LIGER_ROPE = False
            mla = MultiHeadLatentAttention(self.config)
            x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
            
            freqs_cis = precompute_freqs_cis(self.config.qk_rope_head_dim, self.seq_len, self.config.rope_theta)
            cos_cache, sin_cache = precompute_cos_sin(self.config.qk_rope_head_dim, self.seq_len, self.config.rope_theta)
            
            out_fallback = mla(x, freqs_cis=freqs_cis, cos_cache=cos_cache, sin_cache=sin_cache)
            self.assertEqual(out_fallback.shape, x.shape)
            
            # If CUDA is available, we can also test running it on GPU
            if torch.cuda.is_available():
                mla_cuda = mla.cuda()
                x_cuda = x.cuda()
                freqs_cis_cuda = freqs_cis.cuda()
                cos_cache_cuda = cos_cache.cuda()
                sin_cache_cuda = sin_cache.cuda()
                
                out_cuda = mla_cuda(x_cuda, freqs_cis=freqs_cis_cuda, cos_cache=cos_cache_cuda, sin_cache=sin_cache_cuda)
                torch.testing.assert_close(out_fallback, out_cuda.cpu(), rtol=1e-3, atol=1e-3)
                
        finally:
            # Restore
            rope.LIGER_ROPE = orig_liger

if __name__ == '__main__':
    unittest.main()
