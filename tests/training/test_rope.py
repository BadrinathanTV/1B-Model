import torch
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../training")))

from layers.rope import precompute_freqs_cis, precompute_cos_sin, apply_rotary_emb

class TestRoPE(unittest.TestCase):
    def setUp(self):
        self.num_heads = 2
        self.head_dim = 16
        self.dim = self.num_heads * self.head_dim
        self.end = 32
        self.theta = 10000.0
        self.batch_size = 2
        self.seq_len = 8

    def test_precompute_freqs_cis_shape(self):
        freqs_cis = precompute_freqs_cis(self.head_dim, self.end, self.theta)
        self.assertEqual(freqs_cis.shape, (self.end, self.head_dim // 2))
        self.assertTrue(torch.is_complex(freqs_cis))

    def test_precompute_cos_sin_shape(self):
        cos_cache, sin_cache = precompute_cos_sin(self.head_dim, self.end, self.theta)
        self.assertEqual(cos_cache.shape, (self.end, self.head_dim // 2))
        self.assertEqual(sin_cache.shape, (self.end, self.head_dim // 2))

    def test_apply_rotary_emb_shape(self):
        xq = torch.randn(self.batch_size, self.seq_len, self.num_heads, self.head_dim)
        xk = torch.randn(self.batch_size, self.seq_len, self.num_heads, self.head_dim)
        
        freqs_cis = precompute_freqs_cis(self.head_dim, self.end, self.theta)
        cos_cache, sin_cache = precompute_cos_sin(self.head_dim, self.end, self.theta)
        
        xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis, cos_cache, sin_cache)
        
        self.assertEqual(xq_out.shape, xq.shape)
        self.assertEqual(xk_out.shape, xk.shape)

    def test_apply_rotary_emb_backward(self):
        xq = torch.randn(self.batch_size, self.seq_len, self.num_heads, self.head_dim, requires_grad=True)
        xk = torch.randn(self.batch_size, self.seq_len, self.num_heads, self.head_dim, requires_grad=True)
        
        freqs_cis = precompute_freqs_cis(self.head_dim, self.end, self.theta)
        cos_cache, sin_cache = precompute_cos_sin(self.head_dim, self.end, self.theta)
        
        xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis, cos_cache, sin_cache)
        
        loss = xq_out.sum() + xk_out.sum()
        loss.backward()
        
        self.assertIsNotNone(xq.grad)
        self.assertIsNotNone(xk.grad)

    def test_rope_exceeds_cache(self):
        xq = torch.randn(self.batch_size, self.end + 10, self.num_heads, self.head_dim)
        xk = torch.randn(self.batch_size, self.end + 10, self.num_heads, self.head_dim)
        freqs_cis = precompute_freqs_cis(self.head_dim, self.end, self.theta)
        
        with self.assertRaises(ValueError):
            # Fallback path error check
            import layers.rope as rope
            orig_liger = rope.LIGER_ROPE
            rope.LIGER_ROPE = False
            try:
                apply_rotary_emb(xq, xk, freqs_cis)
            finally:
                rope.LIGER_ROPE = orig_liger

    def test_liger_fallback_equivalence(self):
        import layers.rope as rope
        orig_liger = rope.LIGER_ROPE
        
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if device.type != 'cuda':
                self.skipTest("CUDA required for Liger tests")
                
            xq = torch.randn(self.batch_size, self.seq_len, self.num_heads, self.head_dim, device=device, dtype=torch.bfloat16)
            xk = torch.randn(self.batch_size, self.seq_len, self.num_heads, self.head_dim, device=device, dtype=torch.bfloat16)
            
            freqs_cis = precompute_freqs_cis(self.head_dim, self.end, self.theta, device=device)
            cos_cache, sin_cache = precompute_cos_sin(self.head_dim, self.end, self.theta, device=device, dtype=torch.bfloat16)
            
            # Fallback
            rope.LIGER_ROPE = False
            xq_fb, xk_fb = apply_rotary_emb(xq.clone(), xk.clone(), freqs_cis, cos_cache, sin_cache)
            
            # Liger
            rope.LIGER_ROPE = True
            if hasattr(rope, 'LigerRopeFunction'): # Liger installed
                xq_liger, xk_liger = apply_rotary_emb(xq.clone(), xk.clone(), freqs_cis, cos_cache, sin_cache)
                torch.testing.assert_close(xq_fb, xq_liger, rtol=1e-2, atol=1e-2)
                torch.testing.assert_close(xk_fb, xk_liger, rtol=1e-2, atol=1e-2)
                
        finally:
            rope.LIGER_ROPE = orig_liger

if __name__ == '__main__':
    unittest.main()
