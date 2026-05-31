import torch
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../training")))

from layers.norm import RMSNorm

class TestRMSNorm(unittest.TestCase):
    def setUp(self):
        self.dim = 64
        self.eps = 1e-6
        self.batch_size = 2
        self.seq_len = 8

    def test_rmsnorm_output_shape(self):
        norm = RMSNorm(self.dim, self.eps)
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        y = norm(x)
        self.assertEqual(y.shape, x.shape)

    def test_rmsnorm_properties(self):
        """RMSNorm scales the input such that the root-mean-square is 1."""
        norm = RMSNorm(self.dim, self.eps)
        # Initialize weights to 1 to isolate the normalization
        with torch.no_grad():
            norm.weight.fill_(1.0)
            
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        y = norm(x)
        
        # Calculate root mean square of output
        # It should be close to 1.0 (excluding eps effect)
        rms = torch.sqrt(torch.mean(y**2, dim=-1))
        expected_rms = torch.ones_like(rms)
        torch.testing.assert_close(rms, expected_rms, rtol=1e-3, atol=1e-3)

    def test_rmsnorm_scale_invariance(self):
        """Multiplying input by a scalar should not affect the output significantly, except for the scalar being canceled."""
        norm = RMSNorm(self.dim, self.eps)
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        
        y1 = norm(x)
        y2 = norm(x * 10.0)
        
        torch.testing.assert_close(y1, y2, rtol=1e-3, atol=1e-3)

    def test_rmsnorm_backward(self):
        """Ensure gradients flow properly."""
        norm = RMSNorm(self.dim, self.eps)
        x = torch.randn(self.batch_size, self.seq_len, self.dim, requires_grad=True)
        
        y = norm(x)
        loss = y.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(norm.weight.grad)
        self.assertFalse(torch.isnan(x.grad).any())
        self.assertFalse(torch.isnan(norm.weight.grad).any())

    def test_rmsnorm_dtypes(self):
        dtypes = [torch.float32, torch.bfloat16]
        if torch.cuda.is_available():
            dtypes.append(torch.float16)
            
        for dtype in dtypes:
            device = torch.device('cuda' if torch.cuda.is_available() and dtype != torch.bfloat16 else 'cpu')
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                device = torch.device('cpu')
            elif dtype == torch.bfloat16:
                device = torch.device('cuda')

            norm = RMSNorm(self.dim, self.eps).to(dtype=dtype, device=device)
            x = torch.randn(self.batch_size, self.seq_len, self.dim, dtype=dtype, device=device)
            y = norm(x)
            
            self.assertEqual(y.dtype, dtype)
            self.assertFalse(torch.isnan(y).any())

if __name__ == '__main__':
    unittest.main()
