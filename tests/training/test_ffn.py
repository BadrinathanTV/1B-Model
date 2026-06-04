import torch
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../training")))

from config import SLMConfig
from layers.ffn import DenseFFN

class TestDenseFFN(unittest.TestCase):
    def setUp(self):
        self.config = SLMConfig()
        self.config.hidden_size = 64
        self.config.intermediate_size = 128
        self.batch_size = 2
        self.seq_len = 8

    def test_ffn_forward_shape(self):
        ffn = DenseFFN(self.config)
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        y = ffn(x)
        self.assertEqual(y.shape, x.shape)

    def test_ffn_backward(self):
        ffn = DenseFFN(self.config)
        x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size, requires_grad=True)
        y = ffn(x)
        loss = y.sum()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(ffn.gate_up_proj.weight.grad)
        self.assertIsNotNone(ffn.down_proj.weight.grad)

    def test_liger_fallback_equivalence(self):
        import layers.ffn as ffn_module
        orig_liger = ffn_module.LIGER_SWIGLU
        
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if device.type != 'cuda':
                self.skipTest("CUDA required for Liger tests")
                
            ffn = DenseFFN(self.config).to(device)
            x = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size, device=device)
            
            # Fallback
            ffn_module.LIGER_SWIGLU = False
            y_fb = ffn(x)
            
            # Liger
            ffn_module.LIGER_SWIGLU = True
            if hasattr(ffn_module, 'LigerSiLUMulFunction'):
                y_liger = ffn(x)
                torch.testing.assert_close(y_fb, y_liger, rtol=1e-3, atol=1e-3)
        finally:
            ffn_module.LIGER_SWIGLU = orig_liger

if __name__ == '__main__':
    unittest.main()
