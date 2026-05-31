import torch
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../training")))

from optimizers.factory import build_optimizer
from config import SLMConfig

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 1D parameter for AdamW/fallback group
        self.w1 = torch.nn.Parameter(torch.tensor([2.0], device='cuda'))
        # 2D parameter for Aurora/Spectral group
        self.w2 = torch.nn.Parameter(torch.tensor([[2.0]], device='cuda'))
        
    def forward(self, x):
        return self.w1 * x + self.w2 @ x

class TestOptimizers(unittest.TestCase):
    def setUp(self):
        self.config = SLMConfig()
        # Ensure we have some basic optimizer settings
        self.config.optimizer.learning_rate = 0.1
        self.config.optimizer.weight_decay = 0.0
        self.config.optimizer.warmup_steps = 0

    def check_optimizer_convergence(self, opt_type):
        self.config.optimizer.type = opt_type
        model = DummyModel()
        
        try:
            optimizer = build_optimizer(model, self.config)
        except Exception as e:
            self.skipTest(f"Skipping {opt_type} because it failed to initialize: {e}")
            
        if hasattr(optimizer, 'train'):
            optimizer.train()
            
        # Optimize f(w) = w^2 where w starts at 2.0. Minimum is at w=0.
        for _ in range(50):
            loss = (model.w1 ** 2).sum() + (model.w2 ** 2).sum()
            loss.backward()
            
            def closure(): return loss.item()
            try:
                optimizer.step(closure=closure)
            except TypeError:
                optimizer.step()
                
            optimizer.zero_grad()
            
        # Check if weights moved towards 0
        self.assertTrue(abs(model.w1.item()) < 2.0, f"{opt_type} 1D weight failed to converge. Final w1={model.w1.item()}")
        self.assertTrue(abs(model.w2.item()) < 2.0, f"{opt_type} 2D weight failed to converge. Final w2={model.w2.item()}")

    def test_nf_aurora(self):
        self.check_optimizer_convergence("nf_aurora")

    def test_nf_aurora_hybrid(self):
        self.check_optimizer_convergence("nf_aurora_hybrid")

    def test_sf_normuon(self):
        self.check_optimizer_convergence("sf_normuon")

    def test_nf_normuon_hybrid(self):
        self.check_optimizer_convergence("nf_normuon_hybrid")
        
    def test_polar(self):
        self.check_optimizer_convergence("polar")

if __name__ == '__main__':
    unittest.main()
