import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import sys
import os

# Add the directory to sys.path so we can import from the 1B-Model/training folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../training")))

from config import SLMConfig
from train import compute_loss

class TestNTPCELoss(unittest.TestCase):
    def setUp(self):
        self.config = SLMConfig()
        self.config.vocab_size = 100
        self.config.hidden_size = 32
        self.config.z_loss_weight = 1e-4
        self.config.mtp_depth = 2  # Enable MTP for MTP tests

        # Simple projection to compute logits from hidden states
        self.lm_head_weight = nn.Parameter(torch.randn(self.config.vocab_size, self.config.hidden_size))

    def test_ce_loss_single_head(self):
        batch_size = 2
        seq_len = 8
        
        targets = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size, requires_grad=True)
        hidden_states_list = [hidden_states]

        loss, metrics = compute_loss(
            hidden_states_list, targets, self.lm_head_weight, self.config
        )
        
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(loss > 0)
        self.assertIn("main_loss", metrics)
        self.assertIn("z_loss", metrics)
        self.assertIn("mtp_loss", metrics)
        self.assertEqual(metrics["mtp_loss"].item(), 0.0)

        loss.backward()
        self.assertIsNotNone(hidden_states.grad)
        self.assertFalse(torch.isnan(hidden_states.grad).any())

    def test_ce_loss_with_mtp_heads(self):
        batch_size = 2
        seq_len = 8
        
        targets = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        
        hidden_states_main = torch.randn(batch_size, seq_len, self.config.hidden_size, requires_grad=True)
        hidden_states_mtp = torch.randn(batch_size, seq_len, self.config.hidden_size, requires_grad=True)
        hidden_states_list = [hidden_states_main, hidden_states_mtp]

        loss, metrics = compute_loss(
            hidden_states_list, targets, self.lm_head_weight, self.config
        )
        
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(loss > 0)
        self.assertIn("main_loss", metrics)
        self.assertIn("mtp_loss", metrics)
        self.assertTrue(metrics["mtp_loss"] > 0)

        # Manually compute MTP CE to verify it matches
        # MTP head 1 (i=1) uses hidden_states_mtp[:, :-1, :] vs targets[:, 1:]
        # compute_loss trims mtp_hs from the end to align with targets
        mtp_h = hidden_states_mtp[:, :-1, :]
        mtp_targets = targets[:, 1:1 + mtp_h.shape[1]]
        logits = F.linear(mtp_h, self.lm_head_weight)
        expected_mtp_ce = F.cross_entropy(logits.reshape(-1, self.config.vocab_size), mtp_targets.reshape(-1))
        # compute_loss applies mtp_weight=0.3 (step < 67% of max_steps)
        expected_mtp_loss = 0.3 * expected_mtp_ce

        self.assertAlmostEqual(metrics["mtp_loss"].item(), expected_mtp_loss.item(), places=5)

        loss.backward()
        self.assertIsNotNone(hidden_states_main.grad)
        self.assertIsNotNone(hidden_states_mtp.grad)
        self.assertFalse(torch.isnan(hidden_states_main.grad).any())
        self.assertFalse(torch.isnan(hidden_states_mtp.grad).any())

if __name__ == '__main__':
    unittest.main()
