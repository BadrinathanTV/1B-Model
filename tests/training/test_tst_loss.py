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

class TestTSTLoss(unittest.TestCase):
    def setUp(self):
        self.config = SLMConfig()
        self.config.vocab_size = 100
        self.config.hidden_size = 32
        self.config.z_loss_weight = 1e-4
        self.config.training.mtp_loss_weight = 0.3
        self.config.training.mtp_loss_weight_final = 0.1
        self.config.training.mtp_anneal_fraction = 0.67
        self.config.training.max_steps = 100

        # Simple projection to compute logits from hidden states
        self.lm_head_weight = nn.Parameter(torch.randn(self.config.vocab_size, self.config.hidden_size))

    def test_standard_ce_loss_recovery_phase(self):
        batch_size = 2
        seq_len = 8
        
        # In recovery phase, targets is shape (batch_size, seq_len)
        targets = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size, requires_grad=True)
        hidden_states_list = [hidden_states]

        # Calculate loss (is_superposition = False)
        loss, metrics = compute_loss(
            hidden_states_list, targets, self.lm_head_weight, self.config, step=10, is_superposition=False
        )
        
        # Verify shape & properties
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(loss > 0)
        self.assertIn("main_loss", metrics)
        self.assertIn("z_loss", metrics)
        self.assertEqual(metrics["mtp_loss"], 0.0)

        # Backward pass check
        loss.backward()
        self.assertIsNotNone(hidden_states.grad)
        self.assertFalse(torch.isnan(hidden_states.grad).any())

    def test_mce_loss_superposition_phase(self):
        batch_size = 2
        seq_len = 8
        group_size = 4
        
        # In superposition phase, targets is shape (batch_size, seq_len, group_size)
        targets = torch.randint(0, self.config.vocab_size, (batch_size, seq_len, group_size))
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size, requires_grad=True)
        hidden_states_list = [hidden_states]

        # Calculate loss (is_superposition = True)
        loss, metrics = compute_loss(
            hidden_states_list, targets, self.lm_head_weight, self.config, step=10, is_superposition=True
        )
        
        # Verify shape & properties
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(loss > 0)
        self.assertIn("main_loss", metrics)
        self.assertIn("z_loss", metrics)

        # Manually compute MCE to verify it is exactly the average CE
        manual_mce = 0.0
        for j in range(group_size):
            t_j = targets[:, :, j]
            logits = F.linear(hidden_states, self.lm_head_weight)
            manual_mce += F.cross_entropy(logits.view(-1, self.config.vocab_size), t_j.view(-1))
        expected_main_loss = manual_mce / group_size

        self.assertAlmostEqual(metrics["main_loss"].item(), expected_main_loss.item(), places=5)

        # Backward pass check
        loss.backward()
        self.assertIsNotNone(hidden_states.grad)
        self.assertFalse(torch.isnan(hidden_states.grad).any())

    def test_mce_loss_superposition_phase_with_mtp(self):
        batch_size = 2
        seq_len = 8
        group_size = 4
        
        # In superposition phase with MTP, targets is shape (batch_size, seq_len, group_size)
        targets = torch.randint(0, self.config.vocab_size, (batch_size, seq_len, group_size))
        
        hidden_states_main = torch.randn(batch_size, seq_len, self.config.hidden_size, requires_grad=True)
        hidden_states_mtp = torch.randn(batch_size, seq_len, self.config.hidden_size, requires_grad=True)
        hidden_states_list = [hidden_states_main, hidden_states_mtp]

        # Calculate loss (is_superposition = True)
        loss, metrics = compute_loss(
            hidden_states_list, targets, self.lm_head_weight, self.config, step=10, is_superposition=True
        )
        
        # Verify shape & properties
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(loss > 0)
        self.assertIn("main_loss", metrics)
        self.assertIn("mtp_loss", metrics)
        self.assertTrue(metrics["mtp_loss"] > 0)

        # Manually compute MTP MCE to verify it matches
        # For MTP-1 (i=1), targets are targets[:, 1:, :] and mtp_h is hidden_states_mtp[:, :-1, :]
        mtp_h = hidden_states_mtp[:, :-1, :]
        mtp_targets = targets[:, 1:, :]
        manual_mtp_mce = 0.0
        for j in range(group_size):
            t_j = mtp_targets[:, :, j].contiguous()
            logits = F.linear(mtp_h, self.lm_head_weight)
            manual_mtp_mce += F.cross_entropy(logits.view(-1, self.config.vocab_size), t_j.view(-1))
        expected_mtp_loss = manual_mtp_mce / group_size

        self.assertAlmostEqual(metrics["mtp_loss"].item(), expected_mtp_loss.item(), places=5)

        # Backward pass check
        loss.backward()
        self.assertIsNotNone(hidden_states_main.grad)
        self.assertIsNotNone(hidden_states_mtp.grad)
        self.assertFalse(torch.isnan(hidden_states_main.grad).any())
        self.assertFalse(torch.isnan(hidden_states_mtp.grad).any())

if __name__ == '__main__':
    unittest.main()
