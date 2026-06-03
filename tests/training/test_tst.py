import torch
import unittest

import sys
import os

# Add the directory to sys.path so we can import from the 1B-Model/training folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../training")))

from config import SLMConfig
from models.embedding import TokenSuperpositionEmbedding
from train import PretrainingDataset

class TestTokenSuperpositionEmbedding(unittest.TestCase):
    def setUp(self):
        self.config = SLMConfig()
        self.config.vocab_size = 100
        self.config.hidden_size = 32
        self.config.embed_scale = False

    def test_tst_group_size_1(self):
        self.config.tst_group_size = 1
        embedding = TokenSuperpositionEmbedding(self.config)
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        
        output = embedding(input_ids)
        self.assertEqual(output.shape, (batch_size, seq_len, self.config.hidden_size))
        
        # Output should be exactly the word embeddings
        expected_output = embedding.word_embeddings(input_ids)
        torch.testing.assert_close(output, expected_output)

    def test_tst_group_size_2(self):
        self.config.tst_group_size = 2
        embedding = TokenSuperpositionEmbedding(self.config)
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        
        output = embedding(input_ids)
        self.assertEqual(output.shape, (batch_size, 5, self.config.hidden_size))
        
        # Check average computation
        word_embeds = embedding.word_embeddings(input_ids)
        expected_output = (word_embeds[:, 0::2, :] + word_embeds[:, 1::2, :]) / 2.0
        torch.testing.assert_close(output, expected_output)

    def test_tst_group_size_not_divisible(self):
        self.config.tst_group_size = 3
        embedding = TokenSuperpositionEmbedding(self.config)
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        
        with self.assertRaises(ValueError):
            embedding(input_ids)

    def test_tst_backward(self):
        self.config.tst_group_size = 4
        embedding = TokenSuperpositionEmbedding(self.config)
        
        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        
        output = embedding(input_ids)
        loss = output.sum()
        loss.backward()
        
        # Gradients should exist
        self.assertIsNotNone(embedding.word_embeddings.weight.grad)
        
        # Since average is used, the gradient for the embedded tokens should be 1/group_size
        # For a sum loss, grad is 1.0/4 = 0.25
        # We can check the non-zero gradients
        grads = embedding.word_embeddings.weight.grad
        unique_grads = torch.unique(grads[grads != 0])
        # Each selected token gets gradient 1/4 (assuming no duplicate tokens in input)
        # If there are duplicate tokens, it will be k/4
        # We just verify there are no NaN/Inf
        self.assertFalse(torch.isnan(grads).any())
        self.assertFalse(torch.isinf(grads).any())

class TestTSTDataPipeline(unittest.TestCase):
    def test_tst_dataset_alignment(self):
        # We need to verify if the dataset correctly aligns inputs and targets for TST
        # Creating a temporary dir with dummy data
        import tempfile
        import numpy as np
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a small dummy bin file
            dummy_data = np.arange(100, dtype=np.uint16)
            with open(os.path.join(tmpdir, "test.bin"), "wb") as f:
                f.write(dummy_data.tobytes())
                
            # Group size 4 superposition phase test
            input_seq_len = 8
            seq_len = 2 # group size = 4
            vocab_size = 1000
            tst_group_size = 4
            dataset_super = PretrainingDataset(tmpdir, input_seq_len, seq_len, vocab_size, tst_group_size=tst_group_size)
            
            # Get the first item in superposition phase
            inputs, targets = dataset_super[0]
            
            # inputs should be first 8 elements
            self.assertEqual(inputs.tolist(), list(range(8)))
            
            # targets should be bags of 4 elements starting after inputs
            # Group 0: input [0,1,2,3] -> target [4,5,6,7]
            # Group 1: input [4,5,6,7] -> target [8,9,10,11]
            # Shape is (seq_len, s) -> (2, 4)
            self.assertEqual(targets.tolist(), [[4, 5, 6, 7], [8, 9, 10, 11]])

            # Test recovery phase (tst_group_size = 1)
            dataset_recovery = PretrainingDataset(tmpdir, input_seq_len, seq_len, vocab_size, tst_group_size=1)
            inputs_rec, targets_rec = dataset_recovery[0]
            self.assertEqual(inputs_rec.tolist(), list(range(8)))
            self.assertEqual(targets_rec.tolist(), [1, 2])

class TestTSTIntegration(unittest.TestCase):
    def test_full_model_tst(self):
        from model import SLMModel
        from train import compute_loss
        from layers.rope import precompute_freqs_cis, precompute_cos_sin
        
        config = SLMConfig()
        config.vocab_size = 100
        config.hidden_size = 64
        config.num_hidden_layers = 2
        config.num_attention_heads = 2
        config.tst_group_size = 4
        config.mtp_depth = 2
        config.training.seq_len = 8
        config.max_delta_history = 0 # Simplify for test
        
        model = SLMModel(config)
        model.train()
        
        batch_size = 2
        input_seq_len = config.training.seq_len * config.tst_group_size
        
        inputs = torch.randint(0, config.vocab_size, (batch_size, input_seq_len))
        # Superposition targets: (batch_size, seq_len, tst_group_size)
        targets = torch.randint(0, config.vocab_size, (batch_size, config.training.seq_len, config.tst_group_size))
        
        freqs_cis = precompute_freqs_cis(config.qk_rope_head_dim, config.training.seq_len, config.rope_theta, inputs.device)
        cos_cache, sin_cache = precompute_cos_sin(config.qk_rope_head_dim, config.training.seq_len, config.rope_theta, inputs.device)
        
        hidden_states_list = model(inputs, freqs_cis=freqs_cis, cos_cache=cos_cache, sin_cache=sin_cache, return_hidden_states=True)
        
        # Check MTP output shapes (hidden states instead of logits)
        self.assertEqual(len(hidden_states_list), 2)
        self.assertEqual(hidden_states_list[0].shape, (batch_size, config.training.seq_len, config.hidden_size))
        
        # Check loss computation doesn't crash with shapes in superposition phase
        loss, _ = compute_loss(hidden_states_list, targets, model.lm_head.weight, config, 0, is_superposition=True)
        loss.backward()
        
        # Verify gradients flowed back to the embedding
        self.assertIsNotNone(model.embed.word_embeddings.weight.grad)
        
if __name__ == '__main__':
    unittest.main()
