import torch
import unittest

import sys
import os

# Add the directory to sys.path so we can import from the 1B-Model/training folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../training")))

from config import SLMConfig
from models.embedding import Embedding
from train import PretrainingDataset

class TestEmbedding(unittest.TestCase):
    def setUp(self):
        self.config = SLMConfig()
        self.config.vocab_size = 100
        self.config.hidden_size = 32
        self.config.embed_scale = False

    def test_embedding_forward_shape(self):
        embedding = Embedding(self.config)
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        
        output = embedding(input_ids)
        self.assertEqual(output.shape, (batch_size, seq_len, self.config.hidden_size))
        
        # Output should be exactly the word embeddings (no scaling when embed_scale=False)
        expected_output = embedding.word_embeddings(input_ids)
        torch.testing.assert_close(output, expected_output)

    def test_embedding_with_scale(self):
        self.config.embed_scale = True
        embedding = Embedding(self.config)
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        
        output = embedding(input_ids)
        self.assertEqual(output.shape, (batch_size, seq_len, self.config.hidden_size))
        
        # With scaling, output should be embeddings * sqrt(hidden_size)
        expected_output = embedding.word_embeddings(input_ids) * embedding.scale
        torch.testing.assert_close(output, expected_output)

    def test_embedding_backward(self):
        embedding = Embedding(self.config)
        
        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        
        output = embedding(input_ids)
        loss = output.sum()
        loss.backward()
        
        # Gradients should exist
        self.assertIsNotNone(embedding.word_embeddings.weight.grad)
        self.assertFalse(torch.isnan(embedding.word_embeddings.weight.grad).any())
        self.assertFalse(torch.isinf(embedding.word_embeddings.weight.grad).any())

class TestNTPDataPipeline(unittest.TestCase):
    def test_dataset_alignment(self):
        import tempfile
        import numpy as np
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a small dummy bin file
            dummy_data = np.arange(100, dtype=np.uint16)
            with open(os.path.join(tmpdir, "test.bin"), "wb") as f:
                f.write(dummy_data.tobytes())
                
            seq_len = 8
            vocab_size = 1000
            dataset = PretrainingDataset(tmpdir, seq_len, vocab_size)
            
            inputs, targets = dataset[0]
            
            # Standard NTP: inputs = chunk[:-1], targets = chunk[1:]
            self.assertEqual(len(inputs), seq_len)
            self.assertEqual(len(targets), seq_len)
            # targets should be shifted by 1
            for i in range(seq_len):
                self.assertEqual(targets[i].item(), inputs[i].item() + 1)

class TestModelIntegration(unittest.TestCase):
    def test_full_model_forward(self):
        from model import SLMModel
        from train import compute_loss
        from layers.rope import precompute_freqs_cis, precompute_cos_sin
        
        config = SLMConfig()
        config.vocab_size = 100
        config.hidden_size = 64
        config.num_hidden_layers = 2
        config.num_attention_heads = 2
        config.mtp_depth = 2
        config.training.seq_len = 8
        config.max_delta_history = 0
        
        model = SLMModel(config)
        model.train()
        
        batch_size = 2
        
        inputs = torch.randint(0, config.vocab_size, (batch_size, config.training.seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, config.training.seq_len))
        
        freqs_cis = precompute_freqs_cis(config.qk_rope_head_dim, config.training.seq_len, config.rope_theta, inputs.device)
        cos_cache, sin_cache = precompute_cos_sin(config.qk_rope_head_dim, config.training.seq_len, config.rope_theta, inputs.device)
        
        hidden_states_list = model(inputs, freqs_cis=freqs_cis, cos_cache=cos_cache, sin_cache=sin_cache, return_hidden_states=True)
        
        # Check MTP output shapes (hidden states instead of logits)
        self.assertEqual(len(hidden_states_list), 2)
        self.assertEqual(hidden_states_list[0].shape, (batch_size, config.training.seq_len, config.hidden_size))
        
        # Check loss computation doesn't crash
        loss, _ = compute_loss(hidden_states_list, targets, model.lm_head.weight, config)
        loss.backward()
        
        # Verify gradients flowed back to the embedding
        self.assertIsNotNone(model.embed.word_embeddings.weight.grad)
        
if __name__ == '__main__':
    unittest.main()
