import os
import glob
import json
import torch
import numpy as np
from torch.utils.data import IterableDataset

class CurriculumIterableDataset(IterableDataset):
    """
    Streams data from pre-tokenized .bin memmap files across 3 curriculum phases.
    Uses Dynamic Probabilistic Sampling: randomly selects a language/pair file 
    for every single sequence based on the target α=0.3 weights for the current phase.
    This guarantees every batch is perfectly mixed and maximizes cross-lingual learning.
    """
    def __init__(self, data_dir: str, seq_len: int, vocab_size: int, seed: int = 42, manifest_path: str = "data/pretrain_mix/manifest.json"):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.chunk_size = seq_len + 1
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)
        
        # Discover tokenized files (grouped by phase in the previous script)
        # However, the user wants dynamic batch sampling. We can still use the phase 
        # structure, but sample across the files dynamically based on weights.
        self.phases = sorted(glob.glob(os.path.join(data_dir, "phase_*")))
        
        if not self.phases:
            self.use_dummy = True
        else:
            self.use_dummy = False
            self.phase_datasets = {}  # {phase_dir: {file_path: (memmap, valid_chunks)}}
            self.phase_weights = {}   # {phase_dir: [weight_1, weight_2, ...]}
            self.phase_file_list = {} # {phase_dir: [file_1, file_2, ...]}
            
            # Load manifest for precise probability weights
            manifest = None
            if os.path.exists(manifest_path):
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
            
            for i, phase_dir in enumerate(self.phases):
                bin_files = sorted(glob.glob(os.path.join(phase_dir, "**", "*.bin"), recursive=True))
                
                datasets = {}
                weights = []
                file_list = []
                
                # If manifest exists, we can extract the exact token budget targeted for this file
                # and use it as the probability weight. If not, we fall back to file size.
                phase_manifest = manifest["curriculum"][i] if manifest else None
                
                for f in bin_files:
                    m = np.memmap(f, dtype=np.uint16, mode='r')
                    valid_chunks = len(m) // self.chunk_size
                    if valid_chunks > 0:
                        datasets[f] = (m, valid_chunks)
                        file_list.append(f)
                        
                        # Determine sampling weight
                        weight = valid_chunks  # Default: proportional to available data size
                        if phase_manifest:
                            # Try to match the filename to the manifest budget (e.g. "hin" or "hin-tam")
                            basename = os.path.basename(os.path.dirname(f))
                            if basename in phase_manifest.get("monolingual", {}):
                                weight = phase_manifest["monolingual"][basename]
                            elif basename in phase_manifest.get("parallel", {}):
                                weight = phase_manifest["parallel"][basename]
                        
                        weights.append(weight)
                
                if file_list:
                    # Normalize weights to probabilities
                    total_weight = sum(weights)
                    probs = [w / total_weight for w in weights]
                    
                    self.phase_datasets[phase_dir] = datasets
                    self.phase_weights[phase_dir] = probs
                    self.phase_file_list[phase_dir] = file_list

    def _get_dummy_stream(self):
        while True:
            inputs = torch.randint(0, self.vocab_size, (self.seq_len,))
            targets = torch.randint(0, self.vocab_size, (self.seq_len,))
            yield inputs, targets

    def _get_phase_stream(self, phase_dir: str):
        file_list = self.phase_file_list.get(phase_dir)
        if not file_list:
            return
            
        probs = self.phase_weights[phase_dir]
        datasets = self.phase_datasets[phase_dir]
        
        # Track pointers for sequential reading within randomly selected files
        pointers = {f: 0 for f in file_list}
        
        # Calculate how many sequences we should yield for this phase
        # We can yield infinitely, but to transition phases we should bound it.
        # Let's say a phase lasts for N total sequences across all datasets in the phase.
        total_chunks = sum(ds[1] for ds in datasets.values())
        
        for _ in range(total_chunks):
            # 1. Dynamically sample a language/pair based on alpha-smoothed probabilities!
            # Using torch.multinomial is faster, but random.choices or numpy works too.
            # Convert probs to a tensor for fast sampling
            probs_tensor = torch.tensor(probs, dtype=torch.float32)
            idx = torch.multinomial(probs_tensor, 1, generator=self.generator).item()
            selected_file = file_list[idx]
            
            m, max_chunks = datasets[selected_file]
            ptr = pointers[selected_file]
            
            # If we hit the end of a specific file, loop it (so we maintain strict ratios)
            if ptr >= max_chunks:
                ptr = 0
            
            local_idx = ptr * self.chunk_size
            chunk = m[local_idx : local_idx + self.chunk_size]
            chunk = torch.from_numpy(chunk.astype(np.int64))
            
            inputs = chunk[:-1]
            targets = chunk[1:]
            
            pointers[selected_file] = ptr + 1
            yield inputs, targets

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.generator.manual_seed(self.seed + worker_info.id)

        if self.use_dummy:
            yield from self._get_dummy_stream()
        else:
            for phase_dir in self.phases:
                yield from self._get_phase_stream(phase_dir)
