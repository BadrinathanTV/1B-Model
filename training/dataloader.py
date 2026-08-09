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
        
        # Discover tokenized files (grouped by phase in the previous script)
        self.phases = sorted(glob.glob(os.path.join(data_dir, "phase_*")))
        
        if not self.phases:
            self.use_dummy = True
        else:
            self.use_dummy = False
            self.phase_datasets = {}  # {phase_dir: {file_path: valid_chunks}}
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
                
                phase_manifest = manifest["curriculum"][i] if manifest else None
                
                for f in bin_files:
                    # Lazily calculate chunks based on file size (2 bytes per uint16 token)
                    # This prevents creating unpicklable memmap objects in the main process (fixes Windows multiprocessing crash)
                    num_tokens = os.path.getsize(f) // 2
                    valid_chunks = num_tokens // self.chunk_size
                    
                    if valid_chunks > 0:
                        datasets[f] = valid_chunks
                        file_list.append(f)
                        
                        # Determine sampling weight
                        weight = valid_chunks
                        if phase_manifest:
                            basename = os.path.basename(f)
                            lang_code = basename.split("_")[0]
                            if lang_code in phase_manifest.get("monolingual", {}):
                                weight = phase_manifest["monolingual"][lang_code]
                            elif lang_code in phase_manifest.get("parallel", {}):
                                weight = phase_manifest["parallel"][lang_code]
                        
                        weights.append(weight)
                
                if file_list:
                    total_weight = sum(weights)
                    probs = [w / total_weight for w in weights]
                    
                    self.phase_datasets[phase_dir] = datasets
                    self.phase_weights[phase_dir] = probs
                    self.phase_file_list[phase_dir] = file_list

    def _get_dummy_stream(self, generator):
        while True:
            inputs = torch.randint(0, self.vocab_size, (self.seq_len,))
            targets = torch.randint(0, self.vocab_size, (self.seq_len,))
            yield inputs, targets

    def _get_phase_stream(self, phase_dir: str, generator):
        file_list = self.phase_file_list.get(phase_dir)
        if not file_list:
            return
            
        probs = self.phase_weights[phase_dir]
        datasets = self.phase_datasets[phase_dir]
        
        # Open memmaps lazily in the worker process
        memmaps = {f: np.memmap(f, dtype=np.uint16, mode='r') for f in file_list}
        pointers = {f: 0 for f in file_list}
        
        total_chunks = sum(datasets.values())
        probs_tensor = torch.tensor(probs, dtype=torch.float32)
        
        for _ in range(total_chunks):
            idx = torch.multinomial(probs_tensor, 1, generator=generator).item()
            selected_file = file_list[idx]
            
            m = memmaps[selected_file]
            max_chunks = datasets[selected_file]
            ptr = pointers[selected_file]
            
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
        seed = self.seed + (worker_info.id if worker_info else 0)
        generator = torch.Generator().manual_seed(seed)

        if self.use_dummy:
            yield from self._get_dummy_stream(generator)
        else:
            for phase_dir in self.phases:
                yield from self._get_phase_stream(phase_dir, generator)
