import torch
from config import SLMConfig
from model import SLMModel
from accelerate import Accelerator
import torchao
from torchao.float8 import convert_to_float8_training, Float8LinearConfig

config = SLMConfig.from_yaml("fp8_pretraining/configs/default.yaml")
accelerator = Accelerator(mixed_precision="bf16")
model = SLMModel(config).to(accelerator.device)
print(f"After model to GPU: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

fp8_config = Float8LinearConfig()
convert_to_float8_training(model, config=fp8_config)
print(f"After TorchAO FP8: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
