#!/bin/bash
export CUDA_HOME=$(.venv/bin/python -c 'import os, sys; print(os.path.join(os.path.dirname(sys.executable), "..", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia", "cu13"))')
export NCCL_HOME=$(.venv/bin/python -c 'import os, sys; print(os.path.join(os.path.dirname(sys.executable), "..", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia", "nccl"))')
export CUDNN_HOME=$(.venv/bin/python -c 'import os, sys; print(os.path.join(os.path.dirname(sys.executable), "..", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia", "cudnn"))')

export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/include:$NCCL_HOME/include:$CUDNN_HOME/include:$CPATH
export NVTE_FRAMEWORK=pytorch
uv pip install ninja cmake pybind11 setuptools
uv pip install --no-build-isolation "transformer-engine[pytorch]" --extra-index-url https://pypi.nvidia.com
