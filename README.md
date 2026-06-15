# Umbrellm

A small language model framework that lets you train and run your own decoder-only transformer locally.

## Overview

Umbrellm is a language model implementation written in Python and PyTorch. The default model configuration uses 12 transformer layers, 8 attention heads, and a hidden size of 512 dimensions, resulting in roughly 20 million parameters.

The goal of the project is to provide an implementation that is easy to understand and modify. It includes the components needed to train a model from scratch, prepare datasets, run inference, and expose the model through a simple API.

The architecture uses:

* Rotary Position Embeddings (RoPE)
* SwiGLU feed-forward activations
* RMSNorm
* Causal multi-head self-attention

## Features

* Train models from scratch using your own datasets
* Save and resume training from checkpoints
* Learning rate scheduling and gradient clipping
* Optional mixed-precision training
* BPE tokenizer training on custom corpora
* Dataset compilation using the `.ullm` format
* Text generation with temperature, top-k, top-p, and repetition penalty controls
* Token-by-token streaming generation
* FastAPI-based REST API server

## Quick start

```bash
pip install torch fastapi uvicorn

python scripts/compile_dataset.py
python scripts/train.py --epochs 3
python scripts/inference.py --interactive
```

## Python SDK

```python
import umbrellm

umbrellm.utllm.compile()
umbrellm.training.train({"epochs": 3})

response = umbrellm.generate("Explain black holes simply.")
print(response)
```

## Requirements

* Python 3.9 or newer
* PyTorch 2.0 or newer
* FastAPI and uvicorn (only required for the API server)

## License

Released under the MIT License.
