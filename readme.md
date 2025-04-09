#IN PROGRESS Transformer Language Model with Rotary Positional Encoding (RoPE) and SwiGLU MLP 

This project implements a transformer-based language model, combining various modern techniques, such as **Rotary Positional Encoding (RoPE)** and **SwiGLU activation** in MLP layers, for token prediction and language modeling tasks. The model is designed to be flexible and modular, making it possible to extend or integrate it into other NLP applications. The code uses PyTorch and Hugging Face's transformers library.

## Features

- **Rotary Positional Encoding (RoPE)** for improved token position representation.
- **SwiGLU MLP** for the hidden layers of the transformer block.
- **Multi-Head Attention** with optional grouped query attention (GQA).
- **Layer Normalization** using RMSNorm for better convergence.
- **Pre-trained Sentence Transformer (MiniLM-L6)** for token embeddings.

## Model Overview

The model consists of the following components:

1. **Rotary Positional Encoding (RoPE):** A method for improving attention-based models by encoding the positional information directly into the attention mechanism.
2. **SwiGLU MLP:** An MLP with SwiGLU activation, used in the transformer blocks to improve model expressiveness.
3. **Grouped Query Attention (GQA):** Allows using different key-value heads from the attention mechanism to improve performance.
4. **Multi-Head Attention:** Standard attention mechanism with the flexibility to use RoPE for positional encoding.
5. **Transformer Decoder:** A stack of transformer blocks that generate embeddings for token sequences.
6. **Decoder Language Model (DecoderLM):** Uses the transformer decoder to predict the next token in a sequence.

## Dependencies

- `torch`
- `transformers`
- `matplotlib`
- `seaborn`
- `numpy`

Install dependencies via `pip`:

```bash
pip install -r requirements.txt
