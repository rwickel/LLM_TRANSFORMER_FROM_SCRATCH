import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoTokenizer, AutoModel
import math
from dataclasses import dataclass
from typing import Optional

from model.positional_encoding import build_rope_cache, apply_rope
from model.config import TransformerConfig
from model.layers import RMSNorm, SwiGLU, MultiHeadAttention, TransformerBlock

# -----------------------------
# Decoder Model Class
# -----------------------------
class DecoderModel(nn.Module):
    """
    A decoder model consisting of a stack of transformer blocks for generating token sequences.
    
    Args:
        config: A configuration object containing hyperparameters for the model.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        # Embedding layer to convert input token IDs into vectors of size n_embd
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # A list of transformer blocks (each consisting of attention and MLP layers)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        # Final layer normalization to be applied after all transformer blocks
        self.final_norm = RMSNorm(config.n_embd)

    def forward(self, idx):
        """
        Forward pass through the decoder model. The input tensor `idx` is passed through the token embedding layer,
        followed by a stack of transformer blocks, and finally a layer normalization.

        Args:
            idx (Tensor): The input tensor containing token IDs (B, T), where B is batch size and T is sequence length.
        
        Returns:
            Tensor: The output tensor after passing through the transformer layers, with shape (B, T, n_embd).
        """
        B, T = idx.size()
        # Get the embeddings of the input token IDs
        x = self.token_emb(idx)  # (B, T, n_embd)
        # Pass the embeddings through the transformer blocks
        for block in self.blocks:
            x = block(x)
        # Apply final layer normalization
        x = self.final_norm(x)   # (B, T, n_embd)
        return x

# -----------------------------
# Decoder Language Model Class
# -----------------------------
class DecoderLM(nn.Module):
    """
    A language model built on top of the DecoderModel. It includes a final linear layer to generate logits for 
    each token in the vocabulary, which is used for next-token prediction.

    Args:
        config: A configuration object containing hyperparameters for the model.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        # Initialize the transformer decoder model
        self.transformer = DecoderModel(config)
        # Linear layer to project the output embeddings into the vocabulary size space for prediction
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        """
        Forward pass through the decoder language model. The input tensor `idx` is passed through the transformer decoder,
        and the logits for next-token prediction are computed. If targets are provided, the loss is also calculated.
        
        Args:
            idx (Tensor): The input tensor containing token IDs (B, T).
            targets (Optional[Tensor], optional): The ground truth token IDs for calculating the loss. Defaults to None.
        
        Returns:
            logits (Tensor): The output logits, representing the likelihood of each token in the vocabulary.
            loss (Optional[Tensor]): The cross-entropy loss if targets are provided.
        """
        # Pass the input token IDs through the transformer model
        x = self.transformer(idx)
        # Compute logits by projecting the embeddings to vocabulary size
        logits = self.lm_head(x)

        if targets is not None:
            # Compute the loss by comparing the predicted logits with the target tokens
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return logits     # If no targets are provided, return the logits
