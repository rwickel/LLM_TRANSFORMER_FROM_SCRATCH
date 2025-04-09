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

        # # Initialize RoPE cache
        # self.register_buffer(
        #     "rope_cache",
        #     build_rope_cache(
        #         seq_len=config.block_size,
        #         dim=self.config.n_embd // self.config.n_head,
        #         device=config.device
        #     )
        # )

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
        self.config = config
        # Initialize the transformer decoder model
        self.transformer = DecoderModel(config)
        # Linear layer to project the output embeddings into the vocabulary size space for prediction
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)  # bias=True by default

        self.eos_token_id = getattr(config, 'eos_token_id', 50256) 

    def forward(self, input_ids, attention_mask=None, targets=None):
        """
        Forward pass through the decoder language model. The input tensor `idx` is passed through the transformer decoder,
        and the logits for next-token prediction are computed. If targets are provided, the loss is also calculated.
        
        Args:
            input (Tensor): The input tensor containing token IDs (B, T).
            targets (Optional[Tensor], optional): The ground truth token IDs for calculating the loss. Defaults to None.
        
        Returns:
            logits (Tensor): The output logits, representing the likelihood of each token in the vocabulary.
            loss (Optional[Tensor]): The cross-entropy loss if targets are provided.
        """        
        # Pass the input token IDs through the transformer model (returns hidden states/embeddings)
        hidden_states = self.transformer(input_ids)
        # Compute logits by projecting the embeddings to vocabulary size
        logits = self.lm_head(hidden_states)

        if targets is not None:
            if attention_mask is not None:
                # Mask out padding tokens in loss calculation
                loss_mask = attention_mask.view(-1) == 1
                logits = logits.view(-1, logits.size(-1))[loss_mask]
                targets = targets.view(-1)[loss_mask]
                
            loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),  # (batch_size * seq_len, vocab_size)
            targets.view(-1),                  # (batch_size * seq_len)
            ignore_index=self.config.pad_token_id  # Skip [PAD] tokens in loss
        )
            return logits, loss
        return logits
    

    def generate(self, input_ids, max_length=512, temperature=1.0, top_k=50):
        """
        Generate text autoregressively.
        
        Args:
            input_ids (Tensor): The starting input sequence (e.g., the prompt).
            max_length (int): Maximum length of generated sequence.
            temperature (float): Controls randomness of predictions. Higher values = more random.
            top_k (int): Limits the sampling pool to the top k logits.
        
        Returns:
            Tensor: The generated token IDs.
        """
        device = input_ids.device
        generated_ids = input_ids

        # Loop to generate tokens
        for _ in range(max_length):
            # Pass the current input through the model to get logits and hidden states for the next token
            logits = self.forward(generated_ids)
            logits = logits[:, -1, :]  # Get logits for the last token
            logits = logits / temperature  # Scale logits by temperature for randomness control

            # Apply top-k sampling if specified
            if top_k > 0:
                top_k_values, top_k_indices = logits.topk(top_k, dim=-1)
                logits = torch.zeros_like(logits).scatter_(-1, top_k_indices, top_k_values)
            
            # Sample a token from the distribution (sampling the next token)
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            next_token = next_token.squeeze(1)  # Remove extra dimensions

            # Print the generated token and its ID
            #print(f"Generated token: {next_token.item()} (ID: {next_token.item()})")

            # Append the generated token to the input sequence
            generated_ids = torch.cat((generated_ids, next_token.unsqueeze(1)), dim=1)

            # If the token is an end-of-sequence token, break early
            if next_token.item() == 102:
                break

        return generated_ids