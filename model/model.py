# model.py
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
from typing import Optional, Tuple

from model.positional_encoding import build_rope_cache, apply_rope
from model.config import TransformerConfig
from model.layers import RMSNorm, SwiGLU, MultiHeadAttention, TransformerBlock

# -----------------------------
# Decoder Model Class
# -----------------------------

class DecoderModel(nn.Module):
    """The core Transformer decoder stack."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=config.pad_token_id)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = RMSNorm(config.n_embd)
        # Note: RoPE cache is handled within MultiHeadAttention layer

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # This forward pass still accepts attention_mask, but generate won't pass it
        B, T = idx.size()
        x = self.token_emb(idx) # (B, T, n_embd)
        for block in self.blocks:
            # Pass mask if provided (though generate won't provide it now)
            x = block(x)
        x = self.final_norm(x) # (B, T, n_embd)
        return x

# --- DecoderLM class with updated generate method ---
class DecoderLM(nn.Module):
    """Decoder Language Model with a head for predicting token probabilities."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        # Core transformer model
        self.transformer = DecoderModel(config)
        # Linear head to project hidden states to vocabulary logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        # Store EOS token ID from config if available
        self.eos_token_id = config.eos_token_id

        # Optional: Weight tying
        # self.transformer.token_emb.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights of Linear and Embedding layers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


    def forward(self,
                input_ids: torch.Tensor,                
                targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for training or inference.
        (Using GPT-style loss calculation as per previous step)

        Args:
            input_ids: Input token IDs (B, T).
            attention_mask: Mask for padding tokens (B, T).
            targets: Target token IDs for loss calculation (B, T).
                     *** IMPORTANT: Assumes targets are already prepared/shifted
                     correctly for next-token prediction relative to input_ids. ***

        Returns:
            A tuple containing:
            - logits: Output logits of shape (B, T, vocab_size).
            - loss: Calculated cross-entropy loss if targets are provided, else None.
        """
        # Pass input token IDs and mask through the transformer model
        # Note: attention_mask is accepted here but won't be passed by the new generate
        hidden_states = self.transformer(input_ids) # (B, T, n_embd)

        # Compute logits for the entire sequence
        logits = self.lm_head(hidden_states) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # --- GPT-style Loss Calculation ---
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), # Reshape logits to (B*T, V)
                targets.view(-1),                 # Reshape targets to (B*T,)
                ignore_index=-1                   # Use -1 as the ignore index
            )
            return logits, loss
            # --- End GPT-style Loss ---

        return logits, None # Loss is None here

    @torch.no_grad() # Disable gradient calculation during generation
    def generate(self,
                 input_ids: torch.Tensor,
                 max_tokens: int,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None) -> torch.Tensor: # Return only generated IDs
        """
        Generates token sequences autoregressively, combining features.
        Does NOT use attention_mask.

        Args:
            input_ids: Starting sequence of token IDs (B, T_prompt).
            max_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature. Lower values make the output more deterministic.
            top_k: If set, restricts sampling to the top k most likely tokens.

        Returns:
            torch.Tensor: The full sequence including prompt and generated tokens (B, T_prompt + T_generated).
        """
        self.eval() # Set model to evaluation mode
        device = input_ids.device
        B, T = input_ids.size()
        generated_ids = input_ids # Start with the prompt

        for _ in range(max_tokens):
            # --- Context Cropping ---
            current_seq_len = generated_ids.size(1)
            if current_seq_len <= self.config.block_size:
                ids_to_pass = generated_ids
            else:
                # If sequence is too long, crop to the last block_size tokens
                ids_to_pass = generated_ids[:, -self.config.block_size:]
            # --- End Context Cropping ---

            # --- Forward Pass (No Attention Mask) ---
            # Pass None for attention_mask and targets
            logits, _ = self.forward(ids_to_pass, targets=None)
            # Get logits for the very last token position
            next_token_logits = logits[:, -1, :] # Shape: (B, V)
            # --- End Forward Pass ---

            # --- Temperature Scaling ---
            if temperature > 0 and temperature != 1.0:
                 next_token_logits = next_token_logits / temperature
            # --- End Temperature Scaling ---

            # --- Top-K Filtering (Robust version) ---
            if top_k is not None and top_k > 0:
                # Use min() for robustness if top_k > vocab_size
                k = min(top_k, next_token_logits.size(-1))
                top_k_values, _ = torch.topk(next_token_logits, k=k, dim=-1)
                # Get the threshold (k-th value)
                threshold = top_k_values[:, [-1]]
                # Mask logits below the threshold
                next_token_logits[next_token_logits < threshold] = -float('Inf')
            # --- End Top-K Filtering ---

            # --- Sampling (with explicit greedy) ---
            if temperature == 0: # Greedy decoding
                 next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True) # (B, 1)
            else: # Sampling
                 probs = F.softmax(next_token_logits, dim=-1)
                 next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
            # --- End Sampling ---

            # --- Append Token ---
            generated_ids = torch.cat((generated_ids, next_token), dim=1)
            # --- End Append Token ---

            # --- EOS Check ---
            if self.eos_token_id is not None and (next_token == self.eos_token_id).all():
                break
            # --- End EOS Check ---

        self.train() # Set model back to training mode
        return generated_ids # Return only the generated IDs

# --- End of Updated DecoderLM class ---
