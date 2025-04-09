# layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.positional_encoding import apply_rope, build_rope_cache
import math

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm / math.sqrt(x.size(-1))
        return self.weight * (x / (rms + self.eps))

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.proj(F.silu(self.w1(x)) * self.w2(x))

# -----------------------------
# Attention with GQA + RoPE
# -----------------------------
class MultiHeadAttentionOld(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.head_dim = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head or config.n_head  # Grouped Query Attention
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        self.rope_freqs = build_rope_cache(config.block_size, self.head_dim, device=config.device)

    def forward(self, x, rope_freqs=None, attention_mask=None):
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, n_embd)
            attention_mask: optional mask of shape (batch_size, seq_len)
                           where 1 = keep token, 0 = mask out (e.g., for [PAD] tokens)
        """
        B, T, C = x.size()
        
        # Projections
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if rope_freqs is not None:
            q = apply_rope(q, rope_freqs)
            k = apply_rope(k, rope_freqs)

        # Grouped Query Attention
        if self.n_head != self.n_kv_head:
            k = k.repeat(1, self.n_head // self.n_kv_head, 1, 1)
            v = v.repeat(1, self.n_head // self.n_kv_head, 1, 1)

        # Attention scores
        att = (q @ k.transpose(-2, -1)) * self.scale

        # Masking
        if attention_mask is not None:
            # Combine causal mask with padding mask
            causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            combined_mask = causal_mask & (padding_mask != 0)
            att = att.masked_fill(~combined_mask, float('-inf'))
        else:
            # Default causal mask
            att = att.masked_fill(torch.tril(torch.ones(T, T, device=x.device)) == 0, float('-inf'))

        # Softmax and output
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Ensure embedding dimension is divisible by number of heads
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        
        # Attention parameters
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if hasattr(config, 'n_kv_head') else config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim ** -0.5
        
        # Projection layers
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # Regularization
        self.dropout = nn.Dropout(config.dropout)
        
        # Positional embeddings (optional)
        self.use_rope = getattr(config, 'use_rope', True)
        if self.use_rope:
            self.rope_freqs = build_rope_cache(
                config.block_size,
                self.head_dim,
                device=config.device
            )

    def forward(self, x, attention_mask=None, use_cache=False, cache_position=0):
        B, T, C = x.size()
        
        # Project queries, keys, values
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Apply rotary positional embeddings if enabled
        if self.use_rope:
            rope_freqs = self.rope_freqs[:T]  # Use only frequencies up to sequence length
            q = apply_rope(q, rope_freqs)
            k = apply_rope(k, rope_freqs)

        # Handle grouped query attention
        if self.n_head != self.n_kv_head:
            k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
            v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        # Attention scores
        att = (q @ k.transpose(-2, -1)) * self.scale

        # Masking logic
        if attention_mask is not None:
            # Create causal mask
            causal_mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            
            # Combine with padding mask
            padding_mask = attention_mask.view(B, 1, 1, T)
            combined_mask = causal_mask & padding_mask
            
            # Apply mask
            att = att.masked_fill(~combined_mask, float('-inf'))
        else:
            # Default causal mask
            att = att.masked_fill(~torch.ones(T, T, device=x.device).tril(), float('-inf'))

        # Softmax and attention output
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(out)

# -----------------------------
# Transformer Block
# -----------------------------
class TransformerBlock(nn.Module):
    """
    A single transformer block that consists of a multi-head attention layer followed by an MLP (with SwiGLU activation),
    and residual connections with layer normalization applied before each operation.

    Args:
        config: A configuration object that contains hyperparameters like embedding size, number of heads, etc.
    """
    def __init__(self, config):
        super().__init__()
        # Layer normalization before the attention layer
        self.ln1 = RMSNorm(config.n_embd)
        # Multi-head attention layer
        self.attn = MultiHeadAttention(config)
        # Layer normalization before the MLP layer
        self.ln2 = RMSNorm(config.n_embd)
        # MLP with SwiGLU activation
        hidden_dim = int(8 / 3 * config.n_embd)
        self.mlp = SwiGLU(config.n_embd, hidden_dim)

    def forward(self, x, rope_freqs=None, attention_mask=None):
        """
        Forward pass through the transformer block. The input tensor `x` passes through the following operations:
        1. Layer Normalization -> Multi-head Attention -> Residual connection
        2. Layer Normalization -> MLP -> Residual connection
        
        Args:
            x (Tensor): The input tensor of shape (B, T, n_embd), where B is batch size, T is sequence length, and n_embd is the embedding dimension.
        
        Returns:
            Tensor: The output tensor after applying the attention and MLP layers, with the same shape as the input.
        """
        # Apply the multi-head attention with residual connection
        x = x + self.attn(self.ln1(x), rope_freqs=rope_freqs, attention_mask=attention_mask)

        # Apply the MLP with residual connection
        x = x + self.mlp(self.ln2(x))  # (B, T, n_embd)
        return x