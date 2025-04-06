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
class MultiHeadAttention(nn.Module):
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

    def forward(self, x):
        B, T, C = x.size() # B =1, T = 13, C = 384
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        freqs = self.rope_freqs[:, :T, :, :].to(x.device)# torch.Size([1, 1, 512, 48])
        #q.shape torch.Size([1, 8, 13, 48])
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        if self.n_head != self.n_kv_head:
            # repeat keys and values across heads
            k = k.repeat(1, self.n_head // self.n_kv_head, 1, 1)
            v = v.repeat(1, self.n_head // self.n_kv_head, 1, 1)

        att = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        att = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0) * att
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = torch.matmul(att, v).transpose(1, 2).contiguous().view(B, T, C)
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
        self.mlp = SwiGLU(config.n_embd, 4 * config.n_embd)

    def forward(self, x):
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
        x = x + self.attn(self.ln1(x))  # (B, T, n_embd)
        # Apply the MLP with residual connection
        x = x + self.mlp(self.ln2(x))  # (B, T, n_embd)
        return x