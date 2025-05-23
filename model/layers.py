# layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.config import TransformerConfig
import math
from typing import Optional # Import Optional

# --- Positional Encoding ---
def build_rope_cache(seq_len: int, dim: int, device: torch.device, base: int = 10000) -> torch.Tensor:
    """
    Builds rotary positional embedding cache.

    Args:
        seq_len: Maximum sequence length.
        dim: Dimension of the embeddings for RoPE (usually head_dim).
        device: The torch device to place the cache on.
        base: The base value for the geometric progression of frequencies.

    Returns:
        A tensor of shape (seq_len, 1, dim / 2, 2) containing the cosine and sine frequencies.
    """
    assert dim % 2 == 0, "Dimension must be even for RoPE."
    theta = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    seq_idx = torch.arange(seq_len, device=device).float()
    idx_theta = torch.einsum("n,d->nd", seq_idx, theta)
    freqs = torch.stack((torch.cos(idx_theta), torch.sin(idx_theta)), dim=-1)
    print(f"Built RoPE cache with shape: {freqs.unsqueeze(1).shape} on device: {device}")
    return freqs.unsqueeze(1) # Shape: (seq_len, 1, dim/2, 2)

def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embedding to the input tensor.

    Args:
        x: Input tensor, typically query or key projections.
           Shape: (batch_size, num_heads, seq_len, head_dim)
        freqs: RoPE frequency cache.
               Shape: (seq_len, 1, head_dim / 2, 2)

    Returns:
        Tensor with RoPE applied, same shape as input x.
    """
    B, H, T, D = x.shape
    head_dim_half = D // 2
    assert D % 2 == 0, f"Head dimension ({D}) must be even for RoPE."
    assert freqs.shape[2] == head_dim_half, \
        f"RoPE cache dim ({freqs.shape[2]}) does not match half head dim ({head_dim_half})."

    freqs = freqs[:T] # Limit freqs to the current sequence length T
    x_pairs = x.float().reshape(B, H, T, head_dim_half, 2)
    x_complex = torch.view_as_complex(x_pairs)
    freqs_complex = torch.view_as_complex(freqs)

    # Reshape freqs for broadcasting: (T, 1, D/2) -> (1, 1, T, D/2)
    freqs_complex_broadcast = freqs_complex.squeeze(1).unsqueeze(0).unsqueeze(0)

    x_rotated_complex = x_complex * freqs_complex_broadcast
    x_rotated_pairs = torch.view_as_real(x_rotated_complex)
    x_out = x_rotated_pairs.reshape(B, H, T, D)
    return x_out.type_as(x)

# --- Layer Definitions ---
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
    """SwiGLU activation function."""
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)
        self.proj = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):
        return self.proj(F.silu(self.w1(x)) * self.w2(x))

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with optional Grouped Query Attention and RoPE."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = config.n_head
        # Handle potential None for n_kv_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim ** -0.5
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.num_key_value_groups = self.n_head // self.n_kv_head

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        self.use_rope = config.use_rope
        self.use_rope = config.use_rope
        if self.use_rope:
             self._rope_freqs_buffer_name = "rope_freqs"
             # Register buffer, but initialize lazily in forward
             self.register_buffer(self._rope_freqs_buffer_name, None, persistent=False)
             

    def forward(self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            use_cache: bool = False,
            cache_position: int = 0) -> torch.Tensor:

        B, T, C = x.size()

        # Project to queries, keys, values
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Rotary embedding (if used)
        if self.use_rope:
            current_rope_freqs = self.rope_freqs.to(x.device)
            q = apply_rope(q, current_rope_freqs)
            k = apply_rope(k, current_rope_freqs)

        # Repeat keys/values for multi-query attention
        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        # Compute attention scores
        att_scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # --- Construct Causal Mask ---
        causal_mask = torch.ones(T, T, dtype=torch.bool, device=att_scores.device).tril()
        causal_mask = causal_mask.view(1, 1, T, T)  # (1, 1, T, T) for broadcasting

        # --- Combine with Padding Mask (key-side masking only) ---
        if attention_mask is not None:
            key_mask = attention_mask.bool().view(B, 1, 1, T)  # (B, 1, 1, T)
            full_mask = causal_mask & key_mask  # final mask: (B, 1, T, T)
        else:
            full_mask = causal_mask  # (1, 1, T, T)

        # Apply mask: set masked positions to -inf
        att_scores = att_scores.masked_fill(~full_mask, float('-inf'))

        # Compute attention weights
        att_weights = F.softmax(att_scores, dim=-1)
        att_weights = self.dropout(att_weights)

        # Optionally zero out weights for padded queries (optional for safety)
        if attention_mask is not None:
            query_mask = attention_mask.bool().view(B, 1, T, 1)  # (B, 1, T, 1)
            att_weights = att_weights * query_mask  # suppress query-side pads

        # Apply attention to values
        att_output = att_weights @ v  # (B, H, T, D_h)
        att_output = att_output.transpose(1, 2).contiguous().view(B, T, C)

        # Final output projection
        att_output = self.out_proj(att_output)

        return att_output
    # --- End of MultiHeadAttention ---

class TransformerBlock(nn.Module):
    """A single block of the Transformer model."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.attn = MultiHeadAttention(config) # Uses the updated MultiHeadAttention
        self.ln2 = RMSNorm(config.n_embd)
        if config.intermediate_size is None:
             # Fallback if not set in config (though it should be)
             print("Warning: config.intermediate_size not set, calculating MLP hidden_dim.")
             hidden_dim = 4 * config.n_embd # A common default
        else:
             hidden_dim = config.intermediate_size
        self.mlp = SwiGLU(config.n_embd, hidden_dim, bias=config.bias)
        self.config = config

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (B, T, n_embd).
            attention_mask: Optional mask for attention of shape (B, T).
                            *** NOTE: The current MultiHeadAttention ignores this mask! ***

        Returns:
            Output tensor of the same shape as input.
        """
        # Pass x to attention, note that the provided attention_mask is ignored by the new self.attn.forward
        attn_output = self.attn(self.ln1(x), attention_mask=attention_mask)
        x = x + attn_output
        mlp_output = self.mlp(self.ln2(x))
        x = x + mlp_output
        return x
