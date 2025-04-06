# positional_encoding.py

import torch
import math

# -----------------------------
# Rotary Positional Embedding
# -----------------------------
def rotate_half(x):
    """
    Rotates the second half of the input tensor `x` and swaps it with the first half.
    
    This function is used to apply the rotary positional encoding, which involves 
    rotating the second half of the tensor to create a positional encoding effect.
    
    Args:
        x (Tensor): Input tensor of shape (B, T, 2*head_dim), 
                    where B is batch size, T is sequence length, and 2*head_dim is the feature dimension 
                    (must be even for splitting into two equal parts).
                    
    Returns:
        Tensor: The rotated tensor of the same shape as `x`, where the second half of the tensor is rotated 
                and concatenated with the first half.
    """
    # Split the input tensor `x` into two equal chunks along the last dimension (head_dim).
    x1, x2 = x.chunk(2, dim=-1)  # `x1` and `x2` have shapes (B, T, head_dim)
    
    # Concatenate the negative of `x2` (second half) with `x1` (first half) along the last dimension.
    # This operation swaps and negates the second half of the tensor, which is used for RoPE.
    return torch.cat([-x2, x1], dim=-1)  # The output has the same shape as the input (B, T, 2*head_dim)


def apply_rope(x, rope_freqs):
    """
    Applies Rotary Positional Embedding (RoPE) to the input tensor `x`.
    
    Args:
        x (Tensor): Input tensor of shape (B, n_head, T, head_dim), 
                    where B is batch size, n_head is the number of attention heads, 
                    T is the sequence length, and head_dim is the dimension of each head.
        rope_freqs (Tensor): The rotary positional frequencies of shape (1, n_head, max_len, head_dim), 
                              where max_len is the maximum sequence length and head_dim is the head dimension.
                              
    Returns:
        Tensor: The input tensor `x` after applying the rotary positional encoding, 
                with the same shape as the input tensor.
    """
    B, n_head, T, head_dim = x.size()  # B = batch size, n_head = number of attention heads, T = sequence length, head_dim = dimension of each head
    
    # Slice `rope_freqs` to match the sequence length T. We take the first T positions of the frequencies.
    # `rope_freqs` is assumed to have shape [1, n_head, max_len, head_dim], so we slice it to [B, n_head, T, head_dim].
    freqs = rope_freqs[:, :, :T, :].to(x.device)  # Ensure the tensor is moved to the correct device (GPU or CPU)
    
    # Apply the rotary positional encoding: element-wise cosine and sine of the frequencies
    # We multiply the input tensor `x` by the cosine of the frequencies and the rotated version of `x` by the sine of the frequencies.
    # `rotate_half(x)` rotates the second half of the tensor to prepare for the positional encoding.
    x_rot = x * freqs.cos() + rotate_half(x) * freqs.sin()  # Apply RoPE (cosine to first half, sine to second half of x)
    
    return x_rot  # Return the rotated tensor after applying RoPE


def build_rope_cache(seq_len, head_dim, base=10000.0, device="cuda"):
    """
    Builds the rotary positional encoding cache (rope_freqs) for a given sequence length and head dimension.
    
    This function computes the frequencies used for Rotary Positional Embedding (RoPE), which are 
    essential for applying RoPE in attention mechanisms. The frequencies are based on the formula 
    used in RoPE, where the frequency values are derived from a base (commonly set to 10000.0) and 
    the head dimension.

    Args:
        seq_len (int): The length of the sequence (T), which will define how many positions to encode.
        head_dim (int): The dimensionality of each attention head, which determines the number of features in the encoding.
        base (float, optional): The base used for the positional encoding frequency calculation. Defaults to 10000.0.
        device (str, optional): The device on which the tensor will be created ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        Tensor: A tensor of shape (1, 1, seq_len, head_dim) containing the positional encoding frequencies 
                to be applied to the input tensor during the attention computation.
    """
    # Compute the frequencies for the rotary positional encoding (RoPE) based on the base and head_dim.
    # The `theta` values represent the angular frequencies for each dimension in RoPE.
    # We take every other element (`torch.arange(0, head_dim, 2)`) and divide by head_dim.
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))

    # Generate a tensor `t` representing the sequence positions (0 to seq_len-1).
    t = torch.arange(seq_len, device=device, dtype=torch.float32)

    # Compute the frequency values by performing a matrix multiplication of `t` and `theta`.
    # `torch.einsum("i,j->ij", t, theta)` performs the outer product, generating a grid of frequencies.
    freqs = torch.einsum("i,j->ij", t, theta)  # Shape: [seq_len, head_dim // 2]

    # Concatenate the frequencies to form the full frequency tensor (doubling the columns).
    # This results in the full frequency tensor with shape [seq_len, head_dim].
    freqs = torch.cat([freqs, freqs], dim=-1)  # Shape: [seq_len, head_dim]

    # Add batch and head dimensions to the frequency tensor to match the expected shape for RoPE.
    # The shape becomes [1, 1, seq_len, head_dim].
    freqs = freqs.unsqueeze(0).unsqueeze(1)  # Shape: [1, 1, seq_len, head_dim]

    return freqs
