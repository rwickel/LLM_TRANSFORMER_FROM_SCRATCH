# config.py

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class TransformerConfig:
    n_embd: int                   # Embedding dimension
    n_head: int                   # Number of attention heads
    n_kv_head: Optional[int] = None  # Number of KV heads (for GQA); defaults to n_head if None    
    n_layer: int = 12             # Number of transformer blocks
    use_rope: bool = False        # Whether to use RoPE (Rotary Positional Embeddings)
    vocab_size: int = 50257       # Vocabulary size
    block_size: int = 2048        # Max sequence length
    dropout: float = 0.1          # Dropout rate
    device: str = "cuda"          # Device
    pad_token_id: int = 0         # Padding token ID


def check_device():
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use GPU
        print("CUDA is available. Using GPU.")
        
        # Check which GPU is available
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        device = torch.device("cpu")   # Use CPU if no GPU is available
        print("CUDA is not available. Using CPU.")
    
    return device    

def get_model_config(embed_model, tokenizer, device):
    """
    Create a TransformerConfig object with the specified parameters.
    
    Args:
        embed_model: The pre-trained embedding model.
        tokenizer: The tokenizer for the model.
        device: The device to use (CPU or GPU).
    
    Returns:
        TransformerConfig: A configuration object for the transformer model.
    """
    return TransformerConfig(
        n_embd=embed_model.config.hidden_size,  # 384 for MiniLM-L6-v2
        n_head=8,
        n_kv_head=4,
        n_layer=4,
        use_rope=False,   
        vocab_size=tokenizer.vocab_size,
        block_size=embed_model.config.max_position_embeddings, #512        
        dropout=0.1,
        device=device.type,
        pad_token_id=tokenizer.pad_token_id, #ignored in the model loss function
    )

