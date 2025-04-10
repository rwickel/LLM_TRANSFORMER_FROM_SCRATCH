# config.py

from dataclasses import dataclass
from typing import Optional
import torch
from transformers import PretrainedConfig, AutoTokenizer # Added PretrainedConfig type hint

@dataclass
class TransformerConfig:
    """Configuration class for the Transformer model."""
    block_size: int = 128 # Max sequence length
    vocab_size: int = 30522 # Placeholder, will be updated by tokenizer
    n_layer: int = 6      # Number of transformer blocks
    n_head: int = 6       # Number of attention heads
    n_embd: int = 384     # Embedding dimension
    n_kv_head: Optional[int] = None # Number of key/value heads for Grouped Query Attention (if None, defaults to n_head)
    dropout: float = 0.1  # Dropout rate
    bias: bool = False    # Use bias in Linear layers?
    device: str = 'cpu'   # Device to run on ('cpu', 'cuda', 'mps')
    pad_token_id: int = 0 # Padding token ID, updated by tokenizer
    use_rope: bool = True # Use Rotary Positional Embeddings?
    eos_token_id: Optional[int] = None # End-of-sequence token ID, updated by tokenizer


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

def get_model_config(base_config: PretrainedConfig, tokenizer: AutoTokenizer,seq_lenght=None, device=check_device()) -> TransformerConfig:
    """Creates TransformerConfig based on embedding model and tokenizer."""
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must have a pad_token_id defined.")
    
    # *** Derive block_size from base_config ***
    block_size = base_config.max_position_embeddings
    print(f"Deriving block_size from base_config.max_position_embeddings: {block_size}")   


    if seq_lenght is None or seq_lenght > base_config.max_position_embeddings:
         seq_lenght = base_config.max_position_embeddings

    config_data = {
        'block_size': seq_lenght,
        'vocab_size': tokenizer.vocab_size,
        'n_layer': 6,
        'n_head': 6,
        'n_embd': base_config.hidden_size,
        'n_kv_head': None,
        'dropout': 0.1,
        'bias': False,
        'device': str(device), # Store device as string
        'pad_token_id': tokenizer.pad_token_id,
        'use_rope': False,
        'eos_token_id': tokenizer.eos_token_id
    }
    return TransformerConfig(**config_data)
