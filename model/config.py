# config.py

from dataclasses import dataclass
from typing import Optional
import torch
from transformers import PretrainedConfig, AutoTokenizer # Added PretrainedConfig type hint

@dataclass
class TransformerConfig:
    """Configuration class for the Transformer model."""
    # --- Core Dimensions ---
    block_size: int = 512
    vocab_size: int = 30522
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    n_kv_head: Optional[int] = None
    intermediate_size: Optional[int] = 4* n_embd
    dropout: float = 0.1
    bias: bool = False

    device: Optional[str] = None

    # --- Positional Embeddings ---
    use_rope: bool = True
    rope_theta: float = 10000.0

    # --- Initialization & Misc ---
    tie_weights: bool = True
    init_std: float = 0.02
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None



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
        'n_layer': 4,
        'n_head': 6,
        'n_embd': base_config.hidden_size,
        'n_kv_head': None,
        'dropout': 0.1,
        'intermediate_size': 4 * base_config.hidden_size,
        'bias': False,
        'device': str(device), # Store device as string
        'pad_token_id': tokenizer.pad_token_id,
        'use_rope': False,
        'eos_token_id': tokenizer.sep_token_id
    }
    return TransformerConfig(**config_data)
