# trainer/config.py
from dataclasses import dataclass, field
import torch

@dataclass
class TrainingConfig:  
    # Data    
    dataset_name: str = "bookcorpus"
    # Use 'wikitext-103-v1' for predefined val split, 'wikitext-2-v1' needs manual split
    dataset_config_name: str = "plain_text"
    validation_split_percentage: float = 0.01 # Used only if dataset has no 'validation' split
    max_seq_length: int = 512 # Max sequence length for tokenization/model context

    train_data_subset_fraction: float = 0.0001 # between 0 and 1, 1 for using the whole dataset
    vram_log_interval: int =1
    # Training Hyperparameters
    num_epochs: int = 3 # Adjust total epochs
    batch_size: int = 4 # Per device batch size
    gradient_accumulation_steps: int = 8 # Effective batch size = batch_size * accumulation_steps
    base_learning_rate: float = 5e-5
    weight_decay: float = 0.01
    gradient_clipping_norm: float = 1.0

    # LR Schedule
    decay_lr: bool = True
    warmup_ratio: float = 0.05 # % of total steps for warmup
    min_lr_ratio: float = 0.1 # min_lr = base_learning_rate * min_lr_ratio

    # Technical
    use_mixed_precision: bool = True # Use AMP (bfloat16 or float16)
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu" # Auto-detect device

    # Logging & Saving
    log_interval: int = 20 # Log training loss every N * grad_accum steps
    eval_interval: int = 100 # Evaluate on validation set every N * grad_accum steps
    save_path: str = "checkpoints" # Directory to save checkpoints
    checkpoint_filename_latest: str = "latest_checkpoint.pt"
    checkpoint_filename_best: str = "best_model.pt"

    # Dataloader
    num_workers: int = 2 # Dataloader workers

    # To be calculated later
    total_train_steps: int = 0
    warmup_steps: int = 0
    min_lr: float = 0.0