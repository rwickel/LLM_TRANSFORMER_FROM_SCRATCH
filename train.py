# train.py

import os
# Set the environment variable BEFORE importing tensorflow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
import torch
import torch.optim as optim
# ****** Import AutoModel explicitly ******
from transformers import AutoTokenizer, AutoConfig, AutoModel

import random
import numpy as np
import math

# --- Custom Model Imports ---
from model.config import TransformerConfig, get_model_config
from model.model import DecoderLM

# --- Trainer Imports ---
from trainer.config import TrainingConfig
from trainer.utils import get_lr, load_model_from_checkpoint
from trainer.data_utils import load_and_prepare_data,load_data_with_caching, create_dataloaders
from trainer.trainer import Trainer

def set_seed(seed_value):
    """Sets the seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #
    device = "cpu" 

    # --- Configuration ---
    config = TrainingConfig()
    config.device = device # Update config with actual device
    config.resume_from_checkpoint = True # Set to True if you want to resume from a checkpoint

    # Override tokenizer setting in config if needed to match the model used below
    config.tokenizer_name_or_path = 'sentence-transformers/all-MiniLM-L6-v2'
    # config.base_model_config_name is now less relevant if using AutoModel below

    # --- Setup ---
    set_seed(config.seed)     
    os.makedirs(config.save_path, exist_ok=True)

    # --- Define the specific model name ---
    emb_model = 'sentence-transformers/all-MiniLM-L6-v2'
    print(f"Using '{emb_model}' to derive model config via get_model_config.")

    # --- Tokenizer ---
    # Load tokenizer directly from the specified model name
    print(f"Loading tokenizer: {emb_model}")
    # Ensure config reflects the tokenizer being used
   
    tokenizer = AutoTokenizer.from_pretrained(emb_model)

    # Handle special tokens
    if tokenizer.pad_token is None:
        print("Adding pad token (using EOS token)")
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    if tokenizer.eos_token is None:
         # MiniLM might have SEP/CLS instead of EOS, add EOS if needed for GPT-style
         print("Adding EOS token '<|endoftext|>'")
         tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # --- Model Config & Initialization ---
       # Load ONLY the configuration object (efficient)
    print(f"Loading base AutoConfig: {emb_model}")
    base_config_obj = AutoConfig.from_pretrained(emb_model)
   
    print("Calling get_model_config...")
    # Pass the loaded AutoConfig instance to the updated function
    model_config = get_model_config(
        base_config=base_config_obj, # Pass the config object
        tokenizer=tokenizer,
        device=device
        # No max_seq_length arg needed if get_model_config derives it internally
    )

    # Validate the returned config type
    if not isinstance(model_config, TransformerConfig):
         raise TypeError(f"get_model_config must return an instance of TransformerConfig, but got {type(model_config)}")

    # Optional: Sanity check block_size vs tokenizer max_length
    if model_config.block_size < config.max_seq_length:
        print(f"Warning: Model block_size ({model_config.block_size}) is less than data tokenization max_seq_length ({config.max_seq_length}). Data will be truncated to model's block_size during training if not handled by model.")
    elif model_config.block_size > config.max_seq_length:
         print(f"Info: Model block_size ({model_config.block_size}) is larger than data tokenization max_seq_length ({config.max_seq_length}). Padding/truncation handled by data prep.")


    print(f"Initializing model with config from get_model_config: {model_config}")
    model = DecoderLM(model_config).to(device)
    
    print(f"Model initialized on device: {next(model.parameters()).device}")
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M trainable parameters.")
    

    # --- Data Preparation ---
    # Data loading now uses the tokenizer derived from model_name_for_config
    config.cache_dir = ".\\my_cache_dir" 
    #tokenized_train_data, tokenized_val_data = load_and_prepare_data(config, tokenizer)  
    
    tokenized_train_data, tokenized_val_data = load_data_with_caching(config, tokenizer)

    # Check if datasets are valid before proceeding
    if tokenized_train_data is None or tokenized_val_data is None:
        print("Fatal: Data preparation failed. Exiting.")
        exit()
    if len(tokenized_train_data) == 0 or len(tokenized_val_data) == 0:
        print("Fatal: Train or validation dataset is empty after processing. Exiting.")
        exit()  

    if 0.0 < config.train_data_subset_fraction < 1.0:
        num_train_samples = len(tokenized_train_data)
        subset_size = math.ceil(num_train_samples * config.train_data_subset_fraction)
        print(f"Using a subset of the training data: {subset_size} samples ({config.train_data_subset_fraction*100:.1f}%)")
        # Ensure shuffling happens before subsetting if using datasets library's map/filter
        # If tokenized_train_data is a simple list or array:
        # random.shuffle(tokenized_train_data) # Optional: shuffle before taking subset
        tokenized_train_data = tokenized_train_data.select(range(subset_size)) # If using Hugging Face Dataset
        # Or for a list: tokenized_train_data = tokenized_train_data[:subset_size]
    elif config.train_data_subset_fraction >= 1.0:
        print("Using the full training dataset.")
    else:
        raise ValueError("train_data_subset_fraction must be between 0.0 and 1.0 (or >= 1.0 to use all data)")    
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # For Causal Language Modeling (like GPT) shift is done automatically by the DataCollatorForLanguageModeling when you use it with mlm=False
        return_tensors="pt"
    )

    train_loader = DataLoader(
        tokenized_train_data,
        batch_size=config.batch_size,
        collate_fn=data_collator,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        tokenized_val_data,
        batch_size=config.batch_size,
        collate_fn=data_collator,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # --- Optimizer ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate, # Initial LR, scheduler will adjust
        weight_decay=config.weight_decay
    )

    # --- Initialize Trainer ---
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        lr_scheduler_func=get_lr, # Pass the function itself
        config=config,
        device=device,
        tokenizer=tokenizer
    )

    # --- Start Training ---
    trainer.train(config.resume_from_checkpoint)

    