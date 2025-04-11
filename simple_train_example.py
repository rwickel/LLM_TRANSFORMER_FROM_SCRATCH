import os
# Set the environment variable BEFORE importing tensorflow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModel
from dataclasses import dataclass, field # Import field
from typing import Optional # Import Optional

from model.config import TransformerConfig, check_device, get_model_config
from model.layers import RMSNorm, SwiGLU, MultiHeadAttention, TransformerBlock
from model.positional_encoding import build_rope_cache, apply_rope
from model.model import DecoderLM
import time
import math
import torch.nn as nn


# --- Main Training Script ---
# --- Learning Rate Scheduler (Cosine with Warmup) ---
def get_lr(it, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)  # Linear warmup
    if it > lr_decay_iters:
        return min_lr  # Minimum learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Cosine decay
    return min_lr + coeff * (learning_rate - min_lr)

if __name__ == "__main__":
    # --- Configuration ---
    num_epochs = 210
    max_iters = num_epochs
    base_learning_rate = 5e-4
    weight_decay = 0.0
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'

    gradient_clipping_norm = 1.0
    early_stopping_patience = 10
    decay_lr = True
    warmup_iters = 15
    lr_decay_iters = max_iters
    min_lr = base_learning_rate / 10

    # --- Setup ---
    device = check_device()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embed_model_for_config = AutoModel.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad token. Adding '[PAD]'.")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
        print(f"Set pad_token_id to: {tokenizer.pad_token_id}")
    if tokenizer.eos_token is None:
        print("Tokenizer does not have an EOS token. Adding '<|endoftext|>'.")
        tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')
        print(f"Set eos_token_id to: {tokenizer.eos_token_id}")

    print(f"Loading base AutoConfig: {model_name}")
    base_config_obj = AutoConfig.from_pretrained(model_name)

    # ----- REMOVED AutoModel loading -----
    # print(f"Loading base AutoModel for config: {model_name_for_config}")
    # print("(Note: Loading full AutoModel for config is memory-intensive)")
    # embed_model_for_config = AutoModel.from_pretrained(model_name_for_config)
    # ----- END REMOVAL -----

    print("Calling get_model_config...")
    # Pass the loaded AutoConfig instance to the updated function
    config = get_model_config(
        base_config=base_config_obj, # Pass the config object
        tokenizer=tokenizer,
        device=device
        # No max_seq_length arg needed if get_model_config derives it internally
    )      
    
    print(f"Model Config: {config}")
    model = DecoderLM(config).to(device)
    print(f"Model initialized on device: {next(model.parameters()).device}")
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M trainable parameters.")

    # --- Data Preparation ---
    wiki_sample_text = "A transformer is a deep learning model introduced in 2017, used primarily in the field of natural language processing (NLP). It is based on a self-attention mechanism that allows the model to weigh the importance of different words in a sentence, regardless of their position. This architecture has led to significant advancements in various NLP tasks, including translation, summarization, and text generation."
    encoding = tokenizer(wiki_sample_text, return_tensors='pt', padding='max_length', truncation=True, max_length=config.block_size, add_special_tokens=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask'] # <-- Gets the MASK (1s and 0s)

    # Prepare Targets for GPT-style Loss (shifted input_ids)
    targets = input_ids.clone()
    targets = torch.roll(targets, shifts=-1, dims=1)
    targets[:, -1] = -100  # Set the last token's target to -1 (ignore index)

    # Move to device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device) 
    targets = targets.to(device)
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Targets shape: {targets.shape}")

    # --- Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=base_learning_rate, weight_decay=weight_decay)

    # --- Training Loop ---
    print(f"\n--- Starting Training for up to {num_epochs} Epochs ({max_iters} iterations) on {device} ---")
    total_start_time = time.time()
    best_loss = float('inf')
    epochs_no_improve = 0
    iter_num = 0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()

        lr = get_lr(iter_num, warmup_iters, base_learning_rate, lr_decay_iters, min_lr) if decay_lr else base_learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)

        # Forward pass (no mixed precision)
        logits, loss = model(input_ids, attention_mask=attention_mask, targets=targets)

        if loss is not None:
            loss.backward()

            # Gradient clipping
            if gradient_clipping_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_norm)

            # Optimizer step
            optimizer.step()

            loss_value = loss.item()
        else:
            loss_value = float('nan')
            print(f"Warning: Loss calculation skipped in epoch {epoch+1}.")

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs} | Iter {iter_num} | LR: {lr:.6f} | Loss: {loss_value:.4f} | Duration: {epoch_duration:.2f}s")
        iter_num += 1

        current_loss = loss_value
        if not math.isnan(current_loss) and current_loss < best_loss:
            best_loss = current_loss
            epochs_no_improve = 0
        elif not math.isnan(current_loss):
            epochs_no_improve += 1
            print(f"  Loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

        if iter_num >= max_iters:
            print(f"\nReached max_iters ({max_iters}). Stopping training.")
            break

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"--- Training Finished ---")
    print(f"Total Training Duration: {total_duration:.2f}s")


    # --- Starting Generation Example ---
print("\n--- Starting Generation Example ---")
prompt = "deep learning model"
print(f"Prompt: '{prompt}'")

# --- CORRECT ENCODING ---
# Use the tokenizer's __call__ method or encode()
# Get input_ids directly, usually don't need attention_mask for basic generation start
encoding = tokenizer(prompt, return_tensors='pt', add_special_tokens=False) # Set add_special_tokens based on how you want generation to start
prompt_tensor = encoding['input_ids'].to(config.device)
# --- End Correct Encoding ---

# Alternative using encode (returns list, then convert to tensor)
# prompt_ids_list = tokenizer.encode(prompt, add_special_tokens=False)
# prompt_tensor = torch.tensor([prompt_ids_list], dtype=torch.long).to(config.device)


print(f"Correctly Encoded Prompt IDs: {prompt_tensor}")
print(f"Prompt IDs shape: {prompt_tensor.shape}")

generated_output = model.generate(
    prompt_tensor,
    max_tokens=config.block_size, # Or a smaller desired generation length
    temperature=0.8,
    top_k=5    
)

generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True) # Skip special tokens in output
print(f"Generated IDs: {generated_output}")
print(f"Generated Text: '{generated_text}'")
print("--- Generation Example Complete ---")