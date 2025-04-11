# main.py
import torch
import torch.optim as optim
# ****** Import AutoModel explicitly ******
from transformers import AutoTokenizer, AutoConfig, AutoModel
import os
import random
import numpy as np
import os

# --- Custom Model Imports ---
from model.config import TransformerConfig, get_model_config
from model.model import DecoderLM

# --- Trainer Imports ---
from trainer.utils import load_model_from_checkpoint
from trainer.config import TrainingConfig
from trainer.utils import check_device
# --- Configuration ---
config = TrainingConfig()

device = check_device()

# --- Optional: Load best model and generate ---
print("\n--- Loading best model for generation example ---")
best_model_path = os.path.join(config.save_path, config.checkpoint_filename_best)
loaded_model, loaded_tokenizer, _, loaded_model_config = load_model_from_checkpoint(best_model_path, device=device)

if loaded_model and loaded_tokenizer:
    prompt = "Egypt" # Example prompt
    print(f"Prompt: '{prompt}'")

    encoding = loaded_tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
    prompt_tensor = encoding['input_ids'].to(device)

    loaded_model.eval()
    with torch.no_grad():
            generated_output_ids = loaded_model.generate(
                prompt_tensor,
                max_tokens=min(150, loaded_model_config.block_size),
                temperature=0.7,
                top_k=40                
            )

    generated_text = loaded_tokenizer.decode(generated_output_ids[0], skip_special_tokens=True)
    print(f"Generated Text: '{generated_text}'")
    print("--- Generation Example Complete ---")
else:
    print(f"Could not load the best model from {best_model_path} for generation.")