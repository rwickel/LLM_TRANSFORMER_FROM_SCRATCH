# trainer/utils.py
import math
import torch
import os
from trainer.data_utils import load_and_prepare_data, create_dataloaders
import re 
# --- Learning Rate Scheduler (Cosine with Warmup) ---
# Note: 'it' here refers to training step (batch number), not epoch
def get_lr(it, warmup_steps, total_steps, learning_rate, min_lr):
    # 1) linear warmup for warmup_steps
    if it < warmup_steps:
        return learning_rate * (it + 1) / (warmup_steps + 1)
    # 2) if it >= total_steps, return min learning rate (or slightly before)
    if it >= total_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (total_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (learning_rate - min_lr)


def load_checkpoint(model, optimizer, scaler, checkpoint_dir):
        checkpoint_files = os.listdir(checkpoint_dir)
        if len(checkpoint_files) > 0:
            latest_checkpoint = max(
                checkpoint_files,
                key=lambda f: int(re.search(r'epoch_(\d+)', f).group(1))
            )
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            print(f"Resuming training from checkpoint: {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            iter_num = checkpoint['iter_num']
            return epoch, iter_num, loss
        else:
            print("No checkpoint found. Starting from scratch.")
            return 0, 0, float('inf')



# --- Function to load model and tokenizer from checkpoint ---
def load_model_from_checkpoint2(checkpoint_path: str, device: str = 'cpu'):
    """
    Loads a model and tokenizer from a saved checkpoint.

    Args:
        checkpoint_path (str): Path to the .pt checkpoint file.
        device (str): Device to load the model onto ('cuda' or 'cpu').

    Returns:
        tuple: (model, tokenizer, train_config, model_config) or None if loading fails.
    """
    from transformers import AutoTokenizer
    # Assuming your custom model files are accessible from where this is run
    from model.model import DecoderLM
    from model.config import TransformerConfig # Need this to potentially recreate model

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"Checkpoint loaded from {checkpoint_path}")

        saved_model_config = checkpoint['model_config']
        saved_train_config = checkpoint['train_config']
        tokenizer_save_dir = os.path.dirname(checkpoint_path) # Assume tokenizer saved in same dir

        print("Loading Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_dir)

        print("Initializing Model...")
        # Ensure saved_model_config is the right type (e.g., TransformerConfig instance)
        if isinstance(saved_model_config, dict): # Handle case where it might be saved as dict
             model_config = TransformerConfig(**saved_model_config)
        else:
             model_config = saved_model_config
        model_config.device = device # Ensure model config device is updated
        model = DecoderLM(model_config).to(device)

        print("Loading Model State Dict...")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval() # Set to eval mode by default after loading

        print("Model and Tokenizer loaded successfully.")
        return model, tokenizer, saved_train_config, model_config

    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None, None, None, None
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None
    

 