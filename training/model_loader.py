import torch
from collections import OrderedDict
from model.model import DecoderLM
from model.config import TransformerConfig

def load_checkpoint(checkpoint_path, device):
    """Load model checkpoint with proper error handling"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract config from checkpoint
        config_dict = checkpoint['config'].__dict__ if hasattr(checkpoint['config'], '__dict__') else checkpoint['config']
        config = TransformerConfig(**config_dict)
        
        # Initialize model
        model = DecoderLM(config).to(device)
        
        # Handle state dict (fix potential CUDA/CPU device mismatch)
        state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            if isinstance(v, torch.Tensor):
                state_dict[k] = v.to(device)
            else:
                state_dict[k] = v
                
        # Load state dict
        model.load_state_dict(state_dict)
        model.eval()

        model.device = device
        
        return model, config
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

def verify_model_loading(model, sample_input):
    """Verify the model loads and runs correctly"""
    try:
        with torch.no_grad():
            output = model(sample_input)
        print("✓ Model loaded and ran successfully")
        return True
    except Exception as e:
        print(f"× Model verification failed: {e}")
        return False