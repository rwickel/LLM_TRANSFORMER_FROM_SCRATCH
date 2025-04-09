import torch
from data.dataset import load_triviaqa
from training.util import  preprocess_function, Trainer
from model.config import TransformerConfig,check_device, get_model_config
from model.model import DecoderLM
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
# Load your test dataset (replace with actual loading code)
from data.dataset import load_triviaqa

# Device configuration
device = check_device()

def test_model(checkpoint_path, tokenizer_path, test_dataset, split="test", max_length=512):
    """Complete testing pipeline for your pretrained model"""

    # 1. Select the correct split
    if isinstance(test_dataset, dict):  # If it's a dataset dictionary
        if split not in test_dataset:
            raise ValueError(f"Split '{split}' not found. Available splits: {list(test_dataset.keys())}")
        test_data = test_dataset[split]
    else:
        test_data = test_dataset  # Assume it's already a single split
    
    # 1. Load the pretrained model
    print("Loading model...")
    model = Trainer.load_pretrained_model(checkpoint_path)
    
    # 2. Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 3. Prepare test data - use either config.block_size if available, or fallback to max_length
    block_size = getattr(model.config, 'block_size', max_length)
    print(f"Using sequence length: {block_size}")
    
    print("Preparing test data...")    
    test_data = test_dataset.map(
        lambda x: preprocess_function(x, tokenizer, block_size),
        batched=False
    )
    
    # 4. Create test loader
    test_loader = DataLoader(
        test_data,
        batch_size=4,  # Adjust batch size as needed
        shuffle=False
    )
    
    # 5. Create trainer instance and run tests
    trainer = Trainer(model=model)
    test_results = trainer.test(test_loader, tokenizer, max_length=block_size)
    
    return test_results

if __name__ == "__main__":
    # Example usage
    checkpoint_path = "training_checkpoints/pretrained_model.pth"
    tokenizer_path = checkpoint_path.replace('.pth', '_tokenizer')    
    
    test_dataset = load_triviaqa(sample_size=100)
    
    test_results = test_model(
        checkpoint_path,
        tokenizer_path,
        test_dataset,  # This is a DatasetDict
        split="test",  # Explicitly choose 'test' split       
    )