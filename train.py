import torch
from data.dataset import load_triviaqa
from training.util import  preprocess_function, Trainer
from model.config import TransformerConfig, check_device, get_model_config
from model.model import DecoderLM
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader

def main():
    # Device configuration
    device = check_device()

    # Model configuration
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    embed_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')   

    tokenizer.add_special_tokens = False  # This prevents adding [CLS], [SEP] etc

    config = get_model_config(embed_model, tokenizer, device)
    
    # Load data
    triviaqa_dataset = load_triviaqa(sample_size=100)
    
    # Initialize model
    model = DecoderLM(config).to(device)
    
    # Preprocess data
    train_data = triviaqa_dataset["train"].map(
        lambda x: preprocess_function(x, tokenizer, config.block_size),
        batched=False
    )
    train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Print out some lines of the training data before training
    print("Sample training data examples:")
    for i in range(3):  # Print first 3 examples
        sample = train_data[i]
        print(f"\nExample {i+1}:")       
        print("Decoded input:", tokenizer.decode(sample["input_ids"]))
        print("Attention mask:", sample["attention_mask"])       
        print("Decoded Label:", tokenizer.decode(sample["label"]))
    # DataLoader
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    
    # Training
    trainer = Trainer(model, train_loader, device)
   
    # Train and save
    trainer.train(epochs=20)
    trainer.save_model(config=config, tokenizer=tokenizer)

if __name__ == "__main__":
    main()