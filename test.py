import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from model.config import TransformerConfig, get_model_config
from model.model import DecoderLM


# --- Checkpoints Folder ---
checkpoint_dir = '.\\checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# --- Set Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# --- Load Tokenizer and Model ---
embd_model = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(embd_model)

# Add special tokens if they are missing
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

# --- Load Base Model Config ---
print(f"Loading base AutoConfig: {embd_model}")
base_config_obj = AutoConfig.from_pretrained(embd_model)

config = get_model_config(
        base_config=base_config_obj, # Pass the config object
        tokenizer=tokenizer,        
        # No max_seq_length arg needed if get_model_config derives it internally
    )  

# --- Load the Model ---
# If using a pre-trained model like GPT-2 or others, use AutoModelForCausalLM:
embed_model_for_config = AutoModelForCausalLM.from_pretrained(embd_model, config=config)


print("Calling get_model_config...")
# Pass the loaded AutoConfig instance to the updated function
config = get_model_config(
    base_config=base_config_obj, # Pass the config object
    tokenizer=tokenizer,        
    # No max_seq_length arg needed if get_model_config derives it internally
)      

print(f"Model Config: {config}")
model = DecoderLM(config).to(device)
print(f"Model initialized on device: {next(model.parameters()).device}")
print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M trainable parameters.")
      

checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_388_loss_0.3473.pt')
print(f"Resuming training from checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  

# Move model to the correct device (GPU or CPU)
embed_model_for_config.to(device)

# Put the model in evaluation mode
embed_model_for_config.eval()

# Assuming `model` and `tokenizer` are initialized and configured
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🧠 Interactive Mode: Type your prompt. Type 'exit' to quit.\n")

while True:
    prompt = input("You: ")

    if prompt.strip().lower() == "exit":
        print("👋 Exiting. Bye!")
        break

    # Tokenize the input prompt
    encoding = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids'].to(device)

    # Generate the text using the model with streaming output
    print("Model: ")
    output = model.generate(
        prompt=prompt,        
        temperature=0.7,
        top_k=50       
    )

    # Streaming output: Print token by token
    generated_text = prompt  # Start with the prompt
    for word in output:        
        print(f"{word} ", end="", flush=True)   # Print the token (without newline)
    
    print()  # Move to a new line after the output is done