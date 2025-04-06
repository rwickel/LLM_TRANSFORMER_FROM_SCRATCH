import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from model.config import TransformerConfig, check_device, get_model_config
from model.model import DecoderModel, DecoderLM

device = check_device()

# Load pre-trained sentence transformer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
embed_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Print vocab size to match it with the config
print("Vocab size:", tokenizer.vocab_size)

config = get_model_config(embed_model, tokenizer, device)

model = DecoderLM(config).to(device)

# Define a sentence to test the model
sentence = "The car is red. What is the car color?"

target = "The car is red"

# Tokenize the sentence using the pre-trained tokenizer
inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=config.block_size)

# Extract input IDs (tokenized sentence) and move to the correct device
dummy_input = inputs["input_ids"].to(device)

# Ensure the target is the same for next-token prediction
targets = dummy_input

# Pass the input to the model
logits, loss = model(inputs["input_ids"].to(device), targets=targets)

# Get the logits for the last token
probs = F.softmax(logits, dim=-1) 

# Get the predicted token IDs
predicted_ids = torch.argmax(probs, dim=-1) 

# Decode the predicted IDs back to text
predicted_tokens = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

# Print the predicted tokens (next word prediction)
print(f"Predicted tokens: {predicted_tokens}")

# Print the shapes and the loss
print("Logits shape:", logits.shape)
print("Loss:", loss.item())