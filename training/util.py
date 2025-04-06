import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW


def preprocess_function(example, tokenizer, max_length):
    context = example["search_results"]["search_context"][0] if example["search_results"]["search_context"] else ""
    question = example["question"]
    answer = example["answer"]["value"] if example["answer"]["value"] else ""

    input_text = f"Question: {question} Context: {context}"
    label_text = answer

    inputs = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

    labels = tokenizer(
        label_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

    return {
        "input_ids": inputs["input_ids"].squeeze(0),
        "attention_mask": inputs["attention_mask"].squeeze(0),
        "label": labels["input_ids"].squeeze(0),
    }

class Trainer:
    def __init__(self, model, train_loader, device, learning_rate=2e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
        
        for batch in loop:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["label"].to(self.device)

            _, loss = self.model(input_ids, targets=labels)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        return total_loss / len(self.train_loader)
    
    def train(self, epochs):
        for epoch in range(epochs):
            avg_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch + 1} completed. Avg Loss: {avg_loss:.4f}")