import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AdamW
from model.model import DecoderLM
from model.config import get_model_config
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import OneCycleLR

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
        "targets": labels["input_ids"].squeeze(0),
    }


class Trainer:
    def __init__(self, model, train_loader=None, device=None, use_amp=True, learning_rate=3e-3, checkpoint_dir="training_checkpoints"):
        self.model = model
        self.use_amp = use_amp
        self.train_loader = train_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)

        self.scaler = GradScaler(enabled=self.use_amp)

        # Scheduler - OneCycleLR as an example
        if train_loader is not None:
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=learning_rate,
                steps_per_epoch=len(train_loader),
                epochs=1,  # Set to 1 initially, will update in the `train()` method
                pct_start=0.1,
                anneal_strategy='cos',
                cycle_momentum=False
            )
        else:
            self.scheduler = None

        # Create the directory for saving checkpoints if it doesn't exist
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training tracking attributes
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.global_step = 0
        self.training_history = []

    @classmethod   
    def load_pretrained_model(cls, checkpoint_path, device=None):
        """Load just the model from a checkpoint"""
        # Set device
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # First try loading directly
            checkpoint = torch.load(checkpoint_path, map_location=device)
        except (AttributeError, RuntimeError) as e:
            # If direct loading fails, try loading via BytesIO
            try:
                with open(checkpoint_path, 'rb') as f:
                    buffer = io.BytesIO(f.read())
                checkpoint = torch.load(buffer, map_location=device)
            except Exception as e:
                raise ValueError(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}")
        
        # Handle config (three possible approaches)
        if 'config' in checkpoint:
            # Option 1: Config was saved with model
            config = checkpoint['config']        
        else:
            # Option 3: Manually specify config
            config = get_model_config()
        
        # Initialize model
        model = DecoderLM(config).to(device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to evaluation mode
        model.eval()
        
        return model

    def load_pretrained_model2(checkpoint_path, device=None):
        """Load just the model from a checkpoint"""
        # Set device
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle config (three possible approaches)
        if 'config' in checkpoint:
            # Option 1: Config was saved with model
            config = checkpoint['config']
        elif hasattr(DecoderLM, 'default_config'):
            # Option 2: Use default config from model class
            config = DecoderLM.default_config
        else:
            # Option 3: Manually specify config
            config = None
        
        # Initialize model
        model = DecoderLM(config).to(device)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to evaluation mode
        model.eval()
        
        return model  

    def save_model(self, name="pretrained_model.pth", config=None, tokenizer=None):
        """Save the complete model state"""
        # Ensure the directory exists        
        full_save_path = os.path.join(self.checkpoint_dir, name)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': config if config else getattr(self.model, 'config', None),
            'training_info': {
                'epoch': self.current_epoch,
                'best_loss': self.best_loss,
                'global_step': self.global_step,
                'history': self.training_history
            }
        }

        # Save the checkpoint to the specified path
        torch.save(checkpoint, full_save_path)
        print(f"✓ Model saved to {full_save_path}")

        # Save tokenizer if provided
        if tokenizer:
            tokenizer_save_path = full_save_path.replace('.pth', '_tokenizer')
            tokenizer.save_pretrained(tokenizer_save_path)
            print(f"✓ Tokenizer saved to {tokenizer_save_path}")

    def train(self, epochs):
        """Train the model for the specified number of epochs."""
        if self.scheduler:
            # Update scheduler to match number of steps for all epochs
            self.scheduler.epochs = epochs  # Correctly set the number of epochs
            self.scheduler.total_steps = epochs * len(self.train_loader)  # Calculate total steps based on train loader
        
        for epoch in range(epochs):
            avg_loss = self.train_epoch(epoch)
            
            print(f"Epoch {epoch + 1}/{epochs} | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            self.current_epoch += 1
            
            if hasattr(self, 'val_loader'):
                val_loss = self.validate()
                print(f"Validation Loss: {val_loss:.4f}")

    def train_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        # Initialize tqdm progress bar
        loop = tqdm(self.train_loader, 
                    desc=f"Epoch {epoch + 1}", 
                    leave=False,
                    mininterval=0.5)
        
        for batch in loop:
            # Move data to device
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                _, loss = self.model(input_ids=input_ids, targets=labels)
            
            # Backward pass and optimization
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Update learning rate scheduler if available
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            self.global_step += 1
            
            # Update progress bar
            loop.set_postfix(
                loss=loss.item(),
                lr=self.optimizer.param_groups[0]['lr']
            )
        
        # Calculate average loss
        avg_loss = total_loss / total_samples
        self.training_history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "lr": self.optimizer.param_groups[0]['lr']
        })
        
        # Save best model
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.save_model(f"best_model_epoch_{epoch + 1}.pth")
        
        return avg_loss

    def validate(self):
        """Validate the model on validation set."""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["label"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                _, loss = self.model(input_ids=input_ids,attention_mask=attention_mask, targets=labels)
                
                batch_size = input_ids.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        return total_loss / total_samples
    
    def test(self, test_loader, tokenizer, max_length=512):
        """Test the model on a test dataset"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            loop = tqdm(test_loader, desc="Testing")
            for batch in loop:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["label"].to(self.device)
                
                # Get model outputs
                logits, loss = self.model(input=input_ids, targets=labels)
                total_loss += loss.item()
                
                # Generate predictions
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True
                )
                
                # Decode predictions and references
                predictions = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
                references = [tokenizer.decode(l, skip_special_tokens=True) for l in labels]
                
                all_predictions.extend(predictions)
                all_references.extend(references)
                
                loop.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(test_loader)
        print(f"\nTest Loss: {avg_loss:.4f}")
        
        # Calculate evaluation metrics (e.g., BLEU, ROUGE, exact match)
        # You'll need to implement or import these metrics
        eval_results = self.evaluate_predictions(all_predictions, all_references)
        
        return {
            "loss": avg_loss,
            "predictions": all_predictions,
            "references": all_references,
            "metrics": eval_results
        }
    
    def evaluate_predictions(self, predictions, references):
        """Calculate evaluation metrics"""
        # Implement your evaluation metrics here
        # For example, exact match or BLEU score
        
        exact_matches = sum(1 for p, r in zip(predictions, references) if p.lower() == r.lower())
        exact_match_score = exact_matches / len(predictions)
        
        return {
            "exact_match": exact_match_score,
            # Add other metrics as needed
        }
    
    def interactive_test(trainer, tokenizer, max_length=512):
        """Interactive testing where you can input questions"""
        print("\nInteractive Testing (type 'quit' to exit)")
        while True:
            question = input("\nEnter your question: ")
            if question.lower() == 'quit':
                break
                
            # Format the input
            input_text = f"Question: {question} Context: "
            inputs = tokenizer(
                input_text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            ).to(trainer.device)
            
            # Generate answer
            generated_ids = trainer.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=5,
                early_stopping=True
            )
            
            answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"\nAnswer: {answer}")

