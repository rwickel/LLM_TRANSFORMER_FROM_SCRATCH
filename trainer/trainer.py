# trainer/trainer.py
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os
import math
import time
import datetime # Added for time formatting
from transformers import get_scheduler

# Import components from the trainer package
from .config import TrainingConfig
from .utils import get_lr

class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, lr_scheduler_func, config, device, tokenizer):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        self.global_step = 0
        self.start_epoch = 0
        self.use_amp = self.config.use_amp  # Use AMP if specified in config
        self.pt_dtype = torch.float16 if self.config.use_amp else torch.float32  # Use float16 if AMP is enabled        

        # Set up scheduler
        num_training_steps = self.config.epochs * len(train_loader)
        lr_scheduler_func="linear"
        self.lr_scheduler = get_scheduler(
            lr_scheduler_func,  # it's now just a string like "linear"
            optimizer=self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps
        )

        if config.gradient_clipping > 0:
            self.max_grad_norm = config.gradient_clipping
        else:
            self.max_grad_norm = None

        # Load from checkpoint if required
        if config.resume_from_checkpoint:
            self._load_checkpoint()

    def train(self, resume_from_checkpoint=False):
        self.config.resume_from_checkpoint = resume_from_checkpoint
        print("Starting training...")

        for step, batch in enumerate(self.train_loader):
            print("--- Batch Inspection ---")
            for key, value in batch.items():
                print(f"  {key}: {type(value)}, Shape: {value.shape if isinstance(value, torch.Tensor) else len(value)}")
            print("--- End Batch ---")


        for epoch in range(self.start_epoch, self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")

            # --- Train ---
            self.model.train()
            total_loss = 0.0
            start_time = time.time()

            progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
            for step, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}    
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # Convert device to string and take the first 4 characters (e.g., 'cuda' or 'cpu')
                device_type = str(self.device).lower()[:4]

                # Use autocast context for potential speedup/memory saving during validation too
                with torch.autocast(device_type=device_type, dtype=self.pt_dtype, enabled=self.use_amp):
                    logits, loss = self.model(input_ids, attention_mask=attention_mask, targets=labels)

                # ====== Debug Info: Input, Label, Prediction ======
                predictions = torch.argmax(logits, dim=-1)

                print(f"\n--- Batch {step + 1} ---")
                for i in range(min(2, input_ids.size(0))):  # limit to 2 samples per batch
                    input_sample = input_ids[i]
                    label_sample = labels[i]
                    pred_sample = predictions[i]

                    filtered_labels = [t for t in label_sample.tolist() if t != -100]
                    print(f"\nSample {i + 1}")
                    print("Decoded Input:    ", self.tokenizer.decode(input_sample, skip_special_tokens=True))
                    print("Decoded Labels:   ", self.tokenizer.decode(filtered_labels, skip_special_tokens=True))
                    print("Predicted Tokens: ", self.tokenizer.decode(pred_sample, skip_special_tokens=True))

                loss.backward()

                if self.max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                total_loss += loss.item()               

                # Printing step progress
                progress_bar.set_postfix(loss=loss.item(), step=self.global_step)

            avg_train_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f} | Time: {time.time() - start_time:.2f}s")

            # --- Validation ---
            #val_loss = self.evaluate()
            #print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}")

            # --- Save Checkpoint ---
            self._save_checkpoint(epoch)

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.val_loader, desc="Validating", leave=False)):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                input_ids = batch['input_ids']
                labels = batch['labels']
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                device_type = str(self.device).lower()[:4]

                with torch.autocast(device_type=device_type, dtype=self.pt_dtype, enabled=self.use_amp):
                    logits, loss = self.model(input_ids, attention_mask=attention_mask, targets=labels)

                total_loss += loss.item()

                # Optional: debug print
                print(f"\n--- Validation Batch {step + 1} ---")
                print("Decoded Input: ", self.tokenizer.decode(input_ids[0], skip_special_tokens=True))
                print("Decoded Label: ", self.tokenizer.decode([t for t in labels[0] if t != -100], skip_special_tokens=True))
                predicted_ids = torch.argmax(logits, dim=-1)[0]
                print("Prediction:    ", self.tokenizer.decode(predicted_ids, skip_special_tokens=True))

        return total_loss / len(self.val_loader)

    def _save_checkpoint(self, epoch):
        save_path = os.path.join(self.config.save_path, f"checkpoint_epoch_{epoch+1}.pt")
        print(f"Saving checkpoint to: {save_path}")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "global_step": self.global_step
        }, save_path)

    def _load_checkpoint(self):
        checkpoints = [ckpt for ckpt in os.listdir(self.config.save_path) if ckpt.endswith(".pt")]
        if not checkpoints:
            print("No checkpoint found, starting from scratch.")
            return

        latest_ckpt = sorted(checkpoints)[-1]
        ckpt_path = os.path.join(self.config.save_path, latest_ckpt)
        print(f"Loading checkpoint from: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.start_epoch = checkpoint["epoch"]
        print(f"Resumed from epoch {self.start_epoch}, global step {self.global_step}")