import os
import re
import time
import datetime
import torch
from tqdm import tqdm
from transformers import get_scheduler
from trainer.config import TrainingConfig

class Trainer:
    """
    Manages the training and evaluation loop for a transformer model.

    Handles epoch iteration, batch processing, forward/backward passes,
    optimizer and learning rate scheduler steps, automatic mixed precision (AMP),
    gradient clipping, evaluation, and checkpoint saving/loading.
    """
    def __init__(self, model, optimizer, train_loader, val_loader, config: TrainingConfig, device, tokenizer, lr_scheduler_type="cosine"):
        """
        Initializes the Trainer instance.

        Args:
            model (torch.nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
            val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
            config (TrainingConfig): Configuration object containing training parameters
                (epochs, warmup_steps, gradient_clipping, use_amp, save_path, etc.).
            device (torch.device): The device (CPU or CUDA) to run training on.
            tokenizer: The tokenizer used for decoding sequences in debug prints.
            lr_scheduler_type (str or callable): The name of the Hugging Face scheduler
                type (e.g., "linear", "cosine") or a custom scheduler function.
                Note: Currently using the specified type.
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        self.global_step = 0 # Tracks total optimization steps across epochs
        self.start_epoch = 0 # Starting epoch number (used for resuming)
        self.use_amp = self.config.use_amp # Enable Automatic Mixed Precision
        # Determine PyTorch dtype based on AMP setting
        self.pt_dtype = torch.float16 if self.use_amp else torch.float32
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # --- Learning Rate Scheduler Setup ---
        # Calculate total training steps for scheduler
        num_training_steps = self.config.epochs * len(train_loader)

        print(f"Setting up '{lr_scheduler_type}' LR scheduler...")
        self.lr_scheduler = get_scheduler(
            lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=num_training_steps
        )

        # --- Gradient Clipping Setup ---
        self.max_grad_norm = config.gradient_clipping if config.gradient_clipping > 0 else None
        print(f"Gradient clipping {'enabled' if self.max_grad_norm else 'disabled'} with max_norm: {self.max_grad_norm}")

        # --- Checkpoint Loading ---
        if config.resume_from_checkpoint:
            self._load_checkpoint()
        else:
            print("Starting training from scratch (no checkpoint loaded).")

    def _save_checkpoint(self, epoch ,loss_value):
        """Saves the current state of the model, optimizer, and scheduler."""
        checkpoint_path = os.path.join(self.config.save_path, f"checkpoint_epoch_{epoch + 1}_step_{self.global_step}_loss_{loss_value}.pt")
        torch.save({
            'epoch': epoch + 1,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def _load_checkpoint(self):
        """Loads the latest checkpoint from the save path."""
        checkpoints = [f for f in os.listdir(self.config.save_path) if f.startswith("checkpoint_") and f.endswith(".pt")]
        if not checkpoints:
            print("No checkpoints found. Starting from scratch.")
            return

        checkpoints.sort(key=lambda x: int(re.search(r"epoch_(\d+)", x).group(1)) if re.search(r"epoch_(\d+)", x) else 0)
        latest_checkpoint_path = os.path.join(self.config.save_path, checkpoints[-1])
        print(f"Loading checkpoint from: {latest_checkpoint_path}")
        checkpoint = torch.load(latest_checkpoint_path, map_location=self.device)

        self.start_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        if self.use_amp and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Resumed training from epoch {self.start_epoch} at global step {self.global_step}")

    def train(self, resume_from_checkpoint=False):
        """
        Executes the main training loop over epochs and batches.

        Includes training steps, optional validation, and checkpoint saving.
        Contains extensive debugging prints for batch inspection and decoding.

        Args:
            resume_from_checkpoint (bool): Overrides the initial config setting
                                            for resuming from a checkpoint for this
                                            specific training run.
        """
        # Allow overriding the resume setting specifically for this run
        if resume_from_checkpoint:
            self.config.resume_from_checkpoint = True
            self._load_checkpoint()

        print(f"Starting training for {self.config.epochs} epochs...")
        if self.start_epoch > 0:
            print(f"Resuming from epoch {self.start_epoch} at global step {self.global_step}")

        for epoch in range(self.start_epoch, self.config.epochs):
            print(f"\n===== Epoch {epoch + 1}/{self.config.epochs} =====")

            # --- Training Phase ---
            self.model.train() # Set model to training mode
            total_loss = 0.0
            epoch_start_time = time.time()
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training", leave=False, unit="batch")

            for step, batch in enumerate(progress_bar):
                
                #batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)

                # --- Forward Pass with Automatic Mixed Precision (AMP) ---
                with torch.autocast(device_type=self.device.type, dtype=self.pt_dtype, enabled=self.use_amp):
                    try:                        
                        logits, loss = self.model(input_ids, attention_mask=attention_mask, targets=labels)
                        
                        if loss is None:
                            print(f"Warning: Loss is None at step {step}. Check model forward pass.")
                            continue
                    except Exception as e:
                        print(f"Error during model forward pass at step {step}. Error: {e}")
                        print(f"Input IDs shape: {input_ids.shape}")
                        if attention_mask is not None: print(f"Attention Mask shape: {attention_mask.shape}")
                        print(f"Labels shape: {labels.shape}")
                        continue # Skip this problematic batch

                # --- Backward Pass & Optimization ---
                self.scaler.scale(loss).backward()

                # Gradient Clipping (apply before optimizer step)
                if self.max_grad_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True) # Reset gradients (more efficient)
                self.lr_scheduler.step() # Update learning rate

                self.global_step += 1
                total_loss += loss.item()

                # Update progress bar description
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{self.lr_scheduler.get_last_lr()[0]:.2e}", step=self.global_step)

                
            # --- End of Epoch ---
            progress_bar.close()
            avg_train_loss = total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0.0
            epoch_duration = time.time() - epoch_start_time
            print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f} | Time: {datetime.timedelta(seconds=int(epoch_duration))}") # Formatted time

            # --- Validation Phase ---
            if self.val_loader:
                val_loss = self.evaluate()
                print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}")

            # --- Checkpoint Saving ---
            self._save_checkpoint(epoch, loss.item())

        print("Training finished.")

    def evaluate(self):
        """Evaluates the model on the validation set."""
        self.model.eval() # Set model to evaluation mode
        total_val_loss = 0.0
        progress_bar = tqdm(self.val_loader, desc="Validation", leave=False, unit="batch")

        with torch.no_grad(): # Disable gradient calculations during validation
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['labels'].to(self.config.device)

                with torch.autocast(device_type=self.device.type, dtype=self.pt_dtype, enabled=self.use_amp):
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    total_val_loss += loss.item()
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = total_val_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        return avg_val_loss