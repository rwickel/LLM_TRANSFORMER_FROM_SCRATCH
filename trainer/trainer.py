# trainer/trainer.py
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os
import math
import time

# Import components from the trainer package
from .config import TrainingConfig
from .utils import get_lr

class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, lr_scheduler_func, config: TrainingConfig, device, tokenizer):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr_scheduler_func = lr_scheduler_func
        self.config = config
        self.device = device
        self.tokenizer = tokenizer # Needed for saving with checkpoint

        # Mixed Precision setup
        self.use_amp = config.use_mixed_precision and device.startswith('cuda')
        self.scaler = GradScaler(enabled=self.use_amp)
        self.pt_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() and self.use_amp else torch.float16 if self.use_amp else torch.float32
       
        print(f"Trainer initialized. Using Mixed Precision: {self.use_amp} with dtype: {self.pt_dtype}")

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Calculate total steps and warmup steps for LR scheduling
        self.config.total_train_steps = math.ceil(self.config.num_epochs * len(self.train_loader) / self.config.gradient_accumulation_steps)
        self.config.warmup_steps = int(self.config.total_train_steps * self.config.warmup_ratio)
        self.config.min_lr = self.config.base_learning_rate * self.config.min_lr_ratio
        print(f"Total training steps planned: {self.config.total_train_steps}")
        print(f"Warmup steps: {self.config.warmup_steps}")


    def _save_checkpoint(self, is_best=False):
        """Saves model, optimizer, scaler, and config states."""
        os.makedirs(self.config.save_path, exist_ok=True)

        checkpoint_state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'model_config': self.model.config, # Assumes model has a .config attribute
            'train_config': self.config,
            'epoch': self.current_epoch,
            'step': self.global_step,
            'best_val_loss': self.best_val_loss,
        }

        # Save the latest checkpoint
        latest_save_path = os.path.join(self.config.save_path, self.config.checkpoint_filename_latest)
        torch.save(checkpoint_state, latest_save_path)
        print(f"Latest checkpoint saved to {latest_save_path}")

        # Save the tokenizer state alongside the latest checkpoint
        self.tokenizer.save_pretrained(self.config.save_path) # Saves vocab files etc. in the directory
        # print(f"Tokenizer saved to {self.config.save_path}") # Redundant if dir printed above

        # Save the best checkpoint if applicable
        if is_best:
            best_save_path = os.path.join(self.config.save_path, self.config.checkpoint_filename_best)
            torch.save(checkpoint_state, best_save_path)
            print(f"*** Best checkpoint saved to {best_save_path} (Val Loss: {self.best_val_loss:.4f}) ***")


    def _load_checkpoint(self, path):
         """Loads training state from a checkpoint. (Internal use for resuming)"""
         # Note: Model structure must match the saved checkpoint
         try:
             checkpoint = torch.load(path, map_location=self.device)
             self.model.load_state_dict(checkpoint['model_state_dict'])
             self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
             self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
             self.current_epoch = checkpoint['epoch']
             self.global_step = checkpoint['step']
             self.best_val_loss = checkpoint['best_val_loss']
             # Configs are usually not loaded back directly into existing objects,
             # but good practice to check compatibility if needed.
             print(f"Successfully loaded checkpoint state from {path}")
             print(f"Resuming training from Epoch {self.current_epoch}, Step {self.global_step}")
         except FileNotFoundError:
              print(f"Checkpoint file not found at {path}. Starting training from scratch.")
         except Exception as e:
              print(f"Error loading checkpoint state: {e}. Starting training from scratch.")


    @torch.no_grad()
    def _validate(self):
        """Performs validation on the validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        print("Running validation...")
        pbar = tqdm(self.val_loader, desc=f"Validation Epoch {self.current_epoch}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            with torch.autocast(device_type=self.device[:4], dtype=self.pt_dtype, enabled=self.use_amp): # Use autocast context
                # Ensure model returns loss when labels/targets provided
                logits, loss = self.model(input_ids, attention_mask=attention_mask, targets=labels)

            if loss is not None:
                total_loss += loss.item()
                num_batches += 1
            else:
                print("Warning: Loss is None during validation step.")

        self.model.train() # Set back to train mode
        if num_batches == 0: return float('inf')
        return total_loss / num_batches

    def _train_epoch(self):
        """Runs a single training epoch."""
        self.model.train()
        epoch_loss = 0
        steps_in_epoch = 0
        
        # Initialize tqdm progress bar using user's specified parameters
        batch_iterator = tqdm(self.train_loader,
                              desc=f"Epoch {self.current_epoch}/{self.config.num_epochs}", # Kept informative desc
                              leave=False,          # Added from user request
                              mininterval=0.5,      # Added from user request
                              ncols=100)            # Added ncols for better display control (optional)        


        for step, batch in enumerate(batch_iterator):
            is_accumulating = (step + 1) % self.config.gradient_accumulation_steps != 0
            # Effective step number after accumulation
            actual_step_num = self.global_step // self.config.gradient_accumulation_steps

            # Update learning rate
            if self.config.decay_lr:
                lr = self.lr_scheduler_func(
                    actual_step_num,
                    self.config.warmup_steps,
                    self.config.total_train_steps,
                    self.config.base_learning_rate,
                    self.config.min_lr
                )
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                 lr = self.config.base_learning_rate

            # Forward pass
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            with torch.autocast(device_type=self.device[:4], dtype=self.pt_dtype, enabled=self.use_amp):
                logits, loss = self.model(input_ids, attention_mask=attention_mask, targets=labels)
                if loss is not None:
                    # Scale loss for accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                else:
                    print(f"Warning: Loss is None at step {self.global_step}. Skipping backward.")
                    continue

            # Backward pass (scaled)
            self.scaler.scale(loss).backward()

            current_loss_value = loss.item() * self.config.gradient_accumulation_steps # Unscale for logging
            epoch_loss += current_loss_value
            steps_in_epoch += 1

            # Optimizer step (only when not accumulating)
            if not is_accumulating:
                self.scaler.unscale_(self.optimizer)
                if self.config.gradient_clipping_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                # Logging - Update tqdm postfix here
                if actual_step_num % self.config.log_interval == 0:
                    # Use set_postfix on the tqdm object directly
                    batch_iterator.set_postfix(Loss=f"{current_loss_value:.4f}", LR=f"{lr:.6f}", Step=actual_step_num, refresh=False) # refresh=False reduces flicker

                # Evaluation step
                if actual_step_num > 0 and actual_step_num % self.config.eval_interval == 0:
                    # Temporarily close the training bar to print validation results cleanly
                    batch_iterator.close()
                    val_loss = self._validate() # Assuming _validate also uses tqdm now
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                    print(f"\nStep {actual_step_num}: Validation Loss = {val_loss:.4f} (Best: {self.best_val_loss:.4f})")
                    self._save_checkpoint(is_best=is_best)
                    self.model.train()
                    # Re-open the iterator description for the next steps
                    batch_iterator = tqdm(self.train_loader,
                                          initial=step + 1, # Start from current step
                                          total=len(self.train_loader), # Total steps in loader
                                          desc=f"Epoch {self.current_epoch}/{self.config.num_epochs}",
                                          leave=False,
                                          mininterval=0.5,
                                          ncols=100)


            self.global_step += 1

        # Ensure the loop is closed at the end of the epoch if not already closed by eval
        if hasattr(batch_iterator, 'close'):
             batch_iterator.close()

        # Return average loss for the epoch
        if steps_in_epoch == 0: return 0.0
        return epoch_loss / steps_in_epoch


    def train(self, resume_from_checkpoint=True):
        """Main training loop."""
        print(f"--- Starting Training ---")
        print(f" Config: {self.config}")
        total_start_time = time.time()

        # Attempt to load checkpoint if resuming
        if resume_from_checkpoint:
             latest_checkpoint_path = os.path.join(self.config.save_path, self.config.checkpoint_filename_latest)
             if os.path.exists(latest_checkpoint_path):
                  self._load_checkpoint(latest_checkpoint_path)
             else:
                  print("No checkpoint found to resume from. Starting from scratch.")

        start_epoch = self.current_epoch # Start from loaded epoch or 0

        for epoch in range(start_epoch, self.config.num_epochs):
            self.current_epoch = epoch + 1 # Use 1-based epoch for display
            epoch_start_time = time.time()

            avg_train_loss = self._train_epoch()

            # Validate at the end of each epoch
            val_loss = self._validate()
            is_best = val_loss < self.best_val_loss
            if is_best:
                 self.best_val_loss = val_loss

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            print(f"Epoch {self.current_epoch}/{self.config.num_epochs} Summary | Avg Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} (Best: {self.best_val_loss:.4f}) | Duration: {epoch_duration:.2f}s")

            # Save latest checkpoint (and potentially best again if end-of-epoch validation was better)
            self._save_checkpoint(is_best=is_best)

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        print(f"--- Training Finished ---")
        print(f"Total Training Duration: {total_duration:.2f}s")
        print(f"Best Validation Loss achieved: {self.best_val_loss:.4f}")