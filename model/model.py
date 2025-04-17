# model.py
import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt # Removed unused imports
# import seaborn as sns
# import numpy as np
# from transformers import AutoTokenizer, AutoModel # Removed unused imports
import math
from dataclasses import dataclass
from typing import Optional, Tuple

# Import from local modules
from model.config import TransformerConfig # Updated config
from model.layers import RMSNorm, SwiGLU, MultiHeadAttention, TransformerBlock # Updated layers


# -----------------------------
# Decoder Model Class (Core Stack)
# -----------------------------
class DecoderModel(nn.Module):
    """The core Transformer decoder stack."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        # Padding index handling for Embedding
        self.token_emb = nn.Embedding(
            config.vocab_size,
            config.n_embd,
            padding_idx=config.pad_token_id if config.pad_token_id is not None else None # Use None if not set
        )
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = RMSNorm(config.n_embd)
        # RoPE cache building is now handled lazily within MultiHeadAttention

    def forward(self, idx: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Forward pass for the decoder stack. """
        B, T = idx.size()
        # Ensure input tensor is on the correct device (model's device)
        # device = self.token_emb.weight.device # Infer device from parameters
        # idx = idx.to(device) # Move input if necessary (usually done outside model)

        x = self.token_emb(idx) # (B, T, n_embd)

        # Pass the attention mask to each block
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask) # Pass mask down

        x = self.final_norm(x) # (B, T, n_embd)
        return x

# --- DecoderLM class (Adds Language Model Head) ---
class DecoderLM(nn.Module):
    """Decoder Language Model with a head for predicting token probabilities."""
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        # Core transformer model
        self.transformer = DecoderModel(config)
        # Linear head to project hidden states to vocabulary logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)
        # Store EOS token ID from config if available
        self.eos_token_id = config.eos_token_id

        # Optional: Weight tying based on config
        if config.tie_weights:
            print("Tying input and output embedding weights.")
            self.transformer.token_emb.weight = self.lm_head.weight

        # Initialize weights using config std dev
        self.apply(self._init_weights) # Apply custom initialization

    def _init_weights(self, module):
        """Initialize weights of Linear and Embedding layers."""
        if isinstance(module, nn.Linear):
            # Use init_std from config
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
             # Use init_std from config
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.padding_idx is not None:
                # Ensure padding embedding is zero
                with torch.no_grad():
                     module.weight[module.padding_idx].zero_()


    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None, # Keep attention_mask
                targets: Optional[torch.Tensor] = None,
                ignore_idx: int = -100 # Standard ignore index
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for training or inference.

        Args:
            input_ids: Input token IDs (B, T).
            attention_mask: Mask indicating non-padding tokens (B, T). True for non-padding.
                           *** IMPORTANT: This mask is now passed to the attention layers. ***
            targets: Target token IDs for loss calculation (B, T). Assumed shifted.
            ignore_idx: Index to ignore in the loss calculation (for padding in targets). -100 see generic_tokenize_fn

        Returns:
            A tuple containing:
            - logits: Output logits of shape (B, T, vocab_size).
            - loss: Calculated cross-entropy loss if targets are provided, else None.
        """
        # Pass input token IDs and the attention mask through the transformer model
        hidden_states = self.transformer(input_ids, attention_mask=attention_mask) # (B, T, n_embd)

        # Compute logits for the entire sequence
        logits = self.lm_head(hidden_states) # (B, T, vocab_size)

        preds = logits.argmax(dim=-1)

        
        
        loss = None
        if targets is not None:

            ## DEBUG 
            for i in range(1):  # Just look at 1 sample
                RED = "\033[91m"
                END = "\033[0m"
                input = input_ids[i][:20].tolist()
                target = targets[i][:20].tolist()
                predicted = preds[i][:20].tolist()
                print("Input:   ", input)
                print("Target:   ", target)

                # Print Predicted with red color where it differs from target
                print("Predicted:", end=" ")
                for t, p in zip(target, predicted):
                    if t == ignore_idx:
                        pass
                    if t == p:
                        print(f"{p}", end=" ")
                    else:
                        print(f"{RED}{p}[{self.config.tokenizer.decode(p)}]{END}", end=" ")
                print()  # newline
            ###########

            # Shift logits and targets to align predictions with labels
            if self.config.shift_labels: # check if the train data already shifted 
                # Shift logits and targets to align predictions with labels
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = targets[:, 1:].contiguous()
            else:
                shift_logits = logits.contiguous()
                shift_targets = targets.contiguous()

            
            B, T, V = shift_logits.size()
            logits_flat = shift_logits.view(-1, V)        # (B*T, V)
            targets_flat = shift_targets.view(-1)         # (B*T,)

            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_idx)     

            return logits, loss  # Optionally return unshifted logits + shifted loss

        return logits, None # Return only logits if no targets

    
    def generate(self,
             prompt: str,
             max_tokens: int = 0,
             temperature: float = 1.0,
             top_k: Optional[int] = None,
             ) -> torch.Tensor:
        """
        Generates token sequences autoregressively based on a string input (prompt).
        This version yields token by token for streaming output, starting with the input string.
        """
        max_tokens = max_tokens if max_tokens > 0 else self.config.block_size
        self.eval()  # Set model to evaluation mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use the correct device

        # Tokenize the input string (prompt) into token IDs
        encoding = self.config.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)
        input_ids = encoding['input_ids'].to(device)
        B, T_prompt = input_ids.size()
        generated_ids = input_ids  # Start with the prompt tokens

        stop_token_id = self.config.eos_token_id

        for _ in range(max_tokens):
            # --- Context Cropping ---
            current_seq_len = generated_ids.size(1)
            if current_seq_len <= self.config.block_size:
                ids_to_pass = generated_ids
            else:
                ids_to_pass = generated_ids[:, -self.config.block_size:]

            # --- Forward Pass ---
            logits, _ = self.forward(ids_to_pass, attention_mask=None, targets=None)  # No mask needed for causal gen
            next_token_logits = logits[:, -1, :]  # Shape: (B, V)

            # --- Sampling Logic ---
            if temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            if top_k is not None and top_k > 0:
                # Ensure k is not larger than vocab size
                k = min(top_k, next_token_logits.size(-1))
                top_k_values, top_k_indices = torch.topk(next_token_logits, k=k, dim=-1)
                threshold = top_k_values[:, [-1]]  # Get the k-th value
                next_token_logits[next_token_logits < threshold] = -float('Inf')

                # Print top-k predictions
                #print(f"\n\nTop {k} predicted next tokens:")
                #top_k_tokens = self.config.tokenizer.decode(top_k_indices[0], skip_special_tokens=True)
                #print(top_k_tokens)

            # --- Choose Next Token ---
            if temperature == 0:  # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (B, 1)
            else:  # Sampling
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # --- Append Token ---
            generated_ids = torch.cat((generated_ids, next_token), dim=1)

            # Decode the next token
            decoded_token = self.config.tokenizer.decode(next_token.squeeze().cpu(), skip_special_tokens=True)            
            
            # --- EOS Check ---
            if stop_token_id is not None and (next_token == stop_token_id).all():                
                break

            # Yield the decoded token to the caller for streaming output
            yield decoded_token