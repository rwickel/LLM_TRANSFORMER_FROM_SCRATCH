# trainer/data_utils.py
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
import os
import re

# Import the config class
from .config import TrainingConfig

def load_and_prepare_data(config: TrainingConfig, tokenizer):
    """Loads, cleans, splits, tokenizes the dataset, and saves info."""

    print(f"Loading dataset: {config.dataset_name} ({config.dataset_config_name})")
    # Load the dataset, potentially containing train, validation, and test splits
    try:
        dataset = load_dataset(config.dataset_name, config.dataset_config_name, trust_remote_code=True)
        print(f"Dataset loaded. Original structure: {dataset}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise # Re-raise the exception after logging

    # --- Basic text cleaning (optional) ---
    def clean_text(examples):
        # Ensure 'text' column exists, handle potential missing column gracefully
        if "text" in examples:
            cleaned_texts = [str(text).strip() for text in examples["text"] if text and isinstance(text, str) and str(text).strip()]
            examples["text"] = cleaned_texts
        # If 'text' column missing or different, adapt or log warning
        # else:
        #     print("Warning: 'text' column not found in this batch during cleaning.")
        return examples

    # Apply cleaning to all available splits
    cleaned_dataset = DatasetDict()
    available_splits = list(dataset.keys())
    print(f"Available splits before cleaning: {available_splits}")

    for split_name in available_splits:
      if dataset[split_name] is not None and len(dataset[split_name]) > 0:
          print(f"Cleaning split: {split_name}...")
          cleaned_dataset[split_name] = dataset[split_name].map(
              clean_text,
              batched=True,
              num_proc=os.cpu_count()
          )
          # Filter out examples that became empty after cleaning
          # Check if 'text' column exists before filtering
          if 'text' in cleaned_dataset[split_name].column_names:
              cleaned_dataset[split_name] = cleaned_dataset[split_name].filter(
                  lambda example: example.get('text') and len(example['text']) > 0,
                  num_proc=os.cpu_count()
              )
          else:
              print(f"Warning: 'text' column not found in split '{split_name}' after cleaning, skipping empty filter.")
      else:
           print(f"Skipping cleaning for empty or None split: {split_name}")


    print(f"Dataset structure after cleaning: {cleaned_dataset}")

    # --- Determine Train, Validation, and Test splits ---
    train_data = None
    val_data = None
    test_data = None # Initialize test_data

    if "train" in cleaned_dataset:
        base_split = cleaned_dataset["train"]

        # Check for predefined validation split
        if "validation" in cleaned_dataset:
            print("Using predefined train/validation splits.")
            train_data = base_split
            val_data = cleaned_dataset["validation"]
        else:
            print(f"Creating validation split ({config.validation_split_percentage * 100:.1f}%)...")
            # Ensure base_split is not empty before splitting
            if len(base_split) > 0:
                split = base_split.train_test_split(
                    test_size=config.validation_split_percentage,
                    seed=config.seed
                )
                train_data = split["train"]
                val_data = split["test"] # This is the validation set
            else:
                print("Warning: Train split is empty, cannot create validation split.")
                train_data = base_split # Assign empty dataset
                val_data = base_split # Assign empty dataset


        # Check for predefined test split
        if "test" in cleaned_dataset:
            print("Using predefined test split.")
            test_data = cleaned_dataset["test"]
        else:
            print("No predefined test split found.")
            # test_data remains None

    else:
        # If 'train' split doesn't exist, we cannot proceed with standard training setup
        raise ValueError(f"Critical: 'train' split not found in dataset: {cleaned_dataset}")

    # --- Get sample counts ---
    num_train_samples = len(train_data) if train_data else 0
    num_val_samples = len(val_data) if val_data else 0
    num_test_samples = len(test_data) if test_data else 0 # Get count if test_data exists

    print(f"Train samples: {num_train_samples}")
    print(f"Validation samples: {num_val_samples}")
    print(f"Test samples: {num_test_samples if test_data else 'Not available'}")

    # Sanitize the dataset name to create a valid filename
    sanitized_dataset_name = re.sub(r'[\\/*?:"<>|]',"", config.dataset_name) # Remove invalid OS chars
    sanitized_dataset_name = sanitized_dataset_name.replace("/", "_") # Replace slashes common in HF names

    # Create the filename using the sanitized dataset name + .txt extension
    info_filename = f"{sanitized_dataset_name}.txt"
    dataset_info_path = os.path.join(config.save_path, info_filename)

    print(f"Attempting to save dataset information to: {dataset_info_path}")
    try:
        with open(dataset_info_path, 'w',  encoding='utf-8') as f:
            f.write(f"Dataset Name: {config.dataset_name}\n")
            f.write(f"Dataset Config/ID: {config.dataset_config_name}\n")
            f.write(f"Original Splits Available: {available_splits}\n")
            f.write("--- Sample Counts After Processing ---\n")
            f.write(f"Train samples: {num_train_samples}\n")
            f.write(f"Validation samples: {num_val_samples}\n")
            # Write test sample count conditionally
            if test_data:
                f.write(f"Test samples: {num_test_samples}\n")
            else:
                f.write("Test samples: Not available\n")

            # Save training examples
            f.write("\n--- Sample Training Examples (First 5) ---\n")
            try:
                if train_data and len(train_data) > 0:
                    for i in range(min(5, len(train_data))):
                        example = train_data[i]
                        text = example.get("text", "[No 'text' field]")
                        cleaned_text = text.replace('\n', ' ').strip() if isinstance(text, str) else str(text)
                        f.write(f"[{i+1}] {cleaned_text}\n")
                else:
                    f.write("No training data available to preview.\n")    
            
            except Exception as e:
                f.write(f"Error retrieving training examples: {e}\n")

        print(f"Dataset information saved to: {dataset_info_path}")
    except Exception as e:
        print(f"Warning: Could not write dataset info file. Error: {e}")

    # --- Tokenization function ---
    def tokenize_fn(examples):
        # Ensure 'text' column exists before tokenizing
        if "text" not in examples:
             # Handle cases where 'text' might be missing after cleaning/filtering
             # For example, return empty dict or raise error
             print(f"Warning: 'text' column missing in batch for tokenization. Skipping batch.")
             # Return structure expected by map (can be empty if needed)
             return {'input_ids': [], 'attention_mask': []}

        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_seq_length,
            # padding=True, # Padding handled by DataCollator
        )

    # --- Tokenize Train and Validation Data ---
    print("Tokenizing datasets (this may take a while)...")
    tokenized_train_data = None
    tokenized_val_data = None

    if train_data and len(train_data) > 0:
        # Determine columns to remove - use original columns if available, else current ones
        remove_cols_train = train_data.column_names if train_data.column_names else []
        if 'text' not in remove_cols_train and 'text' in train_data.features:
             # If 'text' wasn't listed but exists, add it if you want it removed
             # remove_cols_train.append('text') # Decide if you want to keep 'text' post-tokenization
             pass
        print(f"Tokenizing train data. Removing columns: {remove_cols_train}")
        tokenized_train_data = train_data.map(
            tokenize_fn,
            batched=True,
            remove_columns=remove_cols_train, # Use determined columns
            num_proc=os.cpu_count(),
            load_from_cache_file=True # Enable caching
        )
    else:
         print("Skipping tokenization for empty train data.")


    if val_data and len(val_data) > 0:
        remove_cols_val = val_data.column_names if val_data.column_names else []
        if 'text' not in remove_cols_val and 'text' in val_data.features:
             pass # Decide if you want to keep 'text' post-tokenization
        print(f"Tokenizing validation data. Removing columns: {remove_cols_val}")
        tokenized_val_data = val_data.map(
            tokenize_fn,
            batched=True,
            remove_columns=remove_cols_val, # Use determined columns
            num_proc=os.cpu_count(),
            load_from_cache_file=True # Enable caching
        )
    else:
         print("Skipping tokenization for empty validation data.")


    # Note: Test data is identified and counted, but not tokenized or returned by this function.
    # If you need tokenized test data later, you'd add similar mapping logic for test_data.

    print("Data loading and preparation complete.")
    if not tokenized_train_data or not tokenized_val_data:
         print("Warning: Resulting train or validation data is empty/None after processing.")


    # Return only tokenized train and validation data as before
    return tokenized_train_data, tokenized_val_data


def create_dataloaders(config: TrainingConfig, tokenizer, train_data, val_data):
    """Creates DataLoaders and the DataCollator."""
    # Check if input data is valid before proceeding
    if train_data is None or val_data is None:
        raise ValueError("Cannot create DataLoaders with None train_data or val_data.")
    if len(train_data) == 0 or len(val_data) == 0:
        print("Warning: Creating DataLoaders with empty train or validation dataset.")


    print("Initializing Data Collator...")
    # mlm=False indicates Causal LM. Collator handles padding and labels.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    print("Creating DataLoaders...")
    # Use max() to prevent num_workers > 0 for empty datasets if needed,
    # though DataLoader might handle it. Safest to use 0 if dataset empty.
    train_num_workers = config.num_workers if len(train_data) > 0 else 0
    val_num_workers = config.num_workers if len(val_data) > 0 else 0

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        collate_fn=data_collator,
        shuffle=True if len(train_data) > 0 else False, # No need to shuffle empty data
        num_workers=train_num_workers,
        pin_memory=True # Can speed up CPU to GPU transfers
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size, # Can often use larger batch size for validation
        collate_fn=data_collator,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=True
    )
    print(f"Train loader: {len(train_loader)} batches, Val loader: {len(val_loader)} batches")

    return train_loader, val_loader, data_collator