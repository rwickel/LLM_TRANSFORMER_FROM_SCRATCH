# trainer/data_utils.py
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
import os

# Import the config class
from .config import TrainingConfig

def load_and_prepare_data(config: TrainingConfig, tokenizer):
    """Loads, cleans, splits, and tokenizes the dataset."""
    print(f"Loading dataset: {config.dataset_name} ({config.dataset_config_name})")
    dataset = load_dataset(config.dataset_name, config.dataset_config_name, trust_remote_code=True)
    print(f"Dataset loaded. Original structure: {dataset}")

    # Basic text cleaning (optional)
    def clean_text(examples):
        examples["text"] = [text.strip() for text in examples["text"] if text.strip()]
        return examples

    dataset = dataset.map(clean_text, batched=True, num_proc=os.cpu_count())
    dataset = dataset.filter(lambda example: len(example['text']) > 0, num_proc=os.cpu_count())

    # Split dataset if validation split doesn't exist
    if "validation" not in dataset and "train" in dataset:
        print(f"Creating validation split ({config.validation_split_percentage * 100:.1f}%)...")
        split = dataset["train"].train_test_split(
            test_size=config.validation_split_percentage,
            seed=config.seed
        )
        train_data = split["train"]
        val_data = split["test"]
    elif "train" in dataset and "validation" in dataset:
        train_data = dataset["train"]
        val_data = dataset["validation"]
        print("Using predefined train/validation splits.")
    else:
        raise ValueError(f"Cannot find 'train' and/or 'validation' splits in dataset: {dataset}")

    print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")

    # Tokenization function
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_seq_length,
        )

    print("Tokenizing datasets (this may take a while)...")
    tokenized_train_data = train_data.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_data.column_names,
        num_proc=os.cpu_count() # Use multiple processes for tokenization
    )
    tokenized_val_data = val_data.map(
        tokenize_fn,
        batched=True,
        remove_columns=val_data.column_names,
        num_proc=os.cpu_count()
    )
    print("Tokenization complete.")

    return tokenized_train_data, tokenized_val_data


def create_dataloaders(config: TrainingConfig, tokenizer, train_data, val_data):
    """Creates DataLoaders and the DataCollator."""
    print("Initializing Data Collator...")
    # mlm=False indicates Causal LM. Collator handles padding and labels.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    print("Creating DataLoaders...")
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        collate_fn=data_collator,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True # Can speed up CPU to GPU transfers
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size, # Can often use larger batch size for validation
        collate_fn=data_collator,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    print(f"Train loader: {len(train_loader)} batches, Val loader: {len(val_loader)} batches")

    return train_loader, val_loader, data_collator