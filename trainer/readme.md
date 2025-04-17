# Custom Transformer Training Framework

## Overview

This package provides a structured framework for training a custom Transformer decoder model (`DecoderLM`) using PyTorch. It leverages the Hugging Face `datasets` and `transformers` libraries for efficient data handling and tokenization. The framework is designed to be modular, allowing for easier configuration and modification.

Key features include:
* Integration with Hugging Face `datasets`.
* Automatic dataset loading, tokenization, and batching.
* A configurable `Trainer` class managing the training loop.
* Support for Mixed Precision training (`torch.cuda.amp`) via `GradScaler`.
* Cosine learning rate scheduling with linear warmup.
* Checkpointing (saving the latest and best model based on validation loss).
* Resuming training from the latest checkpoint.
* Saving model configuration and tokenizer state for easy reloading.
* Progress bars using `tqdm`.

## Directory Structure

The framework assumes the following project structure:
├── model/
│   ├── config.py       # Your existing TransformerConfig, get_model_config, check_device
│   ├── layers.py       # Your existing layer implementations
│   ├── model.py        # Your existing DecoderLM implementation
│   └── positional_encoding.py # Your existing positional encoding
├── trainer/
│   ├── __init__.py     # Empty file to make it a package
│   ├── config.py       # Training configuration dataclass
│   ├── data_utils.py   # Data loading and preparation functions
│   ├── trainer.py      # The main Trainer class
│   └── utils.py        # Helper functions (get_lr)
└── train.py            # Main script to run training


*Note: `check_device` was moved to `trainer/utils.py` in this structure.*

## Core Components (`trainer/`)

* **`config.py`**: Defines the `TrainingConfig` dataclass. Modify this file to change hyperparameters like learning rate, batch size, epochs, dataset names, paths, etc.
* **`data_utils.py`**: Contains functions (`create_dataloaders`) responsible for fetching datasets from Hugging Face Hub, performing train/validation splits, tokenizing the text data using the provided tokenizer, and setting up `DataLoader` instances with a `DataCollatorForLanguageModeling`.
* **`trainer.py`**: Implements the `Trainer` class which orchestrates the entire training process. It handles:
    * Epoch and step iteration.
    * Learning rate scheduling updates.
    * Mixed precision forward/backward passes using `GradScaler`.
    * Gradient accumulation.
    * Optimization steps.
    * Periodic validation loops.
    * Saving checkpoints (`latest_checkpoint.pt`, `best_model.pt`).
    * Resuming from checkpoints.
* **`utils.py`**: Provides utility functions like the `get_lr` scheduler and the `load_model_from_checkpoint` function for loading previously saved models and tokenizers.

## Features

* **Modular Design**: Configuration, data handling, and training logic are separated into distinct files.
* **Hugging Face Integration**: Leverages `datasets` for loading and `transformers` for tokenization and data collation.
* **Checkpointing**:
    * Saves the latest training state (model weights, optimizer state, scaler state, configs) periodically and at the end of epochs to `latest_checkpoint.pt`.
    * Saves the best model state based on validation loss to `best_model.pt`.
    * Saves the tokenizer state alongside checkpoints in the save directory.
* **Resuming**: Training automatically attempts to resume from `latest_checkpoint.pt` if found in the specified save path.
* **Mixed Precision**: Uses `torch.cuda.amp.GradScaler` and `autocast` for faster training and reduced memory usage on compatible GPUs (configurable via `TrainingConfig.use_mixed_precision`).
* **LR Scheduling**: Implements cosine decay with linear warmup.
* **Progress Monitoring**: Uses `tqdm` to display progress bars for training and validation loops.
* **Validation**: Includes a validation loop to monitor generalization performance and determine the best model checkpoint.

## Prerequisites

Ensure you have the necessary libraries installed:

```bash
pip install torch transformers datasets tqdm accelerate # accelerate can sometimes help backend operations