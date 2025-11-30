# =====================
#  TRAINING SCRIPT FOR LANGUAGE MODEL
# =====================
#
# Description:
#   This script trains a character-level transformer-based language model on text files.
#   It supports training on various datasets with configurable hyperparameters.
#
# Code Flow:
#   main() 
#     → train_model() 
#       → load_text_files_from_folder() → loads text data
#       → CharDataset() → creates character-level dataset
#       → DataLoader() → creates batches
#       → SimpleLanguageModel() → initializes model
#       → train_epoch() [for each epoch]
#         → model.forward() → forward pass
#         → loss.backward() → backward pass
#         → optimizer.step() → update weights
#       → torch.save() → saves model.pt
#   
#   setup_logging() [module level] → creates log_file



# =====================
#  EXAMPLE USAGE
# =====================
#
# Basic usage (train on default toy dataset with default parameters):
#   python 01_trainers/train.py
#
# Or explicitly specify the dataset folder:
#   python 01_trainers/train.py 40_training_data/dataset_toy/
#
# Train on anthropology dataset with more epochs:
#   python 01_trainers/train.py 40_training_data/dataset_anthropology/ --epochs 50
#
# Train on machine learning dataset with larger model:
#   python 01_trainers/train.py 40_training_data/dataset_machine_learning/ --embed-dim 256 --num-layers 6 --epochs 30
#
# Train on Bitcoin dataset with custom parameters:
#   python 01_trainers/train.py 40_training_data/dataset_bitcoin/ --epochs 20 --batch-size 64 --lr 0.001
#
# Available arguments:
#   dataset_folder (optional) - Path to folder containing .txt files (default: 40_training_data/dataset_toy/)
#   --epochs - Number of training epochs (default: 10)
#   --batch-size - Batch size (default: 32)
#   --block-size - Sequence length (default: 128)
#   --embed-dim - Embedding dimension (default: 128)
#   --num-heads - Number of attention heads (default: 4)
#   --num-layers - Number of transformer layers (default: 3)
#   --lr - Learning rate (default: 0.001)
#
# Model will be saved to: 50_models/{dataset_name}/embed{embed_dim}_layers{num_layers}_heads{num_heads}_epochs{epochs}.pt
# Example: 50_models/dataset_toy/embed128_layers3_heads4_epochs10.pt


# =====================
#  SETUP
# =====================
#  SET WORKING DIRECTORY TO GIT ROOT
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent / '95_utils')); __import__('path_utils').setup_workspace(__file__)

# IMPORTS
import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import load_text_files_from_folder  # type: ignore
from train_utils import CharDataset, SimpleLanguageModel, train_epoch  # type: ignore

# LOGGING SETUP
from logger_utils import setup_logging  # type: ignore
logger = setup_logging() # Initialize logger


# =====================
#  TRAIN MODEL()
# =====================

def train_model(dataset_folder, epochs=10, batch_size=32, block_size=128, 
                embed_dim=128, num_heads=4, num_layers=3, learning_rate=1e-3,
                device=None):
    """
    Train a language model on all text files in a dataset folder.
    
    Args:
        dataset_folder: Path to folder containing .txt files
        epochs: Number of training epochs
        batch_size: Batch size for training
        block_size: Sequence length for training
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        learning_rate: Learning rate for optimizer
        device: Device to train on (cuda/cpu)
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Training Parameters:")
    logger.info(f"  - Dataset folder: {dataset_folder}")
    logger.info(f"  - Total epochs: {epochs}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Sequence length (block_size): {block_size}")
    logger.info(f"  - Embedding dimension: {embed_dim}")
    logger.info(f"  - Attention heads: {num_heads}")
    logger.info(f"  - Transformer layers: {num_layers}")
    logger.info(f"  - Learning rate: {learning_rate}")
    logger.info("")
    
    # Set device
    logger.info("=" * 80)
    logger.info("STEP 1: DEVICE SELECTION")
    logger.info("=" * 80)
    logger.info("")
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Selected device: {device}")
    if torch.cuda.is_available():
        logger.info(f"  - CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"  - Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.info("  - Using CPU (no CUDA available)")
    logger.info("")
    
    # Load data
    logger.info("=" * 80)
    logger.info("STEP 2: DATA LOADING")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Loading all .txt files from: {dataset_folder}")
    text = load_text_files_from_folder(dataset_folder)
    logger.info("")
    logger.info("Data loaded successfully:")
    logger.info(f"  - Total characters: {len(text):,}")
    logger.info(f"  - Total lines: {text.count(chr(10)) + 1}")
    logger.info("")
    
    # Create dataset
    logger.info("=" * 80)
    logger.info("STEP 3: DATASET PREPARATION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Creating character-level dataset...")
    logger.info(f"  - Processing text into sequences of length {block_size}")
    dataset = CharDataset(text, block_size=block_size)
    logger.info("")
    logger.info("Dataset created:")
    logger.info(f"  - Vocabulary size: {dataset.vocab_size} unique characters")
    logger.info(f"  - Total training sequences: {len(dataset):,}")
    logger.info(f"  - Total tokens: {len(dataset.data):,}")
    logger.info("")
    
    # Create dataloader
    logger.info("=" * 80)
    logger.info("STEP 4: DATALOADER SETUP")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Creating DataLoader with batch_size={batch_size} (shuffling enabled)...")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    logger.info(f"  - Total batches per epoch: {len(dataloader)}")
    logger.info(f"  - Samples per batch: {batch_size}")
    logger.info("")
    
    # Create model
    logger.info("=" * 80)
    logger.info("STEP 5: MODEL INITIALIZATION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Building transformer-based language model...")
    logger.info(f"  - Architecture: {num_layers} transformer layers")
    logger.info(f"  - Embedding dimension: {embed_dim}")
    logger.info(f"  - Attention heads per layer: {num_heads}")
    logger.info(f"  - Vocabulary size: {dataset.vocab_size}")
    logger.info(f"  - Context window: {block_size} tokens")
    model = SimpleLanguageModel(
        vocab_size=dataset.vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        block_size=block_size
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("")
    logger.info("Model initialized:")
    logger.info(f"  - Total parameters: {num_params:,}")
    logger.info(f"  - Trainable parameters: {trainable_params:,}")
    logger.info(f"  - Model location: {next(model.parameters()).device}")
    logger.info("")
    
    # Loss and optimizer
    logger.info("=" * 80)
    logger.info("STEP 6: TRAINING SETUP")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Configuring training components:")
    logger.info("  - Loss function: CrossEntropyLoss")
    logger.info(f"  - Optimizer: AdamW (learning rate: {learning_rate})")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    logger.info("")
    
    # Training loop
    logger.info("=" * 80)
    logger.info("STEP 7: TRAINING LOOP")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Starting training: {epochs} epochs, {len(dataloader)} batches per epoch")
    logger.info("")
    
    for epoch in range(epochs):
        logger.info("")
        logger.info("-" * 80)
        logger.info(f"EPOCH {epoch+1}/{epochs}")
        logger.info("-" * 80)
        logger.info("")
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device, epoch_num=epoch+1)
        logger.info("")
        logger.info(f"Epoch {epoch+1}/{epochs} completed - Average Loss: {avg_loss:.4f}")
        logger.info("")
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    
    # Save model
    logger.info("=" * 80)
    logger.info("STEP 8: MODEL SAVING")
    logger.info("=" * 80)
    logger.info("")
    
    # Extract dataset name from folder path
    dataset_path = Path(dataset_folder)
    dataset_name = dataset_path.name if dataset_path.is_dir() else dataset_path.stem
    
    # Create folder structure: 50_models/{dataset_name}/
    base_model_dir = Path("50_models")
    model_dir = base_model_dir / dataset_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with config: embed{embed_dim}_layers{num_layers}_heads{num_heads}_epochs{epochs}.pt
    model_filename = f"embed{embed_dim}_layers{num_layers}_heads{num_heads}_epochs{epochs}.pt"
    model_path = model_dir / model_filename
    
    logger.info(f"Saving trained model to: {model_path}")
    logger.info("")
    logger.info("Saving model components:")
    logger.info("  - Model weights (state_dict)")
    logger.info("  - Vocabulary mappings (character to index)")
    logger.info("  - Model hyperparameters")
    save_dict = {
        'model_state_dict': model.state_dict(),
        'vocab_size': dataset.vocab_size,
        'stoi': dataset.stoi,
        'itos': dataset.itos,
        'block_size': block_size,
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'dataset_folder': dataset_folder,
        'dataset_name': dataset_name,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
    }
    torch.save(save_dict, model_path)
    file_size = model_path.stat().st_size / (1024 * 1024)  # Size in MB
    logger.info("")
    logger.info(f"Model saved successfully!")
    logger.info(f"  - File path: {model_path}")
    logger.info(f"  - File size: {file_size:.2f} MB")
    logger.info("")
    
    logger.info("=" * 80)
    logger.info("TRAINING SESSION ENDED")
    logger.info("=" * 80)
    logger.info("")
    
    return model, dataset


# =====================
#  MAIN
# =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language model on text files")
    parser.add_argument(
        "dataset_folder",
        type=str,
        nargs='?',
        default="40_training_data/dataset_toy/",
        help="Path to folder containing .txt files to train on (default: 40_training_data/dataset_toy/)"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--block-size", type=int, default=128, help="Sequence length")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    args = parser.parse_args()
    
    logger.info("Command line arguments parsed")
    logger.info("")
    
    # Train the model
    try:
        train_model(
            dataset_folder=args.dataset_folder,
            epochs=args.epochs,
            batch_size=args.batch_size,
            block_size=args.block_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            learning_rate=args.lr
        )
    except Exception as e:
        logger.error("")
        logger.error("")
        logger.error("=" * 80)
        logger.error("TRAINING FAILED")
        logger.error("=" * 80)
        logger.error("")
        logger.error(f"Error: {str(e)}", exc_info=True)
        logger.error("")
        raise

