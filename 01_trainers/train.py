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

# Set working directory to git root
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent / '95_utils')); __import__('path_utils').setup_workspace(__file__)

# Imports
import os
import sys
import signal
import argparse
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import load_text_files_from_folder  # type: ignore
from train_utils import CharDataset, SimpleLanguageModel, train_epoch, evaluate  # type: ignore

# Logging setup
from logger_utils import setup_logging  # type: ignore
logger = setup_logging()


# =====================
#  BATCH SIZE OPTIMIZATION
# =====================

def find_max_batch_size(model, dataset, device, block_size, start_batch_size=32, max_batch_size=2048):
    """
    Automatically find the maximum batch size that fits in memory.
    
    Args:
        model: The model to test with
        dataset: The dataset to test with
        device: Device to test on
        block_size: Sequence length
        start_batch_size: Starting batch size to test
        max_batch_size: Maximum batch size to test (safety limit)
    
    Returns:
        Maximum batch size that fits in memory
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("AUTOMATIC BATCH SIZE DETECTION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Finding maximum batch size that fits in memory...")
    logger.info("")
    
    model.eval()  # Use eval mode for testing (no gradients)
    criterion = nn.CrossEntropyLoss()
    
    # Clear any existing cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    current_batch_size = start_batch_size
    last_working_batch_size = None
    
    # First, try to find an upper bound by doubling
    logger.info("Phase 1: Finding upper bound...")
    while current_batch_size <= max_batch_size:
        try:
            # Create a small dataloader with this batch size
            test_loader = DataLoader(dataset, batch_size=current_batch_size, shuffle=False)
            
            # Try to process one batch
            x, y = next(iter(test_loader))
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            with torch.no_grad():
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # If we get here, it worked!
            last_working_batch_size = current_batch_size
            logger.info(f"  ✓ Batch size {current_batch_size} works")
            
            # Clear memory
            del x, y, logits, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Double and try again
            current_batch_size *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e) or "CUDA out of memory" in str(e):
                logger.info(f"  ✗ Batch size {current_batch_size} failed (OOM)")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                break
            else:
                # Some other error, re-raise
                raise
        except Exception as e:
            logger.warning(f"  Unexpected error at batch size {current_batch_size}: {e}")
            break
    
    if last_working_batch_size is None:
        logger.warning("Could not find a working batch size, using minimum: 1")
        return 1
    
    # Now binary search between last_working and current (which failed)
    lower_bound = last_working_batch_size
    upper_bound = current_batch_size
    
    logger.info("")
    logger.info(f"Phase 2: Binary search between {lower_bound} and {upper_bound}...")
    
    best_batch_size = lower_bound
    
    # Binary search
    while lower_bound < upper_bound - 1:
        test_batch_size = (lower_bound + upper_bound) // 2
        
        try:
            test_loader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False)
            x, y = next(iter(test_loader))
            x, y = x.to(device), y.to(device)
            
            with torch.no_grad():
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # It worked!
            best_batch_size = test_batch_size
            lower_bound = test_batch_size
            logger.info(f"  ✓ Batch size {test_batch_size} works")
            
            del x, y, logits, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e) or "CUDA out of memory" in str(e):
                upper_bound = test_batch_size
                logger.info(f"  ✗ Batch size {test_batch_size} failed (OOM)")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            else:
                raise
    
    logger.info("")
    logger.info(f"Maximum batch size found: {best_batch_size}")
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        logger.info(f"Peak GPU memory used: {peak_memory:.2f} GB")
    logger.info("")
    
    # Clear cache one more time
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    model.train()  # Set back to train mode
    
    return best_batch_size


# =====================
#  CHECKPOINTING FUNCTIONS
# =====================

def save_checkpoint(model, dataset, optimizer, epoch, avg_loss, model_path, 
                   dataset_folder, dataset_name, epochs, batch_size, learning_rate,
                   embed_dim, num_heads, num_layers, block_size):
    """
    Save model checkpoint with all necessary information.
    
    Args:
        model: The model to save
        dataset: The dataset (for vocab mappings)
        optimizer: The optimizer (for resuming training)
        epoch: Current epoch number
        avg_loss: Average loss for this epoch
        model_path: Path to save the checkpoint
        ... (other metadata)
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
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
        'current_epoch': epoch,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'avg_loss': avg_loss,
    }
    torch.save(save_dict, model_path)
    file_size = model_path.stat().st_size / (1024 * 1024)  # Size in MB
    return file_size


# =====================
#  TRAIN MODEL()
# =====================

def train_model(dataset_folder, epochs=10, batch_size=32, block_size=128, 
                embed_dim=128, num_heads=4, num_layers=3, learning_rate=1e-3,
                gradient_accumulation_steps=4, validation_split=0.05, 
                early_stopping_patience=5, device=None):
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
        gradient_accumulation_steps: Number of batches to accumulate gradients over
                                    before updating parameters (default: 4)
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
    
    # Initialize timing - track last timestamp for sequential timing
    last_timestamp = time.time()
    
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
    step_duration = time.time() - last_timestamp
    last_timestamp = time.time()
    logger.info(f"  - Time taken: {step_duration:.3f} seconds")
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
    step_duration = time.time() - last_timestamp
    last_timestamp = time.time()
    logger.info(f"  - Time taken: {step_duration:.3f} seconds")
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
    logger.info(f"  - Total sequences: {len(dataset):,}")
    logger.info(f"  - Total tokens: {len(dataset.data):,}")
    
    # Split dataset into train and validation
    if validation_split > 0:
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
        )
        logger.info("")
        logger.info("Dataset split:")
        logger.info(f"  - Training sequences: {train_size:,} ({100*(1-validation_split):.1f}%)")
        logger.info(f"  - Validation sequences: {val_size:,} ({100*validation_split:.1f}%)")
    else:
        train_dataset = dataset
        val_dataset = None
        logger.info("")
        logger.info("No validation split (validation_split=0)")
    
    step_duration = time.time() - last_timestamp
    last_timestamp = time.time()
    logger.info(f"  - Time taken: {step_duration:.3f} seconds")
    logger.info("")
    
    # Note: Dataloaders will be created AFTER model initialization and batch size detection
    
    # Create model
    logger.info("=" * 80)
    logger.info("STEP 4: MODEL INITIALIZATION")
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
    step_duration = time.time() - last_timestamp
    last_timestamp = time.time()
    logger.info(f"  - Time taken: {step_duration:.3f} seconds")
    logger.info("")
    
    # Auto-detect maximum batch size if batch_size is None, 0, or negative
    if batch_size is None or batch_size == 0 or batch_size < 0:
        batch_size = find_max_batch_size(model, train_dataset, device, block_size)
        logger.info(f"Using auto-detected maximum batch size: {batch_size}")
        logger.info("")
    
    # Now create dataloaders with the (possibly auto-detected) batch size
    logger.info("=" * 80)
    logger.info("STEP 5: DATALOADER SETUP")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Creating DataLoaders with batch_size={batch_size}...")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    logger.info(f"  - Training batches per epoch: {len(train_dataloader)}")
    logger.info(f"  - Samples per batch: {batch_size}")
    
    if val_dataset is not None:
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        logger.info(f"  - Validation batches: {len(val_dataloader)}")
    else:
        val_dataloader = None
    step_duration = time.time() - last_timestamp
    last_timestamp = time.time()
    logger.info(f"  - Time taken: {step_duration:.3f} seconds")
    logger.info("")
    
    # Loss and optimizer
    logger.info("=" * 80)
    logger.info("STEP 6: TRAINING SETUP")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Configuring training components:")
    logger.info("  - Loss function: CrossEntropyLoss")
    logger.info(f"  - Optimizer: AdamW (learning rate: {learning_rate})")
    logger.info(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
    if gradient_accumulation_steps > 1:
        effective_batch_size = batch_size * gradient_accumulation_steps
        logger.info(f"  - Effective batch size: {effective_batch_size} (actual: {batch_size})")
    if validation_split > 0:
        logger.info(f"  - Validation split: {100*validation_split:.1f}%")
        if early_stopping_patience is not None:
            logger.info(f"  - Early stopping patience: {early_stopping_patience} epochs")
        else:
            logger.info(f"  - Early stopping: Disabled")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    step_duration = time.time() - last_timestamp
    last_timestamp = time.time()
    logger.info(f"  - Time taken: {step_duration:.3f} seconds")
    logger.info("")
    
    # Training loop
    logger.info("=" * 80)
    logger.info("STEP 7: TRAINING LOOP")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Starting training: {epochs} epochs, {len(train_dataloader)} batches per epoch")
    logger.info("")
    
    # Record training start time
    training_start_time = last_timestamp
    
    # Early stopping setup
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    best_epoch = 0
    val_loss = None
    
    # Setup checkpoint path (save to same location, overwrite each time)
    dataset_path = Path(dataset_folder)
    dataset_name = dataset_path.name if dataset_path.is_dir() else dataset_path.stem
    base_model_dir = Path("50_models")
    model_dir = base_model_dir / dataset_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Use a consistent checkpoint filename that gets overwritten
    checkpoint_filename = f"embed{embed_dim}_layers{num_layers}_heads{num_heads}_epochs{epochs}_latest.pt"
    checkpoint_path = model_dir / checkpoint_filename
    
    # Store checkpoint metadata for signal handler
    checkpoint_metadata = {
        'dataset_folder': dataset_folder,
        'dataset_name': dataset_name,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'block_size': block_size,
    }
    
    # Track current state for signal handler
    training_state = {
        'model': model,
        'dataset': dataset,
        'optimizer': optimizer,
        'epoch': 0,
        'checkpoint_path': checkpoint_path,
        'metadata': checkpoint_metadata,
    }
    
    # Signal handler for graceful interruption
    def signal_handler(sig, frame):
        logger.info("")
        logger.warning("=" * 80)
        logger.warning("TRAINING INTERRUPTED - SAVING CHECKPOINT")
        logger.warning("=" * 80)
        logger.info("")
        try:
            state = training_state
            file_size = save_checkpoint(
                state['model'], state['dataset'], state['optimizer'], state['epoch'],
                0.0, state['checkpoint_path'], **state['metadata']
            )
            logger.info(f"Checkpoint saved successfully!")
            logger.info(f"  - File path: {state['checkpoint_path']}")
            logger.info(f"  - File size: {file_size:.2f} MB")
            logger.info(f"  - Epoch: {state['epoch']}/{state['metadata']['epochs']}")
            logger.info("")
            logger.info("You can resume training later or use this checkpoint for inference.")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}", exc_info=True)
        logger.info("")
        sys.exit(0)
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Checkpointing enabled:")
    logger.info(f"  - Checkpoint file: {checkpoint_path}")
    logger.info(f"  - Saving after each epoch")
    logger.info(f"  - Will save on interruption (Ctrl+C)")
    logger.info("")
    
    final_avg_loss = 0.0  # Initialize for final save
    for epoch in range(epochs):
        epoch_start_time = time.time()
        logger.info("")
        logger.info("-" * 80)
        logger.info(f"EPOCH {epoch+1}/{epochs}")
        logger.info("-" * 80)
        logger.info("")
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device, 
                                 epoch_num=epoch+1, gradient_accumulation_steps=gradient_accumulation_steps)
        final_avg_loss = train_loss  # Keep track of latest loss
        epoch_duration = time.time() - epoch_start_time
        
        # Evaluate on validation set if available
        val_loss = None
        if val_dataloader is not None:
            logger.info("")
            logger.info("Evaluating on validation set...")
            val_start = time.time()
            val_loss = evaluate(model, val_dataloader, criterion, device)
            val_duration = time.time() - val_start
            logger.info(f"  - Validation loss: {val_loss:.4f}")
            logger.info(f"  - Time taken: {val_duration:.3f} seconds")
        
        logger.info("")
        logger.info(f"Epoch {epoch+1}/{epochs} completed:")
        logger.info(f"  - Training loss: {train_loss:.4f}")
        if val_loss is not None:
            logger.info(f"  - Validation loss: {val_loss:.4f}")
        logger.info(f"  - Time taken: {epoch_duration:.3f} seconds")
        logger.info("")
        
        # Update training state for signal handler
        training_state['epoch'] = epoch + 1
        
        # Early stopping logic
        improved = False
        if val_loss is not None and early_stopping_patience is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                improved = True
                # Save best model state
                best_model_state = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }
                logger.info(f"  ✓ New best validation loss! (improved from {best_val_loss:.4f})")
            else:
                epochs_without_improvement += 1
                logger.info(f"  - No improvement for {epochs_without_improvement}/{early_stopping_patience} epochs")
                logger.info(f"  - Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
        
        # Save checkpoint after each epoch
        logger.info("Saving checkpoint...")
        checkpoint_start = time.time()
        file_size = save_checkpoint(
            model, dataset, optimizer, epoch + 1, train_loss,
            checkpoint_path, dataset_folder, dataset_name, epochs,
            batch_size, learning_rate, embed_dim, num_heads, num_layers, block_size
        )
        checkpoint_duration = time.time() - checkpoint_start
        logger.info(f"  - Checkpoint saved: {checkpoint_path}")
        logger.info(f"  - File size: {file_size:.2f} MB")
        logger.info(f"  - Time taken: {checkpoint_duration:.3f} seconds")
        logger.info("")
        
        # Check for early stopping
        if val_loss is not None and early_stopping_patience is not None:
            if epochs_without_improvement >= early_stopping_patience:
                logger.info("")
                logger.warning("=" * 80)
                logger.warning("EARLY STOPPING TRIGGERED")
                logger.warning("=" * 80)
                logger.warning("")
                logger.warning(f"Validation loss did not improve for {early_stopping_patience} epochs.")
                logger.warning(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
                logger.warning(f"Stopping training at epoch {epoch+1}")
                logger.warning("")
                
                # Restore best model state
                if best_model_state is not None:
                    logger.info("Restoring best model weights...")
                    model.load_state_dict(best_model_state['model_state_dict'])
                    logger.info(f"  - Restored model from epoch {best_epoch}")
                    logger.info("")
                break
        
        last_timestamp = time.time()
    
    # Calculate training duration
    training_duration = time.time() - training_start_time
    hours = int(training_duration // 3600)
    minutes = int((training_duration % 3600) // 60)
    seconds = int(training_duration % 60)
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Training duration: {hours:02d}:{minutes:02d}:{seconds:02d} ({training_duration:.2f} seconds)")
    logger.info(f"Total epochs completed: {epoch+1}/{epochs}")
    if val_loss is not None and early_stopping_patience is not None and best_model_state is not None:
        logger.info(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
        logger.info(f"Final validation loss: {val_loss:.4f}")
    logger.info("")
    
    # Final model save (same as checkpoint, but with final naming)
    logger.info("=" * 80)
    logger.info("STEP 8: FINAL MODEL SAVING")
    logger.info("=" * 80)
    logger.info("")
    
    # If we have a best model from early stopping, use that; otherwise use current model
    model_to_save = model
    epoch_to_save = epoch + 1
    loss_to_save = final_avg_loss
    
    if best_model_state is not None and val_loss is not None:
        logger.info("Saving best model (based on validation loss)...")
        logger.info(f"  - Best epoch: {best_epoch}")
        logger.info(f"  - Best validation loss: {best_val_loss:.4f}")
        # Model state already restored above
        epoch_to_save = best_epoch
        loss_to_save = best_val_loss
    else:
        logger.info("Saving final model...")
    
    # Generate final filename (without _latest suffix)
    final_filename = f"embed{embed_dim}_layers{num_layers}_heads{num_heads}_epochs{epochs}.pt"
    final_model_path = model_dir / final_filename
    
    # If file exists, add version suffix to prevent overwriting
    if final_model_path.exists():
        base_name = final_model_path.stem
        extension = final_model_path.suffix
        version = 2
        while True:
            versioned_filename = f"{base_name}_v{version}{extension}"
            final_model_path = model_dir / versioned_filename
            if not final_model_path.exists():
                logger.info(f"File {final_filename} already exists. Using versioned name: {versioned_filename}")
                break
            version += 1
    
    logger.info(f"Saving model to: {final_model_path}")
    logger.info("")
    logger.info("Saving model components:")
    logger.info("  - Model weights (state_dict)")
    logger.info("  - Vocabulary mappings (character to index)")
    logger.info("  - Model hyperparameters")
    
    # Save the model
    file_size = save_checkpoint(
        model_to_save, dataset, optimizer, epoch_to_save, loss_to_save,
        final_model_path, dataset_folder, dataset_name, epochs,
        batch_size, learning_rate, embed_dim, num_heads, num_layers, block_size
    )
    
    step_duration = time.time() - last_timestamp
    logger.info("")
    logger.info(f"Model saved successfully!")
    logger.info(f"  - File path: {final_model_path}")
    logger.info(f"  - File size: {file_size:.2f} MB")
    logger.info(f"  - Time taken: {step_duration:.3f} seconds")
    logger.info("")
    logger.info(f"Note: Latest checkpoint is also available at: {checkpoint_path}")
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
        default="40_training_data/dataset_machine_learning/AI_Generated_Dataset",
        help="Path to folder containing .txt files to train on (default: 40_training_data/dataset_toy/)"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=-1, 
                       help="Batch size (default: -1 = auto-detect maximum, 0 = auto-detect, positive = use specified)")
    parser.add_argument("--block-size", type=int, default=128, help="Sequence length")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--grad-accum-steps", type=int, default=4, 
                       help="Number of batches to accumulate gradients over before updating (default: 4)")
    parser.add_argument("--val-split", type=float, default=0.05,
                       help="Fraction of data to use for validation (default: 0.05 = 5%%)")
    parser.add_argument("--early-stopping-patience", type=int, default=5,
                       help="Number of epochs to wait for improvement before stopping (default: 5, set to 0 to disable)")
    
    args = parser.parse_args()
    
    # Convert 0 to None for early stopping (0 means disabled)
    early_stopping_patience = args.early_stopping_patience if args.early_stopping_patience > 0 else None
    
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
            learning_rate=args.lr,
            gradient_accumulation_steps=args.grad_accum_steps,
            validation_split=args.val_split,
            early_stopping_patience=early_stopping_patience
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
