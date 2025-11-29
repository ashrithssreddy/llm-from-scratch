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
# Model will be saved to: 50_models/trained_model.pt


# =====================
#  SET WORKING DIRECTORY TO GIT ROOT
# =====================

import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent / '95_utils')); __import__('path_utils').setup_workspace(__file__)


# =====================
#  IMPORTS
# =====================

import os
import argparse
import logging
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from data_loader import load_text_files_from_folder  # type: ignore


# =====================
#  LOGGING SETUP
# =====================

def setup_logging():
    """Set up aggressive logging to both console and file."""
    # Create logs directory if it doesn't exist
    logs_dir = Path("97_logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create log filename with training prefix and timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"training_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()  # Also output to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAINING SESSION STARTED")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info("")
    
    return logger

# Initialize logger
logger = setup_logging()


# =====================
#  SIMPLE CHARACTER-LEVEL LANGUAGE MODEL
# =====================

class CharDataset(Dataset):
    """Dataset for character-level language modeling."""
    
    def __init__(self, text, block_size=128):
        """
        Args:
            text: The full text as a string
            block_size: Length of sequences to use for training
        """
        logger.debug("")
        logger.debug("### FUNCTION: CharDataset.__init__ ###")
        logger.debug("")
        logger.debug(f"Input text length: {len(text)} characters")
        logger.debug(f"Block size: {block_size}")
        
        self.block_size = block_size
        
        # Get unique characters and create mappings
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        logger.debug(f"Vocabulary size: {self.vocab_size}")
        logger.debug(f"Unique characters: {''.join(chars[:50])}{'...' if len(chars) > 50 else ''}")
        
        # Encode the entire text
        logger.debug("Encoding text to integer sequences...")
        self.data = [self.stoi[ch] for ch in text]
        logger.debug(f"Encoded data length: {len(self.data)} tokens")
        
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Get a sequence of block_size characters
        x = torch.tensor(self.data[idx:idx+self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.block_size+1], dtype=torch.long)
        return x, y


class SimpleLanguageModel(nn.Module):
    """Simple transformer-based language model."""
    
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=3, block_size=128):
        super().__init__()
        logger.debug("")
        logger.debug("### FUNCTION: SimpleLanguageModel.__init__ ###")
        logger.debug("")
        logger.debug(f"Initializing model with vocab_size={vocab_size}, embed_dim={embed_dim}, "
                    f"num_heads={num_heads}, num_layers={num_layers}, block_size={block_size}")
        
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        logger.debug(f"Created token embedding: {vocab_size} x {embed_dim}")
        
        # Positional embedding
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        logger.debug(f"Created positional embedding: {block_size} x {embed_dim}")
        
        # Transformer blocks
        logger.debug(f"Creating {num_layers} transformer encoder layers...")
        self.blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        logger.debug(f"Transformer blocks created: {num_layers} layers")
        
        # Output layer
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        logger.debug(f"Output layer: Linear({embed_dim} -> {vocab_size})")
        
    def forward(self, idx):
        B, T = idx.shape
        
        # Token and positional embeddings
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        
        # Apply transformer blocks
        x = self.blocks(x)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits


# =====================
#  TRAINING FUNCTIONS
# =====================

def train_epoch(model, dataloader, optimizer, criterion, device, epoch_num=None):
    """Train for one epoch."""
    logger.debug("")
    logger.debug("### FUNCTION: train_epoch ###")
    logger.debug("")
    if epoch_num is not None:
        logger.info(f"Starting epoch {epoch_num}")
    logger.debug(f"Device: {device}")
    logger.debug(f"Number of batches: {len(dataloader)}")
    
    model.train()
    total_loss = 0
    num_batches = 0
    batch_losses = []
    
    for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc="Training")):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Log gradient norms for debugging
        if batch_idx == 0:
            total_grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
            total_grad_norm = total_grad_norm ** (1. / 2)
            logger.debug(f"Batch {batch_idx}: Gradient norm: {total_grad_norm:.4f}")
        
        optimizer.step()
        
        batch_loss = loss.item()
        total_loss += batch_loss
        batch_losses.append(batch_loss)
        num_batches += 1
        
        # Log every 10 batches
        if (batch_idx + 1) % 10 == 0:
            logger.debug(f"Batch {batch_idx + 1}/{len(dataloader)}: Loss = {batch_loss:.4f}, Avg Loss = {total_loss/num_batches:.4f}")
    
    avg_loss = total_loss / num_batches
    logger.info(f"Epoch complete: Average loss = {avg_loss:.4f}, Min batch loss = {min(batch_losses):.4f}, Max batch loss = {max(batch_losses):.4f}")
    
    return avg_loss


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
    logger.info("### FUNCTION: train_model ###")
    logger.info("")
    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Dataset folder: {dataset_folder}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Block size: {block_size}")
    logger.info(f"Embedding dimension: {embed_dim}")
    logger.info(f"Number of heads: {num_heads}")
    logger.info(f"Number of layers: {num_layers}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info("")
    
    # Set device
    logger.debug("")
    logger.debug("### STEP: Device Selection ###")
    logger.debug("")
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.info("")
    
    # Load data
    logger.info("")
    logger.info("### STEP: Data Loading ###")
    logger.info("")
    logger.info(f"Loading text files from: {dataset_folder}")
    text = load_text_files_from_folder(dataset_folder)
    logger.info(f"Loaded {len(text):,} characters of text")
    logger.info(f"Number of lines: {text.count(chr(10)) + 1}")
    logger.info("")
    
    # Create dataset
    logger.info("")
    logger.info("### STEP: Dataset Creation ###")
    logger.info("")
    logger.info("Creating dataset...")
    dataset = CharDataset(text, block_size=block_size)
    logger.info(f"Vocabulary size: {dataset.vocab_size}")
    logger.info(f"Dataset size: {len(dataset):,} sequences")
    logger.info(f"Total tokens: {len(dataset.data):,}")
    logger.info("")
    
    # Create dataloader
    logger.debug("")
    logger.debug("### STEP: DataLoader Creation ###")
    logger.debug(f"Creating DataLoader with batch_size={batch_size}, shuffle=True")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    logger.debug(f"DataLoader created: {len(dataloader)} batches")
    logger.info("")
    
    # Create model
    logger.info("")
    logger.info("### STEP: Model Initialization ###")
    logger.info("")
    logger.info("Initializing model...")
    logger.debug(f"Model architecture: vocab_size={dataset.vocab_size}, embed_dim={embed_dim}, "
                 f"num_heads={num_heads}, num_layers={num_layers}, block_size={block_size}")
    model = SimpleLanguageModel(
        vocab_size=dataset.vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        block_size=block_size
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_params:,} total parameters")
    logger.info(f"Model has {trainable_params:,} trainable parameters")
    logger.debug(f"Model device: {next(model.parameters()).device}")
    logger.info("")
    
    # Loss and optimizer
    logger.debug("")
    logger.debug("### STEP: Loss and Optimizer Setup ###")
    logger.debug("")
    logger.debug("Using CrossEntropyLoss")
    logger.debug(f"Using AdamW optimizer with lr={learning_rate}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    logger.info("")
    
    # Training loop
    logger.info("")
    logger.info("=" * 80)
    logger.info("STARTING TRAINING LOOP")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Training for {epochs} epochs...")
    logger.info("")
    
    for epoch in range(epochs):
        logger.info("")
        logger.info("")
        logger.info("-" * 80)
        logger.info("")
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device, epoch_num=epoch+1)
        logger.info(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
        logger.info("")
    
    logger.info("")
    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    
    # Save model
    logger.info("")
    logger.info("### STEP: Model Saving ###")
    logger.info("")
    model_dir = Path("50_models")
    model_dir.mkdir(exist_ok=True)
    logger.debug(f"Model directory: {model_dir}")
    
    # Save model and tokenizer info
    model_path = model_dir / "trained_model.pt"
    logger.info(f"Saving model to: {model_path}")
    save_dict = {
        'model_state_dict': model.state_dict(),
        'vocab_size': dataset.vocab_size,
        'stoi': dataset.stoi,
        'itos': dataset.itos,
        'block_size': block_size,
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'num_layers': num_layers,
    }
    torch.save(save_dict, model_path)
    file_size = model_path.stat().st_size / (1024 * 1024)  # Size in MB
    logger.info(f"Model saved successfully! File size: {file_size:.2f} MB")
    logger.info("")
    
    logger.info("")
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
    
    logger.info("")
    logger.info("### FUNCTION: main ###")
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

