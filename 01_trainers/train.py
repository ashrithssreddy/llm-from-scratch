# =====================
#  SET WORKING DIRECTORY TO GIT ROOT
# =====================

import sys
from pathlib import Path

# Add utils to path before importing
utils_path = Path(__file__).parent.parent / '95_utils'
sys.path.insert(0, str(utils_path))

from path_utils import set_working_directory_to_git_root  # type: ignore

# Set working directory to git root
git_root = set_working_directory_to_git_root()
print(f"Working directory set to git root: {git_root}")

# Re-add utils to path after changing directory (in case relative paths changed)
sys.path.insert(0, str(Path.cwd() / '95_utils'))


# =====================
#  IMPORTS
# =====================

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from data_loader import load_text_files_from_folder  # type: ignore


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
        self.block_size = block_size
        
        # Get unique characters and create mappings
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        # Encode the entire text
        self.data = [self.stoi[ch] for ch in text]
        
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
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional embedding
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        
        # Transformer blocks
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
        
        # Output layer
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        
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

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for x, y in tqdm(dataloader, desc="Training"):
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


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
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading text files from: {dataset_folder}")
    text = load_text_files_from_folder(dataset_folder)
    print(f"Loaded {len(text)} characters of text")
    
    # Create dataset
    print("Creating dataset...")
    dataset = CharDataset(text, block_size=block_size)
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Dataset size: {len(dataset)} sequences")
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    print("Initializing model...")
    model = SimpleLanguageModel(
        vocab_size=dataset.vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        block_size=block_size
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    for epoch in range(epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
    
    # Save model
    model_dir = Path("50_models")
    model_dir.mkdir(exist_ok=True)
    
    # Save model and tokenizer info
    model_path = model_dir / "trained_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': dataset.vocab_size,
        'stoi': dataset.stoi,
        'itos': dataset.itos,
        'block_size': block_size,
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'num_layers': num_layers,
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    return model, dataset


# =====================
#  MAIN
# =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a language model on text files")
    parser.add_argument(
        "dataset_folder",
        type=str,
        help="Path to folder containing .txt files to train on"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--block-size", type=int, default=128, help="Sequence length")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    args = parser.parse_args()
    
    # Train the model
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

