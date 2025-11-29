# =====================
#  TRAINING UTILITIES
# =====================

import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =====================
#  DATASET CLASS
# =====================

class CharDataset(Dataset):
    """Dataset for character-level language modeling."""
    
    def __init__(self, text, block_size=128):
        """
        Args:
            text: The full text as a string
            block_size: Length of sequences to use for training
        """
        logger.debug("Creating character-level dataset...")
        logger.debug(f"  - Input text length: {len(text)} characters")
        logger.debug(f"  - Block size: {block_size}")
        
        self.block_size = block_size
        
        # Get unique characters and create mappings
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        
        logger.debug(f"  - Vocabulary size: {self.vocab_size}")
        logger.debug(f"  - Unique characters: {''.join(chars[:50])}{'...' if len(chars) > 50 else ''}")
        
        # Encode the entire text
        logger.debug("  - Encoding text to integer sequences...")
        self.data = [self.stoi[ch] for ch in text]
        logger.debug(f"  - Encoded data length: {len(self.data)} tokens")
        
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Get a sequence of block_size characters
        x = torch.tensor(self.data[idx:idx+self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.block_size+1], dtype=torch.long)
        return x, y


# =====================
#  MODEL CLASS
# =====================

class SimpleLanguageModel(nn.Module):
    """Simple transformer-based language model."""
    
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=3, block_size=128):
        super().__init__()
        logger.debug("Initializing transformer model components...")
        logger.debug(f"  - Vocab size: {vocab_size}, Embed dim: {embed_dim}")
        logger.debug(f"  - Heads: {num_heads}, Layers: {num_layers}, Block size: {block_size}")
        
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        logger.debug(f"  - Created token embedding: {vocab_size} x {embed_dim}")
        
        # Positional embedding
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        logger.debug(f"  - Created positional embedding: {block_size} x {embed_dim}")
        
        # Transformer blocks
        logger.debug(f"  - Creating {num_layers} transformer encoder layers...")
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
        logger.debug(f"  - Transformer blocks created: {num_layers} layers")
        
        # Output layer
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        logger.debug(f"  - Output layer: Linear({embed_dim} -> {vocab_size})")
        
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
    if epoch_num is not None:
        logger.info(f"Processing {len(dataloader)} batches...")
    
    model.train()
    total_loss = 0
    num_batches = 0
    batch_losses = []
    
    for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch_num}")):
        # Move data to device
        x, y = x.to(device), y.to(device)
        
        # Forward pass: predict next character for each position
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # Backward pass: compute gradients
        optimizer.zero_grad()
        loss.backward()
        
        # Log gradient norms for first batch (to monitor training health)
        if batch_idx == 0:
            total_grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
            total_grad_norm = total_grad_norm ** (1. / 2)
            logger.debug(f"  First batch gradient norm: {total_grad_norm:.4f}")
        
        # Update model parameters
        optimizer.step()
        
        batch_loss = loss.item()
        total_loss += batch_loss
        batch_losses.append(batch_loss)
        num_batches += 1
        
        # Log progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            logger.debug(f"  Batch {batch_idx + 1}/{len(dataloader)}: Current loss = {batch_loss:.4f}, Running average = {total_loss/num_batches:.4f}")
    
    avg_loss = total_loss / num_batches
    logger.info(f"Epoch statistics:")
    logger.info(f"  - Average loss: {avg_loss:.4f}")
    logger.info(f"  - Min batch loss: {min(batch_losses):.4f}")
    logger.info(f"  - Max batch loss: {max(batch_losses):.4f}")
    
    return avg_loss

