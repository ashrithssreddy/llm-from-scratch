"""
Model Analysis Script

This script is used to analyze and understand the trained model (.pt file).
It will help explore the model architecture, parameters, and characteristics.

Usage:
    python 01_trainers/train_model_analyze.py 50_models/dataset_toy/embed128_layers3_heads4_epochs10.pt
"""

# =====================
#  SETUP
# =====================

# Set working directory to git root
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent / '95_utils')); __import__('path_utils').setup_workspace(__file__)

# Imports
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict

from train_utils import SimpleLanguageModel  # type: ignore
from logger_utils import setup_logging  # type: ignore

# Logging setup
logger = setup_logging()


# =====================
#  ANALYSIS FUNCTIONS
# =====================

def format_number(num):
    """Format large numbers with commas."""
    return f"{num:,}"


def format_size(size_bytes):
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def analyze_checkpoint(checkpoint):
    """Analyze the checkpoint dictionary and extract all metadata."""
    logger.info("=" * 80)
    logger.info("CHECKPOINT METADATA")
    logger.info("=" * 80)
    logger.info("")
    
    logger.info("Saved Keys in Checkpoint:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            size = len(checkpoint[key])
            logger.info(f"  - {key}: dictionary with {size} entries")
        elif isinstance(checkpoint[key], (int, float)):
            logger.info(f"  - {key}: {checkpoint[key]}")
        elif isinstance(checkpoint[key], str):
            logger.info(f"  - {key}: {checkpoint[key]}")
        elif isinstance(checkpoint[key], torch.Tensor):
            logger.info(f"  - {key}: tensor with shape {list(checkpoint[key].shape)}")
        else:
            logger.info(f"  - {key}: {type(checkpoint[key]).__name__}")
    logger.info("")
    
    # Training configuration
    logger.info("Training Configuration:")
    if 'dataset_name' in checkpoint:
        logger.info(f"  - Dataset: {checkpoint['dataset_name']}")
    if 'dataset_folder' in checkpoint:
        logger.info(f"  - Dataset folder: {checkpoint['dataset_folder']}")
    if 'epochs' in checkpoint:
        logger.info(f"  - Training epochs: {checkpoint['epochs']}")
    if 'batch_size' in checkpoint:
        logger.info(f"  - Batch size: {checkpoint['batch_size']}")
    if 'learning_rate' in checkpoint:
        logger.info(f"  - Learning rate: {checkpoint['learning_rate']}")
    logger.info("")
    
    # Model hyperparameters
    logger.info("Model Hyperparameters:")
    if 'vocab_size' in checkpoint:
        logger.info(f"  - Vocabulary size: {format_number(checkpoint['vocab_size'])} unique characters")
    if 'block_size' in checkpoint:
        logger.info(f"  - Block size (context window): {checkpoint['block_size']} tokens")
    if 'embed_dim' in checkpoint:
        logger.info(f"  - Embedding dimension: {checkpoint['embed_dim']}")
    if 'num_heads' in checkpoint:
        logger.info(f"  - Attention heads: {checkpoint['num_heads']}")
    if 'num_layers' in checkpoint:
        logger.info(f"  - Transformer layers: {checkpoint['num_layers']}")
    logger.info("")
    
    return checkpoint


def analyze_vocabulary(stoi, itos):
    """Analyze the vocabulary mappings."""
    logger.info("=" * 80)
    logger.info("VOCABULARY ANALYSIS")
    logger.info("=" * 80)
    logger.info("")
    
    vocab_size = len(stoi)
    logger.info(f"Vocabulary size: {format_number(vocab_size)} unique characters")
    logger.info("")
    
    # Show all characters in vocabulary
    logger.info("All characters in vocabulary:")
    chars = sorted(stoi.keys())
    
    # Group characters by type for better readability
    printable_chars = [ch for ch in chars if ch.isprintable() and not ch.isspace()]
    whitespace_chars = [ch for ch in chars if ch.isspace()]
    control_chars = [ch for ch in chars if not ch.isprintable()]
    
    if printable_chars:
        logger.info(f"  - Printable characters ({len(printable_chars)}): {''.join(printable_chars)}")
    if whitespace_chars:
        logger.info(f"  - Whitespace characters ({len(whitespace_chars)}): {repr(''.join(whitespace_chars))}")
    if control_chars:
        logger.info(f"  - Control characters ({len(control_chars)}): {[repr(ch) for ch in control_chars]}")
    logger.info("")
    
    # Show first 20 character-to-index mappings
    logger.info("Sample character-to-index mappings (first 20):")
    for i, (char, idx) in enumerate(sorted(stoi.items(), key=lambda x: x[1])[:20]):
        char_repr = repr(char) if not char.isprintable() or char.isspace() else char
        logger.info(f"  - '{char_repr}' -> {idx}")
    if len(stoi) > 20:
        logger.info(f"  ... and {len(stoi) - 20} more")
    logger.info("")
    
    return vocab_size


def analyze_model_architecture(model):
    """Analyze the model architecture and structure."""
    logger.info("=" * 80)
    logger.info("MODEL ARCHITECTURE")
    logger.info("=" * 80)
    logger.info("")
    
    logger.info("Model Structure:")
    logger.info("")
    
    # Print model structure
    model_str = str(model)
    for line in model_str.split('\n'):
        logger.info(f"  {line}")
    logger.info("")
    
    # Count parameters by layer
    logger.info("=" * 80)
    logger.info("PARAMETER BREAKDOWN BY LAYER")
    logger.info("=" * 80)
    logger.info("")
    
    total_params = 0
    trainable_params = 0
    
    logger.info(f"{'Layer Name':<50} {'Parameters':<15} {'Shape':<30}")
    logger.info("-" * 95)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        
        shape_str = str(list(param.shape))
        logger.info(f"{name:<50} {format_number(param_count):<15} {shape_str:<30}")
    
    logger.info("-" * 95)
    logger.info(f"{'TOTAL':<50} {format_number(total_params):<15}")
    logger.info(f"{'TRAINABLE':<50} {format_number(trainable_params):<15}")
    logger.info("")
    
    return total_params, trainable_params


def analyze_weights(model):
    """Analyze weight statistics for each layer."""
    logger.info("=" * 80)
    logger.info("WEIGHT STATISTICS")
    logger.info("=" * 80)
    logger.info("")
    
    logger.info("Weight statistics for each layer:")
    logger.info("")
    logger.info(f"{'Layer Name':<50} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    logger.info("-" * 98)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            weights = param.data.cpu().float()
            mean_val = weights.mean().item()
            std_val = weights.std().item()
            min_val = weights.min().item()
            max_val = weights.max().item()
            
            logger.info(f"{name:<50} {mean_val:>11.6f} {std_val:>11.6f} {min_val:>11.6f} {max_val:>11.6f}")
    
    logger.info("-" * 98)
    logger.info("")
    
    # Overall statistics
    all_weights = torch.cat([p.data.cpu().flatten().float() for p in model.parameters() if p.requires_grad])
    logger.info("Overall Weight Statistics (all parameters combined):")
    logger.info(f"  - Mean: {all_weights.mean().item():.6f}")
    logger.info(f"  - Standard deviation: {all_weights.std().item():.6f}")
    logger.info(f"  - Minimum: {all_weights.min().item():.6f}")
    logger.info(f"  - Maximum: {all_weights.max().item():.6f}")
    logger.info("")


def analyze_model_size(model_path, total_params):
    """Analyze the model file size and memory requirements."""
    logger.info("=" * 80)
    logger.info("MODEL SIZE ANALYSIS")
    logger.info("=" * 80)
    logger.info("")
    
    # File size
    file_size = Path(model_path).stat().st_size
    logger.info(f"Model file size: {format_size(file_size)}")
    logger.info("")
    
    # Memory requirements
    # Each float32 parameter is 4 bytes
    # Each float16 parameter is 2 bytes (if using half precision)
    float32_size = total_params * 4
    float16_size = total_params * 2
    
    logger.info("Memory requirements (approximate):")
    logger.info(f"  - Float32 (full precision): {format_size(float32_size)}")
    logger.info(f"  - Float16 (half precision): {format_size(float16_size)}")
    logger.info("")
    
    # Model size comparison
    logger.info("Model size comparison:")
    if total_params < 1_000_000:
        logger.info(f"  - This is a small model ({format_number(total_params)} parameters)")
    elif total_params < 100_000_000:
        logger.info(f"  - This is a medium-sized model ({format_number(total_params)} parameters)")
    else:
        logger.info(f"  - This is a large model ({format_number(total_params)} parameters)")
    logger.info("")


# =====================
#  MAIN ANALYSIS FUNCTION
# =====================

def analyze_model(model_path, device=None):
    """
    Comprehensive analysis of a trained model.
    
    Args:
        model_path: Path to the saved model file (.pt)
        device: Device to load model on (cuda/cpu)
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("MODEL ANALYSIS")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Analyzing model: {model_path}")
    logger.info("")
    
    # Check if file exists
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info("")
    
    # Load checkpoint
    logger.info("Loading checkpoint...")
    checkpoint = torch.load(model_path, map_location=device)
    logger.info("Checkpoint loaded successfully")
    logger.info("")
    
    # Analyze checkpoint metadata
    analyze_checkpoint(checkpoint)
    
    # Analyze vocabulary
    if 'stoi' in checkpoint and 'itos' in checkpoint:
        analyze_vocabulary(checkpoint['stoi'], checkpoint['itos'])
    
    # Create model and load weights
    logger.info("=" * 80)
    logger.info("LOADING MODEL")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Initializing model architecture...")
    model = SimpleLanguageModel(
        vocab_size=checkpoint['vocab_size'],
        embed_dim=checkpoint['embed_dim'],
        num_heads=checkpoint['num_heads'],
        num_layers=checkpoint['num_layers'],
        block_size=checkpoint['block_size']
    ).to(device)
    
    logger.info("Loading model weights...")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    logger.info("Model loaded successfully")
    logger.info("")
    
    # Analyze model architecture
    total_params, trainable_params = analyze_model_architecture(model)
    
    # Analyze weights
    analyze_weights(model)
    
    # Analyze model size
    analyze_model_size(model_path, total_params)
    
    # Summary
    logger.info("=" * 80)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Key Information:")
    logger.info(f"  - Model type: Transformer-based language model")
    logger.info(f"  - Architecture: {checkpoint['num_layers']} transformer layers")
    logger.info(f"  - Total parameters: {format_number(total_params)}")
    logger.info(f"  - Vocabulary size: {format_number(checkpoint['vocab_size'])}")
    logger.info(f"  - Context window: {checkpoint['block_size']} tokens")
    logger.info(f"  - Embedding dimension: {checkpoint['embed_dim']}")
    logger.info(f"  - Attention heads: {checkpoint['num_heads']}")
    if 'epochs' in checkpoint:
        logger.info(f"  - Training epochs: {checkpoint['epochs']}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    
    return model, checkpoint


# =====================
#  MAIN
# =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a trained language model")
    parser.add_argument(
        "model_path",
        type=str,
        nargs='?',
        default="50_models/dataset_toy/embed128_layers3_heads4_epochs10.pt",
        help="Path to the saved model file (.pt) (default: 50_models/dataset_toy/embed128_layers3_heads4_epochs10.pt)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to load model on (cuda/cpu). Default: auto-detect"
    )
    
    args = parser.parse_args()
    
    # Convert device string to torch.device if provided
    device = None
    if args.device:
        device = torch.device(args.device)
    
    try:
        analyze_model(args.model_path, device=device)
    except Exception as e:
        logger.error("")
        logger.error("=" * 80)
        logger.error("ANALYSIS FAILED")
        logger.error("=" * 80)
        logger.error("")
        logger.error(f"Error: {str(e)}", exc_info=True)
        logger.error("")
        raise
