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
import os
import subprocess
import sys
import torch
import torch.nn as nn
from collections import OrderedDict

from train_utils import SimpleLanguageModel  # type: ignore
from logger_utils import setup_logging  # type: ignore

# Logging setup
logger = setup_logging(prefix="analyze")

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available - cannot create visualizations")


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


def visualize_neural_network(model, checkpoint, output_path=None):
    """Create a visual diagram of the neural network architecture with nodes and connections."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available - skipping visualization")
        logger.info("Install matplotlib to generate visualizations: pip install matplotlib")
        return None
    
    # Suppress matplotlib debug logging
    import logging
    matplotlib_logger = logging.getLogger('matplotlib')
    matplotlib_logger.setLevel(logging.WARNING)
    matplotlib_font_logger = logging.getLogger('matplotlib.font_manager')
    matplotlib_font_logger.setLevel(logging.WARNING)
    
    # Temporarily raise logging level to INFO to skip debug logs during visualization
    original_level = logger.level
    logger.setLevel(logging.INFO)
    
    # Set output path to logs folder if not specified
    if output_path is None:
        import datetime
        logs_dir = Path("97_logs") / "analyze_model"
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = logs_dir / f"model_architecture_{timestamp}.png"
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("CREATING NEURAL NETWORK VISUALIZATION")
    logger.info("=" * 80)
    logger.info("")
    
    vocab_size = checkpoint['vocab_size']
    embed_dim = checkpoint['embed_dim']
    num_heads = checkpoint['num_heads']
    num_layers = checkpoint['num_layers']
    block_size = checkpoint['block_size']
    
    # Create figure with white background - 2x larger for good resolution without huge file size
    fig, ax = plt.subplots(1, 1, figsize=(32, 20))  # 2x larger: 16*2=32, 10*2=20
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    node_color = '#FFA500'  # Orange nodes
    connection_color = '#666666'  # Gray connections
    text_color = 'black'
    
    # Layer positions (x coordinates)
    layer_x_positions = []
    x_start = 1.5
    x_spacing = 1.5
    
    # Get actual characters from vocabulary
    chars_list = []
    if 'stoi' in checkpoint:
        chars_list = sorted(checkpoint['stoi'].keys())
    
    # For input/output layers, show ALL nodes (not just a sample)
    # For embedding/transformer layers, also show ALL nodes
    def get_num_nodes_to_show(actual_size, show_all=False):
        """Get number of nodes to show"""
        if show_all:
            return actual_size
        return actual_size  # Show all nodes for all layers
    
    # Input layer - show ALL nodes
    input_nodes = get_num_nodes_to_show(vocab_size, show_all=True)
    layer_x_positions.append(x_start)
    x_start += x_spacing
    
    # Embedding layer - show ALL 128 nodes
    embed_nodes = get_num_nodes_to_show(embed_dim, show_all=True)
    layer_x_positions.append(x_start)
    x_start += x_spacing
    
    # Transformer layers (each as a group)
    for i in range(num_layers):
        layer_x_positions.append(x_start)
        x_start += x_spacing * 0.8
    
    # Output layer - show ALL nodes
    output_nodes = get_num_nodes_to_show(vocab_size, show_all=True)
    layer_x_positions.append(x_start)
    
    # Function to draw a layer of nodes
    def draw_layer(x, num_nodes, layer_name, actual_size, y_center=5, characters=None, dimension_labels=None):
        """Draw a vertical column of nodes"""
        if num_nodes == 0:
            return []
        
        # Adjust node size and spacing based on number of nodes
        if num_nodes > 100:
            node_radius = 0.05
            node_spacing = 0.08
            label_fontsize = 3
        elif num_nodes > 50:
            node_radius = 0.06
            node_spacing = 0.1
            label_fontsize = 3.5
        elif num_nodes > 30:
            node_radius = 0.08
            node_spacing = 0.15
            label_fontsize = 4
        elif num_nodes > 15:
            node_radius = 0.1
            node_spacing = 0.2
            label_fontsize = 5
        else:
            node_radius = 0.15
            node_spacing = 0.4
            label_fontsize = 6
        
        total_height = (num_nodes - 1) * node_spacing
        y_start = y_center - total_height / 2
        
        nodes = []
        for i in range(num_nodes):
            y = y_start + i * node_spacing
            circle = plt.Circle((x, y), node_radius, color=node_color, 
                              edgecolor='black', linewidth=1, zorder=3)
            ax.add_patch(circle)
            nodes.append((x, y))
            
            # Label each node with its character if provided (for input/output layers)
            if characters and i < len(characters):
                char = characters[i]
                # Use repr for non-printable characters
                char_display = repr(char) if not char.isprintable() or char.isspace() else char
                ax.text(x, y, char_display, ha='center', va='center', 
                       fontsize=label_fontsize, color='black', weight='bold', zorder=4)
            
            # Label each node with dimension number if provided (for embedding/transformer layers)
            if dimension_labels is not None and i < len(dimension_labels):
                dim_label = dimension_labels[i]
                ax.text(x, y, dim_label, ha='center', va='center', 
                       fontsize=label_fontsize, color='black', weight='bold', zorder=4)
        
        # Layer label
        ax.text(x, y_center + total_height/2 + 0.5, layer_name, 
               ha='center', va='bottom', fontsize=9, color=text_color, weight='bold')
        
        # Size label - show actual number of nodes
        size_text = f'{actual_size} nodes'
        ax.text(x, y_center - total_height/2 - 0.3, size_text,
               ha='center', va='top', fontsize=7, color=text_color, style='italic')
        
        # Add dimension range label for embedding/transformer layers
        if dimension_labels is not None:
            # Show range of dimensions (all are shown, so just indicate the range)
            dim_range_text = f'dim 0-{actual_size-1}'
            ax.text(x, y_center - total_height/2 - 0.5, dim_range_text,
                   ha='center', va='top', fontsize=6, color=text_color, style='italic')
        
        return nodes
    
    # Function to draw connections between layers
    def draw_connections(from_nodes, to_nodes, alpha=0.3):
        """Draw connections between two layers - draw ALL connections"""
        # Draw ALL connections, not just a sample
        # Reduce alpha and linewidth if many connections to avoid visual clutter
        num_connections = len(from_nodes) * len(to_nodes)
        if num_connections > 1000:
            alpha = 0.05
            linewidth = 0.1
        elif num_connections > 500:
            alpha = 0.08
            linewidth = 0.15
        elif num_connections > 100:
            alpha = 0.1
            linewidth = 0.2
        else:
            linewidth = 0.3
        
        # Draw all connections
        for fx, fy in from_nodes:
            for tx, ty in to_nodes:
                ax.plot([fx, tx], [fy, ty], color=connection_color, 
                       linewidth=linewidth, alpha=alpha, zorder=1)
    
    # Draw layers
    layer_idx = 0
    
    # Input layer - show ALL 48 nodes with character labels
    input_layer_nodes = draw_layer(layer_x_positions[layer_idx], input_nodes, 
                                   'Input', vocab_size, y_center=5,
                                   characters=chars_list if chars_list else None)
    layer_idx += 1
    
    # Embedding layer - create dimension labels for ALL 128 dimensions (d0, d1, ..., d127)
    embed_dim_labels = [f'd{i}' for i in range(embed_dim)]  # Labels for all 128 nodes
    embed_layer_nodes = draw_layer(layer_x_positions[layer_idx], embed_nodes,
                                   'Embedding', embed_dim, y_center=5,
                                   dimension_labels=embed_dim_labels)
    draw_connections(input_layer_nodes, embed_layer_nodes, alpha=0.2)
    layer_idx += 1
    
    # Transformer layers - show ALL 128 nodes with dimension labels
    prev_layer_nodes = embed_layer_nodes
    for i in range(num_layers):
        # Show ALL nodes for transformer layer
        transformer_nodes = get_num_nodes_to_show(embed_dim, show_all=True)
        transformer_dim_labels = [f'd{j}' for j in range(transformer_nodes)]  # Labels for all nodes d0-d127
        transformer_layer_nodes = draw_layer(layer_x_positions[layer_idx], transformer_nodes,
                                            f'Transformer\nLayer {i+1}', embed_dim, y_center=5,
                                            dimension_labels=transformer_dim_labels)
        draw_connections(prev_layer_nodes, transformer_layer_nodes, alpha=0.2)
        prev_layer_nodes = transformer_layer_nodes
        layer_idx += 1
    
    # Output layer - show ALL 48 nodes with character labels
    output_layer_nodes = draw_layer(layer_x_positions[layer_idx], output_nodes,
                                    'Output', vocab_size, y_center=5,
                                    characters=chars_list if chars_list else None)
    draw_connections(prev_layer_nodes, output_layer_nodes, alpha=0.2)
    
    # Title - at the very top left
    ax.text(0.2, 9.95, 'Neural Network Architecture', 
            ha='left', va='top', fontsize=18, weight='bold', color=text_color)
    
    # Info box - positioned right after title at the very top
    total_params = sum(p.numel() for p in model.parameters())
    total_layers = 2 + num_layers + 1  # Input + Embedding + Transformer layers + Output
    info_text = f'Total Parameters: {format_number(total_params)}\n'
    info_text += f'Total Layers: {total_layers} (Input + Embedding + {num_layers} Transformer + Output)\n'
    info_text += f'Transformer Layers: {num_layers} | Attention Heads: {num_heads} | Embed Dim: {embed_dim}'
    ax.text(0.2, 9.15, info_text, ha='left', va='top', 
            fontsize=10, color=text_color,
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='black', alpha=0.9))
    
    plt.tight_layout()
    # Save at good resolution (2x DPI) - balance between quality and file size
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    # Restore original logging level
    logger.setLevel(original_level)
    
    logger.info(f"Visualization saved to: {output_path}")
    logger.info(f"  - Image size: 32x20 inches at 300 DPI (high resolution)")
    logger.info(f"  - All {vocab_size} input/output nodes displayed")
    logger.info(f"  - All {embed_dim} embedding/transformer nodes displayed")
    logger.info(f"  - All connections between layers drawn")
    logger.info("")
    
    return output_path


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
    # Temporarily suppress debug logs during model initialization
    import logging
    original_level = logger.level
    logger.setLevel(logging.INFO)
    
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
    
    # Restore original logging level
    logger.setLevel(original_level)
    
    logger.info("Model loaded successfully")
    logger.info("")
    
    # Create visualization (saves to logs folder)
    viz_path = visualize_neural_network(model, checkpoint)
    
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
    
    # Open generated files automatically
    try:
        import logging
        # Get the log file path from the logger
        log_file = None
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file = handler.baseFilename
                break
        
        # Open log file
        if log_file and Path(log_file).exists():
            logger.info(f"Opening log file: {log_file}")
            if sys.platform == 'win32':
                os.startfile(log_file)
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['open', log_file])
            else:  # Linux
                subprocess.run(['xdg-open', log_file])
        
        # Open visualization image
        if viz_path and Path(viz_path).exists():
            logger.info(f"Opening visualization: {viz_path}")
            if sys.platform == 'win32':
                os.startfile(viz_path)
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['open', viz_path])
            else:  # Linux
                subprocess.run(['xdg-open', viz_path])
    except Exception as e:
        logger.warning(f"Could not open files automatically: {e}")
    
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
