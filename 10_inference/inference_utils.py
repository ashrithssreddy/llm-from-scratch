# =====================
#  INFERENCE UTILITIES
# =====================
#
# Core inference functions for loading models and generating text.
# This module contains all the main inference logic including model loading,
# text generation, and interactive mode functionality.
#

import logging
import torch
from pathlib import Path
import sys

# Add 01_trainers to path for train_utils import
sys.path.insert(0, str(Path(__file__).parent.parent / '01_trainers'))
from train_utils import SimpleLanguageModel  # type: ignore

# Create logger for utility functions
# Note: This will inherit the root logger configuration set up in inference.py
logger = logging.getLogger(__name__)


# =====================
#  TEXT ENCODING/DECODING
# =====================

def encode_text(text, stoi):
    """
    Encode text string to list of token indices.
    
    Args:
        text: Input text string
        stoi: String-to-index mapping dictionary
        
    Returns:
        List of token indices
    """
    # Handle unknown characters by skipping them or using a default
    encoded = []
    for char in text:
        if char in stoi:
            encoded.append(stoi[char])
        else:
            # Skip unknown characters or use first character in vocab as fallback
            logger.warning(f"Unknown character '{char}' (ord={ord(char)}) - skipping")
    return encoded


def decode_tokens(tokens, itos):
    """
    Decode list of token indices to text string.
    
    Args:
        tokens: List of token indices
        itos: Index-to-string mapping dictionary
        
    Returns:
        Decoded text string
    """
    return ''.join([itos.get(token, '') for token in tokens])


# =====================
#  MODEL LOADING
# =====================

def load_model(model_path, device=None):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model file (.pt)
        device: Device to load model on (cuda/cpu)
        
    Returns:
        Tuple of (model, vocab_size, stoi, itos, block_size, embed_dim, num_heads, num_layers)
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("MODEL LOADING")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Loading model from: {model_path}")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Loading model on device: {device}")
    logger.info("")
    
    # Load checkpoint
    logger.info("Reading model checkpoint...")
    checkpoint = torch.load(model_path, map_location=device)
    logger.info("Checkpoint loaded successfully")
    logger.info("")
    
    # Extract model parameters
    vocab_size = checkpoint['vocab_size']
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    block_size = checkpoint['block_size']
    embed_dim = checkpoint['embed_dim']
    num_heads = checkpoint['num_heads']
    num_layers = checkpoint['num_layers']
    
    logger.info("Model configuration:")
    logger.info(f"  - Vocabulary size: {vocab_size}")
    logger.info(f"  - Block size (context window): {block_size}")
    logger.info(f"  - Embedding dimension: {embed_dim}")
    logger.info(f"  - Attention heads: {num_heads}")
    logger.info(f"  - Transformer layers: {num_layers}")
    logger.info("")
    
    # Create model
    logger.info("Initializing model architecture...")
    model = SimpleLanguageModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        block_size=block_size
    ).to(device)
    
    # Load weights
    logger.info("Loading model weights...")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    logger.info("Model weights loaded successfully")
    logger.info("")
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded:")
    logger.info(f"  - Total parameters: {num_params:,}")
    logger.info(f"  - Model location: {next(model.parameters()).device}")
    logger.info("")
    
    logger.info("=" * 80)
    logger.info("MODEL LOADING COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    
    return model, vocab_size, stoi, itos, block_size, embed_dim, num_heads, num_layers


# =====================
#  TEXT GENERATION
# =====================

def generate_text(model, stoi, itos, block_size, prompt="\n", max_tokens=500, 
                  temperature=1.0, device=None):
    """
    Generate text using the trained model.
    
    Args:
        model: Trained language model
        stoi: String-to-index mapping
        itos: Index-to-string mapping
        block_size: Context window size
        prompt: Starting text prompt
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        device: Device to run inference on
        
    Returns:
        Generated text string
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEXT GENERATION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Generation parameters:")
    logger.info(f"  - Prompt: {repr(prompt)}")
    logger.info(f"  - Max tokens: {max_tokens}")
    logger.info(f"  - Temperature: {temperature}")
    logger.info("")
    
    if device is None:
        device = next(model.parameters()).device
    
    # Encode prompt
    logger.info("Encoding prompt...")
    prompt_tokens = encode_text(prompt, stoi)
    if len(prompt_tokens) == 0:
        logger.warning("Prompt encoded to empty sequence, using newline character")
        prompt_tokens = [stoi.get('\n', 0)]
    
    logger.info(f"  - Prompt length: {len(prompt)} characters")
    logger.info(f"  - Encoded tokens: {len(prompt_tokens)}")
    logger.info("")
    
    # Initialize context
    context = prompt_tokens.copy()
    
    # Truncate if longer than block_size
    if len(context) > block_size:
        logger.warning(f"Prompt is longer than block_size ({block_size}), truncating to last {block_size} tokens")
        context = context[-block_size:]
    
    logger.info("Starting generation...")
    logger.info("")
    
    generated_tokens = []
    
    with torch.no_grad():
        for step in range(max_tokens):
            # Prepare input tensor
            # Take last block_size tokens as context
            input_tokens = context[-block_size:] if len(context) >= block_size else context
            input_tensor = torch.tensor([input_tokens], dtype=torch.long, device=device)
            
            # Forward pass
            logits = model(input_tensor)  # Shape: [1, seq_len, vocab_size]
            
            # Get logits for the last position
            logits = logits[0, -1, :] / temperature  # Shape: [vocab_size]
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Add to context and generated tokens
            context.append(next_token)
            generated_tokens.append(next_token)
            
            # Log progress every 50 tokens
            if (step + 1) % 50 == 0:
                logger.debug(f"  Generated {step + 1}/{max_tokens} tokens...")
    
    logger.info(f"Generation complete: {len(generated_tokens)} tokens generated")
    logger.info("")
    
    # Decode generated text
    logger.info("Decoding generated tokens...")
    generated_text = decode_tokens(generated_tokens, itos)
    logger.info("")
    
    # Combine prompt and generated text
    full_text = prompt + generated_text
    
    logger.info("=" * 80)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    
    return full_text


# =====================
#  INTERACTIVE MODE
# =====================

def interactive_mode(model, stoi, itos, block_size, max_tokens=500, 
                    temperature=1.0, device=None):
    """
    Run interactive text generation mode.
    
    Args:
        model: Trained language model
        stoi: String-to-index mapping
        itos: Index-to-string mapping
        block_size: Context window size
        max_tokens: Maximum number of tokens to generate per turn
        temperature: Sampling temperature
        device: Device to run inference on
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("INTERACTIVE MODE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Enter prompts to generate text. Type 'quit' or 'exit' to stop.")
    logger.info("Type 'clear' to reset the context.")
    logger.info("")
    
    context_history = []
    
    while True:
        try:
            # Get user input
            prompt = input("\nPrompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                logger.info("Exiting interactive mode...")
                break
            
            if prompt.lower() == 'clear':
                context_history = []
                logger.info("Context cleared.")
                continue
            
            if not prompt:
                prompt = "\n"
            
            # Use context history if available
            if context_history:
                full_prompt = ''.join(context_history) + prompt
            else:
                full_prompt = prompt
            
            # Generate text
            generated = generate_text(
                model, stoi, itos, block_size,
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                device=device
            )
            
            # Extract only the newly generated part
            new_text = generated[len(full_prompt):]
            
            # Display result with clear separators
            print("\n" + "=" * 80)
            print("GENERATED TEXT:")
            print("=" * 80)
            print(new_text)
            print("=" * 80 + "\n")
            
            # Log the generated text
            logger.info("")
            logger.info("=" * 80)
            logger.info("INTERACTIVE MODE - GENERATED TEXT")
            logger.info("=" * 80)
            logger.info("")
            logger.info(f"User prompt: {repr(prompt)}")
            logger.info(f"Generated text length: {len(new_text)} characters")
            logger.info("")
            logger.info("--- FULL GENERATED TEXT ---")
            logger.info(new_text)
            logger.info("--- END OF GENERATED TEXT ---")
            logger.info("")
            
            # Update context history (keep last block_size characters)
            context_history.append(prompt + new_text)
            total_context = ''.join(context_history)
            if len(total_context) > block_size * 2:
                # Keep last block_size * 2 characters
                context_history = [total_context[-block_size * 2:]]
            
        except KeyboardInterrupt:
            logger.info("\n\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            logger.error("", exc_info=True)

