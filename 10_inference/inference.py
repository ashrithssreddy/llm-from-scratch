# =====================
#  CODE DESCRIPTION
# =====================
#
# This script performs text generation using a trained character-level language model.
# It loads a saved model checkpoint and generates text based on a user-provided prompt.
#
# Key Features:
#   - Loads trained model from checkpoint file (.pt)
#   - Supports both single-shot and interactive text generation
#   - Configurable temperature for controlling randomness
#   - Character-level tokenization using saved vocabulary mappings
#   - Automatic device selection (CPU/CUDA)
#   - Comprehensive logging to both console and log file
#
# The model uses a transformer-based architecture with positional embeddings,
# multiple transformer encoder layers, and generates text one character at a time
# using autoregressive sampling.
#
#
# Execution Flow (Function Call Graph):
#   __main__ block
#   │
#   ├─→ setup_logging(prefix="inference")
#   │   └─→ Creates logger and log file (inference_*.log)
#   │
#   ├─→ load_model(model_path)
#   │   ├─→ torch.load() - Load checkpoint from disk
#   │   ├─→ SimpleLanguageModel() - Initialize model architecture
#   │   ├─→ model.load_state_dict() - Load trained weights
#   │   └─→ model.eval() - Set to evaluation mode
#   │   └─→ Returns: (model, vocab_size, stoi, itos, block_size, ...)
#   │
#   └─→ Branch based on --interactive flag:
#       │
#       ├─→ IF interactive mode:
#       │   └─→ interactive_mode(model, stoi, itos, block_size, ...)
#       │       └─→ [Loop: Get user input]
#       │           └─→ generate_text(...) [called in loop]
#       │
#       └─→ ELSE single-shot mode:
#           └─→ generate_text(...) [from inference_utils]
#               ├─→ encode_text(prompt, stoi) [from inference_utils]
#               │   └─→ Converts text string to list of token indices
#               │
#               ├─→ [Loop: Generate max_tokens]
#               │   ├─→ model(input_tensor) - Forward pass through transformer
#               │   ├─→ torch.softmax() - Convert logits to probabilities
#               │   └─→ torch.multinomial() - Sample next token
#               │
#               └─→ decode_tokens(generated_tokens, itos) [from inference_utils]
#                   └─→ Converts list of token indices back to text string
#
# All core functions are in inference_utils.py:
#   - load_model() - Loads trained model from checkpoint
#   - generate_text() - Generates text using the model
#   - interactive_mode() - Runs interactive text generation loop
#   - encode_text() - Encodes text to token indices
#   - decode_tokens() - Decodes token indices to text

# =====================
#  EXAMPLE USAGE
# =====================
#
# Basic usage (generate text with default prompt):
#   python 10_inference/inference.py
#
# Generate with custom prompt:
#   python 10_inference/inference.py --prompt "The future of AI"
#
# Generate with custom parameters:
#   python 10_inference/inference.py --prompt "Hello" --max-tokens 200 --temperature 0.8
#
# Interactive mode (keep generating):
#   python 10_inference/inference.py --interactive
#
# Use specific model file:
#   python 10_inference/inference.py --model-path 50_models/trained_model.pt
#
# Available arguments:
#   --model-path - Path to trained model file (default: 50_models/trained_model.pt)
#   --prompt - Starting text prompt (default: "\n")
#   --max-tokens - Maximum number of tokens to generate (default: 500)
#   --temperature - Sampling temperature (default: 1.0, higher = more random)
#   --interactive - Enable interactive mode (default: False)
#   --seed - Random seed for reproducibility (default: None)


# =====================
#  SETUP
# =====================
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent / '95_utils')); __import__('path_utils').setup_workspace(__file__)

#  IMPORTS
import argparse
import logging
import torch

# Import inference utilities
from inference_utils import load_model, generate_text, interactive_mode, trace_forward_pass  # type: ignore

#  LOGGING SETUP
from logger_utils import setup_logging  # type: ignore
logger = setup_logging(prefix="inference")


# =====================
#  MAIN
# =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a trained language model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="50_models/trained_model.pt",
        help="Path to trained model file (default: 50_models/trained_model.pt)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Machine learning",
        help="Starting text prompt (default: newline)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum number of tokens to generate (default: 500)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0, higher = more random)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive mode"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--trace",
        type=str,
        default=None,
        help="Trace forward pass for given input text (e.g., 'ma') - shows what happens at each stage. If not specified, uses the prompt value."
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable tracing (tracing is enabled by default)"
    )
    
    args = parser.parse_args()
    
    logger.info("Command line arguments parsed")
    logger.info("")
    
    # Set random seed if provided
    if args.seed is not None:
        logger.info(f"Setting random seed: {args.seed}")
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info("")
    
    try:
        # Load model
        model, vocab_size, stoi, itos, block_size, embed_dim, num_heads, num_layers = load_model(
            args.model_path
        )
        
        device = next(model.parameters()).device
        
        # Trace forward pass by default (unless --no-trace is specified)
        if not args.no_trace:
            trace_input = args.trace if args.trace else args.prompt
            trace_forward_pass(model, trace_input, stoi, itos, device=device)
        
        # Run inference
        if args.interactive:
            interactive_mode(
                model, stoi, itos, block_size,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                device=device
            )
        else:
            generated_text = generate_text(
                model, stoi, itos, block_size,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                device=device
            )
            
            # Display result with clear separators
            print("\n" + "=" * 80)
            print("INPUT PROMPT:")
            print("=" * 80)
            print(repr(args.prompt))
            print("\n" + "=" * 80)
            print("GENERATED TEXT:")
            print("=" * 80)
            print(generated_text)
            print("=" * 80 + "\n")
            
            # Also log for record keeping
            logger.info("")
            logger.info("=" * 80)
            logger.info("GENERATED TEXT OUTPUT")
            logger.info("=" * 80)
            logger.info("")
            logger.info(f"Input prompt: {repr(args.prompt)}")
            logger.info(f"Generated text length: {len(generated_text)} characters")
            logger.info("")
            logger.info("--- FULL GENERATED TEXT ---")
            logger.info(generated_text)
            logger.info("--- END OF GENERATED TEXT ---")
            logger.info("")
            logger.info("=" * 80)
            logger.info("INFERENCE SESSION ENDED")
            logger.info("=" * 80)
            logger.info("")
            
    except Exception as e:
        logger.error("")
        logger.error("")
        logger.error("=" * 80)
        logger.error("INFERENCE FAILED")
        logger.error("=" * 80)
        logger.error("")
        logger.error(f"Error: {str(e)}", exc_info=True)
        logger.error("")
        raise

