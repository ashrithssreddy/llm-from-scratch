# =====================
#  INFERENCE UTILITIES
# =====================
#
# Helper functions for text encoding/decoding during inference.
# These utilities handle the conversion between text strings and token indices
# using the vocabulary mappings (stoi/itos) from the trained model.
#

import logging

# Create logger for utility functions
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

