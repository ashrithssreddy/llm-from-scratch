# =====================
#  DATA LOADING
# =====================

import os
from pathlib import Path
from typing import List


def load_text_files_from_folder(folder_path: str) -> str:
    """
    Load all text files from a folder and concatenate their contents.
    
    Args:
        folder_path: Path to the folder containing text files.
        
    Returns:
        Concatenated string containing all text from all .txt files in the folder.
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    # Find all .txt files in the folder
    text_files = list(folder.glob("*.txt"))
    
    if not text_files:
        raise ValueError(f"No .txt files found in folder: {folder_path}")
    
    # Read and concatenate all text files
    all_text = []
    for text_file in sorted(text_files):
        with open(text_file, 'r', encoding='utf-8') as f:
            content = f.read()
            all_text.append(content)
    
    return '\n'.join(all_text)


def get_text_file_paths(folder_path: str) -> List[Path]:
    """
    Get list of all text file paths in a folder.
    
    Args:
        folder_path: Path to the folder containing text files.
        
    Returns:
        List of Path objects for all .txt files in the folder.
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    text_files = list(folder.glob("*.txt"))
    return sorted(text_files)

