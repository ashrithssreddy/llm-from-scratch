# =====================
#  DATA LOADING
# =====================

import os
from pathlib import Path
from typing import List


def load_text_files_from_folder(folder_path: str) -> str:
    """
    Load all text files from a folder (recursively) and concatenate their contents.
    Excludes folders containing "scratch_files" in their path.
    
    Args:
        folder_path: Path to the folder containing text files.
        
    Returns:
        Concatenated string containing all text from all .txt files in the folder and subfolders.
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    # Find all .txt files recursively in the folder and subfolders
    text_files = list(folder.rglob("*.txt"))
    
    # Filter out files in folders containing "scratch_files"
    text_files = [f for f in text_files if "scratch_files" not in str(f)]
    
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
    Get list of all text file paths in a folder (recursively).
    Excludes folders containing "scratch_files" in their path.
    
    Args:
        folder_path: Path to the folder containing text files.
        
    Returns:
        List of Path objects for all .txt files in the folder and subfolders.
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    # Find all .txt files recursively in the folder and subfolders
    text_files = list(folder.rglob("*.txt"))
    
    # Filter out files in folders containing "scratch_files"
    text_files = [f for f in text_files if "scratch_files" not in str(f)]
    
    return sorted(text_files)

