# =====================
#  PATH UTILITIES
# =====================

import os
from pathlib import Path


def find_git_root(start_path=None):
    """
    Find the git root directory by crawling up from the current or specified path.
    
    Args:
        start_path: Starting directory path. If None, uses current working directory.
        
    Returns:
        Path object pointing to the git root directory, or None if not found.
    """
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path).resolve()
    
    current = start_path
    
    # Crawl up the directory tree looking for .git folder
    while current != current.parent:
        git_dir = current / '.git'
        if git_dir.exists() and git_dir.is_dir():
            return current
        current = current.parent
    
    # If we reach the root without finding .git, return None
    return None


def set_working_directory_to_git_root(start_path=None):
    """
    Set the working directory to the git root folder.
    
    Args:
        start_path: Starting directory path. If None, uses current working directory.
        
    Returns:
        Path object pointing to the git root directory, or None if not found.
    """
    git_root = find_git_root(start_path)
    
    if git_root is None:
        raise RuntimeError("Could not find git root directory. Make sure you're in a git repository.")
    
    os.chdir(git_root)
    return git_root

