# =====================
#  PATH UTILITIES
# =====================

import os
import sys
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


def setup_workspace(current_file_path):
    """
    Complete workspace setup: find git root, set working directory, and add utils to path.
    
    This function handles everything needed to set up the workspace environment:
    - Adds utils to sys.path (so imports work)
    - Finds git root directory
    - Changes working directory to git root
    - Re-adds utils to sys.path after directory change
    
    Args:
        current_file_path: Path to the current file (usually __file__)
        
    Returns:
        Path object pointing to the git root directory
    """
    current_file = Path(current_file_path)
    
    # Add utils to path before finding git root
    utils_path = current_file.parent.parent / '95_utils'
    if str(utils_path) not in sys.path:
        sys.path.insert(0, str(utils_path))
    
    # Find and set working directory to git root
    git_root = find_git_root(current_file.parent)
    
    if git_root is None:
        raise RuntimeError("Could not find git root directory. Make sure you're in a git repository.")
    
    os.chdir(git_root)
    
    # Re-add utils to path after changing directory (using absolute path from git root)
    utils_path_absolute = git_root / '95_utils'
    if str(utils_path_absolute) not in sys.path:
        sys.path.insert(0, str(utils_path_absolute))
    
    print(f"Working directory set to git root: {git_root}")
    return git_root

