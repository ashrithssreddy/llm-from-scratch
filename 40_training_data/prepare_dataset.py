# =====================
#  DATASET PREPARATION SCRIPT
# =====================
#
# Description:
#   This script downloads Wikipedia articles from the Machine Learning category
#   and its subcategories, extracts clean text, and saves them as .txt files
#   for training the language model.
#
# Code Flow:
#   main()
#     → get_all_category_members() → recursively gets all pages in category
#     → fetch_article_text() → gets article content via API
#     → clean_wikipedia_text() → removes markup and cleans text
#     → save_article_to_file() → writes to .txt file
#
# =====================
#  EXAMPLE USAGE
# =====================
#
# Basic usage (download ML articles to default folder):
#   python 40_training_data/prepare_dataset.py
#
# Specify custom output folder:
#   python 40_training_data/prepare_dataset.py --output 40_training_data/dataset_machine_learning/
#
# Specify starting category:
#   python 40_training_data/prepare_dataset.py --category "Category:Machine_learning"
#
# Limit number of articles (for testing):
#   python 40_training_data/prepare_dataset.py --max-articles 50
#
# Available arguments:
#   --output - Output folder for .txt files (default: 40_training_data/dataset_machine_learning/)
#   --category - Starting category name (default: Category:Machine_learning)
#   --max-articles - Maximum number of articles to download (default: None, download all)
#   --delay - Delay between API requests in seconds (default: 0.1)
#   --overwrite - Overwrite existing files (default: False)


# =====================
#  SETUP
# =====================

# Set working directory to git root
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent / '95_utils')); __import__('path_utils').setup_workspace(__file__)

# Imports
import os
import re
import json
import time
import argparse
import requests
from pathlib import Path
from typing import Set, List, Dict, Optional
from urllib.parse import quote

# Logging setup
from logger_utils import setup_logging  # type: ignore


# =====================
#  WIKIPEDIA API FUNCTIONS
# =====================

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
API_DELAY = 0.1  # Default delay between requests (seconds)


def make_api_request(params: Dict) -> Dict:
    """
    Make a request to Wikipedia API with error handling.
    
    Args:
        params: API parameters dictionary
        
    Returns:
        JSON response from API
        
    Raises:
        requests.RequestException: If API request fails
    """
    params['format'] = 'json'
    params['formatversion'] = '2'
    
    # Wikipedia requires a User-Agent header to identify the client
    headers = {
        'User-Agent': 'LLM-From-Scratch Dataset Preparation Script (https://github.com/yourusername/llm-from-scratch)'
    }
    
    try:
        response = requests.get(WIKIPEDIA_API_URL, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise requests.RequestException(f"API request failed: {e}")


def get_category_members(category_name: str, cmtype: str = "page|subcat") -> List[Dict]:
    """
    Get all members (pages and/or subcategories) of a Wikipedia category.
    
    Args:
        category_name: Name of the category (e.g., "Category:Machine_learning")
        cmtype: Type of members to get - "page", "subcat", or "page|subcat"
        
    Returns:
        List of dictionaries containing member information
    """
    all_members = []
    cmcontinue = None
    
    while True:
        params = {
            'action': 'query',
            'list': 'categorymembers',
            'cmtitle': category_name,
            'cmtype': cmtype,
            'cmlimit': '500',  # Maximum allowed by API
        }
        
        if cmcontinue:
            params['cmcontinue'] = cmcontinue
        
        try:
            data = make_api_request(params)
            
            if 'query' in data and 'categorymembers' in data['query']:
                members = data['query']['categorymembers']
                logger.debug(f"  Found {len(members)} members in this batch")
                all_members.extend(members)
            else:
                logger.debug(f"  No categorymembers found in response. Keys: {list(data.get('query', {}).keys())}")
            
            # Check for continuation
            if 'continue' in data and 'cmcontinue' in data['continue']:
                cmcontinue = data['continue']['cmcontinue']
            else:
                break
                
            # Rate limiting
            time.sleep(API_DELAY)
            
        except Exception as e:
            logger.warning(f"Error fetching category members for {category_name}: {e}")
            break
    
    return all_members


def get_all_category_members(category_name: str, visited_categories: Optional[Set[str]] = None, category_path: str = "") -> Dict[str, str]:
    """
    Recursively get all article titles in a category and its subcategories.
    
    Args:
        category_name: Starting category name (e.g., "Category:Machine_learning")
        visited_categories: Set of already visited categories (to avoid cycles)
        category_path: Path string representing the category hierarchy (for folder structure)
        
    Returns:
        Dictionary mapping article titles to their category folder paths
    """
    if visited_categories is None:
        visited_categories = set()
    
    # Avoid infinite loops
    if category_name in visited_categories:
        logger.debug(f"Skipping already visited category: {category_name}")
        return {}
    
    visited_categories.add(category_name)
    logger.info(f"Processing category: {category_name}")
    
    # Extract category name without "Category:" prefix for folder name
    category_folder_name = category_name.replace('Category:', '').replace(' ', '_')
    if category_path:
        current_path = f"{category_path}/{category_folder_name}"
    else:
        current_path = category_folder_name
    
    all_articles = {}
    
    # Get all members (pages and subcategories)
    members = get_category_members(category_name, cmtype="page|subcat")
    
    logger.debug(f"  Processing {len(members)} members from category {category_name}")
    for member in members:
        title = member.get('title', '')
        # With formatversion=2, type might be in 'ns' (namespace) field
        # Namespace 0 = article, Namespace 14 = category
        member_ns = member.get('ns', 0)
        member_type = member.get('type', '')
        
        # Determine if it's a page or subcategory
        # If title starts with "Category:", it's a category
        # Otherwise, check namespace or type field
        is_category = title.startswith('Category:')
        is_page = not is_category and (member_type == 'page' or member_ns == 0)
        
        if is_page:
            # It's an article page - map it to current category path
            # Only add if not already in dict (first category wins for articles in multiple categories)
            if title not in all_articles:
                all_articles[title] = current_path
                logger.debug(f"  Found article: {title} -> {current_path}")
        elif is_category:
            # It's a subcategory - recurse
            logger.debug(f"  Found subcategory: {title}")
            subcategory_articles = get_all_category_members(title, visited_categories, current_path)
            # Merge subcategory articles, but don't overwrite existing entries
            for article_title, article_path in subcategory_articles.items():
                if article_title not in all_articles:
                    all_articles[article_title] = article_path
    
    logger.info(f"  Category {category_name} contains {len(all_articles)} articles (including subcategories)")
    return all_articles


def fetch_article_text(title: str) -> Optional[str]:
    """
    Fetch the full text content of a Wikipedia article.
    
    Uses prop=revisions to get the full article content (no character limits), then extracts plain text.
    
    Args:
        title: Article title
        
    Returns:
        Plain text content of the article, or None if fetch fails
    """
    # Always use revisions API to get full wikitext (no character limits)
    params_rev = {
        'action': 'query',
        'prop': 'revisions',
        'titles': title,
        'rvprop': 'content',
        'rvslots': 'main',
        'rvlimit': '1',
    }
    
    try:
        data_rev = make_api_request(params_rev)
        
        full_text = None
        if 'query' in data_rev and 'pages' in data_rev['query']:
            pages = data_rev['query']['pages']
            if isinstance(pages, list):
                for page_data in pages:
                    # Check if page is missing or invalid
                    if 'missing' in page_data:
                        logger.warning(f"Article '{title}' does not exist or is missing")
                        return None
                    
                    if 'revisions' in page_data and len(page_data['revisions']) > 0:
                        # Try to get wikitext from slots first (newer API format)
                        revision = page_data['revisions'][0]
                        wikitext = None
                        
                        # Try slots format first - content is in slots['main']['content']
                        if 'slots' in revision and 'main' in revision['slots']:
                            main_slot = revision['slots']['main']
                            wikitext = main_slot.get('content', '') or main_slot.get('*', '')
                        # Try direct content field (older format)
                        elif '*' in revision:
                            wikitext = revision.get('*', '')
                        # Try content field
                        elif 'content' in revision:
                            wikitext = revision.get('content', '')
                        
                        if wikitext:
                            logger.debug(f"Found wikitext for {title}, length: {len(wikitext)}")
                            # Parse wikitext to plain text
                            full_text = parse_wikitext_to_text(wikitext)
                            break
                        else:
                            logger.debug(f"No wikitext found in revision. Keys: {list(revision.keys())}")
            else:
                # Fallback for formatversion=1 (dict format)
                for page_id, page_data in pages.items():
                    if 'missing' in page_data:
                        logger.warning(f"Article '{title}' does not exist or is missing")
                        return None
                    
                    if 'revisions' in page_data and len(page_data['revisions']) > 0:
                        revision = page_data['revisions'][0]
                        wikitext = None
                        
                        # Try slots format first - content is in slots['main']['content']
                        if 'slots' in revision and 'main' in revision['slots']:
                            main_slot = revision['slots']['main']
                            wikitext = main_slot.get('content', '') or main_slot.get('*', '')
                        elif '*' in revision:
                            wikitext = revision.get('*', '')
                        elif 'content' in revision:
                            wikitext = revision.get('content', '')
                        
                        if wikitext:
                            logger.debug(f"Found wikitext for {title}, length: {len(wikitext)}")
                            full_text = parse_wikitext_to_text(wikitext)
                            break
        else:
            logger.debug(f"Unexpected API response structure. Keys: {list(data_rev.keys())}")
        
        if full_text:
            return full_text
        
        logger.warning(f"No text found for article: {title}")
        return None
        
    except Exception as e:
        logger.warning(f"Error fetching article {title}: {e}")
        return None


def parse_wikitext_to_text(wikitext: str) -> str:
    """
    Parse Wikipedia wikitext to plain text, preserving structure and content.
    
    Args:
        wikitext: Raw Wikipedia wikitext
        
    Returns:
        Plain text content with sections preserved
    """
    text = wikitext
    
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # Remove complex nested templates (but keep simple ones that might have content)
    # Remove templates like {{Infobox}}, {{Citation needed}}, etc.
    # This is a simplified approach - for better parsing, use mwparserfromhell
    text = re.sub(r'\{\{[^{}]*\}\}', '', text)  # Simple templates
    # Handle nested templates (up to 3 levels)
    for _ in range(3):
        text = re.sub(r'\{\{[^{}]*\{[^{}]*\{[^{}]*\}[^{}]*\}[^{}]*\}\}', '', text)
    
    # Remove ref tags <ref>...</ref> and <ref name="..."/>
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*/>', '', text)
    
    # Remove other HTML tags but preserve content
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove file/image links [[File:...]] or [[Image:...]]
    text = re.sub(r'\[\[(?:File|Image|Media):[^\]]+\]\]', '', text, flags=re.IGNORECASE)
    
    # Convert section headers (== Header ==) to plain text headers
    # Keep the header text but remove the == markers
    text = re.sub(r'^=+\s*(.+?)\s*=+$', r'\1', text, flags=re.MULTILINE)
    
    # Convert internal links [[Link|Display]] to Display, or [[Link]] to Link
    text = re.sub(r'\[\[([^\|\]]+)\|([^\]]+)\]\]', r'\2', text)  # [[Link|Display]] -> Display
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)  # [[Link]] -> Link
    
    # Remove external links [url text] -> text, or [url] -> remove
    text = re.sub(r'\[https?://[^\s]+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://[^\]]+\]', '', text)
    text = re.sub(r'\[mailto:[^\]]+\]', '', text)
    
    # Remove reference markers [1], [2], etc. (but keep content before them)
    text = re.sub(r'\[\d+\]', '', text)
    
    # Remove category links [[Category:...]]
    text = re.sub(r'\[\[Category:[^\]]+\]\]', '', text, flags=re.IGNORECASE)
    
    # Remove interwiki links [[lang:...]]
    text = re.sub(r'\[\[[a-z]+:[^\]]+\]\]', '', text, flags=re.IGNORECASE)
    
    # Remove navigation templates and other structural elements
    text = re.sub(r'\[\[(?:Template|Help|Wikipedia):[^\]]+\]\]', '', text, flags=re.IGNORECASE)
    
    # Convert bold/italic markup '''text''' -> text, ''text'' -> text
    text = re.sub(r"'''(.*?)'''", r'\1', text)
    text = re.sub(r"''(.*?)''", r'\1', text)
    
    # Remove horizontal rules ----
    text = re.sub(r'^----+$', '', text, flags=re.MULTILINE)
    
    # Remove table markup {| ... |}
    text = re.sub(r'\{[|].*?[|]\}', '', text, flags=re.DOTALL)
    
    # Remove list markers at start of lines (*, #, ;, :)
    text = re.sub(r'^[*#;:]+', '', text, flags=re.MULTILINE)
    
    # Remove leading/trailing pipes from table remnants
    text = re.sub(r'^\|+', '', text, flags=re.MULTILINE)
    
    # Clean up whitespace - preserve paragraph breaks
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if line:  # Only add non-empty lines
            lines.append(line)
    
    text = '\n\n'.join(lines)  # Double newline between paragraphs
    
    # Remove excessive blank lines (more than 2 consecutive)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Final cleanup
    text = text.strip()
    
    return text


# =====================
#  TEXT CLEANING FUNCTIONS
# =====================

def clean_wikipedia_text(text: str) -> str:
    """
    Clean Wikipedia text by removing common artifacts and formatting.
    
    Args:
        text: Raw Wikipedia text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove section headers that are just "== Section =="
    text = re.sub(r'^==+ .+? ==+\n?', '', text, flags=re.MULTILINE)
    
    # Remove standalone "===" lines
    text = re.sub(r'^===+\n?', '', text, flags=re.MULTILINE)
    
    # Remove reference markers like [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    
    # Remove multiple consecutive newlines (more than 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    # Remove empty lines at start and end
    text = text.strip()
    
    return text


def sanitize_filename(title: str) -> str:
    """
    Convert a Wikipedia article title to a valid filename.
    
    Args:
        title: Article title
        
    Returns:
        Sanitized filename (without .txt extension)
    """
    # Replace invalid filename characters
    filename = title.replace('/', '_').replace('\\', '_')
    filename = filename.replace(':', '_').replace('*', '_')
    filename = filename.replace('?', '_').replace('"', '_')
    filename = filename.replace('<', '_').replace('>', '_')
    filename = filename.replace('|', '_')
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    
    return filename


def sanitize_category_path(category_path: str, max_length: int = 200) -> str:
    """
    Sanitize and truncate category path to avoid Windows path length limits.
    
    Args:
        category_path: Category path like "Machine_learning/Neural_networks/Deep_learning"
        max_length: Maximum allowed path length
        
    Returns:
        Sanitized and truncated category path
    """
    if not category_path:
        return ""
    
    # Split into parts
    parts = category_path.split('/')
    
    # Sanitize each part (remove invalid chars, limit length)
    sanitized_parts = []
    for part in parts:
        # Replace invalid path characters
        sanitized = part.replace('/', '_').replace('\\', '_')
        sanitized = sanitized.replace(':', '_').replace('*', '_')
        sanitized = sanitized.replace('?', '_').replace('"', '_')
        sanitized = sanitized.replace('<', '_').replace('>', '_')
        sanitized = sanitized.replace('|', '_')
        sanitized = sanitized.strip('. ')
        
        # Limit each part length
        if len(sanitized) > 50:
            sanitized = sanitized[:50]
        
        if sanitized:  # Only add non-empty parts
            sanitized_parts.append(sanitized)
    
    # Rejoin and check total length
    result = '/'.join(sanitized_parts)
    
    # If still too long, truncate from the beginning (keep most specific categories)
    if len(result) > max_length:
        # Keep the last parts (most specific)
        parts = result.split('/')
        result_parts = []
        current_length = 0
        
        # Add parts from the end until we hit the limit
        for part in reversed(parts):
            if current_length + len(part) + 1 <= max_length:  # +1 for separator
                result_parts.insert(0, part)
                current_length += len(part) + 1
            else:
                break
        
        result = '/'.join(result_parts) if result_parts else parts[-1] if parts else ""
    
    return result


# =====================
#  HIERARCHY CACHE OPERATIONS
# =====================

def save_page_hierarchy(article_to_category: Dict[str, str], output_folder: Path):
    """
    Save the page hierarchy (article to category mapping) to a JSON file.
    
    Args:
        article_to_category: Dictionary mapping article titles to category paths
        output_folder: Base output folder
    """
    scratch_folder = output_folder / "scratch_files"
    scratch_folder.mkdir(parents=True, exist_ok=True)
    
    hierarchy_file = scratch_folder / "page_hierarchy.json"
    
    try:
        with open(hierarchy_file, 'w', encoding='utf-8') as f:
            json.dump(article_to_category, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved page hierarchy to: {hierarchy_file}")
        logger.info(f"  Total articles: {len(article_to_category)}")
    except Exception as e:
        logger.warning(f"Failed to save page hierarchy: {e}")


def load_page_hierarchy(output_folder: Path) -> Optional[Dict[str, str]]:
    """
    Load the page hierarchy (article to category mapping) from a JSON file.
    
    Args:
        output_folder: Base output folder
        
    Returns:
        Dictionary mapping article titles to category paths, or None if file doesn't exist
    """
    scratch_folder = output_folder / "scratch_files"
    hierarchy_file = scratch_folder / "page_hierarchy.json"
    
    if not hierarchy_file.exists():
        return None
    
    try:
        with open(hierarchy_file, 'r', encoding='utf-8') as f:
            article_to_category = json.load(f)
        logger.info(f"Loaded page hierarchy from: {hierarchy_file}")
        logger.info(f"  Total articles: {len(article_to_category)}")
        return article_to_category
    except Exception as e:
        logger.warning(f"Failed to load page hierarchy: {e}")
        return None


# =====================
#  FILE OPERATIONS
# =====================

def save_article_to_file(title: str, text: str, output_folder: Path, category_path: str = "", overwrite: bool = False) -> bool:
    """
    Save an article's text to a .txt file in the appropriate category folder.
    
    Args:
        title: Article title
        text: Article text content
        output_folder: Base folder to save files
        category_path: Subfolder path based on category hierarchy (e.g., "Neural_networks/Deep_learning")
        overwrite: Whether to overwrite existing files
        
    Returns:
        True if file was saved successfully, False otherwise
    """
    if not text or len(text.strip()) < 50:  # Skip very short articles
        logger.debug(f"Skipping {title}: text too short ({len(text)} chars)")
        return False
    
    # Create category subfolder if needed
    if category_path:
        # Sanitize and truncate category path to avoid Windows path length limits
        # Windows MAX_PATH is 260, but we need room for filename and base path
        base_path_len = len(str(output_folder))
        filename_len = len(sanitize_filename(title)) + 4  # +4 for .txt
        max_category_path_len = 200 - base_path_len - filename_len - 10  # -10 for safety margin
        
        sanitized_path = sanitize_category_path(category_path, max_length=max_category_path_len)
        
        if sanitized_path:
            category_folder = output_folder / sanitized_path
            try:
                category_folder.mkdir(parents=True, exist_ok=True)
                save_folder = category_folder
            except (OSError, FileNotFoundError) as e:
                # If path is still too long, use a shorter fallback
                logger.warning(f"Path too long, using fallback: {category_path}")
                # Use just the last part of the category path
                last_part = category_path.split('/')[-1] if '/' in category_path else category_path
                sanitized_last = sanitize_category_path(last_part, max_length=50)
                category_folder = output_folder / sanitized_last
                category_folder.mkdir(parents=True, exist_ok=True)
                save_folder = category_folder
        else:
            save_folder = output_folder
    else:
        save_folder = output_folder
    
    filename = sanitize_filename(title) + '.txt'
    filepath = save_folder / filename
    
    # Check if file already exists - skip download if it does (helps with resuming)
    if filepath.exists() and not overwrite:
        logger.debug(f"Skipping {title}: file already exists at {filepath}")
        return False
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    except Exception as e:
        logger.error(f"Error saving {title} to {filepath}: {e}")
        return False


# =====================
#  MAIN PROCESSING FUNCTION
# =====================

def prepare_dataset(
    output_folder: str = "40_training_data/dataset_machine_learning/",
    category: str = "Category:Machine_learning",
    max_articles: Optional[int] = None,
    delay: float = 0.1,
    overwrite: bool = False
):
    """
    Main function to prepare Wikipedia dataset for training.
    
    Args:
        output_folder: Folder to save .txt files
        category: Starting Wikipedia category
        max_articles: Maximum number of articles to download (None = all)
        delay: Delay between API requests (seconds)
        overwrite: Whether to overwrite existing files
    """
    global API_DELAY
    API_DELAY = delay
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("DATASET PREPARATION CONFIGURATION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  - Output folder: {output_folder}")
    logger.info(f"  - Starting category: {category}")
    logger.info(f"  - Max articles: {max_articles if max_articles else 'All'}")
    logger.info(f"  - API delay: {delay} seconds")
    logger.info(f"  - Overwrite existing: {overwrite}")
    logger.info("")
    
    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder created/verified: {output_path}")
    logger.info("")
    
    # Step 1: Get all article titles (use cache if available)
    logger.info("=" * 80)
    logger.info("STEP 1: GETTING ARTICLE TITLES")
    logger.info("=" * 80)
    logger.info("")
    
    # Try to load existing hierarchy first
    article_to_category = load_page_hierarchy(output_path)
    
    if article_to_category is None:
        # No cache found, generate hierarchy
        logger.info(f"No cached hierarchy found. Recursively traversing category: {category}")
        logger.info("This may take a few minutes...")
        logger.info("")
        
        start_time = time.time()
        article_to_category = get_all_category_members(category)
        elapsed_time = time.time() - start_time
        
        logger.info("")
        logger.info(f"Found {len(article_to_category)} unique articles")
        logger.info(f"Time taken: {elapsed_time:.2f} seconds")
        logger.info("")
        
        # Save hierarchy for future use
        save_page_hierarchy(article_to_category, output_path)
        logger.info("")
    else:
        logger.info(f"Using cached page hierarchy with {len(article_to_category)} articles")
        logger.info("")
    
    # Limit articles if specified
    if max_articles and len(article_to_category) > max_articles:
        limited_items = list(article_to_category.items())[:max_articles]
        article_to_category = dict(limited_items)
        logger.info(f"Limited to {max_articles} articles for processing")
        logger.info("")
    
    # Step 2: Fetch and save articles
    logger.info("=" * 80)
    logger.info("STEP 2: FETCHING AND SAVING ARTICLES")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Fetching {len(article_to_category)} articles...")
    logger.info("")
    
    start_time = time.time()
    saved_count = 0
    skipped_count = 0
    error_count = 0
    
    article_list = sorted(article_to_category.items())
    
    for i, (title, category_path) in enumerate(article_list, 1):
        logger.info(f"[{i}/{len(article_list)}] Processing: {title}")
        if category_path:
            logger.info(f"  Category path: {category_path}")
        
        # Check if file already exists before fetching (skip download if exists)
        # This helps with resuming interrupted downloads
        sanitized_path = sanitize_category_path(category_path, max_length=200) if category_path else ""
        if sanitized_path:
            base_path_len = len(str(output_path))
            filename_len = len(sanitize_filename(title)) + 4
            max_category_path_len = 200 - base_path_len - filename_len - 10
            sanitized_path = sanitize_category_path(category_path, max_length=max_category_path_len)
            if sanitized_path:
                check_folder = output_path / sanitized_path
            else:
                check_folder = output_path
        else:
            check_folder = output_path
        
        filename = sanitize_filename(title) + '.txt'
        filepath = check_folder / filename
        
        if filepath.exists() and not overwrite:
            skipped_count += 1
            logger.info(f"  Skipped: file already exists (resuming from previous run)")
            logger.info("")
            continue
        
        # Fetch article text
        text = fetch_article_text(title)
        time.sleep(API_DELAY)  # Rate limiting
        
        if text is None:
            error_count += 1
            logger.warning(f"  Failed to fetch article")
            continue
        
        # Clean text
        cleaned_text = clean_wikipedia_text(text)
        
        if not cleaned_text or len(cleaned_text.strip()) < 50:
            skipped_count += 1
            logger.debug(f"  Skipped: text too short after cleaning")
            continue
        
        # Save to file in appropriate category folder
        if save_article_to_file(title, cleaned_text, output_path, category_path, overwrite):
            saved_count += 1
            logger.info(f"  Saved: {len(cleaned_text):,} characters")
        else:
            skipped_count += 1
            logger.debug(f"  Skipped: file already exists or save failed")
        
        logger.info("")
    
    elapsed_time = time.time() - start_time
    
    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("DATASET PREPARATION COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Summary:")
    logger.info(f"  - Total articles found: {len(article_to_category)}")
    logger.info(f"  - Articles saved: {saved_count}")
    logger.info(f"  - Articles skipped: {skipped_count}")
    logger.info(f"  - Errors: {error_count}")
    logger.info(f"  - Time taken: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
    logger.info(f"  - Output folder: {output_path}")
    logger.info("")
    
    # Count total files and size (recursively)
    txt_files = list(output_path.rglob("*.txt"))
    total_size = sum(f.stat().st_size for f in txt_files)
    logger.info(f"Total files in output folder (including subfolders): {len(txt_files)}")
    logger.info(f"Total size: {total_size / 1024 / 1024:.2f} MB")
    
    # Show folder structure
    category_folders = sorted(set(f.parent for f in txt_files if f.parent != output_path))
    if category_folders:
        logger.info(f"Category folders created: {len(category_folders)}")
        logger.info("  Sample folders:")
        for folder in category_folders[:10]:  # Show first 10
            relative_path = folder.relative_to(output_path)
            file_count = len(list(folder.glob("*.txt")))
            logger.info(f"    - {relative_path} ({file_count} files)")
        if len(category_folders) > 10:
            logger.info(f"    ... and {len(category_folders) - 10} more")
    logger.info("")


# =====================
#  MAIN
# =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Wikipedia articles from Machine Learning category for training"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="40_training_data/dataset_machine_learning/",
        help="Output folder for .txt files (default: 40_training_data/dataset_machine_learning/)"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="Category:Machine_learning",
        help="Starting Wikipedia category (default: Category:Machine_learning)"
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=None,
        help="Maximum number of articles to download (default: None, download all)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between API requests in seconds (default: 0.1)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files (default: False)"
    )
    parser.add_argument(
        "--test-article",
        type=str,
        default=None,
        help="Test mode: fetch a single article by title (e.g., 'Machine learning')"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(prefix="prepare_dataset")
    
    # Test mode: fetch single article
    if args.test_article:
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST MODE: FETCHING SINGLE ARTICLE")
        logger.info("=" * 80)
        logger.info("")
        logger.info(f"Article title: {args.test_article}")
        logger.info("")
        
        try:
            text = fetch_article_text(args.test_article)
            if text:
                logger.info(f"Success! Retrieved {len(text):,} characters")
                logger.info("")
                logger.info("First 500 characters:")
                logger.info("-" * 80)
                logger.info(text[:500])
                logger.info("-" * 80)
                logger.info("")
                logger.info("Last 500 characters:")
                logger.info("-" * 80)
                logger.info(text[-500:])
                logger.info("-" * 80)
                
                # Optionally save it
                if args.output:
                    output_path = Path(args.output)
                    output_path.mkdir(parents=True, exist_ok=True)
                    filename = sanitize_filename(args.test_article) + '.txt'
                    filepath = output_path / filename
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(text)
                    logger.info(f"")
                    logger.info(f"Saved to: {filepath}")
            else:
                logger.error("Failed to fetch article text")
        except Exception as e:
            logger.error(f"Error: {str(e)}", exc_info=True)
            raise
    else:
        # Run dataset preparation
        try:
            prepare_dataset(
                output_folder=args.output,
                category=args.category,
                max_articles=args.max_articles,
                delay=args.delay,
                overwrite=args.overwrite
            )
        except Exception as e:
            logger.error("")
            logger.error("")
            logger.error("=" * 80)
            logger.error("DATASET PREPARATION FAILED")
            logger.error("=" * 80)
            logger.error("")
            logger.error(f"Error: {str(e)}", exc_info=True)
            logger.error("")
            raise
