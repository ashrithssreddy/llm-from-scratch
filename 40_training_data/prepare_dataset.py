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
    
    Uses prop=revisions to get the full article content, then extracts plain text.
    
    Args:
        title: Article title
        
    Returns:
        Plain text content of the article, or None if fetch fails
    """
    # First try to get full article using extracts with high character limit
    params = {
        'action': 'query',
        'prop': 'extracts',
        'titles': title,
        'explaintext': '1',  # Get plain text, not HTML
        'exintro': '0',  # Get full article, not just intro
        'exsectionformat': 'plain',
        'exchars': '100000',  # Request up to 100k characters (API limit is usually 20k per section)
    }
    
    try:
        data = make_api_request(params)
        
        full_text = None
        if 'query' in data and 'pages' in data['query']:
            pages = data['query']['pages']
            # With formatversion=2, pages is a list, not a dict
            if isinstance(pages, list):
                for page_data in pages:
                    if 'extract' in page_data:
                        full_text = page_data['extract']
                        break
            else:
                # Fallback for formatversion=1 (dict format)
                for page_id, page_data in pages.items():
                    if 'extract' in page_data:
                        full_text = page_data['extract']
                        break
        
        # If extracts didn't return enough (might be truncated), try revisions API
        if not full_text or len(full_text) < 1000:
            # Try using revisions to get full wikitext, then parse it
            params_rev = {
                'action': 'query',
                'prop': 'revisions',
                'titles': title,
                'rvprop': 'content',
                'rvslots': 'main',
                'rvlimit': '1',
            }
            
            data_rev = make_api_request(params_rev)
            
            if 'query' in data_rev and 'pages' in data_rev['query']:
                pages = data_rev['query']['pages']
                if isinstance(pages, list):
                    for page_data in pages:
                        if 'revisions' in page_data and len(page_data['revisions']) > 0:
                            wikitext = page_data['revisions'][0].get('slots', {}).get('main', {}).get('*', '')
                            if wikitext:
                                # Parse wikitext to plain text (basic parsing)
                                full_text = parse_wikitext_to_text(wikitext)
                                break
                else:
                    for page_id, page_data in pages.items():
                        if 'revisions' in page_data and len(page_data['revisions']) > 0:
                            wikitext = page_data['revisions'][0].get('slots', {}).get('main', {}).get('*', '')
                            if wikitext:
                                full_text = parse_wikitext_to_text(wikitext)
                                break
        
        if full_text:
            return full_text
        
        logger.warning(f"No text found for article: {title}")
        return None
        
    except Exception as e:
        logger.warning(f"Error fetching article {title}: {e}")
        return None


def parse_wikitext_to_text(wikitext: str) -> str:
    """
    Basic parsing of Wikipedia wikitext to plain text.
    This is a simplified parser - for full parsing, consider using mwparserfromhell.
    
    Args:
        wikitext: Raw Wikipedia wikitext
        
    Returns:
        Plain text content
    """
    text = wikitext
    
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # Remove templates {{...}}
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    
    # Remove ref tags <ref>...</ref>
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*/>', '', text)
    
    # Remove other HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove file/image links [[File:...]] or [[Image:...]]
    text = re.sub(r'\[\[(?:File|Image):[^\]]+\]\]', '', text)
    
    # Convert internal links [[Link|Display]] to Display, or [[Link]] to Link
    text = re.sub(r'\[\[([^\|\]]+)\|([^\]]+)\]\]', r'\2', text)  # [[Link|Display]] -> Display
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)  # [[Link]] -> Link
    
    # Remove external links [url text] -> text
    text = re.sub(r'\[https?://[^\s]+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://[^\]]+\]', '', text)
    
    # Remove section headers (== Header ==)
    text = re.sub(r'^=+\s*(.+?)\s*=+$', r'\1', text, flags=re.MULTILINE)
    
    # Remove reference markers [1], [2], etc.
    text = re.sub(r'\[\d+\]', '', text)
    
    # Remove category links [[Category:...]]
    text = re.sub(r'\[\[Category:[^\]]+\]\]', '', text)
    
    # Remove interwiki links [[lang:...]]
    text = re.sub(r'\[\[[a-z]+:[^\]]+\]\]', '', text)
    
    # Remove bold/italic markup '''text''' -> text, ''text'' -> text
    text = re.sub(r"'''(.*?)'''", r'\1', text)
    text = re.sub(r"''(.*?)''", r'\1', text)
    
    # Remove horizontal rules ----
    text = re.sub(r'^----+$', '', text, flags=re.MULTILINE)
    
    # Remove multiple consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Clean up whitespace
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
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
        category_folder = output_folder / category_path
        category_folder.mkdir(parents=True, exist_ok=True)
        save_folder = category_folder
    else:
        save_folder = output_folder
    
    filename = sanitize_filename(title) + '.txt'
    filepath = save_folder / filename
    
    if filepath.exists() and not overwrite:
        logger.debug(f"Skipping {title}: file already exists")
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
    
    # Step 1: Get all article titles
    logger.info("=" * 80)
    logger.info("STEP 1: GETTING ARTICLE TITLES")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Recursively traversing category: {category}")
    logger.info("This may take a few minutes...")
    logger.info("")
    
    start_time = time.time()
    article_to_category = get_all_category_members(category)
    elapsed_time = time.time() - start_time
    
    logger.info("")
    logger.info(f"Found {len(article_to_category)} unique articles")
    logger.info(f"Time taken: {elapsed_time:.2f} seconds")
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
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(prefix="prepare_dataset")
    
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
