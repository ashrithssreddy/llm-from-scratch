# =====================
#  LOGGING UTILITIES
# =====================

import logging
import datetime
from pathlib import Path


def setup_logging():
    """Set up aggressive logging to both console and file."""
    # Create logs directory if it doesn't exist
    logs_dir = Path("97_logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create log filename with training prefix and timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"training_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()  # Also output to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("")
    logger.info("=" * 80)
    logger.info("TRAINING SESSION STARTED")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info("")
    
    return logger

