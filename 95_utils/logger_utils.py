# =====================
#  LOGGING UTILITIES
# =====================

import logging
import datetime
from pathlib import Path


def setup_logging(prefix="training"):
    """
    Set up aggressive logging to both console and file.
    
    Args:
        prefix: Prefix for log filename (default: "training", use "inference" for inference scripts, "analyze" for analysis scripts)
        
    Returns:
        Logger instance
    """
    # Create logs directory and subdirectory if they don't exist
    if prefix == "analyze":
        logs_dir = Path("97_logs") / "analyze_model"
    elif prefix == "inference":
        logs_dir = Path("97_logs") / "inference"
    elif prefix == "prepare_dataset":
        logs_dir = Path("97_logs") / "prepare_dataset"
    else:
        logs_dir = Path("97_logs") / "training"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with specified prefix and timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{prefix}_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()  # Also output to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("")
    logger.info("=" * 80)
    if prefix == "inference":
        logger.info("INFERENCE SESSION STARTED")
    elif prefix == "analyze":
        logger.info("ANALYSIS SESSION STARTED")
    elif prefix == "prepare_dataset":
        logger.info("DATASET PREPARATION SESSION STARTED")
    else:
        logger.info("TRAINING SESSION STARTED")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info("")
    
    return logger

