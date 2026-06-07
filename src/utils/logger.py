"""
Logging Utility
===============
Provides centralized logging configuration for console and file output.
"""

import logging
import os
import sys

def setup_logger(
    name: str = "portfolio_rl",
    log_dir: str = "experiments/logs",
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.
    
    Args:
        name: Name of the logger
        log_dir: Directory to save log files
        level: Logging level (e.g. logging.INFO)
        console: Whether to log to stdout
        
    Returns:
        logging.Logger instance configured
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    if logger.hasHandlers():
        return logger
        
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    # File Handler
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"{name}.log")
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to create file handler for logging: {e}", file=sys.stderr)
            
    return logger
