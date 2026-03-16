import logging
import sys
from pathlib import Path


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with the specified name and level

    Args:
        name: Name of the logger
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers if logger already exists
    if logger.handlers:
        return logger

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger


def setup_loggers():
    """Set up all required loggers for the application"""
    loggers = {}

    # Main modules
    modules = [
        'dataset',
        'models',
        'embedding',
        'index',
        'evaluation',
        'analysis',
        'demo',
        'utils',
        'compression'
    ]

    for module in modules:
        loggers[module] = setup_logger(f'multimodal_retrieval.{module}')

    return loggers


# Initialize all loggers
loggers = setup_loggers()

# Convenience functions to access individual loggers
def get_dataset_logger():
    return loggers['dataset']


def get_models_logger():
    return loggers['models']


def get_embedding_logger():
    return loggers['embedding']


def get_index_logger():
    return loggers['index']


def get_evaluation_logger():
    return loggers['evaluation']


def get_analysis_logger():
    return loggers['analysis']


def get_demo_logger():
    return loggers['demo']


def get_utils_logger():
    return loggers['utils']


def get_compression_logger():
    return loggers['compression']