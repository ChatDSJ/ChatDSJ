import sys
from loguru import logger

from config.settings import get_settings

def configure_logging():
    """Configure logging with loguru."""
    settings = get_settings()
    
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    
    # Add file logging in production
    if settings.environment == "production":
        logger.add(
            "logs/chatdsj.log",
            rotation="10 MB",
            retention="1 week",
            level=settings.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        )
    
    return logger