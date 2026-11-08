import logging
import os
from config import Config

def get_logger(name):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, Config.LOG_LEVEL, logging.WARNING))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        os.makedirs(os.path.dirname(Config.LOG_FILE), exist_ok=True)
        fh = logging.FileHandler(Config.LOG_FILE, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception as e:
        print(f"⚠️ Cannot create log file: {e}")
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger
