import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Set


class ForensicLogger:
    def __init__(self, name: str, log_dir: str = "logs", level: int = logging.INFO):
        self.name = name
        self.loggers: Dict[str, logging.Logger] = {}
        self.log_dir = Path(log_dir)
        self.level = level
        self.setup_main_logger()
        
    def setup_main_logger(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        logger.propagate = False
        
        if logger.handlers:
            return
            
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler(sys.stdout)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        self.loggers[self.name] = logger
        
    def get_logger(self, suffix: Optional[str] = None) -> logging.Logger:
        if suffix is None:
            return self.loggers[self.name]
            
        logger_name = f"{self.name}.{suffix}"
        
        if logger_name in self.loggers:
            return self.loggers[logger_name]
            
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.level)
        logger.propagate = True
        
        self.loggers[logger_name] = logger
        return logger


_main_logger: Optional[ForensicLogger] = None
_agent_loggers: Set[str] = set()


def setup_logging(name: str = "forensic", log_dir: str = "logs", level: int = logging.INFO) -> ForensicLogger:
    global _main_logger
    _main_logger = ForensicLogger(name, log_dir, level)
    return _main_logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    global _main_logger
    
    if _main_logger is None:
        # If no main logger exists yet, create a simple console logger
        # This allows modules to get loggers at import time
        console_logger = logging.getLogger(name if name else "default")
        if not console_logger.handlers:  # Only add handler if not already present
            console_logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            console_logger.addHandler(handler)
        return console_logger
        
    if name is None:
        return _main_logger.get_logger()
        
    if name not in _agent_loggers:
        _agent_loggers.add(name)
        
    return _main_logger.get_logger(name)