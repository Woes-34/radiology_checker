import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

from .config import get_config


class LoggerManager:
    _instance = None
    _loggers = {}
    
    def __new__(cls, log_name: str = 'radiology_checker'):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_logger()
        return cls._instance
    
    def _init_logger(self):
        config = get_config().get_log_config()
        
        log_level = config.get('level', 'INFO').upper()
        log_format = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_file_path = config.get('file_path', 'logs/radiology_checker.log')
        max_file_size_mb = config.get('max_file_size_mb', 10)
        backup_count = config.get('backup_count', 5)
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if log_file_path.startswith('.') or log_file_path.startswith('..'):
            log_file_path = os.path.normpath(os.path.join(project_root, log_file_path))
        log_dir = os.path.dirname(os.path.abspath(log_file_path))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        formatter = logging.Formatter(log_format)
        
        handlers = []
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
        
        file_handler = RotatingFileHandler(
            os.path.abspath(log_file_path),
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
        
        self._root_logger = logging.getLogger()
        self._root_logger.setLevel(getattr(logging, log_level, logging.INFO))
        self._root_logger.handlers = []
        for handler in handlers:
            self._root_logger.addHandler(handler)
    
    def get_logger(self, name: str = 'radiology_checker') -> logging.Logger:
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)
        return self._loggers[name]
    
    def set_level(self, level: str):
        log_level = getattr(logging, level.upper(), logging.INFO)
        self._root_logger.setLevel(log_level)
        for logger in self._loggers.values():
            logger.setLevel(log_level)


def get_logger(name: str = 'radiology_checker') -> logging.Logger:
    return LoggerManager().get_logger(name)
