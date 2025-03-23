import logging
import os
import sys
import json
import time
import uuid
import functools
import contextvars
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Set, Callable, TypeVar, Union, List, cast

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

trace_id = contextvars.ContextVar('trace_id', default='')
span_id = contextvars.ContextVar('span_id', default='')
correlation_data = contextvars.ContextVar('correlation_data', default={})

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'name': record.name,
            'level': record.levelname,
            'message': record.getMessage(),
            'path': record.pathname,
            'line': record.lineno,
            'function': record.funcName
        }
        
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms
            
        if hasattr(record, 'trace_id') and record.trace_id:
            log_data['trace_id'] = record.trace_id
            
        if hasattr(record, 'span_id') and record.span_id:
            log_data['span_id'] = record.span_id
            
        if hasattr(record, 'correlation_data') and record.correlation_data:
            log_data.update(record.correlation_data)
            
        if hasattr(record, 'extra_fields') and record.extra_fields:
            log_data.update(record.extra_fields)
            
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)

class StructuredLogRecord(logging.LogRecord):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_id = trace_id.get()
        self.span_id = span_id.get()
        self.correlation_data = correlation_data.get()
        self.extra_fields = {}

class PerformanceTracker:
    def __init__(self, logger: logging.Logger, operation: str, level: int = logging.INFO, 
                 extra_fields: Optional[Dict[str, Any]] = None):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time = None
        self.extra_fields = extra_fields or {}
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        record = StructuredLogRecord(
            self.logger.name, self.level, "", 0, 
            f"Completed {self.operation} in {duration_ms:.2f}ms", 
            (), None
        )
        record.duration_ms = duration_ms
        
        for key, value in self.extra_fields.items():
            record.extra_fields[key] = value
            
        self.logger.handle(record)

class ForensicLogger:
    def __init__(self, name: str, log_dir: str = "logs", level: int = logging.INFO, 
                 structured: bool = True, use_json: bool = True):
        self.name = name
        self.loggers: Dict[str, logging.Logger] = {}
        self.log_dir = Path(log_dir)
        self.level = level
        self.structured = structured
        self.use_json = use_json
        
        logging.setLogRecordFactory(StructuredLogRecord)
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
        
        if self.use_json:
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(trace_id)s:%(span_id)s] - %(message)s'
            )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        self.loggers[self.name] = logger
    
    def get_logger(self, suffix: Optional[str] = None) -> 'StructuredLogger':
        if suffix is None:
            base_logger = self.loggers[self.name]
        else:
            logger_name = f"{self.name}.{suffix}"
            
            if logger_name in self.loggers:
                base_logger = self.loggers[logger_name]
            else:
                logger = logging.getLogger(logger_name)
                logger.setLevel(self.level)
                logger.propagate = True
                
                self.loggers[logger_name] = logger
                base_logger = logger
                
        return StructuredLogger(base_logger)

class StructuredLogger:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        
    def _log(self, level: int, msg: str, *args, **kwargs) -> None:
        extra_fields = kwargs.pop('extra_fields', {})
        
        record = self.logger.makeRecord(
            self.logger.name, level, kwargs.pop('pathname', ''), 
            kwargs.pop('lineno', 0), msg, args, kwargs.pop('exc_info', None),
            kwargs.pop('func', None), kwargs
        )
        
        if extra_fields:
            record.extra_fields = extra_fields
            
        self.logger.handle(record)
    
    def debug(self, msg: str, *args, **kwargs) -> None:
        self._log(logging.DEBUG, msg, *args, **kwargs)
        
    def info(self, msg: str, *args, **kwargs) -> None:
        self._log(logging.INFO, msg, *args, **kwargs)
        
    def warning(self, msg: str, *args, **kwargs) -> None:
        self._log(logging.WARNING, msg, *args, **kwargs)
        
    def error(self, msg: str, *args, **kwargs) -> None:
        self._log(logging.ERROR, msg, *args, **kwargs)
        
    def critical(self, msg: str, *args, **kwargs) -> None:
        self._log(logging.CRITICAL, msg, *args, **kwargs)
        
    def track_performance(self, operation: str, level: int = logging.INFO, 
                         extra_fields: Optional[Dict[str, Any]] = None) -> PerformanceTracker:
        return PerformanceTracker(self.logger, operation, level, extra_fields)
        
    def timed(self, operation: str, level: int = logging.INFO) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.track_performance(operation, level):
                    return func(*args, **kwargs)
            return cast(F, wrapper)
        return decorator
    
    def with_correlation_field(self, key: str, value: Any) -> None:
        current_data = correlation_data.get().copy()
        current_data[key] = value
        correlation_data.set(current_data)
    
    def with_correlation_data(self, **kwargs) -> None:
        current_data = correlation_data.get().copy()
        current_data.update(kwargs)
        correlation_data.set(current_data)
    
    def with_trace(self, trace_id_val: Optional[str] = None, span_id_val: Optional[str] = None) -> None:
        new_trace_id = trace_id_val or str(uuid.uuid4())
        new_span_id = span_id_val or str(uuid.uuid4())
        
        trace_id.set(new_trace_id)
        span_id.set(new_span_id)

_main_logger: Optional[ForensicLogger] = None
_agent_loggers: Set[str] = set()
_log_context_local = threading.local()

def setup_logging(name: str = "forensic", log_dir: str = "logs", level: int = logging.INFO,
                 structured: bool = True, use_json: bool = True) -> ForensicLogger:
    global _main_logger
    _main_logger = ForensicLogger(name, log_dir, level, structured, use_json)
    return _main_logger

def get_logger(name: Optional[str] = None) -> StructuredLogger:
    global _main_logger
    
    if _main_logger is None:
        console_logger = logging.getLogger(name if name else "default")
        if not console_logger.handlers:
            console_logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            console_logger.addHandler(handler)
        return StructuredLogger(console_logger)
    
    if name is None:
        return _main_logger.get_logger()
        
    if name not in _agent_loggers:
        _agent_loggers.add(name)
        
    return _main_logger.get_logger(name)

def start_trace(name: Optional[str] = None) -> str:
    new_trace_id = str(uuid.uuid4())
    new_span_id = str(uuid.uuid4())
    
    trace_id.set(new_trace_id)
    span_id.set(new_span_id)
    
    if name:
        logger = get_logger(name)
        logger.info(f"Started trace", extra_fields={"event": "trace_start"})
    
    return new_trace_id

def set_correlation_id(correlation_id: str, name: Optional[str] = None) -> None:
    with_correlation_data(correlation_id=correlation_id)
    
    if name:
        logger = get_logger(name)
        logger.info(f"Set correlation ID", extra_fields={"correlation_id": correlation_id})

def with_correlation_data(**kwargs) -> None:
    current_data = correlation_data.get().copy()
    current_data.update(kwargs)
    correlation_data.set(current_data)

def get_current_trace_id() -> str:
    return trace_id.get()

def get_current_span_id() -> str:
    return span_id.get()

def get_correlation_data() -> Dict[str, Any]:
    return correlation_data.get()