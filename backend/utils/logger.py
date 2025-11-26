# ============================================
# EchoMind - Logging Utilities
# ============================================

"""
Logging configuration and utilities for EchoMind.

Provides:
    - Structured logging setup
    - Request/response logging
    - Error logging with context
    - Performance logging
"""

import sys
import os
from datetime import datetime
from typing import Any, Dict, Optional
from functools import wraps
import time
import traceback

from loguru import logger as loguru_logger

from backend.config import Config


# Remove default logger
loguru_logger.remove()

# Create custom logger instance
logger = loguru_logger


def setup_logger(app=None) -> None:
    """
    Setup logging configuration for the application.
    
    Args:
        app: Flask application instance (optional)
    """
    # Get configuration
    log_level = Config.LOG_LEVEL if hasattr(Config, "LOG_LEVEL") else "INFO"
    log_dir = Config.LOG_DIR if hasattr(Config, "LOG_DIR") else "logs"
    
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Console handler - colorized output
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level=log_level,
        colorize=True,
    )
    
    # File handler - general logs
    logger.add(
        os.path.join(log_dir, "echomind.log"),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level=log_level,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
    )
    
    # Error file handler - errors only
    logger.add(
        os.path.join(log_dir, "errors.log"),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}\n{exception}",
        level="ERROR",
        rotation="10 MB",
        retention="90 days",
        compression="zip",
        backtrace=True,
        diagnose=True,
    )
    
    # API requests file handler
    logger.add(
        os.path.join(log_dir, "api_requests.log"),
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        level="INFO",
        rotation="50 MB",
        retention="14 days",
        filter=lambda record: record["extra"].get("request_log", False),
    )
    
    # Performance file handler
    logger.add(
        os.path.join(log_dir, "performance.log"),
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        level="INFO",
        rotation="20 MB",
        retention="14 days",
        filter=lambda record: record["extra"].get("performance_log", False),
    )
    
    logger.info("Logger initialized")


def log_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    user_id: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log an API request.
    
    Args:
        method: HTTP method
        path: Request path
        status_code: Response status code
        duration_ms: Request duration in milliseconds
        user_id: Optional user ID
        extra: Optional extra data to log
    """
    log_data = {
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 2),
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    if extra:
        log_data.update(extra)
    
    # Format log message
    status_indicator = "✓" if status_code < 400 else "✗"
    message = f"{status_indicator} {method} {path} -> {status_code} ({duration_ms:.2f}ms)"
    
    if user_id:
        message += f" [user:{user_id}]"
    
    logger.bind(request_log=True).info(message)


def log_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    user_id: Optional[int] = None
) -> None:
    """
    Log an error with context.
    
    Args:
        error: Exception object
        context: Optional context dictionary
        user_id: Optional user ID
    """
    error_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "traceback": traceback.format_exc(),
    }
    
    if context:
        error_data["context"] = context
    
    logger.error(f"Error: {type(error).__name__}: {str(error)}")
    logger.debug(f"Error context: {error_data}")


def log_performance(
    operation: str,
    duration_ms: float,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log performance metrics.
    
    Args:
        operation: Name of the operation
        duration_ms: Duration in milliseconds
        details: Optional additional details
    """
    message = f"PERF | {operation} | {duration_ms:.2f}ms"
    
    if details:
        details_str = " | ".join(f"{k}={v}" for k, v in details.items())
        message += f" | {details_str}"
    
    logger.bind(performance_log=True).info(message)


def log_prediction(
    resource_type: str,
    periods: int,
    duration_ms: float,
    success: bool,
    user_id: Optional[int] = None
) -> None:
    """
    Log prediction operation.
    
    Args:
        resource_type: Type of resource (electricity/water)
        periods: Number of prediction periods
        duration_ms: Duration in milliseconds
        success: Whether prediction was successful
        user_id: Optional user ID
    """
    status = "SUCCESS" if success else "FAILED"
    message = f"Prediction | {resource_type} | {periods} periods | {status} | {duration_ms:.2f}ms"
    
    if user_id:
        message += f" | user:{user_id}"
    
    if success:
        logger.info(message)
    else:
        logger.warning(message)


def log_database(
    operation: str,
    table: str,
    duration_ms: Optional[float] = None,
    rows_affected: Optional[int] = None
) -> None:
    """
    Log database operation.
    
    Args:
        operation: Type of operation (SELECT, INSERT, UPDATE, DELETE)
        table: Table name
        duration_ms: Optional duration in milliseconds
        rows_affected: Optional number of rows affected
    """
    message = f"DB | {operation} | {table}"
    
    if rows_affected is not None:
        message += f" | {rows_affected} rows"
    
    if duration_ms is not None:
        message += f" | {duration_ms:.2f}ms"
    
    logger.debug(message)


def timed(operation_name: Optional[str] = None):
    """
    Decorator to time and log function execution.
    
    Args:
        operation_name: Optional name for the operation (defaults to function name)
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                log_performance(name, duration_ms, {"status": "success"})
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                log_performance(name, duration_ms, {"status": "error", "error": str(e)})
                raise
        
        return wrapper
    return decorator


def async_timed(operation_name: Optional[str] = None):
    """
    Decorator to time and log async function execution.
    
    Args:
        operation_name: Optional name for the operation
        
    Returns:
        Decorated async function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            name = operation_name or func.__name__
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                log_performance(name, duration_ms, {"status": "success"})
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                log_performance(name, duration_ms, {"status": "error", "error": str(e)})
                raise
        
        return wrapper
    return decorator


class LogContext:
    """
    Context manager for logging with additional context.
    
    Usage:
        with LogContext(operation="data_processing", user_id=123):
            # operations here will have context in logs
            process_data()
    """
    
    def __init__(self, **context):
        """
        Initialize log context.
        
        Args:
            **context: Context key-value pairs
        """
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        operation = self.context.get("operation", "operation")
        logger.debug(f"Starting {operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        operation = self.context.get("operation", "operation")
        
        if exc_type is None:
            logger.debug(f"Completed {operation} in {duration_ms:.2f}ms")
        else:
            logger.error(f"Failed {operation} after {duration_ms:.2f}ms: {exc_val}")
        
        return False  # Don't suppress exceptions


def get_request_logger(request_id: str):
    """
    Get a logger bound with request ID.
    
    Args:
        request_id: Unique request identifier
        
    Returns:
        Bound logger instance
    """
    return logger.bind(request_id=request_id)


def log_startup(app_name: str, version: str, environment: str) -> None:
    """
    Log application startup.
    
    Args:
        app_name: Name of the application
        version: Application version
        environment: Environment name
    """
    logger.info("=" * 60)
    logger.info(f"Starting {app_name} v{version}")
    logger.info(f"Environment: {environment}")
    logger.info("=" * 60)


def log_shutdown() -> None:
    """Log application shutdown."""
    logger.info("=" * 60)
    logger.info("Application shutting down")
    logger.info("=" * 60)


# Request ID middleware helper
def generate_request_id() -> str:
    """
    Generate a unique request ID.
    
    Returns:
        Unique request identifier
    """
    import uuid
    return str(uuid.uuid4())[:8]


class RequestLogMiddleware:
    """
    Middleware for logging HTTP requests.
    
    Usage with Flask:
        app.wsgi_app = RequestLogMiddleware(app.wsgi_app)
    """
    
    def __init__(self, app):
        self.app = app
    
    def __call__(self, environ, start_response):
        request_id = generate_request_id()
        start_time = time.time()
        
        method = environ.get("REQUEST_METHOD", "")
        path = environ.get("PATH_INFO", "")
        
        def custom_start_response(status, headers, exc_info=None):
            duration_ms = (time.time() - start_time) * 1000
            status_code = int(status.split()[0])
            
            log_request(
                method=method,
                path=path,
                status_code=status_code,
                duration_ms=duration_ms,
                extra={"request_id": request_id}
            )
            
            # Add request ID to response headers
            headers.append(("X-Request-ID", request_id))
            return start_response(status, headers, exc_info)
        
        return self.app(environ, custom_start_response)