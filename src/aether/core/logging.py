"""
AETHER Structured Logging System

Production-grade logging with:
- Structured JSON output for log aggregation
- Context propagation (trace IDs, spans)
- Performance timing decorators
- Operation tracking
- Multiple output handlers
"""

from __future__ import annotations

import functools
import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

# Context variables for distributed tracing
_trace_id: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
_span_id: ContextVar[Optional[str]] = ContextVar("span_id", default=None)
_operation: ContextVar[Optional[str]] = ContextVar("operation", default=None)
_component: ContextVar[Optional[str]] = ContextVar("component", default=None)


class LogLevel(Enum):
    """Log levels matching standard Python logging."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogContext:
    """
    Context manager for log context propagation.

    Usage:
        with LogContext(trace_id="abc", operation="generate_track"):
            logger.info("Processing...")  # Will include trace_id and operation
    """

    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    _tokens: List[Any] = field(default_factory=list, repr=False)

    def __enter__(self) -> "LogContext":
        # Generate IDs if not provided
        if self.trace_id is None:
            self.trace_id = _trace_id.get() or str(uuid.uuid4())[:8]

        if self.span_id is None:
            self.span_id = str(uuid.uuid4())[:8]

        # Set context variables
        self._tokens.append(_trace_id.set(self.trace_id))
        self._tokens.append(_span_id.set(self.span_id))

        if self.operation:
            self._tokens.append(_operation.set(self.operation))
        if self.component:
            self._tokens.append(_component.set(self.component))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Reset context variables
        for token in self._tokens:
            token.var.reset(token)
        self._tokens.clear()
        return False

    @classmethod
    def current(cls) -> Dict[str, Any]:
        """Get current context as dictionary."""
        return {
            "trace_id": _trace_id.get(),
            "span_id": _span_id.get(),
            "operation": _operation.get(),
            "component": _component.get(),
        }


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs logs as JSON lines for easy parsing by log aggregators.
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_context: bool = True,
        include_location: bool = True,
        indent: Optional[int] = None,
    ):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_context = include_context
        self.include_location = include_location
        self.indent = indent

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if self.include_timestamp:
            log_entry["timestamp"] = datetime.utcnow().isoformat() + "Z"

        if self.include_context:
            ctx = LogContext.current()
            if any(v is not None for v in ctx.values()):
                log_entry["context"] = {k: v for k, v in ctx.items() if v is not None}

        if self.include_location:
            log_entry["location"] = {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Include exception info
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "taskName",
            ):
                try:
                    json.dumps(value)  # Check if serializable
                    extra_fields[key] = value
                except (TypeError, ValueError):
                    extra_fields[key] = str(value)

        if extra_fields:
            log_entry["extra"] = extra_fields

        return json.dumps(log_entry, default=str, indent=self.indent)


class HumanFormatter(logging.Formatter):
    """
    Human-readable formatter for development.

    Includes colors for terminal output.
    """

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True, include_context: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stderr.isatty()
        self.include_context = include_context

    def format(self, record: logging.LogRecord) -> str:
        # Format timestamp
        timestamp = datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]

        # Format level with color
        level = record.levelname
        if self.use_colors:
            color = self.COLORS.get(level, "")
            level = f"{color}{level:8}{self.RESET}"
        else:
            level = f"{level:8}"

        # Format logger name (truncate if too long)
        logger_name = record.name
        if len(logger_name) > 25:
            logger_name = "..." + logger_name[-22:]

        # Build message
        parts = [f"{timestamp} {level} [{logger_name:25}] {record.getMessage()}"]

        # Add context
        if self.include_context:
            ctx = LogContext.current()
            ctx_parts = []
            if ctx.get("trace_id"):
                ctx_parts.append(f"trace={ctx['trace_id']}")
            if ctx.get("operation"):
                ctx_parts.append(f"op={ctx['operation']}")
            if ctx_parts:
                parts.append(f"  ({', '.join(ctx_parts)})")

        # Add exception
        if record.exc_info:
            parts.append("\n" + self.formatException(record.exc_info))

        return "".join(parts)


class AetherLogger(logging.Logger):
    """
    Extended logger with structured logging capabilities.

    Provides:
    - Automatic context injection
    - Performance timing
    - Operation tracking
    - Typed extra fields
    """

    def _log_with_context(
        self,
        level: int,
        msg: str,
        args,
        exc_info=None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Log with automatic context injection."""
        if extra is None:
            extra = {}

        # Inject current context
        ctx = LogContext.current()
        extra.update({k: v for k, v in ctx.items() if v is not None})

        # Add any additional kwargs to extra
        extra.update(kwargs)

        super()._log(level, msg, args, exc_info=exc_info, extra=extra)

    def debug(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.DEBUG, msg, args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.INFO, msg, args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.WARNING, msg, args, **kwargs)

    def error(self, msg: str, *args, exc_info=False, **kwargs):
        self._log_with_context(logging.ERROR, msg, args, exc_info=exc_info, **kwargs)

    def critical(self, msg: str, *args, exc_info=True, **kwargs):
        self._log_with_context(logging.CRITICAL, msg, args, exc_info=exc_info, **kwargs)

    def operation_start(self, operation: str, **details):
        """Log operation start."""
        self.info(f"Operation started: {operation}", operation=operation, **details)

    def operation_end(
        self,
        operation: str,
        success: bool = True,
        duration_ms: Optional[float] = None,
        **details,
    ):
        """Log operation end."""
        status = "completed" if success else "failed"
        msg = f"Operation {status}: {operation}"
        if duration_ms is not None:
            msg += f" ({duration_ms:.2f}ms)"
        self.info(msg, operation=operation, success=success, duration_ms=duration_ms, **details)

    def metric(self, name: str, value: Union[int, float], unit: Optional[str] = None, **tags):
        """Log a metric value."""
        self.info(
            f"Metric: {name}={value}{unit or ''}",
            metric_name=name,
            metric_value=value,
            metric_unit=unit,
            metric_tags=tags,
        )


# Logger cache
_loggers: Dict[str, AetherLogger] = {}


def get_logger(name: str = "aether") -> AetherLogger:
    """
    Get a logger instance.

    Args:
        name: Logger name, typically module path

    Returns:
        AetherLogger instance
    """
    if name not in _loggers:
        # Set custom logger class
        logging.setLoggerClass(AetherLogger)
        logger = logging.getLogger(name)
        if not isinstance(logger, AetherLogger):
            # Wrap existing logger
            logger.__class__ = AetherLogger
        _loggers[name] = logger
        logging.setLoggerClass(logging.Logger)  # Reset

    return _loggers[name]


# Configuration state
_configured = False


def configure_logging(
    level: Union[str, int, LogLevel] = LogLevel.INFO,
    format: str = "human",  # "human" or "json"
    output: str = "stderr",  # "stderr", "stdout", or file path
    include_timestamp: bool = True,
    include_context: bool = True,
    include_location: bool = False,
) -> None:
    """
    Configure the logging system.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Output format ("human" or "json")
        output: Output destination ("stderr", "stdout", or file path)
        include_timestamp: Include timestamp in logs
        include_context: Include trace context in logs
        include_location: Include file/line location in logs
    """
    global _configured

    # Convert level
    if isinstance(level, LogLevel):
        level = level.value
    elif isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Create handler
    if output == "stderr":
        handler = logging.StreamHandler(sys.stderr)
    elif output == "stdout":
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(output)

    # Create formatter
    if format == "json":
        formatter = StructuredFormatter(
            include_timestamp=include_timestamp,
            include_context=include_context,
            include_location=include_location,
        )
    else:
        formatter = HumanFormatter(
            use_colors=True,
            include_context=include_context,
        )

    handler.setFormatter(formatter)

    # Configure root aether logger
    root_logger = logging.getLogger("aether")
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Don't propagate to root logger
    root_logger.propagate = False

    _configured = True


# =============================================================================
# Decorators
# =============================================================================

F = TypeVar("F", bound=Callable[..., Any])


def log_operation(
    operation: Optional[str] = None,
    component: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to log operation execution.

    Args:
        operation: Operation name (defaults to function name)
        component: Component name
        log_args: Log function arguments
        log_result: Log function result

    Usage:
        @log_operation(component="pipeline")
        async def generate_track(...):
            ...
    """
    def decorator(func: F) -> F:
        op_name = operation or func.__name__
        comp = component or func.__module__.split(".")[-1]

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)

            with LogContext(operation=op_name, component=comp):
                start_time = time.perf_counter()

                extra = {}
                if log_args:
                    extra["args"] = str(args)[:200]
                    extra["kwargs"] = str(kwargs)[:200]

                logger.operation_start(op_name, **extra)

                try:
                    result = await func(*args, **kwargs)
                    duration_ms = (time.perf_counter() - start_time) * 1000

                    result_extra = {}
                    if log_result and result is not None:
                        result_extra["result"] = str(result)[:200]

                    logger.operation_end(op_name, success=True, duration_ms=duration_ms, **result_extra)
                    return result

                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    logger.operation_end(
                        op_name,
                        success=False,
                        duration_ms=duration_ms,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)

            with LogContext(operation=op_name, component=comp):
                start_time = time.perf_counter()

                extra = {}
                if log_args:
                    extra["args"] = str(args)[:200]
                    extra["kwargs"] = str(kwargs)[:200]

                logger.operation_start(op_name, **extra)

                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.perf_counter() - start_time) * 1000

                    result_extra = {}
                    if log_result and result is not None:
                        result_extra["result"] = str(result)[:200]

                    logger.operation_end(op_name, success=True, duration_ms=duration_ms, **result_extra)
                    return result

                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    logger.operation_end(
                        op_name,
                        success=False,
                        duration_ms=duration_ms,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


def log_performance(
    name: Optional[str] = None,
    threshold_ms: Optional[float] = None,
) -> Callable[[F], F]:
    """
    Decorator to log function performance.

    Args:
        name: Metric name (defaults to function name)
        threshold_ms: Only log if duration exceeds threshold

    Usage:
        @log_performance(threshold_ms=100)
        def expensive_operation():
            ...
    """
    def decorator(func: F) -> F:
        metric_name = name or f"{func.__module__}.{func.__name__}.duration_ms"

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                if threshold_ms is None or duration_ms >= threshold_ms:
                    logger = get_logger(func.__module__)
                    logger.metric(metric_name, duration_ms, "ms")

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                if threshold_ms is None or duration_ms >= threshold_ms:
                    logger = get_logger(func.__module__)
                    logger.metric(metric_name, duration_ms, "ms")

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


# =============================================================================
# Initialization
# =============================================================================

def _ensure_configured():
    """Ensure logging is configured with defaults."""
    global _configured
    if not _configured:
        configure_logging()


# Configure on import if not already done
_ensure_configured()
