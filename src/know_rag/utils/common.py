"""
Common utilities and functions used across the Know-RAG.

This module provides centralized implementations of commonly used functions
to avoid code duplication and ensure consistency across the codebase.
"""

import functools
import hashlib
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar("T")


def safe_execute(
    func: Callable[..., T],
    *args,
    default: Optional[T] = None,
    log_errors: bool = True,
    reraise: bool = False,
    error_prefix: str = "Safe execution failed",
    **kwargs,
) -> Union[T, Optional[T]]:
    """
    Safely execute a function with comprehensive error handling.

    This is the centralized implementation that should be used throughout
    the codebase for consistent error handling.

    Args:
        func: Function to execute safely
        *args: Positional arguments for the function
        default: Default value to return on error (default: None)
        log_errors: Whether to log errors (default: True)
        reraise: Whether to re-raise the exception after logging (default: False)
        error_prefix: Prefix for error messages (default: "Safe execution failed")
        **kwargs: Keyword arguments for the function

    Returns:
        Function result on success, default value on error

    Raises:
        Original exception if reraise=True
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            func_name = getattr(func, "__name__", str(func))
            logger.error(f"{error_prefix} in {func_name}: {e}", exc_info=True)

        if reraise:
            raise

        return default


def format_exception_context(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    include_traceback: bool = False,
) -> str:
    """
    Format exception with additional context for better debugging.

    Args:
        exception: The exception to format
        context: Additional context information
        include_traceback: Whether to include full traceback

    Returns:
        Formatted exception string
    """
    parts = [f"{type(exception).__name__}: {str(exception)}"]

    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        parts.append(f"Context: {context_str}")

    if include_traceback:
        import traceback

        parts.append(f"Traceback:\n{traceback.format_exc()}")

    return " | ".join(parts)


def validate_file_path(path: Union[str, Path], base_dir: Union[str, Path]) -> bool:
    """
    Validate that a file path is safe and within the allowed base directory.
    Prevents path traversal attacks.

    Args:
        path: The path to validate
        base_dir: The allowed base directory

    Returns:
        True if path is safe, False otherwise
    """
    try:
        # Convert to absolute paths
        abs_path = os.path.abspath(path)
        abs_base = os.path.abspath(base_dir)

        # Use os.path.commonpath for secure comparison
        common_path = os.path.commonpath([abs_path, abs_base])
        return common_path == abs_base
    except (ValueError, TypeError) as e:
        logger.warning(f"Path validation failed: {e}")
        return False


def sanitize_input(
    text: str,
    max_length: int = 10000,
    allowed_chars: Optional[str] = None,
    remove_html: bool = True,
) -> str:
    """
    Sanitize user input to prevent injection attacks and ensure safety.

    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length (default: 10000)
        allowed_chars: Regex pattern of allowed characters (default: alphanumeric + common punctuation)
        remove_html: Whether to remove HTML tags (default: True)

    Returns:
        Sanitized text
    """
    if not text:
        return ""

    # Truncate to max length
    text = text[:max_length]

    # Remove HTML tags if requested
    if remove_html:
        text = re.sub(r"<[^>]+>", "", text)

    # Remove potentially dangerous characters
    if allowed_chars is None:
        # Allow alphanumeric, spaces, and common punctuation
        text = re.sub(r"[^\w\s\-.,!?\'\"()]", "", text)
    else:
        # Use custom allowed characters pattern
        text = re.sub(f"[^{allowed_chars}]", "", text)

    # Normalize whitespace
    text = " ".join(text.split())

    return text.strip()


def create_cache_key(
    *args, prefix: str = "", include_timestamp: bool = False, hash_length: int = 16
) -> str:
    """
    Create a consistent cache key from multiple inputs.

    Args:
        *args: Values to include in the cache key
        prefix: Optional prefix for the key
        include_timestamp: Whether to include timestamp (makes key unique)
        hash_length: Length of the hash portion (default: 16)

    Returns:
        Cache key string
    """
    # Combine all arguments into a single string
    combined = "|".join(str(arg) for arg in args if arg is not None)

    # Add timestamp if requested
    if include_timestamp:
        combined += f"|{datetime.utcnow().isoformat()}"

    # Create hash
    hash_obj = hashlib.sha256(combined.encode("utf-8"))
    hash_str = hash_obj.hexdigest()[:hash_length]

    # Combine prefix and hash
    if prefix:
        return f"{prefix}:{hash_str}"
    return hash_str


def retry_with_backoff(
    func: Callable[..., T],
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    exceptions: Tuple[type, ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[..., T]:
    """
    Decorator to retry a function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for backoff time
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback called on each retry with (exception, attempt_number)

    Returns:
        Decorated function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                last_exception = e

                if attempt < max_retries:
                    # Calculate backoff time
                    backoff_time = backoff_factor**attempt

                    # Call retry callback if provided
                    if on_retry:
                        on_retry(e, attempt + 1)

                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {type(e).__name__}: {e}. "
                        f"Waiting {backoff_time}s..."
                    )

                    import time

                    time.sleep(backoff_time)
                else:
                    # Final attempt failed
                    logger.error(
                        f"All {max_retries} retries failed for {func.__name__}: {e}"
                    )

        # Re-raise the last exception
        if last_exception:
            raise last_exception

    return wrapper


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get file size in megabytes.

    Args:
        file_path: Path to the file

    Returns:
        File size in MB, or 0.0 if file doesn't exist
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except (OSError, IOError):
        return 0.0


def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory: Path to the directory

    Returns:
        Path object for the directory

    Raises:
        OSError: If directory cannot be created
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length with suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add when truncating

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text

    truncate_at = max_length - len(suffix)
    return text[:truncate_at] + suffix


def merge_dicts(
    base: Dict[str, Any], *updates: Dict[str, Any], deep: bool = True
) -> Dict[str, Any]:
    """
    Merge multiple dictionaries, with later dicts overriding earlier ones.

    Args:
        base: Base dictionary
        *updates: Dictionaries to merge into base
        deep: Whether to perform deep merge (recursive)

    Returns:
        Merged dictionary (new instance)
    """
    result = base.copy()

    for update in updates:
        if not update:
            continue

        if deep:
            for key, value in update.items():
                if (
                    key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)
                ):
                    result[key] = merge_dicts(result[key], value, deep=True)
                else:
                    result[key] = value
        else:
            result.update(update)

    return result


def batch_process(
    items: List[T],
    process_func: Callable[[List[T]], Any],
    batch_size: int = 100,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Any]:
    """
    Process items in batches for better performance and memory usage.

    Args:
        items: List of items to process
        process_func: Function that processes a batch of items
        batch_size: Size of each batch
        progress_callback: Optional callback with (processed_count, total_count)

    Returns:
        List of results from all batches
    """
    results = []
    total_items = len(items)

    for i in range(0, total_items, batch_size):
        batch = items[i : i + batch_size]
        batch_result = process_func(batch)
        results.extend(
            batch_result if isinstance(batch_result, list) else [batch_result]
        )

        if progress_callback:
            progress_callback(min(i + batch_size, total_items), total_items)

    return results


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text (remove extra spaces, tabs, newlines).

    Args:
        text: Text to normalize

    Returns:
        Normalized text
    """
    # Replace all whitespace sequences with single space
    normalized = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    return normalized.strip()


def get_timestamp(format_string: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get current timestamp as formatted string.

    Args:
        format_string: strftime format string

    Returns:
        Formatted timestamp
    """
    return datetime.utcnow().strftime(format_string)


# Export all utilities
__all__ = [
    "safe_execute",
    "format_exception_context",
    "validate_file_path",
    "sanitize_input",
    "create_cache_key",
    "retry_with_backoff",
    "get_file_size_mb",
    "ensure_directory_exists",
    "truncate_text",
    "merge_dicts",
    "batch_process",
    "normalize_whitespace",
    "get_timestamp",
]
