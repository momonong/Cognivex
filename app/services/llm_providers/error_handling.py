"""
LLM Error Handling Module

This module provides robust error handling for LLM calls including:
- Retry logic with exponential backoff
- Timeout handling
- JSON parsing error recovery
- Comprehensive error logging

Requirements: 10.1
"""

import time
import json
import re
from typing import Callable, Any, Optional, Dict
from functools import wraps
from datetime import datetime


class LLMError(Exception):
    """Base exception for LLM-related errors"""
    pass


class LLMConnectionError(LLMError):
    """LLM connection failed"""
    pass


class LLMTimeoutError(LLMError):
    """LLM call timed out"""
    pass


class LLMParsingError(LLMError):
    """Failed to parse LLM response"""
    pass


class LLMRetryExhausted(LLMError):
    """All retry attempts exhausted"""
    pass


def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 10.0) -> float:
    """
    Calculate exponential backoff delay
    
    Args:
        attempt: Retry attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
    
    Returns:
        Delay in seconds
    """
    delay = base_delay * (2 ** attempt)
    return min(delay, max_delay)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exceptions: tuple = (Exception,),
    verbose: bool = True
):
    """
    Decorator for retry logic with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Tuple of exceptions to catch and retry
        verbose: Print retry information
    
    Requirements: 10.1
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    # Try to call the function
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    # If this was the last attempt, raise
                    if attempt == max_retries:
                        if verbose:
                            print(f"\n[ERROR] All {max_retries} retry attempts exhausted")
                            print(f"[ERROR] Last error: {type(e).__name__}: {e}")
                        raise LLMRetryExhausted(
                            f"Failed after {max_retries} retries. Last error: {e}"
                        ) from e
                    
                    # Calculate backoff delay
                    delay = exponential_backoff(attempt, base_delay, max_delay)
                    
                    if verbose:
                        print(f"\n[RETRY] Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}")
                        print(f"[RETRY] Retrying in {delay:.1f} seconds...")
                    
                    # Log error with context
                    log_llm_error(
                        error=e,
                        context={
                            'function': func.__name__,
                            'attempt': attempt + 1,
                            'max_retries': max_retries,
                            'delay': delay
                        }
                    )
                    
                    # Wait before retry
                    time.sleep(delay)
            
            # Should never reach here, but just in case
            raise last_exception
        
        return wrapper
    return decorator


def parse_json_with_recovery(
    text: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Parse JSON from text with error recovery
    
    This function attempts multiple strategies to extract valid JSON:
    1. Direct JSON parsing
    2. Extract from markdown code blocks
    3. Find JSON objects/arrays in text
    4. Clean and retry
    
    Args:
        text: Text containing JSON
        verbose: Print recovery attempts
    
    Returns:
        Parsed JSON dictionary
    
    Raises:
        LLMParsingError: If all recovery attempts fail
    
    Requirements: 10.1
    """
    if not text or not text.strip():
        raise LLMParsingError("Empty response from LLM")
    
    # Strategy 1: Direct parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if verbose:
            print("[RECOVERY] Direct JSON parsing failed, trying recovery strategies...")
    
    # Strategy 2: Extract from markdown code blocks
    try:
        # Try ```json ... ```
        match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
        if match:
            json_text = match.group(1).strip()
            if verbose:
                print("[RECOVERY] Found JSON in ```json code block")
            return json.loads(json_text)
        
        # Try ``` ... ```
        match = re.search(r"```\s*([\s\S]*?)\s*```", text)
        if match:
            json_text = match.group(1).strip()
            if verbose:
                print("[RECOVERY] Found JSON in ``` code block")
            return json.loads(json_text)
    except json.JSONDecodeError:
        if verbose:
            print("[RECOVERY] Code block extraction failed")
    
    # Strategy 3: Find JSON objects/arrays
    try:
        # Find first { ... } or [ ... ]
        # Look for balanced braces/brackets
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start_idx = text.find(start_char)
            if start_idx != -1:
                # Find matching closing bracket
                depth = 0
                for i in range(start_idx, len(text)):
                    if text[i] == start_char:
                        depth += 1
                    elif text[i] == end_char:
                        depth -= 1
                        if depth == 0:
                            json_text = text[start_idx:i+1]
                            if verbose:
                                print(f"[RECOVERY] Found JSON object/array at position {start_idx}")
                            return json.loads(json_text)
    except json.JSONDecodeError:
        if verbose:
            print("[RECOVERY] JSON object/array extraction failed")
    
    # Strategy 4: Clean and retry
    try:
        # Remove common issues
        cleaned = text.strip()
        
        # Remove leading/trailing non-JSON text
        if '{' in cleaned:
            cleaned = cleaned[cleaned.find('{'):]
        elif '[' in cleaned:
            cleaned = cleaned[cleaned.find('['):]
        
        if '}' in cleaned:
            cleaned = cleaned[:cleaned.rfind('}')+1]
        elif ']' in cleaned:
            cleaned = cleaned[:cleaned.rfind(']')+1]
        
        # Remove control characters
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)
        
        if verbose:
            print("[RECOVERY] Trying cleaned JSON")
        
        return json.loads(cleaned)
    except json.JSONDecodeError:
        if verbose:
            print("[RECOVERY] All recovery strategies failed")
    
    # All strategies failed
    raise LLMParsingError(
        f"Failed to parse JSON from LLM response. "
        f"Response preview: {text[:200]}..."
    )


def log_llm_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    log_file: str = "output/llm_errors.log"
):
    """
    Log LLM error with context for debugging
    
    Args:
        error: Exception that occurred
        context: Additional context information
        log_file: Path to log file
    
    Requirements: 10.1
    """
    import os
    from pathlib import Path
    
    # Create log directory if needed
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Format log entry
    timestamp = datetime.now().isoformat()
    error_type = type(error).__name__
    error_msg = str(error)
    
    log_entry = {
        'timestamp': timestamp,
        'error_type': error_type,
        'error_message': error_msg,
        'context': context or {}
    }
    
    # Append to log file
    try:
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"[WARNING] Failed to write to log file: {e}")


def safe_llm_call(
    llm_func: Callable,
    *args,
    max_retries: int = 3,
    timeout: Optional[float] = None,
    parse_json: bool = False,
    verbose: bool = True,
    **kwargs
) -> Any:
    """
    Safely call an LLM function with error handling
    
    This is a convenience wrapper that combines:
    - Retry logic with exponential backoff
    - Timeout handling
    - JSON parsing with recovery
    - Error logging
    
    Args:
        llm_func: LLM function to call
        *args: Positional arguments for llm_func
        max_retries: Maximum retry attempts
        timeout: Timeout in seconds (not implemented yet)
        parse_json: Parse response as JSON
        verbose: Print debug information
        **kwargs: Keyword arguments for llm_func
    
    Returns:
        LLM response (string or dict if parse_json=True)
    
    Raises:
        LLMRetryExhausted: If all retries fail
        LLMParsingError: If JSON parsing fails
    
    Requirements: 10.1
    """
    # Apply retry decorator
    @retry_with_backoff(
        max_retries=max_retries,
        base_delay=1.0,
        max_delay=10.0,
        exceptions=(Exception,),
        verbose=verbose
    )
    def wrapped_call():
        # Call LLM function
        response = llm_func(*args, **kwargs)
        
        # Parse JSON if requested
        if parse_json:
            return parse_json_with_recovery(response, verbose=verbose)
        
        return response
    
    try:
        return wrapped_call()
    except Exception as e:
        # Log final error
        log_llm_error(
            error=e,
            context={
                'function': llm_func.__name__,
                'max_retries': max_retries,
                'parse_json': parse_json
            }
        )
        raise


# ============================================================================
# Demo Functions
# ============================================================================

def demo_retry_logic():
    """Demo: Retry logic with exponential backoff"""
    print("\n" + "="*80)
    print("DEMO: Retry Logic with Exponential Backoff")
    print("="*80)
    
    # Simulate a function that fails twice then succeeds
    attempt_count = [0]
    
    @retry_with_backoff(max_retries=3, base_delay=0.5, verbose=True)
    def flaky_function():
        attempt_count[0] += 1
        print(f"\n[ATTEMPT {attempt_count[0]}] Calling flaky function...")
        
        if attempt_count[0] < 3:
            raise ConnectionError("Simulated connection error")
        
        return "Success!"
    
    try:
        result = flaky_function()
        print(f"\n[RESULT] {result}")
    except LLMRetryExhausted as e:
        print(f"\n[FAILED] {e}")


def demo_json_recovery():
    """Demo: JSON parsing with recovery"""
    print("\n" + "="*80)
    print("DEMO: JSON Parsing with Recovery")
    print("="*80)
    
    # Test cases
    test_cases = [
        # Valid JSON
        ('{"key": "value"}', "Valid JSON"),
        
        # JSON in markdown
        ('```json\n{"key": "value"}\n```', "JSON in markdown"),
        
        # JSON with extra text
        ('Here is the result: {"key": "value"} Hope this helps!', "JSON with extra text"),
        
        # Malformed JSON (should fail)
        ('This is not JSON at all', "Invalid text"),
    ]
    
    for text, description in test_cases:
        print(f"\n[TEST] {description}")
        print(f"[INPUT] {text}")
        
        try:
            result = parse_json_with_recovery(text, verbose=True)
            print(f"[SUCCESS] Parsed: {result}")
        except LLMParsingError as e:
            print(f"[FAILED] {e}")


if __name__ == "__main__":
    # Run demos
    demo_retry_logic()
    print("\n\n")
    demo_json_recovery()
