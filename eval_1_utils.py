import resource
import os
import time

def increase_file_limit():
    """Increase the system's file descriptor limit"""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f"Successfully increased file descriptor limit to {hard}")
    except ValueError as e:
        print(f"Warning: Could not increase file descriptor limit: {e}")

def measure_timing(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    return wrapper

def debug_print(message, data=None):
    """Helper function for debug printing"""
    print(f"\nDEBUG: {message}")
    if data is not None:
        print(f"Data: {data}") 