# pyPRT/_compat.py
"""Python Version compatibility support"""

import sys
import warnings
import functools

PYTHON_VERSION = sys.version_info

# deprecated decorators
if PYTHON_VERSION >= (3, 9) and PYTHON_VERSION < (3, 12):
    from functools import deprecated
else:
    # Custom implementation
    def deprecated(arg=None):
        if callable(arg):
            func = arg
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                warnings.warn(
                    f"Function '{func.__name__}' is deprecated",
                    FutureWarning,
                    stacklevel=2
                )
                return func(*args, **kwargs)
            return wrapper
        else:
            message = arg or "Function is deprecated"
            def decorator(func):
                @functools.wraps(func)
                def wrapper(*args, **kwargs):
                    warnings.warn(message, FutureWarning, stacklevel=2)
                    return func(*args, **kwargs)
                return wrapper
            return decorator

# Other compatibility functions...
__all__ = ['deprecated']