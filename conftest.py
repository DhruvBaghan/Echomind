"""
Pytest configuration for EchoMind tests.
Handles environment setup and fixtures.
"""

import sys
import os
import warnings

# Monkey-patch to suppress SQLAlchemy typing errors on Python 3.13
try:
    import sqlalchemy.util.langhelpers as langhelpers
    original_init_subclass = langhelpers.__init_subclass__
    
    def patched_init_subclass(cls, *args, **kwargs):
        try:
            return original_init_subclass(cls, *args, **kwargs)
        except AssertionError:
            # Ignore typing-related assertion errors
            pass
    
    langhelpers.__init_subclass__ = patched_init_subclass
except Exception:
    pass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set testing environment
os.environ['FLASK_ENV'] = 'testing'
os.environ['TESTING'] = 'True'
