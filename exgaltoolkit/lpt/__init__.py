"""
LPT module with backward compatibility.

This module preserves the original Cube interface while providing optional access
to the new refactored LPT services.
"""

# Original modules
from . import cube
from . import multihost_rfft

# Original classes (preserved for backward compatibility)
from .cube import Cube

# For users who want to try the new architecture, provide easy access
from ..services.lpt_service import LPTCalculator
from ..services.grid_service import GridManager, FFTProcessor

__all__ = [
    # Legacy classes
    'Cube',
    
    # Legacy modules  
    'cube',
    'multihost_rfft',
    
    # New architecture (optional usage)
    'LPTCalculator',
    'GridManager',
    'FFTProcessor',
]

