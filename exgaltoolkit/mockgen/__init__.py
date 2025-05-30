"""
Mock generation module with backward compatibility.

This module preserves the original interface while providing optional access
to the new refactored implementation.
"""

# Original modules
from . import cosmo 
from . import ics 
from . import sky
from . import defaults

# Original classes (preserved for backward compatibility)
from .cosmo import CosmologyInterface
from .ics import ICs
from .sky import Sky

# For users who want to try the new architecture, provide easy access
from ..api import SimulationFactory, MockGenerationSimulation

__all__ = [
    # Legacy classes
    'Sky',
    'CosmologyInterface', 
    'ICs',
    
    # Legacy modules
    'cosmo',
    'ics', 
    'sky',
    'defaults',
    
    # New architecture (optional usage)
    'SimulationFactory',
    'MockGenerationSimulation',
]