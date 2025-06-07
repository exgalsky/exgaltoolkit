"""
ExGalToolkit: A Python-based, distributed, JAX-accelerated toolkit for 
generating cosmological initial conditions.

Simplified API:
- exgaltoolkit.initial_conditions: Main IC generation interface
- exgaltoolkit.cosmology: Cosmological parameters and growth factors
- exgaltoolkit.grid: Grid operations and LPT calculations

Legacy Compatibility:
- exgaltoolkit.mockgen: Original mock generation module
- exgaltoolkit.lpt: Original LPT module  
- All original classes and interfaces preserved
"""

# Simplified API - Import new modular components
try:
    from .initial_conditions import ICGenerator, ICWriter
    from .cosmology import CosmologyService, CosmologicalParameters
    from .grid import GridOperations, LPTCalculator
    SIMPLIFIED_API_AVAILABLE = True
except ImportError:
    # Simplified API not available
    ICGenerator = None
    ICWriter = None
    CosmologyService = None
    CosmologicalParameters = None
    GridOperations = None
    LPTCalculator = None
    SIMPLIFIED_API_AVAILABLE = False

# Legacy compatibility - import original classes
try:
    from .mockgen import Sky, CosmologyInterface, ICs
    from .lpt import Cube
    LEGACY_AVAILABLE = True
except ImportError:
    # Even original classes not available
    Sky = None
    Cube = None
    CosmologyInterface = None
    ICs = None
    LEGACY_AVAILABLE = False

# For backward compatibility, also expose legacy modules
from . import mockgen
from . import lpt

__version__ = "2.0.0"

__all__ = []

# Add simplified API exports if available
if SIMPLIFIED_API_AVAILABLE:
    __all__.extend([
        'ICGenerator', 
        'ICWriter',
        'CosmologyService',
        'CosmologicalParameters',
        'GridOperations', 
        'LPTCalculator'
    ])

# Add legacy exports if available
if LEGACY_AVAILABLE:
    __all__.extend(['Sky', 'Cube', 'CosmologyInterface', 'ICs'])

# Always expose legacy modules for backward compatibility
__all__.extend(['mockgen', 'lpt'])