"""
ExGalToolkit: A Python-based, distributed, JAX-accelerated toolkit for 
generating cosmological initial conditions.

Clean Modular API:
- exgaltoolkit.initial_conditions: Main IC generation interface (ICGenerator, ICWriter)
- exgaltoolkit.cosmology: Cosmological parameters and services (CosmologyService, CosmologicalParameters)
- exgaltoolkit.grid: Grid operations and LPT calculations (GridOperations, LPTCalculator)
- exgaltoolkit.util: Core utilities (JAX, MPI, logging)
- exgaltoolkit.mathutil: Mathematical utilities and random number generation
- exgaltoolkit.data: Data files and resources

Legacy Modules (for compatibility):
- exgaltoolkit.lpt: Legacy LPT implementation
- exgaltoolkit.mockgen: Legacy mock generation utilities
"""

# Import main API components
try:
    from .initial_conditions import ICGenerator, ICWriter
    from .cosmology import CosmologyService, CosmologicalParameters
    from .grid import GridOperations, LPTCalculator
    API_AVAILABLE = True
except ImportError:
    # API not available
    ICGenerator = None
    ICWriter = None
    CosmologyService = None
    CosmologicalParameters = None
    GridOperations = None
    LPTCalculator = None
    API_AVAILABLE = False

__version__ = "2.0.0"

__all__ = []

# Add API exports if available
if API_AVAILABLE:
    __all__.extend([
        'ICGenerator', 
        'ICWriter',
        'CosmologyService',
        'CosmologicalParameters',
        'GridOperations', 
        'LPTCalculator'
    ])