"""
ExGalToolkit: A toolkit for extragalactic simulations.

This package provides both a modern, refactored architecture and backward-compatible
legacy interfaces for existing code.

New Architecture:
- exgaltoolkit.api: High-level simulation interfaces
- exgaltoolkit.core: Core configuration and data models
- exgaltoolkit.services: Modular business logic services  
- exgaltoolkit.workflow: Step-based workflow execution

Legacy Compatibility:
- exgaltoolkit.mockgen: Original mock generation module
- exgaltoolkit.lpt: Original LPT module
- All original classes and interfaces preserved
"""

# Try to import new API, fall back gracefully if not available
try:
    from .api import SimulationFactory, MockGenerationSimulation
    NEW_API_AVAILABLE = True
except ImportError:
    # New API not available, define placeholder variables
    SimulationFactory = None
    MockGenerationSimulation = None
    NEW_API_AVAILABLE = False

# Try to import core components, fall back gracefully if not available
try:
    from .core.config import (
        SimulationConfig,
        CosmologicalParameters, 
        PowerSpectrum,
        GridConfiguration,
        SimulationParameters,
        ParallelConfig,
        OutputConfig
    )
    from .core.data_models import (
        SimulationResults,
        ParticleData,
        LPTDisplacements,
        GridData
    )
    CORE_AVAILABLE = True
except ImportError:
    # Core not available, define placeholder variables
    SimulationConfig = None
    CosmologicalParameters = None
    PowerSpectrum = None
    GridConfiguration = None
    SimulationParameters = None
    ParallelConfig = None
    OutputConfig = None
    SimulationResults = None
    ParticleData = None
    LPTDisplacements = None
    GridData = None
    CORE_AVAILABLE = False

# Try to import legacy compatibility wrappers, fall back to original modules
try:
    from .legacy import Sky, Cube, CosmologyInterface, ICs
    LEGACY_WRAPPERS_AVAILABLE = True
except ImportError:
    # Legacy wrappers not available, import original classes
    try:
        from .mockgen import Sky, CosmologyInterface, ICs
        from .lpt import Cube
        LEGACY_WRAPPERS_AVAILABLE = False
    except ImportError:
        # Even original classes not available
        Sky = None
        Cube = None
        CosmologyInterface = None
        ICs = None
        LEGACY_WRAPPERS_AVAILABLE = False

# For backward compatibility, also expose legacy modules
from . import mockgen
from . import lpt

__version__ = "2.0.0"

__all__ = []

# Add new API exports if available
if NEW_API_AVAILABLE:
    __all__.extend(['SimulationFactory', 'MockGenerationSimulation'])

# Add core exports if available  
if CORE_AVAILABLE:
    __all__.extend([
        'SimulationConfig',
        'CosmologicalParameters',
        'PowerSpectrum', 
        'GridConfiguration',
        'SimulationParameters',
        'ParallelConfig',
        'OutputConfig',
        'SimulationResults',
        'ParticleData',
        'LPTDisplacements',
        'GridData',
    ])

# Add legacy exports if available
if Sky is not None:
    __all__.append('Sky')
if Cube is not None:
    __all__.append('Cube')
if CosmologyInterface is not None:
    __all__.append('CosmologyInterface')
if ICs is not None:
    __all__.append('ICs')

# Always try to expose legacy modules for backward compatibility
__all__.extend(['mockgen', 'lpt'])