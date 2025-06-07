"""
Exception classes for the exgaltoolkit.
"""

class ExgalToolkitError(Exception):
    """Base exception for exgaltoolkit."""
    pass

class ConfigurationError(ExgalToolkitError):
    """Configuration validation errors."""
    pass

class SimulationError(ExgalToolkitError):
    """Simulation execution errors."""
    pass

class GridError(ExgalToolkitError):
    """Grid operation errors.""" 
    pass
