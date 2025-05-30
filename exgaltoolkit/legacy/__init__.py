"""
Legacy compatibility layer providing backward-compatible interfaces.
"""
from .sky import Sky
from .cube import Cube
from .cosmo import CosmologyInterface
from .ics import ICs

__all__ = [
    'Sky',
    'Cube', 
    'CosmologyInterface',
    'ICs'
]
