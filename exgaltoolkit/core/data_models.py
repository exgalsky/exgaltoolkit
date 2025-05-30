"""
Data models for simulation results and intermediate data.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import jax.numpy as jnp

@dataclass
class LPTDisplacements:
    """Container for LPT displacement fields."""
    s1x: jnp.ndarray
    s1y: jnp.ndarray  
    s1z: jnp.ndarray
    s2x: Optional[jnp.ndarray] = None
    s2y: Optional[jnp.ndarray] = None
    s2z: Optional[jnp.ndarray] = None

@dataclass
class ParticleData:
    """Container for particle data."""
    positions: jnp.ndarray
    velocities: jnp.ndarray
    masses: jnp.ndarray
    ids: Optional[jnp.ndarray] = None

@dataclass
class GrowthFactors:
    """Container for cosmological growth factors."""
    z: jnp.ndarray
    hubble: jnp.ndarray
    d1: jnp.ndarray
    d2: jnp.ndarray
    f1: jnp.ndarray
    f2: jnp.ndarray

@dataclass
class SimulationContext:
    """Context object that carries state between simulation steps."""
    config: 'SimulationConfig'
    data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}
    
    def get(self, key: str, default=None):
        """Get data from context."""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set data in context."""
        self.data[key] = value
    
    def has(self, key: str) -> bool:
        """Check if key exists in context."""
        return key in self.data

@dataclass
class StepResult:
    """Result of a single simulation step."""
    success: bool
    step_name: str
    message: str = ""
    data: Optional[Dict[str, Any]] = None

@dataclass
class SimulationResult:
    """Result of complete simulation."""
    success: bool
    final_step: str
    step_results: list
    context: SimulationContext
    message: str = ""

@dataclass
class GridData:
    """Container for grid-based data."""
    density: Optional[jnp.ndarray] = None
    potential: Optional[jnp.ndarray] = None
    lpt_s1: Optional[LPTDisplacements] = None
    lpt_s2: Optional[LPTDisplacements] = None

@dataclass 
class SimulationResults:
    """Container for complete simulation results."""
    particle_data: Optional[ParticleData] = None
    grid_data: Optional[GridData] = None
    density_grid: Optional[jnp.ndarray] = None
    lpt_displacements: Optional[LPTDisplacements] = None
    growth_factors: Optional[GrowthFactors] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
