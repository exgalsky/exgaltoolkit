"""
Configuration data structures.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import jax.numpy as jnp

@dataclass
class CosmologicalParameters:
    """Immutable cosmological parameters."""
    h: float = 0.7
    omega_m: float = 0.276
    omega_k: float = 0.0
    omega_lambda: float = field(init=False)
    
    def __post_init__(self):
        object.__setattr__(self, 'omega_lambda', 1 - self.omega_m - self.omega_k)

@dataclass
class PowerSpectrum:
    """Power spectrum data container."""
    k: jnp.ndarray
    power: jnp.ndarray
    
    def interpolate_to_grid(self, k_grid: jnp.ndarray) -> jnp.ndarray:
        """Interpolate power spectrum to given k grid."""
        return jnp.interp(k_grid, self.k, self.power, left=0.0, right=0.0)

@dataclass
class GridConfiguration:
    """Grid and box configuration."""
    N: int
    Lbox: float
    parallel_config: Optional['ParallelConfig'] = None

@dataclass
class SimulationParameters:
    """Simulation-specific parameters."""
    seed: int = 13579
    n_iterations: int = 1
    final_step: str = 'all'
    lpt_order: int = 2
    initial_redshift: float = 50.0
    model_id: str = "MockgenDefaultID"
    input_type: str = "delta"
    nside: int = 1024
    write_ics: bool = False
    
@dataclass 
class ParallelConfig:
    """Parallel processing configuration."""
    use_mpi: bool = True
    use_gpu: bool = True
    max_gpu_memory_gb: float = 40.0

@dataclass
class OutputConfig:
    """Output configuration."""
    format: str = 'nyx'
    base_filename: str = 'testics'
    output_dir: str = './output/'

@dataclass
class SimulationConfig:
    """Master configuration for simulation."""
    cosmology: CosmologicalParameters
    grid: GridConfiguration
    simulation: SimulationParameters
    output: OutputConfig
    parallel: ParallelConfig
    power_spectrum: Optional[PowerSpectrum] = None
    
    @classmethod
    def from_legacy_kwargs(cls, **kwargs) -> 'SimulationConfig':
        """Create from legacy kwargs (for backward compatibility)."""
        # Extract cosmology parameters
        cosmo_params = CosmologicalParameters(
            h=kwargs.get('h', 0.7),
            omega_m=kwargs.get('omegam', 0.276),
            omega_k=kwargs.get('omegak', 0.0)
        )
        
        # Extract grid configuration
        grid_config = GridConfiguration(
            N=kwargs.get('N', 512),
            Lbox=kwargs.get('Lbox', 7700.0)
        )
        
        # Extract simulation parameters
        sim_params = SimulationParameters(
            seed=kwargs.get('seed', 13579),
            n_iterations=kwargs.get('Niter', 1),
            final_step=kwargs.get('laststep', 'all'),
            lpt_order=kwargs.get('nlpt', 2),
            initial_redshift=kwargs.get('zInit', 50.0),
            model_id=kwargs.get('ID', "MockgenDefaultID"),
            input_type=kwargs.get('input', "delta"),
            nside=kwargs.get('Nside', 1024),
            write_ics=kwargs.get('icw', False)
        )
        
        # Extract parallel configuration
        parallel_config = ParallelConfig(
            use_mpi=kwargs.get('mpi', True),
            use_gpu=kwargs.get('gpu', True)
        )
        
        # Extract output configuration
        output_config = OutputConfig(
            format='nyx',
            base_filename='testics'
        )
        
        # Handle power spectrum if provided
        power_spectrum = None
        if 'pspec' in kwargs:
            pspec_data = kwargs['pspec']
            power_spectrum = PowerSpectrum(
                k=pspec_data['k'],
                power=pspec_data['pofk']
            )
        
        return cls(
            cosmology=cosmo_params,
            grid=grid_config,
            simulation=sim_params,
            output=output_config,
            parallel=parallel_config,
            power_spectrum=power_spectrum
        )
