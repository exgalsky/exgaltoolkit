"""
Configuration data structures.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import jax.numpy as jnp

@dataclass
class CosmologicalParameters:
    """Immutable cosmological parameters."""
    # Support both h and H0 parameter names
    h: float = 0.7
    omega_m: float = 0.276
    omega_k: float = 0.0
    omega_b: float = 0.049
    n_s: float = 0.965
    sigma_8: float = 0.81
    omega_lambda: float = field(init=False)
    
    def __init__(self, h=None, H0=None, omega_m=0.276, Omega_m=None, 
                 omega_b=0.049, Omega_b=None, n_s=0.965, sigma_8=0.81, omega_k=0.0):
        """Initialize with flexible parameter names."""
        # Handle h vs H0
        if H0 is not None:
            self.h = H0 / 100.0
        elif h is not None:
            self.h = h
        else:
            self.h = 0.7
            
        # Handle omega_m vs Omega_m
        if Omega_m is not None:
            self.omega_m = Omega_m
        else:
            self.omega_m = omega_m
            
        # Handle omega_b vs Omega_b
        if Omega_b is not None:
            self.omega_b = Omega_b
        else:
            self.omega_b = omega_b
            
        self.omega_k = omega_k
        self.n_s = n_s
        self.sigma_8 = sigma_8
        self.omega_lambda = 1 - self.omega_m - self.omega_k
    
    @classmethod
    def from_legacy_kwargs(cls, **kwargs):
        """Create parameters from legacy keyword arguments."""
        return cls(
            h=kwargs.get('h', 0.7),
            omega_m=kwargs.get('omega_m', 0.276),
            omega_b=kwargs.get('omega_b', 0.049),
            n_s=kwargs.get('n_s', 0.965),
            sigma_8=kwargs.get('sigma_8', 0.81),
            omega_k=kwargs.get('omega_k', 0.0)
        )

@dataclass
class PowerSpectrum:
    """Power spectrum configuration and data container."""
    z_initial: float = 99.0
    k_min: float = 1e-4
    k_max: float = 10.0
    n_points: int = 1000
    k: Optional[jnp.ndarray] = None
    power: Optional[jnp.ndarray] = None
    
    @classmethod
    def from_legacy_kwargs(cls, **kwargs):
        """Create power spectrum from legacy keyword arguments."""
        return cls(
            z_initial=kwargs.get('zInit', 99.0),
            k_min=kwargs.get('k_min', 1e-4),
            k_max=kwargs.get('k_max', 10.0),
            n_points=kwargs.get('n_points', 1000)
        )
    
    def interpolate_to_grid(self, k_grid: jnp.ndarray) -> jnp.ndarray:
        """Interpolate power spectrum to given k grid."""
        if self.k is None or self.power is None:
            raise ValueError("Power spectrum data not loaded")
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
    base_path: str = './output/'
    save_grids: bool = True
    save_particles: bool = True
    format: str = 'nyx'
    base_filename: str = 'testics'
    output_dir: Optional[str] = None
    
    def __post_init__(self):
        """Set output_dir to base_path if not provided for backward compatibility."""
        if self.output_dir is None:
            self.output_dir = self.base_path

@dataclass
class SimulationConfig:
    """Master configuration for simulation."""
    cosmology: CosmologicalParameters
    grid: GridConfiguration
    simulation: SimulationParameters
    output: OutputConfig
    parallel: Optional[ParallelConfig] = None
    power_spectrum: Optional[PowerSpectrum] = None
    
    def __post_init__(self):
        """Set default parallel configuration if not provided."""
        if self.parallel is None:
            self.parallel = ParallelConfig()
    
    @classmethod
    def from_legacy_kwargs(cls, **kwargs) -> 'SimulationConfig':
        """Create from legacy kwargs (for backward compatibility)."""
        # Extract cosmology parameters
        cosmo_params = CosmologicalParameters.from_legacy_kwargs(**kwargs)
        
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
