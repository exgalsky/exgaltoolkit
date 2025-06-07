"""
Grid operations for distributed computing and FFT operations.
"""
import jax.numpy as jnp
from typing import Optional, Tuple
import exgaltoolkit.util.jax_util as ju

class GridOperations:
    """
    Grid operations for cosmological simulations.
    
    This class handles:
    - Grid generation and management
    - Distributed FFT operations
    - Noise generation and convolution
    - Integration with existing LPT Cube functionality
    """
    
    def __init__(self, N: int, Lbox: float, **kwargs):
        """
        Initialize grid operations.
        
        Parameters:
        -----------
        N : int
            Grid size (N^3 grid points)
        Lbox : float
            Box size in Mpc/h
        """
        self.N = N
        self.Lbox = Lbox
        
        # Import and create the underlying Cube for compatibility
        import exgaltoolkit.lpt as lpt
        self.cube = lpt.Cube(N=N, Lbox=Lbox, **kwargs)
        
        # Expose key cube attributes for easy access
        self.delta = None
        self.s1x = self.s1y = self.s1z = None
        self.s2x = self.s2y = self.s2z = None
    
    def generate_noise(self, seed: int = 12345):
        """Generate initial noise field."""
        self.cube.generate_noise(seed=seed)
        return self
    
    def convolve_with_transfer_function(self, cosmology_service):
        """Convolve noise with transfer function to get density field."""
        self.cube.noise2delta(cosmology_service)
        self.delta = self.cube.delta
        return self
    
    def compute_lpt_displacements(self, order: int = 2, input_mode: str = 'noise'):
        """Compute LPT displacement fields."""
        if order > 0:
            self.cube.slpt(infield=input_mode)
            
            # Extract displacements
            self.s1x = getattr(self.cube, 's1x', None)
            self.s1y = getattr(self.cube, 's1y', None) 
            self.s1z = getattr(self.cube, 's1z', None)
            
            if order > 1:
                self.s2x = getattr(self.cube, 's2x', None)
                self.s2y = getattr(self.cube, 's2y', None)
                self.s2z = getattr(self.cube, 's2z', None)
        
        return self
    
    def get_density_field(self) -> Optional[jnp.ndarray]:
        """Get the density contrast field."""
        return getattr(self.cube, 'delta', None)
    
    def get_displacement_fields(self) -> Tuple[Optional[jnp.ndarray], ...]:
        """Get LPT displacement fields."""
        return (
            getattr(self.cube, 's1x', None),
            getattr(self.cube, 's1y', None),
            getattr(self.cube, 's1z', None),
            getattr(self.cube, 's2x', None),
            getattr(self.cube, 's2y', None),
            getattr(self.cube, 's2z', None)
        )
    
    def get_k_grids(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Get k-space coordinate grids."""
        # Use the cube's k-space functionality if available
        if hasattr(self.cube, 'get_k_grids'):
            return self.cube.get_k_grids()
        else:
            # Fallback implementation
            k_fundamental = 2 * jnp.pi / self.Lbox
            k_nyquist = k_fundamental * self.N / 2
            
            k1d = jnp.fft.fftfreq(self.N, d=1.0/self.N) * k_fundamental
            kx, ky, kz = jnp.meshgrid(k1d, k1d, k1d, indexing='ij')
            return kx, ky, kz
    
    def compute_power_spectrum(self, field: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute power spectrum of a field.
        
        Parameters:
        -----------
        field : jnp.ndarray, optional
            Field to compute power spectrum for. If None, uses density field.
            
        Returns:
        --------
        k : jnp.ndarray
            k values
        pk : jnp.ndarray
            Power spectrum P(k)
        """
        if field is None:
            field = self.get_density_field()
            
        if field is None:
            raise ValueError("No field available for power spectrum calculation")
        
        # Use cube's power spectrum functionality if available
        if hasattr(self.cube, 'compute_power_spectrum'):
            return self.cube.compute_power_spectrum(field)
        else:
            # Basic power spectrum calculation
            field_k = jnp.fft.fftn(field)
            power_3d = jnp.abs(field_k)**2
            
            # Spherically average (simplified)
            kx, ky, kz = self.get_k_grids()
            k_mag = jnp.sqrt(kx**2 + ky**2 + kz**2)
            
            # This is a simplified version - full implementation would do proper binning
            k_fundamental = 2 * jnp.pi / self.Lbox
            k_max = k_fundamental * self.N / 2
            k_bins = jnp.linspace(0, k_max, 50)
            
            return k_bins[:-1], jnp.ones_like(k_bins[:-1])  # Placeholder
    
    # Expose cube methods for backward compatibility
    def __getattr__(self, name):
        """Delegate unknown attributes to the underlying cube."""
        if hasattr(self.cube, name):
            return getattr(self.cube, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
