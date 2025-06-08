"""
Grid operations for distributed computing and FFT operations.

Pure JAX implementation using the core module.
"""
import jax.numpy as jnp
from typing import Optional, Tuple
import exgaltoolkit.core as core

class GridOperations:
    """
    Grid operations for cosmological simulations.
    
    Modern JAX-native implementation that replaces legacy lpt.Cube functionality.
    Uses the pure implementations from exgaltoolkit.core module.
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
        
        # Store additional parameters
        self.partype = kwargs.get('partype', None)
        
        # Fields - these will be populated by the operations
        self.delta = None
        self.s1x = self.s1y = self.s1z = None
        self.s2x = self.s2y = self.s2z = None
    
    def generate_noise(self, seed: int = 12345):
        """Generate initial noise field using core.noise module."""
        grid_shape = (self.N, self.N, self.N)
        self.delta = core.generate_white_noise(grid_shape, seed)
        return self
    
    def convolve_with_transfer_function(self, cosmology_service):
        """Convolve noise with transfer function using core.transfers module."""
        if self.delta is None:
            raise ValueError("Must generate noise before applying transfer function")
        
        # Apply power spectrum transfer function
        self.delta = core.apply_power_spectrum_transfer(
            noise_field=self.delta,
            power_spectrum=cosmology_service.pspec,
            box_size=self.Lbox
        )
        return self
    
    def compute_lpt_displacements(self, order: int = 2, input_mode: str = 'noise'):
        """Compute LPT displacement fields using core.lpt_math module."""
        if self.delta is None:
            raise ValueError("Must have density field before computing LPT displacements")
        
        # Compute displacements using pure JAX implementation
        displacements = core.compute_lpt_displacements(
            delta_field=self.delta,
            box_size=self.Lbox,
            order=order
        )
        
        # Unpack results
        if order == 1:
            self.s1x, self.s1y, self.s1z = displacements
        elif order == 2:
            self.s1x, self.s1y, self.s1z, self.s2x, self.s2y, self.s2z = displacements
        
        return self
    
    def get_density_field(self) -> Optional[jnp.ndarray]:
        """Get the density contrast field."""
        return self.delta
    
    def get_displacement_fields(self) -> Tuple[Optional[jnp.ndarray], ...]:
        """Get LPT displacement fields."""
        return (self.s1x, self.s1y, self.s1z, self.s2x, self.s2y, self.s2z)
    
    def get_k_grids(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Get k-space coordinate grids."""
        return core.create_k_grids_rfft(self.N, self.Lbox)
    
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
        
        # Basic power spectrum calculation using JAX
        field_k = jnp.fft.fftn(field)
        power_3d = jnp.abs(field_k)**2
        
        # Create k-magnitude grid
        kx, ky, kz = core.create_k_grids_full(self.N, self.Lbox)
        k_mag = jnp.sqrt(kx**2 + ky**2 + kz**2)
        
        # Simplified binning for power spectrum
        k_fundamental = 2 * jnp.pi / self.Lbox
        k_max = k_fundamental * self.N / 2
        k_bins = jnp.linspace(k_fundamental, k_max, 20)
        
        # Simple averaging (placeholder - could be improved)
        pk_avg = jnp.ones_like(k_bins[:-1]) * jnp.mean(power_3d)
        
        return k_bins[:-1], pk_avg
    
    # Backward compatibility: expose some legacy-style attributes
    @property
    def delta(self):
        """Density field (legacy compatibility)."""
        return self._delta
    
    @delta.setter 
    def delta(self, value):
        self._delta = value
