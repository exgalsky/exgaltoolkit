"""
Modern JAX-Native Grid Operations
Phase 2B: Clean Implementation without Legacy Dependencies
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Dict, Any
import gc

# Import JAX-native core functions
from .. import core

# Will replace this with pure JAX implementation
# import exgaltoolkit.lpt as lpt


class ModernGridOperations:
    """
    JAX-native grid operations implementation.
    
    This is a complete rewrite of the legacy GridOperations that:
    1. Uses pure JAX without legacy dependencies
    2. Preserves exact numerical behavior 
    3. Implements clean, modern patterns
    4. Supports distributed computing via JAX
    """
    
    def __init__(self, N: int, Lbox: float, partype: Optional[str] = None):
        """
        Initialize grid operations.
        
        Parameters:
        -----------
        N : int
            Grid size (N³ total points)
        Lbox : float
            Box size in Mpc
        partype : str, optional
            Parallelization type ('jaxshard' for distributed, None for serial)
        """
        self.N = N
        self.Lbox = Lbox
        self.partype = partype
        
        # Derived quantities
        self.dk = 2 * jnp.pi / self.Lbox  # k-space resolution
        self.d3k = self.dk ** 3           # k-space volume element
        
        # Grid shapes
        self.rshape = (self.N, self.N, self.N)                    # Real space shape
        self.cshape = (self.N, self.N, self.N // 2 + 1)           # Complex (rfft) shape
        
        # Initialize distributed computing setup
        self._setup_distributed_computing()
        
        # Field storage
        self.delta = None      # Density contrast field
        self.s1x = self.s1y = self.s1z = None  # 1st order displacements
        self.s2x = self.s2y = self.s2z = None  # 2nd order displacements
    
    def _setup_distributed_computing(self):
        """Setup distributed computing parameters."""
        if self.partype == 'jaxshard':
            # Multi-GPU distributed setup
            self.ngpus = jax.device_count() 
            self.host_id = jax.process_index()
            self.start = self.host_id * self.N // self.ngpus
            self.end = (self.host_id + 1) * self.N // self.ngpus
            
            # Local shapes for distributed execution
            self.rshape_local = (self.N, self.N // self.ngpus, self.N)
            self.cshape_local = (self.N, self.N // self.ngpus, self.N // 2 + 1)
        else:
            # Serial execution
            self.ngpus = 1
            self.host_id = 0
            self.start = 0
            self.end = self.N
            self.rshape_local = self.rshape
            self.cshape_local = self.cshape
    
    def k_axis(self, r: bool = False, slab_axis: bool = False) -> jnp.ndarray:
        """
        Generate k-space frequency grid.
        
        Parameters:
        -----------
        r : bool
            If True, use rfftfreq (for real FFTs)
        slab_axis : bool  
            If True, return only local slab for distributed computing
            
        Returns:
        --------
        k : jnp.ndarray
            Frequency grid in units of dk
        """
        if r:
            k_i = (jnp.fft.rfftfreq(self.N) * self.dk * self.N).astype(jnp.float32)
        else:
            k_i = (jnp.fft.fftfreq(self.N) * self.dk * self.N).astype(jnp.float32)
            
        if slab_axis:
            return k_i[self.start:self.end].astype(jnp.float32)
        return k_i
    
    def k_square(self, kx: jnp.ndarray, ky: jnp.ndarray, kz: jnp.ndarray) -> jnp.ndarray:
        """
        Compute k² = kx² + ky² + kz² on 3D grid.
        
        Parameters:
        -----------
        kx, ky, kz : jnp.ndarray
            k-space grids for each axis
            
        Returns:
        --------
        k2 : jnp.ndarray
            k-squared grid
        """
        kxa, kya, kza = jnp.meshgrid(kx, ky, kz, indexing='ij')
        k2 = (kxa**2 + kya**2 + kza**2).astype(jnp.float32)
        return k2
    
    def generate_noise(self, seed: int = 13579, noisetype: str = 'white') -> 'ModernGridOperations':
        """
        Generate noise field with reproducible random numbers.
        
        Parameters:
        -----------
        seed : int
            Random seed for reproducibility
        noisetype : str
            Type of noise ('white', 'uniform')
            
        Returns:
        --------
        self : ModernGridOperations
            For method chaining
        """
        if self.partype == 'jaxshard':
            self.delta = self._generate_sharded_noise(seed, noisetype)
        else:
            self.delta = self._generate_serial_noise(seed, noisetype)
        
        return self
    
    def _generate_serial_noise(self, seed: int, noisetype: str) -> jnp.ndarray:
        """Generate noise field for serial execution."""
        key = jax.random.PRNGKey(seed)
        
        if noisetype == 'white':
            # Gaussian white noise
            noise = jax.random.normal(key, shape=self.rshape, dtype=jnp.float32)
        elif noisetype == 'uniform':
            # Uniform noise [-1, 1]
            noise = 2.0 * jax.random.uniform(key, shape=self.rshape, dtype=jnp.float32) - 1.0
        else:
            raise ValueError(f"Unknown noise type: {noisetype}")
        
        return noise
    
    def _generate_sharded_noise(self, seed: int, noisetype: str) -> jnp.ndarray:
        """Generate noise field for distributed execution."""
        # Create unique but reproducible seed for each process
        process_seed = seed + self.host_id
        key = jax.random.PRNGKey(process_seed)
        
        if noisetype == 'white':
            noise = jax.random.normal(key, shape=self.rshape_local, dtype=jnp.float32)
        elif noisetype == 'uniform':
            noise = 2.0 * jax.random.uniform(key, shape=self.rshape_local, dtype=jnp.float32) - 1.0
        else:
            raise ValueError(f"Unknown noise type: {noisetype}")
        
        return noise
    
    def convolve_with_transfer_function(self, cosmology_service) -> 'ModernGridOperations':
        """
        Convert noise field to density field using cosmological transfer function.
        
        Parameters:
        -----------
        cosmology_service : CosmologyService
            Service providing power spectrum data
            
        Returns:
        --------
        self : ModernGridOperations
            For method chaining
        """
        if self.delta is None:
            raise ValueError("No noise field available. Run generate_noise() first.")
        
        # Get power spectrum from cosmology service
        k_1d = cosmology_service.pspec['k']
        pk_1d = cosmology_service.pspec['pofk']
        
        # Compute transfer function
        p_whitenoise = (2 * jnp.pi)**3 / (self.d3k * self.N**3)
        transfer_1d = jnp.sqrt(pk_1d / p_whitenoise)
        
        # Apply transfer function in k-space
        self.delta = self._apply_transfer_function(self.delta, k_1d, transfer_1d)
        
        return self
    
    def _apply_transfer_function(self, field: jnp.ndarray, k_1d: jnp.ndarray, 
                                transfer_1d: jnp.ndarray) -> jnp.ndarray:
        """Apply transfer function to field in k-space."""
        # FFT to k-space
        field_k = self._fft(field, direction='r2c')
        
        # Create 3D transfer function grid
        transfer_3d = self._interpolate_to_k_grid(k_1d, transfer_1d)
        
        # Apply transfer function
        field_k_transferred = field_k * transfer_3d
        
        # FFT back to real space
        field_real = self._fft(field_k_transferred, direction='c2r')
        
        return field_real
    
    def _interpolate_to_k_grid(self, k_1d: jnp.ndarray, f_1d: jnp.ndarray) -> jnp.ndarray:
        """Interpolate 1D function to 3D k-grid."""
        # Create k-space grids
        kx = self.k_axis()
        ky = self.k_axis(slab_axis=True) 
        kz = self.k_axis(r=True)
        
        # Compute |k| on 3D grid
        k_mag = jnp.sqrt(self.k_square(kx, ky, kz)).ravel()
        
        # Interpolate to k-grid
        f_3d = jnp.interp(k_mag, k_1d, f_1d, left=0.0, right=0.0)
        
        # Reshape to grid
        if self.partype == 'jaxshard':
            f_3d = f_3d.reshape(self.cshape_local)
        else:
            f_3d = f_3d.reshape(self.cshape)
            
        return f_3d.astype(jnp.float32)
    
    def _fft(self, field: jnp.ndarray, direction: str) -> jnp.ndarray:
        """
        Perform FFT operation using JAX.
        
        Parameters:
        -----------
        field : jnp.ndarray
            Input field
        direction : str
            'r2c' for real-to-complex, 'c2r' for complex-to-real
            
        Returns:
        --------
        result : jnp.ndarray
            Transformed field
        """
        if self.partype == 'jaxshard':
            return self._fft_distributed(field, direction)
        else:
            return self._fft_serial(field, direction)
    
    def _fft_serial(self, field: jnp.ndarray, direction: str) -> jnp.ndarray:
        """Serial FFT using JAX."""
        if direction == 'r2c':
            return jnp.fft.rfftn(field)
        elif direction == 'c2r': 
            return jnp.fft.irfftn(field, s=self.rshape)
        else:
            raise ValueError(f"Unknown FFT direction: {direction}")
    
    def _fft_distributed(self, field: jnp.ndarray, direction: str) -> jnp.ndarray:
        """
        Distributed FFT using JAX experimental features.
        
        TODO: Implement using jax.experimental.multihost_utils
        For now, fall back to serial FFT
        """
        # Placeholder: Use serial FFT for now
        # In full implementation, this would use JAX distributed FFT
        return self._fft_serial(field, direction)
    
    def compute_lpt_displacements(self, order: int = 2, input_mode: str = 'noise') -> 'ModernGridOperations':
        """
        Compute LPT displacement fields.
        
        Parameters:
        -----------
        order : int
            LPT order (1 or 2)
        input_mode : str
            Input field mode ('noise', 'delta')
            
        Returns:
        --------
        self : ModernGridOperations
            For method chaining
        """
        if order <= 0:
            return self
        
        # Setup k-space grids using 3D meshgrids for proper broadcasting
        kx, ky, kz = core.create_k_grids_3d(self.N, self.Lbox)
        
        # Compute k² from 3D grids
        k2 = kx**2 + ky**2 + kz**2
        
        # Find k=0 mode
        k_zero_mask = (k2 == 0.0)
        
        # Get density field in k-space
        if input_mode == 'noise':
            delta_k = self._fft(self.delta, direction='r2c')
        elif input_mode == 'delta':
            delta_k = self._fft(self.delta, direction='r2c')
        else:
            raise ValueError(f"Unknown input mode: {input_mode}")
        
        # Compute 1st order displacements
        self.s1x = self._compute_displacement_component(kx, delta_k, k2, k_zero_mask)
        self.s1y = self._compute_displacement_component(ky, delta_k, k2, k_zero_mask)
        self.s1z = self._compute_displacement_component(kz, delta_k, k2, k_zero_mask)
        
        # Compute 2nd order displacements if requested
        if order > 1:
            delta2_k = self._compute_second_order_density(kx, ky, kz, delta_k, k2, k_zero_mask)
            self.s2x = self._compute_displacement_component(kx, delta2_k, k2, k_zero_mask)
            self.s2y = self._compute_displacement_component(ky, delta2_k, k2, k_zero_mask)
            self.s2z = self._compute_displacement_component(kz, delta2_k, k2, k_zero_mask)
        
        return self
    
    def _compute_displacement_component(self, ki: jnp.ndarray, delta_k: jnp.ndarray, 
                                      k2: jnp.ndarray, k_zero_mask: jnp.ndarray) -> jnp.ndarray:
        """Compute single component of displacement field."""
        # S_i(k) = i k_i δ(k) / k²
        s_k = (1j) * ki * delta_k / jnp.where(k_zero_mask, 1.0, k2)
        
        # Set k=0 mode to zero (no monopole displacement)
        if self.host_id == 0:
            s_k = jnp.where(k_zero_mask, 0.0 + 0.0j, s_k)
        
        # Transform back to real space
        s_real = self._fft(s_k, direction='c2r')
        
        return s_real
    
    def _compute_second_order_density(self, kx: jnp.ndarray, ky: jnp.ndarray, kz: jnp.ndarray,
                                    delta_k: jnp.ndarray, k2: jnp.ndarray, 
                                    k_zero_mask: jnp.ndarray) -> jnp.ndarray:
        """
        Compute second-order density contrast for 2nd order LPT.
        
        δ²(k) = Σ[∂S_i/∂q_i ∂S_j/∂q_j - (∂S_i/∂q_j)²]
        """
        # Define shear factor computation
        def compute_shear_factor(ki: jnp.ndarray, kj: jnp.ndarray) -> jnp.ndarray:
            # ∂²S_i/∂x_i∂x_j = k_i k_j δ(k) / k²
            shear_k = ki * kj * delta_k / jnp.where(k_zero_mask, 1.0, k2)
            if self.host_id == 0:
                shear_k = jnp.where(k_zero_mask, 0.0 + 0.0j, shear_k)
            return self._fft(shear_k, direction='c2r')
        
        # Compute all required shear terms
        sxx = compute_shear_factor(kx, kx)  # ∂²S_x/∂x∂x  
        syy = compute_shear_factor(ky, ky)  # ∂²S_y/∂y∂y
        szz = compute_shear_factor(kz, kz)  # ∂²S_z/∂z∂z
        sxy = compute_shear_factor(kx, ky)  # ∂²S_x/∂x∂y
        sxz = compute_shear_factor(kx, kz)  # ∂²S_x/∂x∂z  
        syz = compute_shear_factor(ky, kz)  # ∂²S_y/∂y∂z
        
        # Combine shear terms: δ² = diagonal terms - off-diagonal terms  
        delta2_real = (sxx * syy + sxx * szz + syy * szz - 
                       sxy * sxy - sxz * sxz - syz * syz)
        
        # Transform to k-space for return
        delta2_k = self._fft(delta2_real, direction='r2c')
        
        return delta2_k
    
    def get_displacement_fields(self) -> Tuple[Optional[jnp.ndarray], ...]:
        """Get LPT displacement fields."""
        return (self.s1x, self.s1y, self.s1z, self.s2x, self.s2y, self.s2z)
    
    def get_density_field(self) -> Optional[jnp.ndarray]:
        """Get the density contrast field."""
        return self.delta
    
    def get_field_statistics(self) -> Dict[str, Any]:
        """Get statistics of current density field."""
        if self.delta is None:
            return {}
        
        return {
            'mean': float(jnp.mean(self.delta)),
            'std': float(jnp.std(self.delta)),
            'min': float(jnp.min(self.delta)),
            'max': float(jnp.max(self.delta))
        }
