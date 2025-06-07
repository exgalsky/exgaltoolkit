"""
Grid management and operations services.
"""
import jax
import jax.numpy as jnp
import gc
import os
from typing import Tuple, Optional
from ..core.config import GridConfiguration, ParallelConfig
from ..core.exceptions import GridError
import exgaltoolkit.mathutil as mu

class GridManager:
    """Manages 3D grid operations and sharding."""
    
    def __init__(self, config: GridConfiguration):
        self.config = config
        self.N = config.N
        self.Lbox = config.Lbox
        self.shape = (config.N, config.N, config.N)
        
        # Grid spacing and k-space parameters
        self.dk = 2 * jnp.pi / self.Lbox
        self.d3k = self.dk * self.dk * self.dk
        
        # Set up parallelization strategy
        self._setup_parallelization()
        
    def _setup_parallelization(self):
        """Set up parallelization parameters."""
        # Check if we're running in parallel mode
        self.parallel = False
        self.ngpus = 1        
        self.host_id = 0
        self.start = 0
        self.end = self.N
        
        # Check for JAX sharding environment
        if hasattr(jax, 'process_index'):
            try:
                ngpus_env = os.environ.get("XGSMENV_NGPUS")
                if ngpus_env:
                    self.ngpus = int(ngpus_env)
                    self.host_id = jax.process_index()
                    self.start = self.host_id * self.N // self.ngpus
                    self.end = (self.host_id + 1) * self.N // self.ngpus
                    self.parallel = True
            except:
                pass  # Fall back to serial mode
        
        # Set up shapes
        self.rshape = (self.N, self.N, self.N)
        self.cshape = (self.N, self.N, self.N//2 + 1)
        
        if self.parallel:
            self.rshape_local = (self.N, self.N // self.ngpus, self.N)
            self.cshape_local = (self.N, self.N // self.ngpus, self.N // 2 + 1)
        else:
            self.rshape_local = self.rshape
            self.cshape_local = self.cshape
    
    def create_k_grids(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Create k-space grids for FFT operations."""
        kx = (jnp.fft.fftfreq(self.N) * self.dk * self.N).astype(jnp.float32)
        
        if self.parallel:
            ky = (jnp.fft.fftfreq(self.N) * self.dk * self.N)[self.start:self.end].astype(jnp.float32)
        else:
            ky = (jnp.fft.fftfreq(self.N) * self.dk * self.N).astype(jnp.float32)
            
        kz = (jnp.fft.rfftfreq(self.N) * self.dk * self.N).astype(jnp.float32)
        
        return kx, ky, kz
    
    def compute_k_squared(self, kx: jnp.ndarray, ky: jnp.ndarray, kz: jnp.ndarray) -> jnp.ndarray:
        """Compute k^2 grid."""
        kxa, kya, kza = jnp.meshgrid(kx, ky, kz, indexing='ij')
        del kx, ky, kz
        gc.collect()

        k2 = (kxa**2 + kya**2 + kza**2).astype(jnp.float32)
        del kxa, kya, kza
        gc.collect()

        return k2
    
    def interpolate_to_k_grid(self, k_1d: jnp.ndarray, f_1d: jnp.ndarray) -> jnp.ndarray:
        """Interpolate function to k-space grid."""
        kx, ky, kz = self.create_k_grids()
        k2 = self.compute_k_squared(kx, ky, kz)
        
        interp_fcn = jnp.sqrt(k2).ravel()
        del kx, ky, kz
        gc.collect()

        interp_fcn = jnp.interp(interp_fcn, k_1d, f_1d, left=0.0, right=0.0)
        return jnp.reshape(interp_fcn, self.cshape_local).astype(jnp.float32)

class NoiseGenerator:
    """Generates noise fields with proper seeding."""
    
    def __init__(self, grid_manager: GridManager):
        self.grid_manager = grid_manager
    
    def generate_white_noise(self, seed: int, noise_type: str = 'white', nsub: int = 1024**3) -> jnp.ndarray:
        """Generate white noise field."""
        N = self.grid_manager.N
        
        if self.grid_manager.parallel:
            return self._generate_sharded_noise(N, noise_type, seed, nsub)
        else:
            return self._generate_serial_noise(N, noise_type, seed, nsub)
    
    def _generate_sharded_noise(self, N: int, noise_type: str, seed: int, nsub: int) -> jnp.ndarray:           
        """Generate noise for sharded/parallel execution."""
        start = self.grid_manager.start
        end = self.grid_manager.end

        stream = mu.Stream(seedkey=seed, nsub=nsub)
        noise = stream.generate(start=start*N**2, size=(end-start)*N**2).astype(jnp.float32)
        noise = jnp.reshape(noise, (end-start, N, N))
        return jnp.transpose(noise, (1, 0, 2)) 

    def _generate_serial_noise(self, N: int, noise_type: str, seed: int, nsub: int) -> jnp.ndarray:
        """Generate noise for serial execution."""
        stream = mu.Stream(seedkey=seed, nsub=nsub)
        noise = stream.generate(start=0, size=N**3).astype(jnp.float32)
        noise = jnp.reshape(noise, (N, N, N))
        return jnp.transpose(noise, (1, 0, 2))

class FFTProcessor:
    """Handles FFT operations with proper parallelization."""
    
    def __init__(self, grid_manager: GridManager):
        self.grid_manager = grid_manager
    
    def forward_transform(self, field: jnp.ndarray) -> jnp.ndarray:
        """Forward FFT."""
        # Import the existing multihost_rfft module
        from ..lpt import multihost_rfft as mfft
        return mfft.fft(field, direction='r2c')
    
    def inverse_transform(self, field: jnp.ndarray) -> jnp.ndarray:
        """Inverse FFT."""
        # Import the existing multihost_rfft module  
        from ..lpt import multihost_rfft as mfft
        return mfft.fft(field, direction='c2r')
    
    def apply_transfer_function(self, field: jnp.ndarray, transfer_data: jnp.ndarray) -> jnp.ndarray:
        """Apply transfer function to field."""
        transfer_grid = self.grid_manager.interpolate_to_k_grid(transfer_data[0], transfer_data[1])
        del transfer_data
        gc.collect()
        return field * transfer_grid
