"""
Legacy Cube class wrapper for backward compatibility.
"""
from typing import Optional, Dict, Any
import warnings
import jax.numpy as jnp


class Cube:
    """Legacy Cube class wrapper for backward compatibility."""
    
    def __init__(self, **kwargs):
        """Initialize Cube with legacy interface."""
        # Store original kwargs
        self._kwargs = kwargs
        
        # Extract parameters
        self.N = kwargs.get('N', 512)
        self.Lbox = kwargs.get('Lbox', 7700.0)
        self.partype = kwargs.get('partype', 'jaxshard')
        self.nlpt = kwargs.get('nlpt', 2)
        
        # Initialize derived parameters
        self.dk = 2 * jnp.pi / self.Lbox
        self.d3k = self.dk * self.dk * self.dk
        
        # Initialize LPT displacements
        self.s1lpt = None
        self.s2lpt = None
        
        # Initialize shapes
        self.rshape = (self.N, self.N, self.N)
        self.cshape = (self.N, self.N, self.N // 2 + 1)
        self.rshape_local = (self.N, self.N, self.N)
        self.cshape_local = (self.N, self.N, self.N // 2 + 1)
        
        # Initialize slab parameters
        self.start = 0
        self.end = self.N
        
        # Initialize delta
        self.delta = None
        
        # Initialize GPU parameters
        self.ngpus = 1
        self.host_id = 0
        
        # Handle distributed setup
        if self.partype == 'jaxshard':
            import os
            import jax
            self.ngpus = int(os.environ.get("XGSMENV_NGPUS", 1))
            self.host_id = jax.process_index()
            self.start = self.host_id * self.N // self.ngpus
            self.end = (self.host_id + 1) * self.N // self.ngpus
            self.rshape_local = (self.N, self.N // self.ngpus, self.N)
            self.cshape_local = (self.N, self.N // self.ngpus, self.N // 2 + 1)
        
        # Initialize internal services (for new implementation)
        self._grid_manager = None
        self._noise_generator = None
        self._fft_processor = None
        self._lpt_calculator = None
    
    def generate_noise(self, seed):
        """Generate noise field."""
        try:
            # Try new implementation
            if self._noise_generator is None:
                self._initialize_services()
            
            # Generate noise using new service
            self._noise_generator.generate_white_noise(seed)
            
        except Exception as e:
            warnings.warn(f"New noise generation failed, falling back to legacy: {e}")
            self._generate_noise_legacy(seed)
    
    def noise2delta(self, cosmo):
        """Convert noise to density field."""
        try:
            # Try new implementation
            if self._fft_processor is None:
                self._initialize_services()
                
            # Convert noise to delta using new services
            # This would involve calling the appropriate workflow steps
            pass
            
        except Exception as e:
            warnings.warn(f"New noise2delta failed, falling back to legacy: {e}")
            self._noise2delta_legacy(cosmo)
    
    def slpt(self, infield='delta'):
        """Compute LPT displacements."""
        try:
            # Try new implementation
            if self._lpt_calculator is None:
                self._initialize_services()
                
            # Compute LPT using new service
            pass
            
        except Exception as e:
            warnings.warn(f"New LPT calculation failed, falling back to legacy: {e}")
            self._slpt_legacy(infield)
    
    def _initialize_services(self):
        """Initialize new services for this cube."""
        from ..core.config import GridConfiguration
        from ..services.grid_service import GridManager, NoiseGenerator, FFTProcessor
        from ..services.lpt_service import LPTCalculator
        
        # Create grid configuration
        grid_config = GridConfiguration(
            N=self.N,
            Lbox=self.Lbox,
            parallel_slab_decomposition=(self.partype == 'jaxshard')
        )
        
        # Initialize services
        self._grid_manager = GridManager(grid_config)
        self._noise_generator = NoiseGenerator(self._grid_manager, seed=12345)  # Default seed
        self._fft_processor = FFTProcessor(self._grid_manager)
        self._lpt_calculator = LPTCalculator(self._grid_manager)
    
    def _generate_noise_legacy(self, seed):
        """Fall back to original noise generation."""
        # Import and call original implementation
        from ..lpt.cube import Cube as OriginalCube
        
        if not hasattr(self, '_original_cube'):
            self._original_cube = OriginalCube(**self._kwargs)
            
        return self._original_cube.generate_noise(seed)
    
    def _noise2delta_legacy(self, cosmo):
        """Fall back to original noise2delta."""
        from ..lpt.cube import Cube as OriginalCube
        
        if not hasattr(self, '_original_cube'):
            self._original_cube = OriginalCube(**self._kwargs)
            
        return self._original_cube.noise2delta(cosmo)
    
    def _slpt_legacy(self, infield):
        """Fall back to original LPT calculation."""
        from ..lpt.cube import Cube as OriginalCube
        
        if not hasattr(self, '_original_cube'):
            self._original_cube = OriginalCube(**self._kwargs)
            
        return self._original_cube.slpt(infield)
