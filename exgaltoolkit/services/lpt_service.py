"""
LPT (Lagrangian Perturbation Theory) calculation service.
"""
import jax.numpy as jnp
import gc
from typing import Optional
from ..core.data_models import LPTDisplacements
from ..core.config import GridConfiguration
from .grid_service import GridManager, FFTProcessor

class LPTCalculator:
    """Calculates Lagrangian Perturbation Theory displacements."""
    
    def __init__(self, grid_manager: GridManager, order: int = 2):
        self.grid_manager = grid_manager
        self.order = order
        self.fft_processor = FFTProcessor(grid_manager)
    
    def compute_displacements(self, delta_field: jnp.ndarray, input_mode: str = 'noise') -> LPTDisplacements:
        """Compute LPT displacement fields."""
        if self.order <= 0:
            raise ValueError("LPT order must be positive")
        
        # Get k-space grids
        kx, ky, kz = self.grid_manager.create_k_grids()
        k2 = self.grid_manager.compute_k_squared(kx, ky, kz)
        
        # Set zero modes to avoid division by zero
        kx = kx.at[self.grid_manager.N//2].set(0.0)
        kz = kz.at[-1].set(0.0)

        if (self.grid_manager.start <= self.grid_manager.N//2 and 
            self.grid_manager.end > self.grid_manager.N//2):
            ky = ky.at[self.grid_manager.N//2 - self.grid_manager.start].set(0.0)

        kx = kx[:, None, None]
        ky = ky[None, :, None]
        kz = kz[None, None, :]

        index0 = jnp.nonzero(k2 == 0.0)

        # Get delta in Fourier space
        if input_mode == 'delta':
            delta_k = self.fft_processor.forward_transform(delta_field)
        else:
            # For noise mode, delta_field is already the noise field
            delta_k = self.fft_processor.forward_transform(delta_field)
        
        # Compute first-order displacements
        s1x = self._delta_to_displacement(kx, k2, delta_k, index0)
        s1y = self._delta_to_displacement(ky, k2, delta_k, index0)
        s1z = self._delta_to_displacement(kz, k2, delta_k, index0)
        
        s2x = s2y = s2z = None
        
        # Compute second-order displacements if requested
        if self.order > 1:
            s2x, s2y, s2z = self._compute_second_order_displacements(
                kx, ky, kz, k2, delta_k, index0
            )
        
        return LPTDisplacements(
            s1x=s1x, s1y=s1y, s1z=s1z,
            s2x=s2x, s2y=s2y, s2z=s2z
        )
    
    def _delta_to_displacement(self, ki: jnp.ndarray, k2: jnp.ndarray, 
                             delta_k: jnp.ndarray, index0: tuple) -> jnp.ndarray:
        """Convert delta to displacement using LPT formula."""
        # Convention:
        #   Y_k = Sum_j=0^n-1 [ X_j * e^(- 2pi * sqrt(-1) * j * k / n)]
        # where Y_k is complex transform of real X_j
        arr = (0 + 1j) * ki / k2 * delta_k
        if self.grid_manager.host_id == 0: 
            arr = arr.at[index0].set(0.0 + 0.0j)
        arr = self.fft_processor.inverse_transform(arr)
        return arr
    
    def _get_shear_factor(self, ki: jnp.ndarray, kj: jnp.ndarray, k2: jnp.ndarray,
                         delta_k: jnp.ndarray, index0: tuple) -> jnp.ndarray:
        """Compute shear factor for second-order LPT."""
        arr = ki * kj / k2 * delta_k
        if self.grid_manager.host_id == 0: 
            arr = arr.at[index0].set(0.0 + 0.0j)
        return self.fft_processor.inverse_transform(arr)
    
    def _compute_second_order_displacements(self, kx: jnp.ndarray, ky: jnp.ndarray, 
                                          kz: jnp.ndarray, k2: jnp.ndarray,
                                          delta_k: jnp.ndarray, index0: tuple) -> tuple:
        """Compute second-order LPT displacements."""
        # Use the 'lean' mode computation for memory efficiency
        # This follows the original logic from cube.py
        
        # Calculate all shear components
        sxx = self._get_shear_factor(kx, kx, k2, delta_k, index0)
        syy = self._get_shear_factor(ky, ky, k2, delta_k, index0)
        szz = self._get_shear_factor(kz, kz, k2, delta_k, index0)
        
        sxy = self._get_shear_factor(kx, ky, k2, delta_k, index0)
        sxz = self._get_shear_factor(kx, kz, k2, delta_k, index0)
        syz = self._get_shear_factor(ky, kz, k2, delta_k, index0)
        
        # Compute second-order source term
        delta2 = (sxx * syy + sxx * szz + syy * szz - 
                 sxy * sxy - sxz * sxz - syz * syz)
        
        # Clean up intermediate arrays
        del sxx, syy, szz, sxy, sxz, syz
        gc.collect()
        
        # Transform to k-space
        delta2_k = self.fft_processor.forward_transform(delta2)
        del delta2
        gc.collect()
        
        # Compute second-order displacements
        s2x = self._delta_to_displacement(kx, k2, delta2_k, index0)
        s2y = self._delta_to_displacement(ky, k2, delta2_k, index0)
        s2z = self._delta_to_displacement(kz, k2, delta2_k, index0)
        
        del delta2_k
        gc.collect()
        
        return s2x, s2y, s2z
