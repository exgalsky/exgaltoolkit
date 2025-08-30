"""
Pure JAX implementations of core cosmological simulation algorithms.

This module contains efficient, modern implementations of the core algorithms
for cosmological simulation. All functions are JAX-native for performance.
"""

from .noise import generate_white_noise
from .transfers import noise2delta
from .lpt_math import compute_first_order_lpt, compute_second_order_lpt
from .fft_ops import rfft_with_normalization, irfft_with_normalization
from .k_grids import create_k_grids_rfft, create_k_grids_full, create_k_grids_3d, handle_nyquist_modes, compute_k_squared

# High-level wrapper functions for convenience
def compute_lpt_displacements(delta_field, box_size, order=2):
    """
    Compute LPT displacement fields.
    
    Parameters:
    -----------
    delta_field : jnp.ndarray
        Density contrast field
    box_size : float
        Box size in Mpc/h  
    order : int
        LPT order (1 or 2)
        
    Returns:
    --------
    tuple
        Displacement fields (s1x, s1y, s1z) or (s1x, s1y, s1z, s2x, s2y, s2z)
    """
    # Compute first order displacements
    s1x, s1y, s1z = compute_first_order_lpt(delta_field, box_size)
    
    if order == 1:
        return (s1x, s1y, s1z)
    elif order == 2:
        # Compute second order displacements
        s2x, s2y, s2z = compute_second_order_lpt(s1x, s1y, s1z, box_size)
        return (s1x, s1y, s1z, s2x, s2y, s2z)
    else:
        raise ValueError(f"Unsupported LPT order: {order}")

__all__ = [
    'generate_white_noise',
    'noise2delta', 
    'compute_first_order_lpt',
    'compute_second_order_lpt',
    'compute_lpt_displacements',
    'rfft_with_normalization',
    'irfft_with_normalization',
    'create_k_grids_rfft',
    'create_k_grids_full',
    'handle_nyquist_modes',
    'compute_k_squared'
    'irfft_with_normalization',
    'create_k_grids_rfft',
    'create_k_grids_full',
    'handle_nyquist_modes'
]
