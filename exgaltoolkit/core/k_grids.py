"""
Pure JAX implementation of k-space grid utilities.

Efficient k-space grid generation for cosmological simulations.
"""
import jax
import jax.numpy as jnp
from typing import Tuple
from .fft_ops import fft_frequencies, apply_nyquist_treatment


def create_k_grids_rfft(
    N: int, 
    box_size: float,
    distributed_slice: slice = None,
    apply_nyquist_zeroing: bool = False # New parameter
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Create k-space coordinate grids for real FFT.
    
    Parameters:
    -----------
    N : int
        Grid size (N^3 total points)
    box_size : float
        Physical box size
    distributed_slice : slice, optional
        Y-axis slice for distributed computing
    apply_nyquist_zeroing : bool, optional
        If True, applies Nyquist frequency zeroing to the 1D k-vectors.
        Defaults to False, returning "raw" frequencies.
        
    Returns:
    --------
    kx, ky, kz : Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        1D k-space coordinate arrays.
    """
    # Generate 1D frequency arrays
    kx_1d = fft_frequencies(N, box_size, rfft_axis=False)  # Complex FFT
    ky_1d_full = fft_frequencies(N, box_size, rfft_axis=False)  # Complex FFT  
    kz_1d = fft_frequencies(N, box_size, rfft_axis=True)   # Real FFT
    
    if apply_nyquist_zeroing:
        kx_1d = apply_nyquist_treatment(kx_1d, N, 'complex')
        kz_1d = apply_nyquist_treatment(kz_1d, N, 'real')
        # Nyquist for full ky before slicing
        ky_1d_full_treated = apply_nyquist_treatment(ky_1d_full, N, 'complex')
    else:
        ky_1d_full_treated = ky_1d_full

    if distributed_slice is not None:
        ky_1d = ky_1d_full_treated[distributed_slice]
        # If applying zeroing and this slice contains the Nyquist frequency for y
        if apply_nyquist_zeroing and (N % 2 == 0) and (N//2 >= distributed_slice.start and N//2 < distributed_slice.stop):
            # This condition for ky_1d_full_treated having Nyquist zeroed is already handled if apply_nyquist_zeroing is true.
            # The slicing itself is the main thing here.
            # If ky_1d_full_treated was not treated, but we still want to treat the slice if it contains Nyquist:
            # This part of logic might need refinement if apply_nyquist_zeroing=False but distributed slice needs it.
            # For now, assume if apply_nyquist_zeroing=True, ky_1d_full_treated is correct.
            # If apply_nyquist_zeroing=False, ky_1d is just a slice of raw ky_1d_full.
            pass # Covered by ky_1d_full_treated being sliced.
    else:
        ky_1d = ky_1d_full_treated
            
    return kx_1d, ky_1d, kz_1d


def compute_k_squared(
    kx_1d: jnp.ndarray, 
    ky_1d: jnp.ndarray, 
    kz_1d: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute k² from 1D k-vectors.
    
    Broadcasts 1D k-vectors to create 3D k² grid.
    
    Parameters:
    -----------
    kx_1d, ky_1d, kz_1d : jnp.ndarray
        1D k-space frequency vectors
        
    Returns:
    --------
    k_squared : jnp.ndarray
        k² values with shape (len(kx_1d), len(ky_1d), len(kz_1d))
    """
    # Broadcast 1D vectors to 3D grid
    kxa, kya, kza = jnp.meshgrid(kx_1d, ky_1d, kz_1d, indexing='ij')
    
    # Compute k² magnitude
    k_squared = (kxa**2 + kya**2 + kza**2).astype(jnp.float32)
    
    return k_squared


def handle_nyquist_modes(
    field_k: jnp.ndarray,
    k_squared: jnp.ndarray
) -> jnp.ndarray:
    """
    Handle k=0 and Nyquist modes for physical fields.
    
    Sets modes to zero where k^2 = 0 to avoid singularities.
    
    Parameters:
    -----------
    field_k : jnp.ndarray
        Field in k-space
    k_squared : jnp.ndarray
        k^2 values
        
    Returns:
    --------
    field_k_treated : jnp.ndarray
        Field with proper mode handling
    """
    # Set k=0 mode to zero (no monopole)
    field_k_treated = jnp.where(k_squared == 0, 0.0, field_k)
    
    return field_k_treated


def create_k_grids_full(
    N: int, 
    box_size: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Create k-space coordinate grids for full complex FFT.
    
    Used for power spectrum calculations and other analyses.
    
    Parameters:
    -----------
    N : int
        Grid size
    box_size : float
        Physical box size
        
    Returns:
    --------
    kx, ky, kz : Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        k-space coordinate arrays for full FFT
    """
    # All axes use complex FFT frequencies
    kx_1d = fft_frequencies(N, box_size, rfft_axis=False)
    ky_1d = fft_frequencies(N, box_size, rfft_axis=False)
    kz_1d = fft_frequencies(N, box_size, rfft_axis=False)
    
    # Apply Nyquist treatment
    kx_1d = apply_nyquist_treatment(kx_1d, N, 'complex')
    ky_1d = apply_nyquist_treatment(ky_1d, N, 'complex')
    kz_1d = apply_nyquist_treatment(kz_1d, N, 'complex')
    
    # Create 3D meshgrids
    kx, ky, kz = jnp.meshgrid(kx_1d, ky_1d, kz_1d, indexing='ij')
    
    return kx, ky, kz


def create_k_grids_3d(
    N: int, 
    box_size: float,
    distributed_slice: slice = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Create 3D meshgrid k-space coordinate grids for LPT displacement calculations.
    
    This function creates 3D meshgrids that can be directly used for broadcasting
    with FFT arrays in displacement field calculations.
    Nyquist frequencies ARE zeroed out here as LPT calculations often involve 1/k^2.
    
    Parameters:
    -----------
    N : int
        Grid size (N^3 total points)
    box_size : float
        Physical box size
    distributed_slice : slice, optional
        Y-axis slice for distributed computing
        
    Returns:
    --------
    kx, ky, kz : Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        3D k-space coordinate meshgrids with proper shape for broadcasting
    """
    # Get 1D k-vectors first, WITH Nyquist treatment for LPT
    kx_1d, ky_1d, kz_1d = create_k_grids_rfft(N, box_size, distributed_slice, apply_nyquist_zeroing=True)
    
    # Create 3D meshgrids for broadcasting with FFT arrays
    kx, ky, kz = jnp.meshgrid(kx_1d, ky_1d, kz_1d, indexing='ij')
    
    return kx, ky, kz
