"""
Pure JAX implementation of FFT operations with proper normalization.

Efficient FFT implementations for cosmological simulations.
"""
import jax
import jax.numpy as jnp
from typing import Tuple, Optional


def rfft_with_normalization(
    field: jnp.ndarray,
    axes: Optional[Tuple[int, ...]] = None
) -> jnp.ndarray:
    """
    Perform real FFT with proper normalization.
    
    Efficient real FFT operations with JAX implementation.
    
    Parameters:
    -----------
    field : jnp.ndarray
        Real-valued input field
    axes : Optional[Tuple[int, ...]]
        Axes over which to compute FFT. If None, uses all axes.
        
    Returns:
    --------
    field_k : jnp.ndarray
        Complex FFT coefficients
    """
    if axes is None:
        # For 3D field, use rfftn over all axes
        field_k = jnp.fft.rfftn(field)
    else:
        # Use specified axes
        field_k = jnp.fft.rfftn(field, axes=axes)
    
    return field_k


def irfft_with_normalization(
    field_k: jnp.ndarray,
    shape: Optional[Tuple[int, ...]] = None,
    axes: Optional[Tuple[int, ...]] = None
) -> jnp.ndarray:
    """
    Perform inverse real FFT with proper normalization.
    
    Parameters:
    -----------
    field_k : jnp.ndarray
        Complex FFT coefficients
    shape : Optional[Tuple[int, ...]]
        Shape of the output real field
    axes : Optional[Tuple[int, ...]]
        Axes over which to compute inverse FFT
        
    Returns:
    --------
    field : jnp.ndarray
        Real-valued field
    """
    if axes is None:
        # For 3D field, use irfftn over all axes
        field = jnp.fft.irfftn(field_k, s=shape)
    else:
        # Use specified axes
        field = jnp.fft.irfftn(field_k, s=shape, axes=axes)
    
    return field


def fft_frequencies(N: int, box_size: float, rfft_axis: bool = False) -> jnp.ndarray:
    """
    Generate FFT frequency arrays.
    
    Parameters:
    -----------
    N : int
        Grid size
    box_size : float
        Physical box size
    rfft_axis : bool
        If True, use rfftfreq (for the last axis in real FFT)
        
    Returns:
    --------
    freqs : jnp.ndarray
        Frequency array in appropriate units
    """
    if rfft_axis:
        # For real FFT last axis: [0, 1, 2, ..., N//2]
        freqs = jnp.fft.rfftfreq(N, d=1.0/N)
    else:
        # For complex FFT: [0, 1, ..., N//2-1, -N//2, ..., -1]
        freqs = jnp.fft.fftfreq(N, d=1.0/N)
    
    # Convert to physical units
    dk = 2 * jnp.pi / box_size
    return freqs * dk * N


def apply_nyquist_treatment(
    k_array: jnp.ndarray,
    N: int,
    axis_type: str = 'complex'
) -> jnp.ndarray:
    """
    Apply Nyquist frequency treatment for numerical stability.
    
    Sets Nyquist modes to zero for better numerical behavior.
    
    Parameters:
    -----------
    k_array : jnp.ndarray
        Frequency array
    N : int
        Grid size
    axis_type : str
        'complex' for standard FFT axis, 'real' for real FFT axis
        
    Returns:
    --------
    k_treated : jnp.ndarray
        Frequency array with Nyquist modes set to zero
    """
    k_treated = k_array.copy()
    
    if axis_type == 'complex':
        # For complex FFT: set k[N//2] = 0
        if N % 2 == 0:  # Only for even N
            k_treated = k_treated.at[N//2].set(0.0)
    elif axis_type == 'real':
        # For real FFT: set last element to 0 if it's Nyquist
        if N % 2 == 0:
            k_treated = k_treated.at[-1].set(0.0)
    
    return k_treated
