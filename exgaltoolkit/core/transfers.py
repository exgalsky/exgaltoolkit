"""
Pure JAX implementation of transfer function application.

Core implementation of the noise to delta field transformation.
"""
import jax
import jax.numpy as jnp
import numpy as np
import gc
from typing import Dict, Any, Optional, Tuple

from .k_grids import create_k_grids_rfft
from .fft_ops import rfft_with_normalization, irfft_with_normalization


def noise2delta(
    noise_field: jnp.ndarray,
    cosmo_pspec: Dict[str, jnp.ndarray],
    N: int,
    Lbox: float,
    partype: Optional[str] = None,
    host_id: int = 0,
    ngpus: int = 1,
    start: int = 0,
    end: int = None
) -> jnp.ndarray:
    """
    Transform white noise field into density contrast field.
    
    Parameters:
    -----------
    noise_field : jnp.ndarray
        White noise field (already generated)
    cosmo_pspec : Dict[str, jnp.ndarray]
        Cosmology power spectrum with 'k' and 'pofk' keys
    N : int
        Grid size
    Lbox : float
        Box size
    partype : str, optional
        Parallelization type ('jaxshard' or None)
    host_id : int
        Host ID for distributed computation
    ngpus : int
        Number of GPUs
    start : int
        Start index for Y-axis slicing
    end : int
        End index for Y-axis slicing
        
    Returns:
    --------
    delta_field : jnp.ndarray
        Density contrast field
    """
    if end is None:
        end = N
        
    # Process power spectrum data
    power = np.asarray([cosmo_pspec['k'], cosmo_pspec['pofk']])
    transfer = power.copy()
    
    # Compute white noise power spectrum normalization
    dk = 2*jnp.pi/Lbox
    d3k = dk * dk * dk
    p_whitenoise = (2*np.pi)**3/(d3k*N**3)
    
    # Apply transfer function scaling
    transfer[1] = (power[1] / p_whitenoise)**0.5
    transfer = jnp.asarray(transfer)
    
    # Forward FFT: real to complex
    delta_k = _mfft_fft(noise_field, direction='r2c', partype=partype, 
                       host_id=host_id, ngpus=ngpus)
    
    # Apply grid transfer function
    delta_k = _apply_grid_transfer_function(delta_k, transfer, N, Lbox, 
                                                  partype, host_id, ngpus, start, end)
    
    # Inverse FFT: complex to real
    delta_field = _mfft_fft(delta_k, direction='c2r', partype=partype,
                           host_id=host_id, ngpus=ngpus)
    
    return delta_field


def _mfft_fft(x_np: jnp.ndarray, direction: str = 'r2c', partype: Optional[str] = None,
             host_id: int = 0, ngpus: int = 1) -> jnp.ndarray:
    """
    FFT operations with support for distributed computation.
    
    Parameters:
    -----------
    x_np : jnp.ndarray
        Input array
    direction : str
        'r2c' for real-to-complex or 'c2r' for complex-to-real
    partype : str, optional
        Parallelization type
    host_id : int
        Host ID for distributed computation
    ngpus : int
        Number of GPUs
    """
    if partype == 'jaxshard' and ngpus > 1:
        # Multi-GPU support would be implemented here
        pass
    
    # Standard FFT operations
    if direction == 'r2c':
        # Real to complex: rfftn
        return jnp.fft.rfftn(x_np)
    elif direction == 'c2r':
        # Complex to real: irfftn  
        return jnp.fft.irfftn(x_np)
    else:
        raise ValueError(f"Unknown direction: {direction}")


def _apply_grid_transfer_function(field: jnp.ndarray, transfer_data: jnp.ndarray,
                                 N: int, Lbox: float, partype: Optional[str], 
                                 host_id: int, ngpus: int, start: int, end: int) -> jnp.ndarray:
    """
    Apply transfer function to grid in k-space.
    
    Parameters:
    -----------
    field : jnp.ndarray
        Input field in k-space
    transfer_data : jnp.ndarray
        Transfer function data [k, transfer_values]
    N : int
        Grid size
    Lbox : float
        Box size
    partype : str, optional
        Parallelization type
    host_id : int
        Host ID
    ngpus : int
        Number of GPUs
    start : int
        Start index for slicing
    end : int
        End index for slicing
    """
    transfer_cdm = _interp2kgrid(transfer_data[0], transfer_data[1], N, Lbox,
                                partype, host_id, ngpus, start, end)
    del transfer_data
    gc.collect()
    
    return field * transfer_cdm


def _interp2kgrid(k_1d: jnp.ndarray, f_1d: jnp.ndarray, N: int, Lbox: float,
                 partype: Optional[str], host_id: int, ngpus: int, 
                 start: int, end: int) -> jnp.ndarray:
    """
    Interpolate 1D function onto 3D k-grid.
    
    Parameters:
    -----------
    k_1d : jnp.ndarray
        1D k values for interpolation
    f_1d : jnp.ndarray
        1D function values
    N : int
        Grid size
    Lbox : float
        Box size
    partype : str, optional
        Parallelization type
    host_id : int
        Host ID
    ngpus : int
        Number of GPUs
    start : int
        Start index for slicing
    end : int
        End index for slicing
    """
    # Generate k-space grids
    kx = _k_axis(N, Lbox, r=False, slab_axis=False, partype=partype, 
                host_id=host_id, ngpus=ngpus, start=start, end=end)
    
    ky = _k_axis(N, Lbox, r=False, slab_axis=True, partype=partype,
                host_id=host_id, ngpus=ngpus, start=start, end=end)
    
    kz = _k_axis(N, Lbox, r=True, slab_axis=False, partype=partype,
                host_id=host_id, ngpus=ngpus, start=start, end=end)
    
    # Compute k magnitude grid
    interp_fcn = jnp.sqrt(_k_square(kx, ky, kz)).ravel()
    del kx, ky, kz
    gc.collect()
    
    # Interpolate with extrapolation
    interp_fcn = jnp.interp(interp_fcn, k_1d, f_1d, left='extrapolate', right='extrapolate')
    
    # Reshape to local complex shape
    cshape_local = _get_cshape_local(N, partype, ngpus, start, end)
    return jnp.reshape(interp_fcn, cshape_local).astype(jnp.float32)


def _k_axis(N: int, Lbox: float, r: bool = False, slab_axis: bool = False,
           partype: Optional[str] = None, host_id: int = 0, ngpus: int = 1,
           start: int = 0, end: int = None) -> jnp.ndarray:
    """
    Generate k-axis values for FFT grids.
    
    Parameters:
    -----------
    N : int
        Grid size
    Lbox : float
        Box size
    r : bool
        If True, use rfft frequencies (for real FFT)
    slab_axis : bool
        If True, return sliced axis for distributed computation
    partype : str, optional
        Parallelization type
    host_id : int
        Host ID
    ngpus : int
        Number of GPUs
    start : int
        Start index for slicing
    end : int
        End index for slicing
    """
    if end is None:
        end = N
        
    dk = 2*jnp.pi/Lbox
    
    if r:
        k_i = (jnp.fft.rfftfreq(N) * dk * N).astype(jnp.float32)
    else:
        k_i = (jnp.fft.fftfreq(N) * dk * N).astype(jnp.float32)
    
    if slab_axis:
        return (k_i[start:end]).astype(jnp.float32)
    
    return k_i


def _k_square(kx: jnp.ndarray, ky: jnp.ndarray, kz: jnp.ndarray) -> jnp.ndarray:
    """
    Compute k-squared magnitude on grid.
    
    Parameters:
    -----------
    kx : jnp.ndarray
        k values in x direction
    ky : jnp.ndarray
        k values in y direction
    kz : jnp.ndarray
        k values in z direction
    
    Returns:
    --------
    k2 : jnp.ndarray
        k-squared values on 3D grid
    """
    kxa, kya, kza = jnp.meshgrid(kx, ky, kz, indexing='ij')
    del kx, ky, kz
    gc.collect()
    
    k2 = (kxa**2 + kya**2 + kza**2).astype(jnp.float32)
    del kxa, kya, kza
    gc.collect()
    
    return k2


def _get_cshape_local(N: int, partype: Optional[str], ngpus: int, start: int, end: int) -> Tuple[int, int, int]:
    """
    Get the local complex array shape for distributed computation.
    
    Parameters:
    -----------
    N : int
        Grid size
    partype : str, optional
        Parallelization type
    ngpus : int
        Number of GPUs
    start : int
        Start index
    end : int
        End index
    
    Returns:
    --------
    shape : Tuple[int, int, int]
        Local complex array shape
    """
    if partype == 'jaxshard':
        return (N, end - start, N // 2 + 1)
    else:
        return (N, N, N // 2 + 1)
