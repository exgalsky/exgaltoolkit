"""
Pure JAX implementation of transfer function application.

Core implementation of the noise to delta field transformation.
"""
import jax
import jax.numpy as jnp
import numpy as np
import gc
from typing import Dict, Any, Optional, Tuple

# JAX distributed computing imports
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental.multihost_utils import sync_global_devices

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
        print(f"⚡ DISTRIBUTED FFT: Using {ngpus} GPUs with JAX sharding for {direction} FFT")
        return _distributed_fft(x_np, direction, ngpus)
    else:
        if partype == 'jaxshard':
            print(f"⚠️  SERIAL FFT: partype='jaxshard' requested but ngpus={ngpus}, falling back to serial")
        else:
            print(f"🔧 SERIAL FFT: Using single-GPU JAX FFT for {direction} operation")
    
    # Standard single-GPU FFT operations
    if direction == 'r2c':
        return jnp.fft.rfftn(x_np)
    elif direction == 'c2r':
        return jnp.fft.irfftn(x_np)
    else:
        raise ValueError(f"Unknown direction: {direction}")


def _distributed_fft(x_np: jnp.ndarray, direction: str, ngpus: int) -> jnp.ndarray:
    """
    Distributed FFT implementation using JAX sharding.
    
    Implements Y-slab decomposition across multiple GPUs for large-scale FFT operations.
    
    Parameters:
    -----------
    x_np : jnp.ndarray
        Input array (local slab)
    direction : str
        FFT direction ('r2c' or 'c2r')
    ngpus : int
        Number of GPUs for distribution
        
    Returns:
    --------
    jnp.ndarray
        Local result after distributed FFT
    """
    if jax.process_count() != ngpus:
        raise ValueError(f"JAX process count ({jax.process_count()}) != ngpus ({ngpus})")
    
    # Calculate global shape from local slab
    local_shape = x_np.shape
    global_shape = (local_shape[0], local_shape[1] * ngpus, local_shape[2])
    
    print(f"Process {jax.process_index()}: Local shape {local_shape}, Global shape {global_shape}")
    
    # Create device mesh for sharding
    devices = mesh_utils.create_device_mesh((ngpus,))
    mesh = Mesh(devices, axis_names=('gpus',))
    
    with mesh:
        # Convert local array to JAX and create distributed array
        x_local = jax.device_put(x_np).block_until_ready()
        
        # Create sharded array from local slabs
        x_sharded = jax.make_array_from_single_device_arrays(
            global_shape,
            NamedSharding(mesh, P(None, "gpus")),  # Shard along Y-axis
            [x_local]
        ).block_until_ready()
        
        print(f"Process {jax.process_index()}: Sharded array created with sharding: {x_sharded.sharding}")
        
        # Set up sharding specifications
        shard_spec = NamedSharding(mesh, P(None, "gpus"))
        
        # Define the distributed FFT computation functions
        def distributed_rfft(x):
            # Z-axis FFT (real-to-complex)
            x = jnp.fft.rfft(x, axis=2)
            # XY-plane FFT
            x = jnp.fft.fftn(x, axes=[0, 1])
            return x
                
        def distributed_irfft(x):
            # XY-plane inverse FFT
            x = jnp.fft.ifftn(x, axes=[0, 1])
            # Z-axis inverse FFT (complex-to-real)
            x = jnp.fft.irfft(x, axis=2)
            return x
        
        # Execute distributed FFT with SPMD mode and explicit sharding
        sync_global_devices("Starting distributed FFT computation")
        
        with jax.spmd_mode('allow_all'):
            if direction == 'r2c':
                # JIT compile with explicit sharding
                rfft_jit = jax.jit(distributed_rfft, in_shardings=shard_spec, out_shardings=shard_spec)
                result_sharded = rfft_jit(x_sharded).block_until_ready()
            elif direction == 'c2r':
                # JIT compile with explicit sharding  
                irfft_jit = jax.jit(distributed_irfft, in_shardings=shard_spec, out_shardings=shard_spec)
                result_sharded = irfft_jit(x_sharded).block_until_ready()
            else:
                raise ValueError(f"Unsupported direction: {direction}")
                
        sync_global_devices("FFT computation complete")
        
        # Extract local result from distributed array
        local_result = result_sharded.addressable_data(0)
        
        # Validate that we got a local slab, not the global array
        if direction == 'r2c':
            # r2c: real input (128, 32, 128) -> complex output (128, 32, 65)
            expected_local_shape = (local_shape[0], local_shape[1], local_shape[2]//2 + 1)
        else:
            # c2r: complex input (128, 32, 65) -> real output (128, 32, 128)
            # For c2r, we need to calculate the expected real output size
            if local_shape[2] == local_shape[0]//2 + 1:  # Input is complex
                expected_local_shape = (local_shape[0], local_shape[1], (local_shape[2]-1)*2)
            else:  # Input is already real-shaped, expect same size
                expected_local_shape = local_shape
        
        # For distributed computation, check that Y dimension is local slab size
        expected_y_slab_size = global_shape[1] // ngpus
        if local_result.shape[1] != expected_y_slab_size:
            raise RuntimeError(f"DISTRIBUTED FFT FAILED: Got Y-dimension {local_result.shape[1]}, "
                             f"expected local slab Y-size {expected_y_slab_size}. "
                             f"This indicates data gathering occurred instead of distributed computation.")
        
        print(f"Process {jax.process_index()}: SUCCESS - Distributed FFT returned local slab {local_result.shape}")
        
        return local_result


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
