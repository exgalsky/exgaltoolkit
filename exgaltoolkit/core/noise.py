"""
Pure JAX implementation of noise generation for cosmological simulations.

Provides efficient white noise generation using JAX's PRNG system.
"""
import jax
import jax.numpy as jnp
import jax.random as rnd
from typing import Tuple

# Default nsub value for noise subsequence generation
DEFAULT_NSUB = 1024**3

def generate_white_noise(
    grid_shape: Tuple[int, int, int],
    seed: int = 13579,
    dtype: jnp.dtype = jnp.float32,
    nsub: int = DEFAULT_NSUB
) -> jnp.ndarray:
    """
    Generate a white noise field for cosmological simulations.

    Parameters:
    -----------
    grid_shape : Tuple[int, int, int]
        Shape of the 3D grid (N, N, N). Must be cubic.
    seed : int
        Random seed for reproducible generation.
    dtype : jnp.dtype
        Data type for the noise field.
    nsub : int
        The nsub parameter controlling the size of intermediate random number blocks.

    Returns:
    --------
    noise : jnp.ndarray
        White noise field with shape grid_shape, transposed as (1,0,2)
        from an initial (N,N,N) C-ordered generation.
    """
    N = grid_shape[0]
    if not (grid_shape[0] == grid_shape[1] == grid_shape[2]):
        raise ValueError(
            "grid_shape must be (N, N, N) for this function."
        )

    _PRNGkey = rnd.PRNGKey(seed)

    # Parameters for noise generation
    gen_start_offset = 0
    gen_size = N**3

    # Generate subsequences of random numbers
    # Use integer division for sequence ID calculation
    start_seqID = gen_start_offset // nsub
    # The end offset for the last element is gen_start_offset + gen_size - 1
    # Handle gen_size = 0 case where gen_start_offset + gen_size - 1 < 0
    if gen_size == 0:
        end_seqID = start_seqID -1 # This will make seqIDs empty
    else:
        end_seqID = (gen_start_offset + gen_size - 1) // nsub


    all_subsequences = []
    if start_seqID <= end_seqID: # Proceed only if there are sequences to generate
        seqIDs = jnp.arange(start_seqID, end_seqID + 1, dtype=jnp.int32)
        
        # Generate keys for each sub-sequence
        sub_keys = jax.vmap(lambda key, val: rnd.fold_in(key, jnp.uint32(val)), in_axes=(None, 0), out_axes=0)(_PRNGkey, seqIDs)

        for i, seqID_val in enumerate(seqIDs): # seqID_val is an actual ID number
            current_key = sub_keys[i]
            
            subseq_full = rnd.normal(current_key, shape=(nsub,), dtype=dtype)
            
            current_block_abs_start = seqID_val * nsub
            
            slice_start_in_subseq = jnp.maximum(0, gen_start_offset - current_block_abs_start)
            # gen_start_offset + gen_size - 1 is the index of the last element needed from the stream
            slice_end_in_subseq = jnp.minimum(nsub - 1, (gen_start_offset + gen_size - 1) - current_block_abs_start)

            if slice_start_in_subseq <= slice_end_in_subseq:
                all_subsequences.append(subseq_full[slice_start_in_subseq : slice_end_in_subseq + 1])
    
    if not all_subsequences:
        noise_1d = jnp.array([], dtype=dtype)
    else:
        noise_1d = jnp.concatenate(all_subsequences)

    if noise_1d.shape[0] != gen_size:
        raise RuntimeError(
            f"Internal error: Generated noise_1d size {noise_1d.shape[0]} "
            f"does not match expected gen_size {gen_size}."
        )
    # End noise generation

    if N == 0: # Handles grid_shape (0,0,0) -> gen_size 0
        return jnp.empty((0,0,0), dtype=dtype)
        
    noise_3d_c_order = jnp.reshape(noise_1d, (N, N, N))
    noise_final = jnp.transpose(noise_3d_c_order, (1, 0, 2))
    
    return noise_final


def generate_distributed_noise(
    grid_shape: Tuple[int, int, int],
    slab_slice: slice, 
    seed: int = 13579,
    dtype: jnp.dtype = jnp.float32,
    nsub: int = DEFAULT_NSUB
) -> jnp.ndarray:
    """
    Generate a slab of a white noise field for distributed computation.

    Parameters:
    -----------
    grid_shape : Tuple[int, int, int]
        Shape of the full 3D grid (N, N, N). Must be cubic.
    slab_slice : slice
        The slice object (e.g., slice(start, end)) defining the range of
        indices along the second axis (axis 1 of the C-ordered N,N,N array
        before final transpose) for this distributed slab.
    seed : int
        Random seed for reproducible generation (must be the same for all slabs).
    dtype : jnp.dtype
        Data type for the noise field.
    nsub : int
        The nsub parameter controlling the size of intermediate random number blocks.

    Returns:
    --------
    noise_slab : jnp.ndarray
        White noise field for the specified slab. If the C-ordered slab data
        (before final transpose) has shape (slab_axis_size, N, N), the
        returned slab is transposed to (N, slab_axis_size, N).
    """
    N = grid_shape[0]
    if not (grid_shape[0] == grid_shape[1] == grid_shape[2]):
        raise ValueError(
            "grid_shape must be (N, N, N) for this function."
        )

    slab_axis_start_idx = slab_slice.start 
    slab_axis_end_idx = slab_slice.stop
    slab_axis_size = slab_axis_end_idx - slab_axis_start_idx

    if slab_axis_size < 0:
        raise ValueError("slab_slice results in a negative size.")
    if N == 0: # Full grid is (0,0,0)
         # slab_axis_size would be 0 if slab_slice is slice(0,0)
        return jnp.empty((0, slab_axis_size, 0), dtype=dtype)
    if slab_axis_size == 0:
        return jnp.empty((N, 0, N), dtype=dtype)

    _PRNGkey = rnd.PRNGKey(seed)

    gen_start_offset = slab_axis_start_idx * (N * N)
    gen_size = slab_axis_size * (N * N)

    # Generate subsequences of random numbers
    start_seqID = gen_start_offset // nsub
    if gen_size == 0: # Should be covered by slab_axis_size == 0 or N == 0, but as safety
        end_seqID = start_seqID - 1
    else:
        end_seqID = (gen_start_offset + gen_size - 1) // nsub

    all_subsequences = []
    if start_seqID <= end_seqID:
        seqIDs = jnp.arange(start_seqID, end_seqID + 1, dtype=jnp.int32)
        sub_keys = jax.vmap(lambda key, val: rnd.fold_in(key, jnp.uint32(val)), in_axes=(None, 0), out_axes=0)(_PRNGkey, seqIDs)
        
        for i, seqID_val in enumerate(seqIDs):
            current_key = sub_keys[i]
            subseq_full = rnd.normal(current_key, shape=(nsub,), dtype=dtype)
            
            current_block_abs_start = seqID_val * nsub
            
            slice_start_in_subseq = jnp.maximum(0, gen_start_offset - current_block_abs_start)
            slice_end_in_subseq = jnp.minimum(nsub - 1, (gen_start_offset + gen_size - 1) - current_block_abs_start)

            if slice_start_in_subseq <= slice_end_in_subseq:
                all_subsequences.append(subseq_full[slice_start_in_subseq : slice_end_in_subseq + 1])
    
    if not all_subsequences:
        noise_1d = jnp.array([], dtype=dtype)
    else:
        noise_1d = jnp.concatenate(all_subsequences)

    if noise_1d.shape[0] != gen_size:
        raise RuntimeError(
            f"Internal error: Generated noise_1d size {noise_1d.shape[0]} "
            f"does not match expected gen_size {gen_size} for slab "
            f"slice {slab_slice} (start_offset={gen_start_offset})."
        )
    # End noise generation
    
    # Reshape based on slab_axis_size, N, N for C-order
    # then transpose to (N, slab_axis_size, N)
    noise_3d_slab_c_order = jnp.reshape(noise_1d, (slab_axis_size, N, N))
    noise_final_slab = jnp.transpose(noise_3d_slab_c_order, (1, 0, 2))
    
    return noise_final_slab
