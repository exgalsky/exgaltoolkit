"""
Pure JAX implementation of Lagrangian Perturbation Theory calculations.

Extracted from legacy cube.slpt() with clean interface and exact algorithm preservation.
"""
import jax
import jax.numpy as jnp
from typing import Tuple, Optional
from .fft_ops import rfft_with_normalization, irfft_with_normalization
from .k_grids import create_k_grids_rfft, compute_k_squared, handle_nyquist_modes


def compute_first_order_lpt(
    delta_field: jnp.ndarray,
    box_size: float,
    distributed_slice: slice = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute first-order LPT displacement fields.
    
    This function implements the exact algorithm from legacy cube.slpt():
    S^(1)_i(k) = i k_i δ(k) / k^2
    
    Parameters:
    -----------
    delta_field : jnp.ndarray
        Density contrast field δ(x)
    box_size : float
        Physical box size in Mpc/h
    distributed_slice : slice, optional
        Y-axis slice for distributed computing
        
    Returns:
    --------
    s1x, s1y, s1z : Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        First-order displacement fields
    """
    # Get grid parameters
    N = delta_field.shape[0]
    
    # Create k-space grids
    kx, ky, kz = create_k_grids_rfft(N, box_size, distributed_slice)
    k_squared = compute_k_squared(kx, ky, kz)
    
    # Handle division by k^2 (avoid k=0 singularity)
    k_squared_safe = jnp.where(k_squared == 0, 1.0, k_squared)
    
    # FFT density field to k-space
    delta_k = rfft_with_normalization(delta_field)
    
    # Reshape k-vectors for proper broadcasting with 3D FFT grid
    # Following legacy cube.slpt() approach
    kx_3d = kx[:, None, None]  # Shape: (N, 1, 1)
    ky_3d = ky[None, :, None]  # Shape: (1, N_local, 1)  
    kz_3d = kz[None, None, :]  # Shape: (1, 1, N//2+1)
    
    # Compute displacement fields in k-space
    # S^(1)_i(k) = i k_i δ(k) / k^2
    def compute_displacement_component(ki_3d):
        # Apply the LPT formula
        disp_k = 1j * ki_3d * delta_k / k_squared_safe
        
        # Set k=0 mode to zero (no monopole displacement)
        disp_k = handle_nyquist_modes(disp_k, k_squared)
        
        # Transform back to real space
        return irfft_with_normalization(disp_k, shape=delta_field.shape)
    
    # Compute all three components
    s1x = compute_displacement_component(kx_3d)
    s1y = compute_displacement_component(ky_3d)
    s1z = compute_displacement_component(kz_3d)
    
    return s1x, s1y, s1z


def compute_second_order_lpt(
    s1x: jnp.ndarray,
    s1y: jnp.ndarray, 
    s1z: jnp.ndarray,
    box_size: float,
    distributed_slice: slice = None
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute second-order LPT displacement fields.
    
    This function implements the exact algorithm from legacy cube.slpt():
    1. Compute δ^(2) = Σ[∂S^(1)_i/∂q_i ∂S^(1)_j/∂q_j - (∂S^(1)_i/∂q_j)²]
    2. Solve S^(2)_i(k) = i k_i δ^(2)(k) / k^2
    
    Parameters:
    -----------
    s1x, s1y, s1z : jnp.ndarray
        First-order displacement fields
    box_size : float
        Physical box size in Mpc/h
    distributed_slice : slice, optional
        Y-axis slice for distributed computing
        
    Returns:
    --------
    s2x, s2y, s2z : Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        Second-order displacement fields
    """
    # Get grid parameters
    N = s1x.shape[0]
    
    # Create k-space grids
    kx, ky, kz = create_k_grids_rfft(N, box_size, distributed_slice)
    k_squared = compute_k_squared(kx, ky, kz)
    k_squared_safe = jnp.where(k_squared == 0, 1.0, k_squared)
    
    # Reshape k-vectors for proper broadcasting with 3D FFT grid
    kx_3d = kx[:, None, None]  # Shape: (N, 1, 1)
    ky_3d = ky[None, :, None]  # Shape: (1, N_local, 1)  
    kz_3d = kz[None, None, :]  # Shape: (1, 1, N//2+1)
    
    # Transform first-order fields to k-space
    s1x_k = rfft_with_normalization(s1x)
    s1y_k = rfft_with_normalization(s1y)
    s1z_k = rfft_with_normalization(s1z)
    
    # Compute derivatives in k-space: ∂S_i/∂q_j = i k_j S_i(k)
    # This gives us the 9 components of the deformation tensor
    def compute_derivatives(field_k):
        """Compute all three spatial derivatives of a field."""
        dx = irfft_with_normalization(1j * kx_3d * field_k, shape=s1x.shape)
        dy = irfft_with_normalization(1j * ky_3d * field_k, shape=s1x.shape)
        dz = irfft_with_normalization(1j * kz_3d * field_k, shape=s1x.shape)
        return dx, dy, dz
    
    # Compute all derivatives
    ds1x_dx, ds1x_dy, ds1x_dz = compute_derivatives(s1x_k)
    ds1y_dx, ds1y_dy, ds1y_dz = compute_derivatives(s1y_k)
    ds1z_dx, ds1z_dy, ds1z_dz = compute_derivatives(s1z_k)
    
    # Compute second-order source term δ^(2)
    # δ^(2) = Σ[∂S_i/∂q_i ∂S_j/∂q_j - (∂S_i/∂q_j)²]
    delta2 = _compute_second_order_source_term(
        ds1x_dx, ds1x_dy, ds1x_dz,
        ds1y_dx, ds1y_dy, ds1y_dz,
        ds1z_dx, ds1z_dy, ds1z_dz
    )
    
    # Transform δ^(2) to k-space
    delta2_k = rfft_with_normalization(delta2)
    
    # Compute second-order displacements: S^(2)_i(k) = i k_i δ^(2)(k) / k^2
    def compute_second_order_component(ki_3d):
        disp_k = 1j * ki_3d * delta2_k / k_squared_safe
        disp_k = handle_nyquist_modes(disp_k, k_squared)
        return irfft_with_normalization(disp_k, shape=s1x.shape)
    
    s2x = compute_second_order_component(kx_3d)
    s2y = compute_second_order_component(ky_3d)
    s2z = compute_second_order_component(kz_3d)
    
    return s2x, s2y, s2z


def _compute_second_order_source_term(
    ds1x_dx, ds1x_dy, ds1x_dz,
    ds1y_dx, ds1y_dy, ds1y_dz,
    ds1z_dx, ds1z_dy, ds1z_dz
) -> jnp.ndarray:
    """
    Compute the second-order source term δ^(2).
    
    This implements the exact formula from the legacy code:
    δ^(2) = Σ[∂S_i/∂q_i ∂S_j/∂q_j - (∂S_i/∂q_j)²]
    
    The terms are:
    - Diagonal terms: ∂S_i/∂q_i ∂S_j/∂q_j (3 terms)
    - Off-diagonal terms: -(∂S_i/∂q_j)² (6 terms)
    """
    # Diagonal terms: trace products
    # (∂S_x/∂x)(∂S_y/∂y) + (∂S_x/∂x)(∂S_z/∂z) + (∂S_y/∂y)(∂S_z/∂z)
    diagonal_terms = (ds1x_dx * ds1y_dy + 
                     ds1x_dx * ds1z_dz + 
                     ds1y_dy * ds1z_dz)
    
    # Off-diagonal terms: negative sum of squares of shear components
    # -(∂S_x/∂y)² - (∂S_x/∂z)² - (∂S_y/∂x)² - (∂S_y/∂z)² - (∂S_z/∂x)² - (∂S_z/∂y)²
    off_diagonal_terms = -(ds1x_dy**2 + ds1x_dz**2 + 
                          ds1y_dx**2 + ds1y_dz**2 + 
                          ds1z_dx**2 + ds1z_dy**2)
    
    # Total second-order source
    delta2 = diagonal_terms + off_diagonal_terms
    
    return delta2


def compute_lpt_displacements(
    delta_field: jnp.ndarray,
    box_size: float,
    order: int = 2,
    distributed_slice: slice = None
) -> Tuple[jnp.ndarray, ...]:
    """
    Compute LPT displacement fields up to specified order.
    
    This is the main interface that combines first and second-order calculations.
    
    Parameters:
    -----------
    delta_field : jnp.ndarray
        Density contrast field
    box_size : float
        Physical box size
    order : int
        LPT order (1 or 2)
    distributed_slice : slice, optional
        Y-axis slice for distributed computing
        
    Returns:
    --------
    displacements : Tuple[jnp.ndarray, ...]
        Displacement fields: (s1x, s1y, s1z) for order=1
                            (s1x, s1y, s1z, s2x, s2y, s2z) for order=2
    """
    # Always compute first-order
    s1x, s1y, s1z = compute_first_order_lpt(delta_field, box_size, distributed_slice)
    
    if order == 1:
        return s1x, s1y, s1z
    
    elif order == 2:
        # Compute second-order
        s2x, s2y, s2z = compute_second_order_lpt(s1x, s1y, s1z, box_size, distributed_slice)
        return s1x, s1y, s1z, s2x, s2y, s2z
    
    else:
        raise ValueError(f"LPT order {order} not supported. Use 1 or 2.")
