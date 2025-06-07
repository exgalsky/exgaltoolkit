"""
LPT (Lagrangian Perturbation Theory) calculations.
"""
import jax.numpy as jnp
from typing import Tuple, Optional
from .operations import GridOperations

class LPTCalculator:
    """
    Calculator for LPT displacement fields.
    
    This wraps and enhances the existing LPT functionality from the Cube class.
    """
    
    def __init__(self, grid_ops: GridOperations, order: int = 2):
        """
        Initialize LPT calculator.
        
        Parameters:
        -----------
        grid_ops : GridOperations
            Grid operations instance
        order : int
            LPT order (1 or 2)
        """
        self.grid_ops = grid_ops
        self.order = order
    
    def compute_displacements(self, input_mode: str = 'noise') -> Tuple[jnp.ndarray, ...]:
        """
        Compute LPT displacement fields.
        
        Parameters:
        -----------
        input_mode : str
            Input field mode ('noise', 'delta')
            
        Returns:
        --------
        displacements : tuple
            (s1x, s1y, s1z, s2x, s2y, s2z) displacement fields
        """
        # Use the grid operations to compute displacements
        self.grid_ops.compute_lpt_displacements(order=self.order, input_mode=input_mode)
        
        # Return the displacement fields
        return self.grid_ops.get_displacement_fields()
    
    def compute_particle_positions(self, grid_positions: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Compute particle positions from LPT displacements.
        
        Parameters:
        -----------
        grid_positions : jnp.ndarray, optional
            Initial grid positions. If None, generates regular grid.
            
        Returns:
        --------
        positions : jnp.ndarray
            Final particle positions
        """
        if grid_positions is None:
            # Generate regular grid positions
            x1d = jnp.linspace(0, self.grid_ops.Lbox, self.grid_ops.N, endpoint=False)
            xx, yy, zz = jnp.meshgrid(x1d, x1d, x1d, indexing='ij')
            grid_positions = jnp.stack([xx, yy, zz], axis=-1)
        
        # Get displacement fields
        s1x, s1y, s1z, s2x, s2y, s2z = self.grid_ops.get_displacement_fields()
        
        if s1x is None:
            raise ValueError("No LPT displacements available. Run compute_displacements first.")
        
        # Apply displacements
        positions = grid_positions.copy()
        
        # First-order displacements
        positions = positions.at[..., 0].add(s1x)
        positions = positions.at[..., 1].add(s1y) 
        positions = positions.at[..., 2].add(s1z)
        
        # Second-order displacements if available
        if self.order > 1 and s2x is not None:
            positions = positions.at[..., 0].add(s2x)
            positions = positions.at[..., 1].add(s2y)
            positions = positions.at[..., 2].add(s2z)
        
        # Apply periodic boundary conditions
        positions = jnp.mod(positions, self.grid_ops.Lbox)
        
        return positions
    
    def compute_particle_velocities(self, cosmology_service, z_initial: float = 99.0) -> jnp.ndarray:
        """
        Compute particle velocities from LPT displacements.
        
        Parameters:
        -----------
        cosmology_service : CosmologyService
            Cosmology service for growth factors
        z_initial : float
            Initial redshift
            
        Returns:
        --------
        velocities : jnp.ndarray
            Particle velocities
        """
        # Get growth factors
        growth_factors = cosmology_service.get_growth_factors()
        
        # Interpolate growth rates at initial redshift
        f1 = jnp.interp(z_initial, growth_factors.z, growth_factors.f1)
        f2 = jnp.interp(z_initial, growth_factors.z, growth_factors.f2) if self.order > 1 else 0.0
        
        # Get displacement fields
        s1x, s1y, s1z, s2x, s2y, s2z = self.grid_ops.get_displacement_fields()
        
        if s1x is None:
            raise ValueError("No LPT displacements available. Run compute_displacements first.")
        
        # Compute velocities
        H_z = cosmology_service.get_hubble_parameter(z_initial)
        
        velocities = jnp.zeros_like(jnp.stack([s1x, s1y, s1z], axis=-1))
        
        # First-order velocities
        velocities = velocities.at[..., 0].set(H_z * f1 * s1x)
        velocities = velocities.at[..., 1].set(H_z * f1 * s1y)
        velocities = velocities.at[..., 2].set(H_z * f1 * s1z)
        
        # Second-order velocities if available
        if self.order > 1 and s2x is not None:
            velocities = velocities.at[..., 0].add(H_z * f2 * s2x)
            velocities = velocities.at[..., 1].add(H_z * f2 * s2y)
            velocities = velocities.at[..., 2].add(H_z * f2 * s2z)
        
        return velocities
