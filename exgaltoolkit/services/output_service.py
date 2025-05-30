"""
Output and file I/O services.
"""
import jax.numpy as jnp
import numpy as np
from typing import Optional
from ..core.data_models import ParticleData
from ..core.config import OutputConfig

class OutputManager:
    """Manages output operations."""
    
    def __init__(self, config: OutputConfig):
        self.config = config
    
    def write_initial_conditions(self, particles: ParticleData, 
                                filename: Optional[str] = None,
                                format: str = None) -> None:
        """Write initial conditions in specified format."""
        if filename is None:
            filename = self.config.base_filename
        if format is None:
            format = self.config.format
            
        formatter = OutputFormatters.get_formatter(format)
        full_path = f"{self.config.output_dir}/{filename}"
        formatter(particles, full_path)

class OutputFormatters:
    """Collection of output formatters."""
    
    @staticmethod
    def get_formatter(format_name: str):
        """Get formatter function for specified format."""
        formatters = {
            'nyx': OutputFormatters.nyx_format,
            'gadget': OutputFormatters.gadget_format
        }
        
        if format_name not in formatters:
            raise ValueError(f"Unknown output format: {format_name}")
        
        return formatters[format_name]
    
    @staticmethod
    def nyx_format(particles: ParticleData, filename: str) -> None:
        """Write in Nyx format."""
        # This preserves the exact logic from the original writenyx method
        x, y, z = particles.positions[0], particles.positions[1], particles.positions[2]
        vx, vy, vz = particles.velocities[0], particles.velocities[1], particles.velocities[2]
        mass = particles.masses
        
        fid = open(filename, 'wb')

        # Determine npart from data shape
        npart = x.size
        ndim = 3
        nx = 4

        mass_array = np.repeat(mass, npart) if np.isscalar(mass) else mass

        np.asarray([npart], dtype='long').tofile(fid)
        np.asarray([ndim], dtype='int32').tofile(fid)
        np.asarray([nx], dtype='int32').tofile(fid)
    
        (np.asarray([x.flatten(), y.flatten(), z.flatten(), mass_array, 
                    vx.flatten(), vy.flatten(), vz.flatten()], 
                   dtype='float32').T).tofile(fid)

        fid.close()
    
    @staticmethod 
    def gadget_format(particles: ParticleData, filename: str) -> None:
        """Write in Gadget format (placeholder - not implemented in original)."""
        raise NotImplementedError("Gadget format not implemented")
