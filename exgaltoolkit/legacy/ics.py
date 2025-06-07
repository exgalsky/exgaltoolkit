"""
Legacy ICs class wrapper for backward compatibility.
"""
from typing import Optional, Dict, Any
import warnings


class ICs:
    """Legacy ICs class wrapper for backward compatibility."""
    
    def __init__(self, sky, cube, cosmo, **kwargs):
        """Initialize ICs with legacy interface."""
        self.sky = sky
        self.cube = cube
        self.cosmo = cosmo
        self._kwargs = kwargs
        
        # Extract filename
        self.fname = kwargs.get('fname', 'default_ics')
    
    def writeics(self):
        """Write initial conditions."""
        try:
            # Try new implementation using output service
            from ..services.output_service import OutputManager
            from ..core.config import OutputConfig
            from ..core.data_models import ParticleData, SimulationResults
            
            # Create output configuration
            output_config = OutputConfig(
                base_path='output',
                format='binary',
                save_grids=True,
                save_particles=True
            )
            
            output_manager = OutputManager(output_config)
            
            # Create simulation results from cube data
            if hasattr(self.cube, 's1lpt') and self.cube.s1lpt is not None:
                # Extract particle data from cube
                particle_data = ParticleData(
                    positions=self.cube.s1lpt,  # This would need proper extraction
                    velocities=getattr(self.cube, 's2lpt', None)
                )
                
                results = SimulationResults(
                    particle_data=particle_data,
                    density_grid=getattr(self.cube, 'delta', None)
                )
                
                # Save using new output manager
                output_manager.save_simulation_results(results, self.fname)
                return
                
        except Exception as e:
            warnings.warn(f"New ICs writing failed, falling back to legacy: {e}")
        
        # Fall back to original implementation
        from ..mockgen.ics import ICs as OriginalICs
        
        original_ics = OriginalICs(self.sky, self.cube, self.cosmo, **self._kwargs)
        return original_ics.writeics()
