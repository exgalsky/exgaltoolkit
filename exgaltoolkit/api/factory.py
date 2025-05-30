"""
Factory for creating simulations with clean interfaces.
"""
from typing import Optional, Dict, Any
from ..core.config import (
    SimulationConfig, 
    CosmologicalParameters, 
    PowerSpectrum,
    GridConfiguration
)
from ..services.cosmology_service import PowerSpectrumService
from .simulation import MockGenerationSimulation

class SimulationFactory:
    """Factory for creating simulations."""
    
    @staticmethod
    def from_config(config: SimulationConfig) -> MockGenerationSimulation:
        """Create simulation from configuration object."""
        return MockGenerationSimulation(config)
    
    @staticmethod
    def create_mock_generation_simulation(
        cosmology_params: CosmologicalParameters,
        power_spectrum: PowerSpectrum,
        grid_config: GridConfiguration,
        **kwargs
    ) -> MockGenerationSimulation:
        """Create a mock generation simulation."""
        from ..core.config import SimulationParameters, ParallelConfig, OutputConfig
        
        # Extract additional parameters from kwargs
        sim_params = SimulationParameters(**{k: v for k, v in kwargs.items() 
                                           if hasattr(SimulationParameters, k)})
        parallel_config = ParallelConfig(**{k: v for k, v in kwargs.items() 
                                          if hasattr(ParallelConfig, k)})
        output_config = OutputConfig(**{k: v for k, v in kwargs.items() 
                                      if hasattr(OutputConfig, k)})
        
        config = SimulationConfig(
            cosmology=cosmology_params,
            grid=grid_config,
            simulation=sim_params,
            parallel=parallel_config,
            output=output_config,
            power_spectrum=power_spectrum
        )
        
        return MockGenerationSimulation(config)
    
    @staticmethod
    def from_legacy_kwargs(**kwargs) -> MockGenerationSimulation:
        """Create simulation from legacy kwargs (for backward compatibility)."""
        config = SimulationConfig.from_legacy_kwargs(**kwargs)
        return MockGenerationSimulation(config)
    
    @staticmethod
    def from_camb_results(camb_results, redshift: float = 0.0, **kwargs) -> MockGenerationSimulation:
        """Create simulation from CAMB results."""
        power_service = PowerSpectrumService.from_camb(camb_results, redshift)
        
        # Create config from kwargs
        config = SimulationConfig.from_legacy_kwargs(**kwargs)
        config.power_spectrum = power_service.power_spectrum
        
        return MockGenerationSimulation(config)
