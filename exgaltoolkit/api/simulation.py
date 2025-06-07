"""
High-level simulation classes providing clean interfaces.
"""
from typing import Optional, Dict, Any, Tuple
import numpy as np

from ..core.config import SimulationConfig
from ..core.data_models import SimulationResults, SimulationResult
from ..core.exceptions import SimulationError
from ..workflow.workflow_engine import WorkflowEngine
from ..services.cosmology_service import CosmologyService, PowerSpectrumService
from ..services.grid_service import GridManager, NoiseGenerator, FFTProcessor
from ..services.lpt_service import LPTCalculator
from ..services.output_service import OutputManager


class MockGenerationSimulation:
    """High-level interface for mock generation simulations."""
    
    def __init__(self, config: SimulationConfig):
        """Initialize simulation with configuration."""
        self.config = config
        self._workflow_engine = None
        self._results = None
        
    def _initialize_services(self):
        """Initialize all required services."""
        # Create services
        cosmology_service = CosmologyService(self.config.cosmology)
        power_spectrum_service = PowerSpectrumService(
            self.config.power_spectrum
        )
        grid_manager = GridManager(self.config.grid)
        noise_generator = NoiseGenerator(grid_manager)
        fft_processor = FFTProcessor(grid_manager)
        lpt_calculator = LPTCalculator(grid_manager)
        output_manager = OutputManager(self.config.output)
        
        # Create workflow engine
        self._workflow_engine = WorkflowEngine(self.config)
        
        # Store services in context for steps to access
        self._workflow_engine.context.cosmology_service = cosmology_service
        self._workflow_engine.context.power_spectrum_service = power_spectrum_service
        self._workflow_engine.context.grid_manager = grid_manager
        self._workflow_engine.context.noise_generator = noise_generator
        self._workflow_engine.context.fft_processor = fft_processor
        self._workflow_engine.context.lpt_calculator = lpt_calculator
        self._workflow_engine.context.output_manager = output_manager
    
    def run(self) -> SimulationResult:
        """Run the complete simulation."""
        if self._workflow_engine is None:
            self._initialize_services()
            
        try:
            self._results = self._workflow_engine.execute()
            return self._results
        except Exception as e:
            raise SimulationError(f"Simulation failed: {str(e)}") from e
    
    def get_results(self) -> Optional[SimulationResult]:
        """Get simulation results if available."""
        return self._results
    
    def get_grid_data(self) -> Optional[np.ndarray]:
        """Get density grid data if available."""
        if self._results and self._results.success:
            # Get density data from workflow context
            # Check for the last iteration density field
            config = self._results.context.config
            last_iteration = config.simulation.n_iterations - 1
            density_key = f'delta_field_{last_iteration}'
            
            if self._results.context.has(density_key):
                return np.asarray(self._results.context.get(density_key))
            
            # Fallback to generic 'delta' key
            if self._results.context.has('delta'):
                return np.asarray(self._results.context.get('delta'))
        
        return None
    
    def get_particle_data(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get particle positions and velocities if available."""
        if self._results and self._results.success:
            # Get particle data from workflow context
            particles = self._results.context.get('particles')
            if particles is not None:
                return np.asarray(particles.positions), np.asarray(particles.velocities)
        
        return None, None
    
    def save_results(self, output_path: Optional[str] = None) -> None:
        """Save simulation results to file."""
        if self._results is None:
            raise SimulationError("No results to save. Run simulation first.")
            
        if self._workflow_engine is None:
            raise SimulationError("Workflow engine not initialized.")
            
        # Use output manager to save results
        if output_path:
            # Temporarily override output path
            original_path = self.config.output.base_path
            self.config.output.base_path = output_path
            
        try:
            self._workflow_engine.output_manager.save_simulation_results(self._results)
        finally:
            if output_path:
                # Restore original path
                self.config.output.base_path = original_path
