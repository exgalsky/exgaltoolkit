"""
Main workflow engine for orchestrating simulation execution.
"""
from typing import List, Optional
from ..core.config import SimulationConfig
from ..core.data_models import SimulationContext, SimulationResult, StepResult
from .simulation_steps import (
    SimulationStep, 
    NoiseGenerationStep,
    ConvolutionStep, 
    LPTDisplacementStep,
    InitialConditionsWriteStep
)

class SimulationWorkflow:
    """Orchestrates the simulation pipeline with clear step definitions."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.context = SimulationContext(config)
        self.steps = self._build_pipeline()
        self._step_map = {step.name: step for step in self.steps}
    
    def execute(self, until_step: Optional[str] = None) -> SimulationResult:
        """Execute workflow until specified step."""
        if until_step is None:
            until_step = self.config.simulation.final_step
        
        step_results = []
        
        # Handle special case for 'all' step
        if until_step == 'all':
            target_steps = self.steps
        else:
            target_steps = self._get_steps_until(until_step)
        
        try:
            for step in target_steps:
                # Check prerequisites
                if not step.validate_prerequisites(self.context):
                    result = StepResult(
                        success=False,
                        step_name=step.name,
                        message=f"Prerequisites not met for step {step.name}"
                    )
                    step_results.append(result)
                    return SimulationResult(
                        success=False,
                        final_step=step.name,
                        step_results=step_results,
                        context=self.context,
                        message=f"Failed at step {step.name}"
                    )
                
                # Execute step
                result = step.execute(self.context)
                step_results.append(result)
                
                if not result.success:
                    return SimulationResult(
                        success=False,
                        final_step=step.name,
                        step_results=step_results,
                        context=self.context,
                        message=f"Failed at step {step.name}: {result.message}"
                    )
                
                # Check if this is our target step
                if step.name == until_step:
                    break
            
            return SimulationResult(
                success=True,
                final_step=until_step,
                step_results=step_results,
                context=self.context,
                message="Simulation completed successfully"
            )
            
        except Exception as e:
            return SimulationResult(
                success=False,
                final_step=until_step,
                step_results=step_results,
                context=self.context,
                message=f"Simulation failed with exception: {str(e)}"
            )
    
    def _build_pipeline(self) -> List[SimulationStep]:
        """Build the ordered list of simulation steps."""
        steps = [
            NoiseGenerationStep(),
            ConvolutionStep(),
            LPTDisplacementStep(),
            InitialConditionsWriteStep()
        ]
        return steps
    
    def _get_steps_until(self, step_name: str) -> List[SimulationStep]:
        """Get all steps up to and including the named step."""
        target_steps = []
        for step in self.steps:
            target_steps.append(step)
            if step.name == step_name:
                break
        
        if not target_steps or target_steps[-1].name != step_name:
            raise ValueError(f"Unknown step: {step_name}")
        
        return target_steps
